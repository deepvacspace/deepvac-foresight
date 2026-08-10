#!/usr/bin/env python3
"""Two-phase live chamber run: GP-chosen far-band PID, then GRU+MPC close in.

Chamber I/O (temp_ref jobs, PID writes with readback verification, preconditioning,
CSV artifacts) follows optimization/tocero_3band.py. The scheduling is different:

  Phase 1 -- "gp_far", while abs(temp - target) > --far-band (10 deg by default).
    One PID triplet, chosen once by the far-band Gaussian Process from
    optimization/band_bo_gp.py, held for the whole approach. There is only this
    one band, so the GP's far-band cost (approach MAE + time to reach the band
    edge) is exactly the objective this phase is judged on.

  Phase 2 -- "gru_mpc", from the first time the chamber is inside --far-band.
    The GRU plant model plus receding-horizon MPC re-infer the PID every
    --mpc-hold-s seconds (5 by default) from the live state, and each decision is
    written to the chamber.

The MPC uses deepvac.mpc_batch, which scores the whole candidate population in
one batched GRU forward. The scalar deepvac.mpc path costs ~281 s per default
CEM decision on CPU, which cannot hold a 5 s cadence; batched it is ~3 s.

SAFETY: this writes PID values and temp_ref jobs to a real chamber over TCP and
has no independent interlock or emergency stop. --dry-run exercises the whole
schedule without writing anything.

Example:

    python -m optimization.tocero_gp_mpc \
        --duration-s 1800 --target-temp 0 --far-band 10 --mpc-hold-s 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac import mpc as _mpc  # noqa: E402
from deepvac import mpc_batch  # noqa: E402
from deepvac.artifacts import append_row_csv, append_rows_csv, history_run_file, make_run_id, save_json  # noqa: E402
from deepvac.metrics import append_mae_column, compute_tail_cost  # noqa: E402
from deepvac.pid import parse_bounds, read_pid_from_tcp  # noqa: E402
from gru.gru_common import DEFAULT_CHECKPOINT, DEFAULT_FEATURE_NAMES, ChamberPID, CodesysDiff, load_model  # noqa: E402
from gru.mpc_gru import horizon_cost  # noqa: E402
from optimization import band_bo_gp as band_bo  # noqa: E402
from tcp.tcp_common import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
)

PIDTriplet = Tuple[int, int, int]

PHASE_GP = "gp_far"
PHASE_MPC = "gru_mpc"

OUTPUT_DIR = Path(__file__).with_name("output")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Live chamber run: GP far-band PID, then GRU+MPC replanning inside the band.",
    )

    # --dt-s, --target-temp, --duration-s, PID bounds, CEM/optimizer settings,
    # history-seeding and --max-abs-temp all come from the shared MPC flag set.
    _mpc.add_common_mpc_args(ap)

    # Horizon/hold and cost weights are mpc_gru.py's design, restated here because
    # mpc_gru.build_arg_parser() also owns --checkpoint/--output-dir defaults we replace.
    ap.add_argument("--mpc-horizon-s", type=float, default=80.0,
                    help="Future horizon optimized at every MPC decision.")
    ap.add_argument("--mpc-hold-s", type=float, default=5.0,
                    help="Seconds between MPC re-inferences once inside the far band.")
    ap.add_argument("--w-overshoot-max", type=float, default=80.0)
    ap.add_argument("--w-abs-error", type=float, default=2.0)
    ap.add_argument("--w-motion", type=float, default=20.0)
    ap.add_argument("--motion-error-scale", type=float, default=5.0,
                    help="Larger values make the controller start braking earlier.")
    ap.add_argument("--w-near-std", type=float, default=1.0)

    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--candidate-table", default="",
                    help="Prebuilt CSV from gru/mpc_build.py. Empty disables history seeding.")
    ap.add_argument(
        "--mpc-time-budget-s",
        type=float,
        default=None,
        help="Stop CEM refinement after this many seconds. Default: 60%% of --mpc-hold-s.",
    )
    ap.add_argument(
        "--mpc-min-warmup-samples",
        type=int,
        default=None,
        help=(
            "Real samples required before MPC may take over, so the GRU window is not "
            "reading synthetic warmup rows. Default: the checkpoint's window_steps."
        ),
    )

    # --- Phase boundary --------------------------------------------------------
    ap.add_argument("--far-band", type=float, default=10.0,
                    help="Absolute error in degrees that separates the GP phase from the MPC phase.")

    # --- Far-band Gaussian Process --------------------------------------------
    ap.add_argument("--gp-history-root", default="run_history",
                    help="Run history the far-band GP is fitted from.")
    ap.add_argument("--gp-acquisition", choices=["ei", "lcb", "eig"], default="lcb")
    ap.add_argument("--gp-lcb-kappa", type=float, default=0.3)
    ap.add_argument("--gp-xi", type=float, default=0.01)
    ap.add_argument("--gp-min-band-samples", type=int, default=8)
    ap.add_argument("--gp-n-candidates", type=int, default=100000)
    ap.add_argument("--gp-n-restarts-optimizer", type=int, default=5)
    ap.add_argument("--gp-seed", type=int, default=0)
    ap.add_argument("--gp-kp-bounds", default="1,20")
    ap.add_argument("--gp-ki-bounds", default="1,1000")
    ap.add_argument("--gp-kd-bounds", default="1,50")
    ap.add_argument("--gp-mae-weight", type=float, default=0.5)
    ap.add_argument("--gp-time-weight", type=float, default=0.03)
    ap.add_argument(
        "--far-pid",
        default=None,
        help="Skip the GP fit and use this 'kp,ki,kd' for the far band instead.",
    )

    # --- Chamber run shape (tocero_3band.py) -----------------------------------
    ap.add_argument("--heatup-temp-ref", type=float, default=25.0, help="Pre-test heatup temp_ref")
    ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0, help="Seconds at heatup temp_ref")
    ap.add_argument("--post-heatup-cooldown", type=float, default=3.0 * 60.0,
                    help="Seconds after heatup before the test begins")
    ap.add_argument("--condition-initial", action="store_true",
                    help="Run heatup and post-heatup cooldown before the test")
    ap.add_argument("--skip-preconditioning", action="store_true",
                    help="Skip heatup and cooldown entirely")
    ap.add_argument("--progress-every", type=float, default=60.0)
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)
    ap.add_argument("--val", action="store_true", help="Save the run with a val_* id.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the full schedule without writing temp_ref or PID to the chamber.")

    ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4]")
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)

    ap.add_argument("--history-root", default="run_history_gp_mpc")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--decisions-csv", default="mpc_decisions.csv")
    ap.add_argument("--all-runs-csv", default=str(OUTPUT_DIR / "gp_mpc_all_runs.csv"))

    return ap


# -----------------------------------------------------------------------------
# Phase 1: far-band PID from the Gaussian Process
# -----------------------------------------------------------------------------


def build_gp_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """A band_bo_gp namespace configured for a single far band at --far-band.

    band_bo_gp's own parser is the source of truth for every knob it reads; this
    only overrides the ones this script exposes. --far-threshold is pinned to
    --far-band so the GP scores exactly the region phase 1 controls.
    """
    gp_args = band_bo.build_arg_parser(band_mode=3).parse_args([])

    gp_args.history_root = args.gp_history_root
    gp_args.far_threshold = float(args.far_band)
    # near_threshold only splits the mid/near bands, which this script never uses,
    # but it must stay below far_threshold for classify_bands to be well formed.
    gp_args.near_threshold = min(float(gp_args.near_threshold), float(args.far_band) / 2.0)
    gp_args.min_band_samples = int(args.gp_min_band_samples)
    gp_args.acquisition = args.gp_acquisition
    gp_args.lcb_kappa = float(args.gp_lcb_kappa)
    gp_args.xi = float(args.gp_xi)
    gp_args.n_candidates = int(args.gp_n_candidates)
    gp_args.n_restarts_optimizer = int(args.gp_n_restarts_optimizer)
    gp_args.seed = int(args.gp_seed)
    gp_args.far_kp_bounds = args.gp_kp_bounds
    gp_args.far_ki_bounds = args.gp_ki_bounds
    gp_args.far_kd_bounds = args.gp_kd_bounds
    gp_args.far_mae_weight = float(args.gp_mae_weight)
    gp_args.far_time_weight = float(args.gp_time_weight)
    gp_args.suggest_target_temp = float(args.target_temp)
    gp_args.top_k = 1

    return gp_args


def plan_far_pid(args: argparse.Namespace, start_temp: Optional[float]) -> Tuple[PIDTriplet, str, Dict[str, object]]:
    """Fit the far-band GP over history and return its suggested triplet."""
    if args.far_pid:
        parts = [p.strip() for p in str(args.far_pid).split(",")]
        if len(parts) != 3:
            raise ValueError(f"--far-pid must be 'kp,ki,kd', got {args.far_pid!r}")
        triplet = (int(round(float(parts[0]))), int(round(float(parts[1]))), int(round(float(parts[2]))))
        return triplet, "far-pid-argument", {}

    gp_args = build_gp_namespace(args)
    if start_temp is not None:
        gp_args.suggest_start_temp = float(start_temp)

    print(f"[gp] fitting far-band model over {gp_args.history_root} (abs_error > {args.far_band:g})")
    training_table = band_bo.compute_all_run_metrics(gp_args)

    model = band_bo.fit_one_band_gp(
        training_table=training_table,
        band="far",
        n_restarts_optimizer=gp_args.n_restarts_optimizer,
    )
    if model is None:
        raise RuntimeError(
            f"No usable far-band training rows under {gp_args.history_root}. "
            f"Runs need at least {gp_args.min_band_samples} samples with "
            f"abs_error > {args.far_band:g} (--gp-min-band-samples), or pass --far-pid."
        )

    context = band_bo.resolve_suggestion_context(training_table, gp_args)
    suggestion = band_bo.suggest_for_band(
        model=model,
        bounds=band_bo.bounds_for_band(gp_args, "far"),
        start_temp=float(context["start_temp"]),
        target_temp=float(context["target_temp"]),
        n_candidates=gp_args.n_candidates,
        top_k=1,
        acquisition=gp_args.acquisition,
        xi=gp_args.xi,
        lcb_kappa=gp_args.lcb_kappa,
        seed=gp_args.seed,
    )

    lo_kp, hi_kp = parse_bounds(gp_args.far_kp_bounds)
    lo_ki, hi_ki = parse_bounds(gp_args.far_ki_bounds)
    lo_kd, hi_kd = parse_bounds(gp_args.far_kd_bounds)

    def to_int(value: float, lo: float, hi: float) -> int:
        return int(min(max(int(np.floor(float(value))), int(np.ceil(lo))), int(np.floor(hi))))

    triplet = (
        to_int(suggestion["kp"], lo_kp, hi_kp),
        to_int(suggestion["ki"], lo_ki, hi_ki),
        to_int(suggestion["kd"], lo_kd, hi_kd),
    )

    info = {
        "n_training_samples": int(model["n_samples"]),
        "best_observed_cost": float(model["best_cost"]),
        "best_observed_triplet": model["best_x"],
        "pred_cost": float(suggestion["pred_cost"]),
        "pred_std": float(suggestion["pred_std"]),
        "acquisition": gp_args.acquisition,
        "raw_suggestion": {k: float(suggestion[k]) for k in ("kp", "ki", "kd")},
        "suggestion_context": context,
        "history_root": str(Path(gp_args.history_root).resolve()),
    }
    print(
        f"[gp] far-band suggestion kp={triplet[0]} ki={triplet[1]} kd={triplet[2]} "
        f"(pred_cost={info['pred_cost']:.4f} +/- {info['pred_std']:.4f}, "
        f"n={info['n_training_samples']} runs)"
    )
    return triplet, f"gp_far_{gp_args.acquisition}", info


# -----------------------------------------------------------------------------
# Chamber I/O
# -----------------------------------------------------------------------------


def read_snapshot(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    """One state read with retries. Returns None once every attempt has failed."""
    last_exc: Optional[Exception] = None
    for _ in range(max(1, args.read_retries + 1)):
        try:
            return request_temperature_states(
                host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout
            )
        except Exception as exc:
            last_exc = exc
            if args.read_retry_delay_s > 0:
                time.sleep(args.read_retry_delay_s)
    if last_exc is not None:
        print(f"[read] failed after {args.read_retries + 1} attempts: {last_exc}")
    return None


def write_pid(
    args: argparse.Namespace,
    *,
    label: str,
    run_id: str,
    pid: PIDTriplet,
    events: List[Dict[str, object]],
) -> PIDTriplet:
    if args.dry_run:
        print(f"[run {run_id}] DRY-RUN PID update ({label}): kp={pid[0]}, ki={pid[1]}, kd={pid[2]}")
        events.append({"event": label, "kp": pid[0], "ki": pid[1], "kd": pid[2], "timestamp": time.time()})
        return pid
    return apply_pid_update(
        label=label, run_id=run_id, row=args.pid_row, pid=pid, args=args, events=events
    )


def publish_job(args: argparse.Namespace, *, temp_ref: float, duration_s: float) -> None:
    if args.dry_run:
        print(f"[dry-run] would publish temp_ref={temp_ref:.3f} for {duration_s:.1f}s")
        return
    publish_temp_ref_job(
        temp_ref=float(temp_ref), duration_s=duration_s,
        host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout,
    )


def precondition(args: argparse.Namespace, run_id: str) -> None:
    if args.skip_preconditioning or not args.condition_initial:
        print(f"[run {run_id}] skipping preconditioning and post-heatup cooldown")
        return

    print(
        f"[run {run_id}] preconditioning: temp_ref={args.heatup_temp_ref:.3f} "
        f"for {args.heatup_duration:.1f}s (no logging)"
    )
    publish_job(args, temp_ref=args.heatup_temp_ref, duration_s=args.heatup_duration)
    time.sleep(args.heatup_duration)

    if args.post_heatup_cooldown > 0:
        print(f"[run {run_id}] post-heatup cooldown for {args.post_heatup_cooldown:.1f}s (no logging)")
        time.sleep(args.post_heatup_cooldown)


# -----------------------------------------------------------------------------
# Live GRU state tracking
# -----------------------------------------------------------------------------


class LiveWindow:
    """Rolling GRU feature window fed from real chamber telemetry.

    Seeded with synthetic warmup rows so a decision is structurally possible from
    the first sample; --mpc-min-warmup-samples is what actually stops MPC from
    acting before the window holds real data.

    The PID integrator is re-synced from the chamber's reported temp_u_i every
    sample rather than simulated forward, so rollouts start from the controller's
    true integral state. The derivative filter is only approximated: it is updated
    once per --dt-s sample, while the chamber runs it at --pid-period-s.
    """

    def __init__(self, args: argparse.Namespace, feature_names: List[str], window_steps: int, temp: float) -> None:
        self.args = args
        self.feature_names = feature_names
        self.window_steps = window_steps
        self.feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)

        self.window = _mpc.initialize_feature_window(
            feature_names, window_steps, temp, float(args.target_temp), float(args.dt_s),
            float(args.kp_init), float(args.ki_init), float(args.kd_init),
        )
        self.pid = ChamberPID(u_min=float(args.u_min), u_max=float(args.u_max),
                              pid_i_reverse_mul=float(args.pid_i_reverse_mul))
        self.diff = CodesysDiff()
        self.diff.prev_value = float(temp)
        self.temp = float(temp)
        self.previous_temp = float(temp)
        self.real_samples = 0

    def update(self, snap: Dict[str, float], pid: PIDTriplet, elapsed_s: float) -> None:
        temp = float(snap["temp"])
        row = _mpc.make_feature_row(
            self.feature_names,
            temp=temp,
            temp_ref=float(snap["temp_ref"]),
            previous_temp=self.temp,
            dt_s=float(self.args.dt_s),
            u=float(snap["temp_u"]),
            u_p=float(snap["temp_u_p"]),
            u_i=float(snap["temp_u_i"]),
            u_d=float(snap["temp_u_d"]),
            kp=float(pid[0]), ki=float(pid[1]), kd=float(pid[2]),
        )
        self.window = np.roll(self.window, shift=-1, axis=0)
        self.window[-1, :] = row

        self.diff.update(temp)
        self.pid.i_part = float(snap["temp_u_i"]) / self.feature_scale
        self.pid.p_part = float(snap["temp_u_p"]) / self.feature_scale
        self.pid.d_part = float(snap["temp_u_d"]) / self.feature_scale

        self.previous_temp = self.temp
        self.temp = temp
        self.elapsed_s = elapsed_s
        self.real_samples += 1

    def sim_state(self, pid: PIDTriplet, elapsed_s: float) -> _mpc.SimState:
        return _mpc.SimState(
            elapsed_s=float(elapsed_s),
            temp=self.temp,
            previous_temp=self.previous_temp,
            feature_window=self.window,
            pid=self.pid,
            diff=self.diff,
            kp=float(pid[0]), ki=float(pid[1]), kd=float(pid[2]),
        )


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, object]:
    run_id = make_run_id(prefix="val" if args.val else "gpmpc")
    target_temp = float(args.target_temp)
    far_band = float(args.far_band)

    print(f"[run {run_id}] starting {'DRY-RUN ' if args.dry_run else ''}two-phase test")
    precondition(args, run_id)

    first = read_snapshot(args)
    if first is None:
        raise RuntimeError("Could not read chamber state before the test started")
    start_temp = float(first["temp"])
    args.start_temp = start_temp
    print(f"[run {run_id}] start_temp={start_temp:.3f} target={target_temp:.3f} far_band={far_band:g}")

    far_pid, far_source, gp_info = plan_far_pid(args, start_temp)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))
    candidate_table = _mpc.load_candidate_table(args)
    warmup_needed = int(args.mpc_min_warmup_samples if args.mpc_min_warmup_samples is not None else window_steps)
    time_budget = float(args.mpc_time_budget_s) if args.mpc_time_budget_s is not None else 0.6 * float(args.mpc_hold_s)

    print(f"[mpc] checkpoint={args.checkpoint} device={device} window_steps={window_steps}")
    print(
        f"[mpc] horizon={args.mpc_horizon_s:g}s hold={args.mpc_hold_s:g}s "
        f"optimizer={args.optimizer} population={args.cem_population} "
        f"iterations={args.cem_iterations} time_budget={time_budget:.1f}s"
    )
    print(f"[mpc] warmup before MPC may take over: {warmup_needed} real samples "
          f"({warmup_needed * float(args.dt_s):.0f}s)")
    print(
        f"[mpc] PID bounds kp=({args.kp_min:g},{args.kp_max:g}) "
        f"ki=({args.ki_min:g},{args.ki_max:g}) kd=({args.kd_min:g},{args.kd_max:g}) "
        f"| GP far-band bounds kp={args.gp_kp_bounds} ki={args.gp_ki_bounds} kd={args.gp_kd_bounds}"
    )

    # Decisions can only fire on a sampling tick, so a hold that is not a whole
    # number of samples rounds up in practice.
    hold_in_samples = float(args.mpc_hold_s) / float(args.dt_s)
    if abs(hold_in_samples - round(hold_in_samples)) > 1e-9:
        print(
            f"[mpc] NOTE --mpc-hold-s {args.mpc_hold_s:g}s is not a multiple of --dt-s "
            f"{args.dt_s:g}s; decisions will land on the next sample, averaging "
            f"{args.mpc_hold_s:g}s but alternating between "
            f"{np.floor(hold_in_samples) * args.dt_s:g}s and {np.ceil(hold_in_samples) * args.dt_s:g}s."
        )

    tcp_pid_before = read_pid_from_tcp(row=args.pid_row, args=args)
    print(
        f"[run {run_id}] PID before (TCP): kp={tcp_pid_before['kp']:.3f}, "
        f"ki={tcp_pid_before['ki']:.3f}, kd={tcp_pid_before['kd']:.3f}"
    )

    # Prime the TCP state-name cache before the job starts (see tocero_3band.py).
    _ = read_snapshot(args)

    publish_job(args, temp_ref=target_temp, duration_s=float(args.duration_s))
    print(f"[run {run_id}] published temp_ref job for {args.duration_s:.1f}s")

    pid_events: List[Dict[str, object]] = []
    current_pid = write_pid(args, label=f"{PHASE_GP}/start", run_id=run_id, pid=far_pid, events=pid_events)

    live = LiveWindow(args, feature_names, window_steps, start_temp)
    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, object]] = []
    decisions: List[Dict[str, object]] = []
    phase = PHASE_GP
    phase_switch_elapsed: Optional[float] = None
    consecutive_failures = 0
    abs_error_sum = 0.0
    overruns = 0

    dt_s = float(args.dt_s)
    t0 = time.time()
    next_sample = t0
    next_decision = float("inf")
    next_progress = t0 + args.progress_every if args.progress_every > 0 else float("inf")

    print(f"[run {run_id}] sampling every {dt_s:.3f}s for {args.duration_s:.1f}s")

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= float(args.duration_s):
            break

        snap = read_snapshot(args)
        if snap is None:
            consecutive_failures += 1
            print(f"[run {run_id}] read failed ({consecutive_failures}/{args.max_consecutive_failures})")
            if consecutive_failures >= args.max_consecutive_failures:
                raise RuntimeError(f"Too many consecutive state read failures ({consecutive_failures})")
            time.sleep(dt_s)
            continue
        consecutive_failures = 0

        live.update(snap, current_pid, elapsed)
        abs_err = abs(float(snap["temp"]) - target_temp)
        abs_error_sum += abs(float(snap["temp_ref"]) - float(snap["temp"]))

        if phase == PHASE_GP and abs_err <= far_band and live.real_samples >= warmup_needed:
            phase = PHASE_MPC
            phase_switch_elapsed = elapsed
            next_decision = now
            print(
                f"[run {run_id}] entering {PHASE_MPC} at t={elapsed:.1f}s "
                f"temp={snap['temp']:.3f} abs_error={abs_err:.3f}"
            )

        decision: Optional[Dict[str, float]] = None
        if phase == PHASE_MPC and now >= next_decision:
            decision = mpc_batch.optimize_pid_batched(
                state=live.sim_state(current_pid, elapsed),
                model=model, checkpoint=checkpoint, feature_names=feature_names,
                device=device, args=args, rng=rng, cost_fn=horizon_cost,
                candidate_table=candidate_table, decision_idx=len(decisions) + 1,
                time_budget_s=time_budget,
            )
            new_pid = (int(decision["kp"]), int(decision["ki"]), int(decision["kd"]))
            if bool(decision["changed"]) and new_pid != tuple(current_pid):
                current_pid = write_pid(
                    args, label=f"{PHASE_MPC}/decision_{len(decisions) + 1}",
                    run_id=run_id, pid=new_pid, events=pid_events,
                )
            decision_row = {
                "run_id": run_id, "decision_idx": len(decisions) + 1, "elapsed_s": elapsed,
                "temp": float(snap["temp"]), "abs_error": abs_err,
                "kp": new_pid[0], "ki": new_pid[1], "kd": new_pid[2],
                **{k: decision[k] for k in decision if k not in ("kp", "ki", "kd")},
            }
            decisions.append(decision_row)

            took = float(decision["decision_seconds"])
            if took > float(args.mpc_hold_s):
                overruns += 1
                print(
                    f"[run {run_id}] WARNING decision {len(decisions)} took {took:.2f}s "
                    f"> --mpc-hold-s {args.mpc_hold_s:g}s; cadence is slipping"
                )
            # Advance on a fixed schedule so compute time does not accumulate into
            # the interval. Only resync if a decision overran its whole slot.
            next_decision += float(args.mpc_hold_s)
            behind = time.time() - next_decision
            if behind > 0:
                next_decision += (int(behind // float(args.mpc_hold_s)) + 1) * float(args.mpc_hold_s)

        sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
        rows.append({
            "run_id": run_id, "timestamp": now, "elapsed_s": elapsed, "phase": phase,
            "kp": float(current_pid[0]), "ki": float(current_pid[1]), "kd": float(current_pid[2]),
            **snap, "sq_error": sq_error,
            "mpc_decision": 0 if decision is None else 1,
        })

        if now >= next_progress:
            print(
                f"[run {run_id}] phase={phase} samples={len(rows)} elapsed={elapsed:.1f}s "
                f"temp={snap['temp']:.3f} mae={abs_error_sum / max(1, len(rows)):.6f} "
                f"kp={current_pid[0]} ki={current_pid[1]} kd={current_pid[2]} "
                f"decisions={len(decisions)}"
            )
            next_progress += args.progress_every

        # Hold the sampling cadence; a long decision skips the samples it ate.
        next_sample += dt_s
        sleep_s = next_sample - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            missed = int((-sleep_s) // dt_s) + 1
            next_sample += missed * dt_s

    return finalize(args, run_id, rows, decisions, pid_events, far_pid, far_source, gp_info,
                    start_temp, phase_switch_elapsed, overruns)


def finalize(
    args: argparse.Namespace,
    run_id: str,
    rows: List[Dict[str, object]],
    decisions: List[Dict[str, object]],
    pid_events: List[Dict[str, object]],
    far_pid: PIDTriplet,
    far_source: str,
    gp_info: Dict[str, object],
    start_temp: float,
    phase_switch_elapsed: Optional[float],
    overruns: int,
) -> Dict[str, object]:
    df_samples = pd.DataFrame(rows)
    if df_samples.empty:
        raise RuntimeError("No TCP samples were collected during the run")

    df_samples = append_mae_column(df_samples)
    df_samples["start_temp"] = start_temp

    cost_info = compute_tail_cost(
        df_samples, entry_band=args.entry_band, overshoot_weight=args.overshoot_weight
    )
    df_mpc = df_samples[df_samples["phase"] == PHASE_MPC]

    run_summary: Dict[str, object] = {
        "run_id": run_id,
        "start_ts": float(df_samples["timestamp"].iloc[0]),
        "end_ts": float(df_samples["timestamp"].iloc[-1]),
        "duration_s": float(df_samples["timestamp"].iloc[-1] - df_samples["timestamp"].iloc[0]),
        "num_samples": int(len(df_samples)),
        "start_temp": start_temp,
        "temp_ref": float(df_samples["temp_ref"].iloc[0]),
        "far_band": float(args.far_band),
        "pid_source": far_source,
        "far_kp": int(far_pid[0]), "far_ki": int(far_pid[1]), "far_kd": int(far_pid[2]),
        "mpc_decisions": int(len(decisions)),
        "mpc_pid_changes": int(sum(1 for e in pid_events if str(e["event"]).startswith(PHASE_MPC))),
        "mpc_overruns": int(overruns),
        "phase_switch_s": None if phase_switch_elapsed is None else float(phase_switch_elapsed),
        "gp_samples": int(len(df_samples) - len(df_mpc)),
        "mpc_samples": int(len(df_mpc)),
        "mse": float(df_samples["sq_error"].mean()),
        "mae": float(df_samples["mae"].mean()),
        "mpc_mae": None if df_mpc.empty else float(df_mpc["mae"].mean()),
        "cost": float(cost_info["cost"]),
        "tail_mae": None if cost_info["tail_mae"] is None else float(cost_info["tail_mae"]),
        "overshoot": None if cost_info["overshoot"] is None else float(cost_info["overshoot"]),
    }

    samples_out = history_run_file(run_id, str(Path(args.history_root) / args.samples_csv), args.history_root)
    runs_out = history_run_file(run_id, str(Path(args.history_root) / args.runs_csv), args.history_root)
    decisions_out = history_run_file(run_id, str(Path(args.history_root) / args.decisions_csv), args.history_root)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])
    if decisions:
        append_rows_csv(decisions_out, decisions)
    save_json(Path(decisions_out).with_name("gp_far_band.json"), {
        "run_id": run_id, "far_pid": list(far_pid), "source": far_source, **gp_info,
    })

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_row_csv(args.all_runs_csv, run_summary)

    print(f"\n[run {run_id}] === summary ===")
    for key in ("num_samples", "phase_switch_s", "gp_samples", "mpc_samples", "mpc_decisions",
                "mpc_pid_changes", "mpc_overruns", "mae", "mpc_mae", "cost", "tail_mae", "overshoot"):
        print(f"[run {run_id}] {key}={run_summary[key]}")
    print(f"[run {run_id}] samples_csv={samples_out}")
    print(f"[run {run_id}] runs_csv={runs_out}")
    if decisions:
        print(f"[run {run_id}] decisions_csv={decisions_out}")
    print(f"[run {run_id}] all_runs_csv={args.all_runs_csv}")

    return run_summary


def validate_args(args: argparse.Namespace) -> None:
    if args.dt_s <= 0:
        raise ValueError("--dt-s must be > 0")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    if args.far_band <= 0:
        raise ValueError("--far-band must be > 0")
    if args.mpc_hold_s <= 0:
        raise ValueError("--mpc-hold-s must be > 0")
    if args.mpc_horizon_s <= 0:
        raise ValueError("--mpc-horizon-s must be > 0")
    if args.mpc_time_budget_s is not None and args.mpc_time_budget_s <= 0:
        raise ValueError("--mpc-time-budget-s must be > 0")
    if args.heatup_duration < 0:
        raise ValueError("--heatup-duration must be >= 0")
    if args.post_heatup_cooldown < 0:
        raise ValueError("--post-heatup-cooldown must be >= 0")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    if args.read_retries < 0:
        raise ValueError("--read-retries must be >= 0")
    if args.read_retry_delay_s < 0:
        raise ValueError("--read-retry-delay-s must be >= 0")
    if args.max_consecutive_failures <= 0:
        raise ValueError("--max-consecutive-failures must be > 0")
    if not (0 <= args.pid_row <= 4):
        raise ValueError("--pid-row must be in range [0, 4]")
    if args.mpc_min_warmup_samples is not None and args.mpc_min_warmup_samples < 0:
        raise ValueError("--mpc-min-warmup-samples must be >= 0")
    for name in ("gp_kp_bounds", "gp_ki_bounds", "gp_kd_bounds"):
        parse_bounds(getattr(args, name))


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    torch.manual_seed(int(args.seed))
    run(args)


if __name__ == "__main__":
    main()
