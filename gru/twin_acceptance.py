#!/usr/bin/env python3
"""Measure how well a trained GRU checkpoint acts as a digital twin.

    --mode replay    Re-simulate whole logged runs closed-loop and compare against
                     what the chamber actually did. No hardware needed. Beyond
                     trajectory error this scores decision fidelity -- overshoot,
                     time to band, tail MAE and cost -- and the rank correlation
                     between predicted and measured cost, which is what determines
                     whether MPC picks the same PID the chamber would.

    --mode horizon   From many start points within logged runs, measure how
                     prediction error grows with lookahead: mae/p90/divergence at
                     each --lookahead-marks-s, both overall and by how close the
                     run already was to target when the prediction started. The
                     output is the trust horizon an MPC-style advisor should plan
                     within.

    --mode chamber   For each experiment, predict the whole run first, then drive
                     the real chamber over TCP with the same PID triplet held for
                     the entire run, then compare. One triplet per experiment.

Examples:

    python -m gru.twin_acceptance --mode replay --max-runs 40
    python -m gru.twin_acceptance --mode horizon --max-runs 40
    python -m gru.twin_acceptance --mode chamber --num-tests 6 --pid-sets "6,997,16;10,500,20"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from deepvac.artifacts import append_row_csv, append_rows_csv, history_run_file, make_run_id  # noqa: E402
from deepvac.metrics import append_mae_column, compute_tail_cost  # noqa: E402
from deepvac.mpc import SimState, make_feature_row, run_pid_substeps, step_state  # noqa: E402
from tcp.tcp_common import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
)

from gru.gru_common import ChamberPID, CodesysDiff, load_model, predict_delta_t1  # noqa: E402

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "validation_t1" / "gru_t1.pt"
DEFAULT_HISTORY_ROOT = ROOT / "optimization" / "run_history"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "twin_acceptance"

DEFAULT_PID_SETS = "6,997,16;10,500,20;15,750,10;20,200,40;4,900,8;12,300,25"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Score a GRU checkpoint as a whole-run digital twin of the chamber.",
    )

    ap.add_argument("--mode", choices=["replay", "horizon", "chamber"], default="replay")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--session-name", default=None, help="Output subfolder name. Default: generated id.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--window-steps", type=int, default=60, help="Fallback only if the checkpoint lacks it.")
    ap.add_argument("--no-plots", action="store_true")

    # --- Twin simulation -------------------------------------------------------
    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--pid-period-s", type=float, default=0.1)
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--max-abs-temp", type=float, default=100.0)

    # --- Scoring ---------------------------------------------------------------
    ap.add_argument("--near-band", type=float, default=2.0)
    ap.add_argument("--settle-band", type=float, default=0.5)
    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)
    ap.add_argument("--checkpoints-s", nargs="*", type=float,
                    default=[120.0, 300.0, 600.0, 900.0, 1200.0],
                    help="Elapsed times at which to report signed twin error.")
    ap.add_argument("--far-band", type=float, default=10.0,
                    help="abs_error above this counts as the far region; at/below --near-band is near.")

    # --- Replay mode -----------------------------------------------------------
    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument("--telemetry-names", nargs="+", default=["run_samples.csv"])
    ap.add_argument("--run-ids", nargs="*", default=None, help="Replay only these run folders.")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument(
        "--start-mode",
        choices=["cold", "warm", "both"],
        default="both",
        help=(
            "cold seeds the synthetic flat window MPC uses at deployment. warm seeds "
            "the run's first real rows. both scores each separately."
        ),
    )

    # --- Horizon mode ------------------------------------------------------------
    ap.add_argument("--horizon-stride-s", type=float, default=90.0,
                    help="Elapsed-time spacing between start points sampled within each run.")
    ap.add_argument("--max-horizon-s", type=float, default=600.0,
                    help="Longest lookahead simulated from each start point.")
    ap.add_argument("--lookahead-marks-s", nargs="*", type=float,
                    default=[10.0, 30.0, 60.0, 120.0, 300.0, 450.0, 600.0],
                    help="Lookaheads (seconds ahead of each start point) the report scores.")
    ap.add_argument("--trust-band", type=float, default=None,
                    help="p90 abs error threshold defining the trust horizon. Default: --near-band.")

    # --- Chamber mode ----------------------------------------------------------
    ap.add_argument("--num-tests", type=int, default=6)
    ap.add_argument("--pid-sets", default=DEFAULT_PID_SETS,
                    help="Triplets as 'kp,ki,kd;kp,ki,kd;...'. Use 'random' to sample within bounds.")
    ap.add_argument("--test-duration", type=float, default=20.0 * 60.0)
    ap.add_argument("--dt", type=float, default=2.0, help="Sample and twin step period, seconds.")
    ap.add_argument("--progress-every", type=float, default=60.0)
    ap.add_argument("--cooldown", type=float, default=3.0 * 60.0, help="Seconds between experiments.")
    ap.add_argument("--heatup-temp-ref", type=float, default=25.0)
    ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0)
    ap.add_argument("--post-heatup-cooldown", type=float, default=3.0 * 60.0)
    ap.add_argument("--skip-preconditioning", action="store_true")
    ap.add_argument("--condition-initial", action="store_true",
                    help="Precondition before the first experiment too.")
    ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4].")
    ap.add_argument("--kp-min", type=int, default=1)
    ap.add_argument("--kp-max", type=int, default=50)
    ap.add_argument("--ki-min", type=int, default=1)
    ap.add_argument("--ki-max", type=int, default=1000)
    ap.add_argument("--kd-min", type=int, default=1)
    ap.add_argument("--kd-max", type=int, default=200)

    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)
    ap.add_argument("--save-history-root", default=str(DEFAULT_HISTORY_ROOT),
                    help="Where measured runs are appended, in run_history layout.")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")

    return ap


# -----------------------------------------------------------------------------
# Trajectory scoring
# -----------------------------------------------------------------------------


def first_time_within(times: np.ndarray, temps: np.ndarray, target: float, band: float) -> float | None:
    """Elapsed time when the temperature first enters the band around target."""
    hit = np.where(np.abs(temps - target) <= band)[0]
    return float(times[int(hit[0])]) if len(hit) else None


def overshoot_of(temps: np.ndarray, start_temp: float, target: float) -> float:
    """Peak excursion past the target, in the direction of travel."""
    if target <= start_temp:
        return float(np.max(np.maximum(target - temps, 0.0)))
    return float(np.max(np.maximum(temps - target, 0.0)))


def describe_trajectory(
    times: np.ndarray,
    temps: np.ndarray,
    target: float,
    args: argparse.Namespace,
) -> dict[str, float | None]:
    """Behaviour summary of one trajectory, real or predicted."""
    start_temp = float(temps[0])
    tail_start = max(0.0, float(times[-1]) - float(args.tail_window_s))
    tail = temps[times >= tail_start]
    if len(tail) == 0:
        tail = temps

    return {
        "final_temp": float(temps[-1]),
        "overshoot": overshoot_of(temps, start_temp, target),
        "tail_mae": float(np.mean(np.abs(tail - target))),
        "tail_std": float(np.std(tail)),
        "time_to_near_s": first_time_within(times, temps, target, args.near_band),
        "time_to_settle_s": first_time_within(times, temps, target, args.settle_band),
    }


def trajectory_cost(times: np.ndarray, temps: np.ndarray, target: float,
                    args: argparse.Namespace) -> float | None:
    """Tail cost used to rank PID candidates, computed the same way for both curves.

    None when the trajectory never reaches the entry band, rather than the caller's
    large sentinel, so it stays out of the aggregates.
    """
    frame = pd.DataFrame({"timestamp": times, "temp": temps, "temp_ref": target})
    info = compute_tail_cost(frame, entry_band=args.entry_band,
                             overshoot_weight=args.overshoot_weight)
    return None if info["tail_mae"] is None else float(info["cost"])


def compare_trajectories(
    real_t: np.ndarray,
    real_temp: np.ndarray,
    pred_t: np.ndarray,
    pred_temp: np.ndarray,
    target: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Score a predicted trajectory against a measured one on a shared time base."""
    grid = real_t[(real_t >= pred_t[0]) & (real_t <= pred_t[-1])]
    if len(grid) < 2:
        raise ValueError("Predicted and measured trajectories do not overlap in time.")

    real_on_grid = np.interp(grid, real_t, real_temp)
    pred_on_grid = np.interp(grid, pred_t, pred_temp)
    err = pred_on_grid - real_on_grid

    aligned = pd.DataFrame({
        "elapsed_s": grid,
        "real_temp": real_on_grid,
        "pred_temp": pred_on_grid,
        "error_temp": err,
        "temp_ref": target,
    })

    real_desc = describe_trajectory(grid, real_on_grid, target, args)
    pred_desc = describe_trajectory(grid, pred_on_grid, target, args)

    metrics: dict[str, object] = {
        "n_steps": int(len(grid)),
        "duration_s": float(grid[-1] - grid[0]),
        "mae_temp": float(np.mean(np.abs(err))),
        "rmse_temp": float(np.sqrt(np.mean(err ** 2))),
        "bias_temp": float(np.mean(err)),
        "p90_abs_error_temp": float(np.percentile(np.abs(err), 90)),
        "max_abs_error_temp": float(np.max(np.abs(err))),
        "final_error_temp": float(err[-1]),
    }

    for mark in args.checkpoints_s:
        key = f"error_at_{int(mark)}s"
        metrics[key] = float(np.interp(mark, grid, err)) if grid[0] <= mark <= grid[-1] else None

    for name, real_value in real_desc.items():
        pred_value = pred_desc[name]
        metrics[f"real_{name}"] = real_value
        metrics[f"pred_{name}"] = pred_value
        if real_value is None or pred_value is None:
            metrics[f"err_{name}"] = None
        else:
            metrics[f"err_{name}"] = float(pred_value - real_value)

    real_cost = trajectory_cost(grid, real_on_grid, target, args)
    pred_cost = trajectory_cost(grid, pred_on_grid, target, args)
    metrics["real_cost"] = real_cost
    metrics["pred_cost"] = pred_cost
    metrics["err_cost"] = (None if real_cost is None or pred_cost is None
                           else float(pred_cost - real_cost))
    metrics["reached_band_real"] = real_cost is not None
    metrics["reached_band_pred"] = pred_cost is not None

    return metrics, aligned


def rank_values(values: Sequence[float]) -> np.ndarray:
    """Average ranks, ties shared."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def rank_correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman correlation between two score sequences."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True)
             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    ra = rank_values([p[0] for p in pairs])
    rb = rank_values([p[1] for p in pairs])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float(np.sum(ra ** 2)) * float(np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / denom) if denom > 0 else None


# -----------------------------------------------------------------------------
# Twin simulation
# -----------------------------------------------------------------------------


def pid_driven_start(
    feature_names: Sequence[str],
    window_steps: int,
    start_temp: float,
    target_temp: float,
    dt_s: float,
    kp: float,
    ki: float,
    kd: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Seed window and controller terms for when no real telemetry exists at all --
    an offline what-if prediction with no chamber run behind it.

    Holds temperature at start_temp while stepping a real ChamberPID/CodesysDiff
    forward for window_steps ticks, so the seeded control terms reflect how long
    that error has actually been sitting.
    """
    pid = ChamberPID()
    diff = CodesysDiff()
    diff.prev_value = start_temp

    rows = []
    terms: dict[str, float] = {}
    for _ in range(window_steps):
        terms = run_pid_substeps(
            pid=pid, diff=diff, temp_start=start_temp,
            temp_ref=target_temp, kp=kp, ki=ki, kd=kd, dt_s=dt_s,
            period_s=0.1, feature_scale=100.0,
        )
        rows.append(make_feature_row(
            feature_names, temp=start_temp, temp_ref=target_temp,
            previous_temp=start_temp, dt_s=dt_s,
            u=terms["u"], u_p=terms["u_p"], u_i=terms["u_i"], u_d=terms["u_d"],
            kp=kp, ki=ki, kd=kd,
        ))
    window = np.vstack(rows).astype(np.float32)
    warm_terms = {"temp_u_p": terms["u_p"], "temp_u_i": terms["u_i"], "temp_u_d": terms["u_d"]}
    return window, warm_terms


def simulate_twin(
    *,
    model,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    window_steps: int,
    device: torch.device,
    args: argparse.Namespace,
    start_temp: float,
    target_temp: float,
    duration_s: float,
    dt_s: float,
    pid_schedule: Sequence[tuple[float, float, float]],
    warm_window: np.ndarray | None = None,
    warm_terms: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run the twin closed-loop for a whole run through the deployed MPC step path.

    pid_schedule supplies (kp, ki, kd) per step, so replayed runs keep the gain
    switching the chamber actually used. warm_window seeds the feature window from
    real rows; without it, seeding falls back to pid_driven_start.
    """
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)
    total_steps = max(1, int(round(float(duration_s) / float(dt_s))))
    kp0, ki0, kd0 = pid_schedule[0]

    if warm_window is not None:
        feature_window = np.asarray(warm_window, dtype=np.float32).copy()
    else:
        feature_window, synthetic_terms = pid_driven_start(
            feature_names=feature_names,
            window_steps=window_steps,
            start_temp=start_temp,
            target_temp=target_temp,
            dt_s=dt_s,
            kp=kp0,
            ki=ki0,
            kd=kd0,
        )
        if warm_terms is None:
            warm_terms = synthetic_terms

    pid = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    diff = CodesysDiff()
    diff.prev_value = start_temp

    if warm_terms is not None:
        pid.p_part = float(warm_terms.get("temp_u_p", 0.0)) / feature_scale
        pid.i_part = float(warm_terms.get("temp_u_i", 0.0)) / feature_scale
        pid.d_part = float(warm_terms.get("temp_u_d", 0.0)) / feature_scale
        if float(kd0) != 0.0:
            diff_out = -(pid.d_part * float(kp0)) / float(kd0)
            diff.out = diff_out
            diff.filter_out = diff_out / 10.0

    state = SimState(
        elapsed_s=0.0,
        temp=float(start_temp),
        previous_temp=float(start_temp),
        feature_window=feature_window,
        pid=pid,
        diff=diff,
        kp=float(kp0),
        ki=float(ki0),
        kd=float(kd0),
    )

    rows: list[dict[str, float]] = []
    for step in range(total_steps):
        kp, ki, kd = pid_schedule[min(step, len(pid_schedule) - 1)]
        state.kp, state.ki, state.kd = float(kp), float(ki), float(kd)

        state, info, valid, reason = step_state(
            state=state,
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            device=device,
            target_temp=float(target_temp),
            dt_s=float(dt_s),
            pid_period_s=float(args.pid_period_s),
            feature_scale=feature_scale,
            max_abs_temp=float(args.max_abs_temp),
            predict_fn=predict_delta_t1,
        )

        rows.append({
            "elapsed_s": float(state.elapsed_s),
            "temp": float(state.temp),
            "temp_ref": float(target_temp),
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            "u": float(info["u"]),
            "pred_delta": float(info["pred_delta"]),
            "valid": bool(valid),
        })
        if not valid:
            print(f"   twin diverged at t={state.elapsed_s:.0f}s: {reason}")
            break

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Replay mode
# -----------------------------------------------------------------------------


def load_replay_runs(args: argparse.Namespace) -> list[tuple[str, pd.DataFrame]]:
    """Historical runs eligible for replay, cleaned by the shared loader."""
    from deepvac.datasets import find_run_csvs, prepare_run_dataframe

    loader_args = argparse.Namespace(min_samples=args.min_samples, min_duration_s=0.0)
    runs: list[tuple[str, pd.DataFrame]] = []

    for csv_path in find_run_csvs(Path(args.history_root), args.telemetry_names):
        run_id = csv_path.parent.name
        if args.run_ids and run_id not in args.run_ids:
            continue
        try:
            df, _ = prepare_run_dataframe(csv_path, loader_args)
        except Exception as exc:
            print(f"[skip] {run_id}: {exc}")
            continue
        runs.append((run_id, df))
        if args.max_runs is not None and len(runs) >= int(args.max_runs):
            break

    if not runs:
        raise RuntimeError(f"No replayable runs under {args.history_root}")
    return runs


def warm_start_at(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    window_steps: int,
    offset: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Feature window and controller terms as of a real row, for seeding the twin
    from that point instead of the synthetic flat window MPC starts a run with."""
    warm_window = df.iloc[offset - window_steps + 1 : offset + 1][list(feature_names)].to_numpy(np.float32)
    warm_terms = {c: float(df[c].iloc[offset]) for c in ("temp_u_p", "temp_u_i", "temp_u_d")}
    return warm_window, warm_terms


def replay_one(
    run_id: str,
    df: pd.DataFrame,
    start_mode: str,
    *,
    model,
    checkpoint,
    feature_names,
    window_steps,
    device,
    args,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Score the twin against one logged run."""
    elapsed = df["elapsed_s"].to_numpy(dtype=float)
    temps = df["temp"].to_numpy(dtype=float)
    target = float(df["temp_ref"].iloc[0])
    dt_s = float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else 2.0

    if start_mode == "warm":
        if len(df) <= window_steps + 1:
            raise ValueError(f"run too short to warm-start: {len(df)} <= {window_steps + 1}")
        offset = window_steps - 1
        warm_window, warm_terms = warm_start_at(df, feature_names, window_steps, offset)
    else:
        offset = 0
        warm_window = None
        warm_terms = None

    schedule = list(zip(
        df["kp"].to_numpy(dtype=float)[offset:],
        df["ki"].to_numpy(dtype=float)[offset:],
        df["kd"].to_numpy(dtype=float)[offset:],
        strict=True,
    ))
    real_t = elapsed[offset:] - elapsed[offset]
    real_temp = temps[offset:]

    pred = simulate_twin(
        model=model,
        checkpoint=checkpoint,
        feature_names=feature_names,
        window_steps=window_steps,
        device=device,
        args=args,
        start_temp=float(temps[offset]),
        target_temp=target,
        duration_s=float(real_t[-1]),
        dt_s=dt_s,
        pid_schedule=schedule,
        warm_window=warm_window,
        warm_terms=warm_terms,
    )

    metrics, aligned = compare_trajectories(
        real_t, real_temp,
        pred["elapsed_s"].to_numpy(dtype=float), pred["temp"].to_numpy(dtype=float),
        target, args,
    )
    metrics.update({
        "run_id": run_id,
        "mode": "replay",
        "start_mode": start_mode,
        "start_temp": float(temps[offset]),
        "target_temp": target,
        "kp": float(df["kp"].iloc[offset]),
        "ki": float(df["ki"].iloc[offset]),
        "kd": float(df["kd"].iloc[offset]),
        "twin_valid": bool(pred["valid"].all()),
    })
    return metrics, aligned


def run_replay_mode(session_dir: Path, *, model, checkpoint, feature_names, window_steps, device, args):
    runs = load_replay_runs(args)
    modes = ["cold", "warm"] if args.start_mode == "both" else [args.start_mode]
    print(f"[replay] {len(runs)} runs x {len(modes)} start mode(s)")

    rows: list[dict[str, object]] = []
    for idx, (run_id, df) in enumerate(runs, start=1):
        for start_mode in modes:
            try:
                metrics, aligned = replay_one(
                    run_id, df, start_mode,
                    model=model, checkpoint=checkpoint, feature_names=feature_names,
                    window_steps=window_steps, device=device, args=args,
                )
            except Exception as exc:
                print(f"[{idx}/{len(runs)}] {run_id} ({start_mode}) FAILED: {exc}")
                continue

            rows.append(metrics)
            save_trajectory(session_dir, f"{run_id}__{start_mode}", aligned, args)
            print(f"[{idx}/{len(runs)}] {run_id} ({start_mode:4s}) "
                  f"mae={metrics['mae_temp']:.3f} bias={metrics['bias_temp']:+.3f} "
                  f"final_err={metrics['final_error_temp']:+.3f} "
                  f"overshoot_err={fmt(metrics['err_overshoot'])}")
    return rows


# -----------------------------------------------------------------------------
# Horizon mode -- how far ahead a prediction from any point in a run can be trusted
# -----------------------------------------------------------------------------


def error_band(abs_error: float, args: argparse.Namespace) -> str:
    if abs_error <= float(args.near_band):
        return "near"
    if abs_error <= float(args.far_band):
        return "mid"
    return "far"


def horizon_samples_for_run(
    run_id: str,
    df: pd.DataFrame,
    *,
    model,
    checkpoint,
    feature_names: Sequence[str],
    window_steps: int,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Lookahead error at each --lookahead-marks-s from every sampled start point
    in one run. One record per (start point, mark)."""
    elapsed = df["elapsed_s"].to_numpy(dtype=float)
    temps = df["temp"].to_numpy(dtype=float)
    dt_s = float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else 2.0
    target = float(df["temp_ref"].iloc[0])
    max_mark = max(args.lookahead_marks_s)

    stride_steps = max(1, int(round(float(args.horizon_stride_s) / dt_s)))
    last_start = len(df) - 2
    offsets = range(window_steps - 1, last_start + 1, stride_steps)

    records: list[dict[str, object]] = []
    for offset in offsets:
        start_elapsed = float(elapsed[offset])
        start_abs_error = abs(float(temps[offset]) - target)
        real_t = elapsed[offset:] - start_elapsed
        real_temp = temps[offset:]
        if real_t[-1] < min(args.lookahead_marks_s):
            continue

        horizon_s = min(float(args.max_horizon_s), max_mark, float(real_t[-1]))
        warm_window, warm_terms = warm_start_at(df, feature_names, window_steps, offset)
        schedule = list(zip(
            df["kp"].to_numpy(dtype=float)[offset:],
            df["ki"].to_numpy(dtype=float)[offset:],
            df["kd"].to_numpy(dtype=float)[offset:],
            strict=True,
        ))

        pred = simulate_twin(
            model=model, checkpoint=checkpoint, feature_names=feature_names,
            window_steps=window_steps, device=device, args=args,
            start_temp=float(temps[offset]), target_temp=target,
            duration_s=horizon_s, dt_s=dt_s, pid_schedule=schedule,
            warm_window=warm_window, warm_terms=warm_terms,
        )
        if pred.empty:
            continue

        pred_t = pred["elapsed_s"].to_numpy(dtype=float)
        pred_temp = pred["temp"].to_numpy(dtype=float)
        pred_valid = pred["valid"].to_numpy(dtype=bool)
        diverged_at = float(pred_t[np.argmax(~pred_valid)]) if not pred_valid.all() else float("inf")

        for mark in args.lookahead_marks_s:
            if mark > real_t[-1]:
                continue
            record = {
                "run_id": run_id, "start_offset": int(offset),
                "start_elapsed_s": start_elapsed, "start_abs_error": start_abs_error,
                "start_band": error_band(start_abs_error, args), "lookahead_s": float(mark),
            }
            if mark > diverged_at or mark > pred_t[-1]:
                record.update({"error": None, "abs_error": None, "diverged": True})
            else:
                err = float(np.interp(mark, pred_t, pred_temp) - np.interp(mark, real_t, real_temp))
                record.update({"error": err, "abs_error": abs(err), "diverged": False})
            records.append(record)

    return records


def aggregate_horizon(rows: pd.DataFrame, group_cols: list[str], trust_band: float) -> pd.DataFrame:
    """Per-lookahead mae/bias/p90/diverged-fraction, within each group_cols group."""
    out = []
    for key, group in rows.groupby(group_cols + ["lookahead_s"]):
        keys = key if isinstance(key, tuple) else (key,)
        alive = group[~group["diverged"]]
        row = dict(zip(group_cols + ["lookahead_s"], keys, strict=True))
        row["n_started"] = int(len(group))
        row["diverged_fraction"] = float(group["diverged"].mean())
        if len(alive):
            row["mae"] = float(alive["abs_error"].mean())
            row["bias"] = float(alive["error"].mean())
            row["p90_abs_error"] = float(np.percentile(alive["abs_error"], 90))
        else:
            row["mae"] = row["bias"] = row["p90_abs_error"] = float("nan")
        row["p90_trusted"] = bool(row["p90_abs_error"] <= trust_band) if len(alive) else False
        out.append(row)
    return pd.DataFrame(out).sort_values(group_cols + ["lookahead_s"]).reset_index(drop=True)


def trust_horizon(curve: pd.DataFrame) -> float | None:
    """Longest lookahead, in ascending mark order, before the curve first fails
    trust (p90 error over threshold, or any divergence)."""
    curve = curve.sort_values("lookahead_s")
    ok = curve["p90_trusted"] & (curve["diverged_fraction"] == 0.0)
    if ok.all():
        return float(curve["lookahead_s"].max())
    first_fail = curve.index[~ok][0]
    prior = curve.loc[: first_fail - 1] if first_fail > curve.index[0] else None
    return float(prior["lookahead_s"].max()) if prior is not None and len(prior) else 0.0


def plot_horizon_curve(curve: pd.DataFrame, trust_band: float, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_err, ax_div) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for band, group in curve.groupby("start_band"):
        group = group.sort_values("lookahead_s")
        ax_err.plot(group["lookahead_s"], group["mae"], marker="o", label=f"{band} (mae)")
        ax_err.plot(group["lookahead_s"], group["p90_abs_error"], marker=".",
                    linestyle="--", alpha=0.6, label=f"{band} (p90)")
        ax_div.plot(group["lookahead_s"], 100.0 * group["diverged_fraction"], marker="o", label=band)

    ax_err.axhline(trust_band, color="#888888", linestyle=":", label=f"trust band ({trust_band:g}C)")
    ax_err.set_ylabel("Absolute error (deg C)")
    ax_err.set_title("How far ahead a prediction can be trusted, by starting error band")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(fontsize=8)

    ax_div.set_xlabel("Lookahead (s)")
    ax_div.set_ylabel("Twin diverged (%)")
    ax_div.grid(True, alpha=0.25)
    ax_div.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_horizon_mode(session_dir: Path, *, model, checkpoint, feature_names, window_steps, device, args):
    runs = load_replay_runs(args)
    trust_band = float(args.trust_band) if args.trust_band is not None else float(args.near_band)
    print(f"[horizon] {len(runs)} runs, marks={args.lookahead_marks_s}s, trust_band={trust_band:g}C")

    all_records: list[dict[str, object]] = []
    for idx, (run_id, df) in enumerate(runs, start=1):
        try:
            records = horizon_samples_for_run(
                run_id, df, model=model, checkpoint=checkpoint, feature_names=feature_names,
                window_steps=window_steps, device=device, args=args,
            )
        except Exception as exc:
            print(f"[{idx}/{len(runs)}] {run_id} FAILED: {exc}")
            continue
        all_records.extend(records)
        n_starts = len({r["start_offset"] for r in records})
        print(f"[{idx}/{len(runs)}] {run_id}: {n_starts} start points, {len(records)} lookahead samples")

    if not all_records:
        raise RuntimeError("No horizon samples were produced.")

    raw = pd.DataFrame(all_records)
    raw.to_csv(session_dir / "horizon_samples.csv", index=False)

    overall = aggregate_horizon(raw, [], trust_band)
    overall["start_band"] = "all"
    by_band = aggregate_horizon(raw, ["start_band"], trust_band)
    curve = pd.concat([overall, by_band], ignore_index=True)
    curve.to_csv(session_dir / "horizon_curve.csv", index=False)

    if not args.no_plots:
        plot_horizon_curve(curve, trust_band, session_dir / "plots" / "horizon_curve.png")

    trust: dict[str, float | None] = {"all": trust_horizon(overall)}
    for band, group in by_band.groupby("start_band"):
        trust[str(band)] = trust_horizon(group)

    print("\n=== Trust horizon (p90 error within band, twin never diverges) ===")
    for band, value in trust.items():
        print(f"  {band:6s} {'n/a' if value is None else f'{value:.0f}s'}")

    print(f"\n{'lookahead_s':>12s} {'band':>6s} {'mae':>8s} {'p90':>8s} {'diverged%':>10s} {'n':>6s}")
    for _, row in curve.sort_values(["start_band", "lookahead_s"]).iterrows():
        print(f"{row['lookahead_s']:12.0f} {row['start_band']:>6s} {row['mae']:8.3f} "
              f"{row['p90_abs_error']:8.3f} {100.0 * row['diverged_fraction']:9.1f}% {row['n_started']:6.0f}")

    return {
        "n_runs": int(len(runs)),
        "n_start_points": int(raw.drop_duplicates(["run_id", "start_offset"]).shape[0]),
        "n_lookahead_samples": int(len(raw)),
        "trust_band_c": trust_band,
        "trust_horizon_s": trust,
        "horizon_samples_csv": str(session_dir / "horizon_samples.csv"),
        "horizon_curve_csv": str(session_dir / "horizon_curve.csv"),
    }


# -----------------------------------------------------------------------------
# Chamber mode
# -----------------------------------------------------------------------------


def parse_pid_sets(args: argparse.Namespace, rng: random.Random) -> list[tuple[int, int, int]]:
    """PID triplets to test, one held for each whole experiment."""
    if str(args.pid_sets).strip().lower() == "random":
        return [
            (rng.randint(args.kp_min, args.kp_max),
             rng.randint(args.ki_min, args.ki_max),
             rng.randint(args.kd_min, args.kd_max))
            for _ in range(int(args.num_tests))
        ]

    triplets: list[tuple[int, int, int]] = []
    for chunk in str(args.pid_sets).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--pid-sets entries must be kp,ki,kd; got {chunk!r}")
        triplets.append((int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))))

    if not triplets:
        raise ValueError("--pid-sets produced no triplets")
    return [triplets[i % len(triplets)] for i in range(int(args.num_tests))]


def read_state_with_retries(args: argparse.Namespace) -> dict[str, float] | None:
    """One TCP state snapshot, retried per --read-retries."""
    for _ in range(max(1, args.read_retries + 1)):
        try:
            return request_temperature_states(
                host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout
            )
        except Exception:
            if args.read_retry_delay_s > 0:
                time.sleep(args.read_retry_delay_s)
    return None


def precondition_chamber(run_id: str, args: argparse.Namespace) -> None:
    """Heat the chamber up and let it settle so every experiment starts alike."""
    print(f"[{run_id}] preconditioning: temp_ref={args.heatup_temp_ref:.2f} for {args.heatup_duration:.0f}s")
    publish_temp_ref_job(
        temp_ref=float(args.heatup_temp_ref),
        duration_s=args.heatup_duration,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    time.sleep(args.heatup_duration)
    if args.post_heatup_cooldown > 0:
        print(f"[{run_id}] post-heatup settle for {args.post_heatup_cooldown:.0f}s")
        time.sleep(args.post_heatup_cooldown)


def measure_chamber_run(
    run_id: str,
    pid: tuple[int, int, int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Drive one cooldown over TCP with a single PID triplet and log every sample."""
    kp, ki, kd = pid
    target = float(args.target_temp)

    _ = read_state_with_retries(args)

    publish_temp_ref_job(
        temp_ref=target,
        duration_s=args.test_duration,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    apply_pid_update(
        label="whole-run",
        run_id=run_id,
        row=args.pid_row,
        pid=(kp, ki, kd),
        args=args,
        events=[],
    )

    rows: list[dict[str, float]] = []
    failures = 0
    t0 = time.time()
    next_progress = t0 + args.progress_every if args.progress_every > 0 else float("inf")

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= args.test_duration:
            break

        snap = read_state_with_retries(args)
        if snap is None:
            failures += 1
            print(f"[{run_id}] read failed ({failures}/{args.max_consecutive_failures})")
            if failures >= args.max_consecutive_failures:
                raise RuntimeError(f"Too many consecutive TCP read failures ({failures})")
            time.sleep(args.dt)
            continue

        failures = 0
        rows.append({
            "run_id": run_id,
            "timestamp": now,
            "elapsed_s": elapsed,
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            **snap,
            "sq_error": float((snap["temp_ref"] - snap["temp"]) ** 2),
        })

        if now >= next_progress:
            print(f"[{run_id}] t={elapsed:6.0f}s temp={snap['temp']:7.3f} "
                  f"ref={snap['temp_ref']:6.2f} samples={len(rows)}")
            next_progress += args.progress_every

        time.sleep(args.dt)

    if not rows:
        raise RuntimeError("No TCP samples collected")
    return pd.DataFrame(rows)


def save_measured_run(run_id: str, samples: pd.DataFrame, pid: tuple[int, int, int],
                      args: argparse.Namespace) -> None:
    """Append a measured run to the history root in the standard layout."""
    kp, ki, kd = pid
    samples = append_mae_column(samples)
    start_temp = float(samples["temp"].iloc[0])
    samples["start_temp"] = start_temp

    cost_info = compute_tail_cost(samples, entry_band=args.entry_band,
                                  overshoot_weight=args.overshoot_weight)
    summary = {
        "run_id": run_id,
        "start_ts": float(samples["timestamp"].iloc[0]),
        "end_ts": float(samples["timestamp"].iloc[-1]),
        "duration_s": float(samples["timestamp"].iloc[-1] - samples["timestamp"].iloc[0]),
        "num_samples": int(len(samples)),
        "start_temp": start_temp,
        "temp_ref": float(samples["temp_ref"].iloc[0]),
        "pid_source": "twin_acceptance",
        "far_kp": int(kp), "far_ki": int(ki), "far_kd": int(kd),
        "mid_kp": int(kp), "mid_ki": int(ki), "mid_kd": int(kd),
        "near_kp": int(kp), "near_ki": int(ki), "near_kd": int(kd),
        "mse": float(samples["sq_error"].mean()),
        "mae": float(samples["mae"].mean()),
        "cost": float(cost_info["cost"]),
        "tail_mae": cost_info["tail_mae"],
        "overshoot": cost_info["overshoot"],
    }

    root = args.save_history_root
    append_rows_csv(history_run_file(run_id, str(Path(root) / args.samples_csv), root),
                    samples.to_dict(orient="records"))
    append_row_csv(history_run_file(run_id, str(Path(root) / args.runs_csv), root), summary)


def run_chamber_mode(session_dir: Path, *, model, checkpoint, feature_names, window_steps, device, args):
    rng = random.Random(args.seed)
    schedule = parse_pid_sets(args, rng)
    print(f"[chamber] {len(schedule)} experiments, {args.test_duration:.0f}s each")

    rows: list[dict[str, object]] = []
    for idx, pid in enumerate(schedule, start=1):
        run_id = make_run_id(prefix="twin")
        print(f"\n=== experiment {idx}/{len(schedule)} :: {run_id} :: kp={pid[0]} ki={pid[1]} kd={pid[2]} ===")

        try:
            if not args.skip_preconditioning and (idx > 1 or args.condition_initial):
                precondition_chamber(run_id, args)

            snap = read_state_with_retries(args)
            if snap is None:
                raise RuntimeError("Could not read chamber state before the experiment")
            start_temp = float(snap["temp"])
            print(f"[{run_id}] start_temp={start_temp:.3f}, predicting before running")

            pred = simulate_twin(
                model=model, checkpoint=checkpoint, feature_names=feature_names,
                window_steps=window_steps, device=device, args=args,
                start_temp=start_temp, target_temp=float(args.target_temp),
                duration_s=float(args.test_duration), dt_s=float(args.dt),
                pid_schedule=[(float(pid[0]), float(pid[1]), float(pid[2]))],
            )
            pred.to_csv(session_dir / "predictions" / f"{run_id}.csv", index=False)
            print(f"[{run_id}] predicted final temp={pred['temp'].iloc[-1]:.3f}, now running the chamber")

            samples = measure_chamber_run(run_id, pid, args)
            save_measured_run(run_id, samples, pid, args)

            real_t = samples["elapsed_s"].to_numpy(dtype=float)
            metrics, aligned = compare_trajectories(
                real_t, samples["temp"].to_numpy(dtype=float),
                pred["elapsed_s"].to_numpy(dtype=float), pred["temp"].to_numpy(dtype=float),
                float(args.target_temp), args,
            )
            metrics.update({
                "run_id": run_id,
                "mode": "chamber",
                "start_mode": "cold",
                "start_temp": start_temp,
                "target_temp": float(args.target_temp),
                "kp": int(pid[0]), "ki": int(pid[1]), "kd": int(pid[2]),
                "twin_valid": bool(pred["valid"].all()),
            })
            rows.append(metrics)
            save_trajectory(session_dir, run_id, aligned, args)

            print(f"[{run_id}] mae={metrics['mae_temp']:.3f} bias={metrics['bias_temp']:+.3f} "
                  f"final_err={metrics['final_error_temp']:+.3f} "
                  f"overshoot pred={fmt(metrics['pred_overshoot'])} real={fmt(metrics['real_overshoot'])}")
        except Exception as exc:
            print(f"[{run_id}] FAILED: {exc}")

        if idx < len(schedule) and args.cooldown > 0:
            print(f"waiting {args.cooldown:.0f}s before the next experiment")
            time.sleep(args.cooldown)

    return rows


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def save_trajectory(session_dir: Path, label: str, aligned: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write one aligned real-vs-predicted trajectory, and optionally plot it."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    aligned.to_csv(session_dir / "trajectories" / f"{safe}.csv", index=False)
    if args.no_plots:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    ax_top.plot(aligned["elapsed_s"], aligned["real_temp"], label="chamber")
    ax_top.plot(aligned["elapsed_s"], aligned["pred_temp"], label="twin")
    ax_top.plot(aligned["elapsed_s"], aligned["temp_ref"], linestyle="--", label="target")
    ax_top.set_ylabel("Temperature (deg C)")
    ax_top.set_title(label)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend()

    ax_bot.plot(aligned["elapsed_s"], aligned["error_temp"], color="#d62728")
    ax_bot.axhline(0.0, color="#222222", linewidth=1.0)
    ax_bot.set_xlabel("Elapsed seconds")
    ax_bot.set_ylabel("twin - chamber")
    ax_bot.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(session_dir / "plots" / f"{safe}.png", dpi=150)
    plt.close(fig)


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    """Aggregate per-run scores, split by start mode."""
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {"n_runs": int(len(frame))}

    numeric = ["mae_temp", "rmse_temp", "bias_temp", "p90_abs_error_temp",
               "max_abs_error_temp", "final_error_temp",
               "err_overshoot", "err_tail_mae", "err_time_to_near_s",
               "err_time_to_settle_s", "err_final_temp", "err_cost"]
    numeric += [f"error_at_{int(m)}s" for m in args.checkpoints_s]

    by_mode: dict[str, object] = {}
    for start_mode, group in frame.groupby("start_mode"):
        stats: dict[str, object] = {"n_runs": int(len(group))}
        for column in numeric:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            stats[column] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p90_abs": float(np.percentile(np.abs(values), 90)),
            }
        stats["cost_rank_correlation"] = rank_correlation(
            group["pred_cost"].tolist(), group["real_cost"].tolist()
        )
        stats["twin_valid_fraction"] = float(group["twin_valid"].mean())
        stats["band_reached_real_fraction"] = float(group["reached_band_real"].mean())
        stats["band_reached_pred_fraction"] = float(group["reached_band_pred"].mean())
        by_mode[str(start_mode)] = stats

    summary["by_start_mode"] = by_mode
    return summary


def print_summary(summary: dict[str, object], args: argparse.Namespace) -> None:
    print("\n=== Twin acceptance ===")
    for start_mode, stats in summary["by_start_mode"].items():
        print(f"\n-- start_mode={start_mode}  ({stats['n_runs']} runs) --")
        print(f"{'metric':26s} {'mean':>10s} {'median':>10s} {'p90|x|':>10s}")
        for key, value in stats.items():
            if not isinstance(value, dict):
                continue
            print(f"{key:26s} {value['mean']:10.3f} {value['median']:10.3f} {value['p90_abs']:10.3f}")
        corr = stats.get("cost_rank_correlation")
        print(f"{'cost rank correlation':26s} {'n/a' if corr is None else f'{corr:10.3f}'}")
        print(f"{'twin stayed finite':26s} {100.0 * stats['twin_valid_fraction']:9.0f}%")
        print(f"{'reached band (chamber)':26s} {100.0 * stats['band_reached_real_fraction']:9.0f}%")
        print(f"{'reached band (twin)':26s} {100.0 * stats['band_reached_pred_fraction']:9.0f}%")

    print("\nTrajectory fidelity is mae_temp/bias_temp. Decision fidelity is err_overshoot,")
    print("err_time_to_settle_s and the cost rank correlation, which is what decides")
    print("whether MPC picks the PID the chamber would have preferred.")


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "chamber":
        if args.dt <= 0:
            raise ValueError("--dt must be > 0")
        if args.test_duration <= 0:
            raise ValueError("--test-duration must be > 0")
    if args.mode == "horizon":
        if not args.lookahead_marks_s:
            raise ValueError("--lookahead-marks-s must be non-empty")
        if args.max_horizon_s <= 0 or args.horizon_stride_s <= 0:
            raise ValueError("--max-horizon-s and --horizon-stride-s must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", []))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    session = args.session_name or make_run_id(prefix=f"twin_{args.mode}")
    session_dir = Path(args.output_dir) / session
    subdirs = ("plots",) if args.mode == "horizon" else ("trajectories", "plots", "predictions")
    for sub in subdirs:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}")
    print(f"window     : {window_steps} steps, features {feature_names}")
    print(f"session    : {session_dir}")

    common = dict(model=model, checkpoint=checkpoint, feature_names=feature_names,
                  window_steps=window_steps, device=device, args=args)

    if args.mode == "horizon":
        summary = run_horizon_mode(session_dir, **common)
        summary.update({"mode": args.mode, "checkpoint": str(Path(args.checkpoint).resolve()),
                        "session_dir": str(session_dir.resolve())})
        (session_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsummary : {session_dir / 'summary.json'}")
        return

    rows = (run_replay_mode(session_dir, **common) if args.mode == "replay"
            else run_chamber_mode(session_dir, **common))

    if not rows:
        print("\nNo runs were scored.")
        return

    per_run = pd.DataFrame(rows)
    per_run.to_csv(session_dir / "per_run_metrics.csv", index=False)

    summary = summarize(rows, args)
    summary.update({
        "mode": args.mode,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "session_dir": str(session_dir.resolve()),
        "target_temp": float(args.target_temp),
        "checkpoints_s": [float(m) for m in args.checkpoints_s],
    })
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary, args)
    print(f"\nper-run metrics : {session_dir / 'per_run_metrics.csv'}")
    print(f"summary         : {session_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
