#!/usr/bin/env python3
"""Runs one real-chamber experiment under live PPO advisor control, then scores it
against the digital twin and a CEM-optimized fixed-PID baseline.

The PPO policy re-decides the PID triplet every checkpoint's mpc_hold_s from live
TCP telemetry, writing to the chamber's currently-active gain-scheduled PID band.
Each decision (elapsed_s, kp, ki, kd) is recorded.

After the run, the recorded decisions are replayed through the digital twin for an
"expected" trajectory, scored against the real one for twin accuracy; a CEM search
over the whole run finds a single best-fixed PID triplet as a "baseline" trajectory.
All three trajectories are plotted, and a report.json/console summary covers twin
accuracy and advisor-vs-baseline cost/tail_mae/overshoot improvement.

--gru-checkpoint accepts either family's checkpoint (the twin used is whichever
family it's stamped with) despite the flag name.

Examples:

    python -m advisor.run_advisor_experiment \\
        --gru-checkpoint digitaltwin/gru/validation_rollout/gru_rollout.pt \\
        --ppo-checkpoint advisor/policy_ppo_v5/policy.pt \\
        --target-temp 0 --duration-s 1200

    python -m advisor.run_advisor_experiment --gru-checkpoint ... --ppo-checkpoint ... \\
        --pid-row 2 --tcp-host 172.0.30.10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import make_run_id  
from deepvac.metrics import append_mae_column  
from deepvac.mpc import make_feature_row, optimize_pid_for_state, rollout_constant_pid, run_pid_substeps
from deepvac.pid import clip_pid, pid_bounds  

from digitaltwin.common import ChamberPID, CodesysDiff, load_model, predict_delta_t1
from digitaltwin.twin_acceptance import compare_trajectories, describe_trajectory, simulate_twin, trajectory_cost
from advisor.advise_pid import add_cost_args, build_initial_state, horizon_cost
from advisor.train_policy_ppo import ActorCritic, DIFF_CLIP
from deepvac.mpc import add_common_mpc_args

from tcp.tcp_common import (  
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_states,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run one real-chamber experiment under live PPO advisor control, "
                    "then score it against the digital twin and a CEM baseline."
    )

    ap.add_argument("--gru-checkpoint", required=True,
                    help="Rollout-trained digital-twin checkpoint (GRU or LSTM, despite the flag name).")
    ap.add_argument("--ppo-checkpoint", required=True, help="Trained PPO policy (advisor/policy_ppo_*/policy.pt).")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--session-name", default=None, help="Output subfolder name. Default: generated id.")
    ap.add_argument("--no-plot", action="store_true")

    add_common_mpc_args(ap)
    add_cost_args(ap)
    ap.set_defaults(target_temp=0.0)

    ap.add_argument("--checkpoints-s", nargs="*", type=float, default=[120.0, 300.0, 600.0, 900.0, 1200.0])
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    # --- Real-chamber PID write -------------------------------------------------
    ap.add_argument("--pid-row", type=int, default=None,
                    help="Controller PID row index [0..4] to write decisions to. "
                         "Default: auto-detect the chamber's currently-active band "
                         "(temp_pid_idx) before every decision.")

    # --- Safety ------------------------------------------------------------------
    ap.add_argument("--safety-margin-c", type=float, default=5.0,
                    help="Abort if temp strays this far past [min(start,target), max(start,target)].")
    ap.add_argument("--max-consecutive-failures", type=int, default=10)

    # --- Preconditioning (optional, off by default) -------------------------------
    ap.add_argument("--precondition", action="store_true", help="Heat/settle the chamber before the experiment.")
    ap.add_argument("--heatup-temp-ref", type=float, default=25.0)
    ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0)
    ap.add_argument("--post-heatup-cooldown", type=float, default=3.0 * 60.0)

    # --- TCP -----------------------------------------------------------------------
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--progress-every", type=float, default=60.0)

    return ap


# -----------------------------------------------------------------------------
# PPO advisor: load + deterministic single-observation inference
# -----------------------------------------------------------------------------


def load_ppo_policy(path: Path, device: torch.device) -> tuple[ActorCritic, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(
        int(checkpoint["hidden_dim"]), int(checkpoint["num_layers"]), min_std=float(checkpoint.get("min_std", 0.05))
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def ppo_decide(model: ActorCritic, obs: np.ndarray, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    """Deterministic action: tanh(actor_mean(obs)), de-normalized and rounded to a PID triplet."""
    obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        mean, _, _ = model(obs_t)
    action = np.tanh(mean.cpu().numpy())[0]
    lo, hi = pid_bounds(args)
    pid_raw = lo + (np.clip(action, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
    return clip_pid(pid_raw, args)


def build_observation(
    *,
    temp: float,
    temp_ref: float,
    kp: float,
    ki: float,
    kd: float,
    shadow_pid: ChamberPID,
    shadow_diff: CodesysDiff,
    velocity: float,
    step_idx: int,
    episode_steps: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Builds the policy's 8-dim observation vector from live chamber telemetry
    and the shadow PID/Diff controller state."""
    scale = max(abs(args.temp_min), abs(args.temp_max), 1e-6)
    i_scale = max(abs(float(args.u_min)), abs(float(args.u_max)), 1e-6)
    lo, hi = pid_bounds(args)
    return np.asarray(
        [
            (temp - temp_ref) / scale,
            velocity / scale,
            2.0 * (kp - lo[0]) / (hi[0] - lo[0]) - 1.0,
            2.0 * (ki - lo[1]) / (hi[1] - lo[1]) - 1.0,
            2.0 * (kd - lo[2]) / (hi[2] - lo[2]) - 1.0,
            1.0 - step_idx / max(1, episode_steps),
            shadow_pid.i_part / i_scale,
            shadow_diff.filter_out / DIFF_CLIP,
        ],
        dtype=np.float32,
    )


def step_shadow_controller(
    *, shadow_pid: ChamberPID, shadow_diff: CodesysDiff, temp: float, temp_ref: float,
    kp: float, ki: float, kd: float, dt_s: float, args: argparse.Namespace,
) -> None:
    """Advances the shadow ChamberPID/CodesysDiff controller in place by dt_s,
    using the real measured temperature and gains."""
    run_pid_substeps(
        pid=shadow_pid, diff=shadow_diff, temp_start=temp, temp_end=temp, temp_ref=temp_ref,
        kp=kp, ki=ki, kd=kd, dt_s=dt_s, period_s=float(args.pid_period_s),
        feature_scale=max(abs(float(args.control_feature_scale)), 1e-9),
    )


# -----------------------------------------------------------------------------
# Real chamber I/O
# -----------------------------------------------------------------------------


def read_snapshot(args: argparse.Namespace) -> dict[str, float] | None:
    """One full TCP state snapshot (all states, including temp_pid_idx), retried."""
    for _ in range(max(1, args.read_retries + 1)):
        try:
            return request_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)
        except Exception:
            if args.read_retry_delay_s > 0:
                time.sleep(args.read_retry_delay_s)
    return None


def active_pid_row(snapshot: dict[str, float], args: argparse.Namespace) -> int:
    if args.pid_row is not None:
        return int(args.pid_row)
    if "temp_pid_idx" in snapshot:
        return int(np.clip(round(snapshot["temp_pid_idx"]), 0, 4))
    return 1


def precondition_chamber(run_id: str, args: argparse.Namespace) -> None:
    print(f"[{run_id}] preconditioning: temp_ref={args.heatup_temp_ref:.2f} for {args.heatup_duration:.0f}s")
    publish_temp_ref_job(
        temp_ref=float(args.heatup_temp_ref), duration_s=args.heatup_duration,
        host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout,
    )
    time.sleep(args.heatup_duration)
    if args.post_heatup_cooldown > 0:
        print(f"[{run_id}] post-heatup settle for {args.post_heatup_cooldown:.0f}s")
        time.sleep(args.post_heatup_cooldown)


def run_real_experiment(
    run_id: str, *, ppo_model: ActorCritic, ppo_checkpoint: dict, device: torch.device, args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[tuple[float, int, int, int]], float, bool, str]:
    """Drives the real chamber under live PPO control for args.duration_s.

    Returns (samples_df, decisions[(elapsed_s, kp, ki, kd)], start_temp, aborted, reason).
    """
    mpc_hold_s = float(ppo_checkpoint["mpc_hold_s"])
    dt_s = float(args.dt_s)
    episode_steps = max(1, int(round(float(args.duration_s) / mpc_hold_s)))

    snap0 = read_snapshot(args)
    if snap0 is None:
        raise RuntimeError("Could not read chamber state before the experiment")
    start_temp = float(snap0["temp"])
    safety_lo = min(start_temp, args.target_temp) - args.safety_margin_c
    safety_hi = max(start_temp, args.target_temp) + args.safety_margin_c

    print(f"[{run_id}] start_temp={start_temp:.3f} -> target={args.target_temp:.2f} "
          f"over {args.duration_s:.0f}s, decisions every {mpc_hold_s:.0f}s ({episode_steps} total)")

    publish_temp_ref_job(
        temp_ref=float(args.target_temp), duration_s=float(args.duration_s),
        host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout,
    )

    shadow_pid = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    shadow_diff = CodesysDiff()
    shadow_diff.prev_value = start_temp

    decisions: list[tuple[float, int, int, int]] = []
    rows: list[dict[str, float]] = []
    failures = 0
    t0 = time.time()
    next_decision_ts = t0
    step_idx = 0
    temp_at_last_decision = start_temp
    ts_at_last_decision = t0
    current_pid = (0, 0, 0)
    aborted = False
    abort_reason = ""
    next_progress = t0 + args.progress_every if args.progress_every > 0 else float("inf")

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= args.duration_s:
            break

        snap = read_snapshot(args)
        if snap is None:
            failures += 1
            print(f"[{run_id}] read failed ({failures}/{args.max_consecutive_failures})")
            if failures >= args.max_consecutive_failures:
                raise RuntimeError(f"Too many consecutive TCP read failures ({failures})")
            time.sleep(dt_s)
            continue
        failures = 0

        temp = float(snap["temp"])
        temp_ref = float(snap.get("temp_ref", args.target_temp))
        if temp < safety_lo or temp > safety_hi:
            aborted = True
            abort_reason = f"temp={temp:.2f} outside [{safety_lo:.1f}, {safety_hi:.1f}] safety range"
            print(f"[{run_id}] SAFETY ABORT: {abort_reason}")
            break

        kp = float(snap.get("temp_kp", current_pid[0]))
        ki = float(snap.get("temp_ki", current_pid[1]))
        kd = float(snap.get("temp_kd", current_pid[2]))

        step_shadow_controller(
            shadow_pid=shadow_pid, shadow_diff=shadow_diff, temp=temp, temp_ref=temp_ref,
            kp=kp, ki=ki, kd=kd, dt_s=dt_s, args=args,
        )

        if now >= next_decision_ts:
            velocity = (temp - temp_at_last_decision) / max(now - ts_at_last_decision, 1e-6)
            obs = build_observation(
                temp=temp, temp_ref=temp_ref, kp=kp, ki=ki, kd=kd,
                shadow_pid=shadow_pid, shadow_diff=shadow_diff, velocity=velocity,
                step_idx=step_idx, episode_steps=episode_steps, args=args,
            )
            pid = ppo_decide(ppo_model, obs, device, args)
            current_pid = (int(pid[0]), int(pid[1]), int(pid[2]))

            row = active_pid_row(snap, args)
            apply_pid_update(label=f"t={elapsed:.0f}s", run_id=run_id, row=row, pid=current_pid, args=args, events=[])
            decisions.append((elapsed, *current_pid))

            temp_at_last_decision, ts_at_last_decision = temp, now
            step_idx += 1
            next_decision_ts += mpc_hold_s

        rows.append({
            "run_id": run_id, "timestamp": now, "elapsed_s": elapsed,
            "kp": kp, "ki": ki, "kd": kd, **snap,
            "sq_error": float((temp_ref - temp) ** 2),
        })

        if now >= next_progress:
            print(f"[{run_id}] t={elapsed:6.0f}/{args.duration_s:.0f}s temp={temp:7.3f} ref={temp_ref:6.2f} "
                  f"pid=({current_pid[0]}, {current_pid[1]}, {current_pid[2]}) samples={len(rows)}")
            next_progress += args.progress_every

        time.sleep(dt_s)

    if not rows:
        raise RuntimeError("No TCP samples were collected during the experiment")
    if not decisions:
        raise RuntimeError("No PPO decisions were made during the experiment")

    return pd.DataFrame(rows), decisions, start_temp, aborted, abort_reason


# -----------------------------------------------------------------------------
# Expected (twin replay of the recorded decisions) and CEM baseline
# -----------------------------------------------------------------------------


def cold_start_seed(
    *, feature_names: list, window_steps: int, start_temp: float, target_temp: float, dt_s: float,
    kp: float, ki: float, kd: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Builds a cold-start feature window and controller terms for the twin,
    holding start_temp constant for window_steps ticks."""
    pid = ChamberPID()
    diff = CodesysDiff()
    diff.prev_value = start_temp

    rows = []
    terms: dict[str, float] = {}
    for _ in range(window_steps):
        terms = run_pid_substeps(
            pid=pid, diff=diff, temp_start=start_temp, temp_end=start_temp,
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


def decisions_to_pid_schedule(
    decisions: list[tuple[float, int, int, int]], duration_s: float, dt_s: float,
) -> list[tuple[float, float, float]]:
    """Expands a decision log into one (kp, ki, kd) per simulate_twin() step,
    holding each decision's PID constant until the next decision's timestamp."""
    total_steps = max(1, int(round(duration_s / dt_s)))
    schedule: list[tuple[float, float, float]] = []
    next_idx = 0
    current = tuple(float(v) for v in decisions[0][1:])
    for step in range(total_steps):
        t = step * dt_s
        while (next_idx + 1 < len(decisions)) and (decisions[next_idx + 1][0] <= t):
            next_idx += 1
            current = tuple(float(v) for v in decisions[next_idx][1:])
        schedule.append(current)
    return schedule


def compute_cem_baseline(
    *, model, checkpoint: dict, feature_names: list, window_steps: int, device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], dict, int]:
    """Runs one CEM search over the whole run for a single best-fixed PID
    triplet, then rolls it out."""
    cem_args = argparse.Namespace(**vars(args))
    cem_args.mpc_horizon_s = float(args.duration_s)
    cem_args.context_csv = None

    state = build_initial_state(
        checkpoint=checkpoint, feature_names=feature_names, window_steps=window_steps, args=cem_args, verbose=False,
    )

    rng = np.random.default_rng(int(args.seed))
    decision = optimize_pid_for_state(
        state=state, model=model, checkpoint=checkpoint, feature_names=feature_names, device=device,
        args=cem_args, rng=rng, predict_fn=predict_delta_t1, cost_fn=horizon_cost, decision_idx=1,
    )
    pid = np.asarray([decision["kp"], decision["ki"], decision["kd"]], dtype=float)
    n_evaluated = int(decision["n_evaluated"])

    horizon_steps = max(1, int(math.ceil(float(args.duration_s) / float(args.dt_s))))
    metrics, temps = rollout_constant_pid(
        initial_state=state, candidate_pid=pid, model=model, checkpoint=checkpoint, feature_names=feature_names,
        device=device, args=cem_args, horizon_steps=horizon_steps, predict_fn=predict_delta_t1, cost_fn=horizon_cost,
    )
    times = np.arange(1, len(temps) + 1, dtype=float) * float(args.dt_s)
    return times, np.asarray(temps, dtype=float), (float(pid[0]), float(pid[1]), float(pid[2])), metrics, n_evaluated


# -----------------------------------------------------------------------------
# Reporting / plotting
# -----------------------------------------------------------------------------


def improvement_table(
    real_t: np.ndarray, real_temp: np.ndarray, baseline_t: np.ndarray, baseline_temp: np.ndarray,
    target: float, args: argparse.Namespace,
) -> dict[str, dict[str, float | None]]:
    """Compares the real (advisor) trajectory against the CEM baseline on cost,
    tail MAE, and overshoot, with percent improvement (positive = advisor better)."""
    real_desc = describe_trajectory(real_t, real_temp, target, args)
    base_desc = describe_trajectory(baseline_t, baseline_temp, target, args)
    real_desc["cost"] = trajectory_cost(real_t, real_temp, target, args)
    base_desc["cost"] = trajectory_cost(baseline_t, baseline_temp, target, args)

    table: dict[str, dict[str, float | None]] = {}
    for key in ("cost", "tail_mae", "overshoot"):
        real_v, base_v = real_desc.get(key), base_desc.get(key)
        pct = None
        if real_v is not None and base_v is not None and base_v not in (0, None):
            pct = 100.0 * (base_v - real_v) / abs(base_v)
        table[key] = {"advisor": real_v, "baseline": base_v, "improvement_percent": pct}
    return table


def plot_comparison(
    real: pd.DataFrame, expected: pd.DataFrame, baseline_t: np.ndarray, baseline_temp: np.ndarray,
    decisions: list[tuple[float, int, int, int]], target: float, output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_temp, ax_pid) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_temp.plot(real["elapsed_s"], real["temp"], label="real (chamber)", color="#1f77b4", linewidth=2)
    ax_temp.plot(expected["elapsed_s"], expected["temp"], label="expected (twin)", color="#ff7f0e",
                linestyle="--", linewidth=2)
    ax_temp.plot(baseline_t, baseline_temp, label="CEM baseline (twin)", color="#2ca02c",
                linestyle=":", linewidth=2)
    ax_temp.axhline(target, color="#888888", linestyle="--", alpha=0.6, label="target")
    ax_temp.set_ylabel("Temperature (deg C)")
    ax_temp.set_title("Real advisor-controlled run vs digital-twin expectation vs CEM baseline")
    ax_temp.grid(True, alpha=0.25)
    ax_temp.legend()

    dec_t = [d[0] for d in decisions]
    ax_pid.step(dec_t, [d[1] for d in decisions], where="post", label="kp")
    ax_pid.step(dec_t, [d[3] for d in decisions], where="post", label="kd")
    ax_pid.set_xlabel("Elapsed seconds")
    ax_pid.set_ylabel("kp / kd")
    ax_pid.grid(True, alpha=0.25)
    ax_pid.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.duration_s <= 0 or args.dt_s <= 0:
        raise ValueError("--duration-s and --dt-s must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"gru checkpoint : {args.gru_checkpoint}")
    print(f"ppo checkpoint : {args.ppo_checkpoint}")
    twin_model, twin_checkpoint = load_model(Path(args.gru_checkpoint), device)
    feature_names = list(twin_checkpoint.get("feature_names", []))
    window_steps = int(twin_checkpoint.get("window_steps", args.window_steps))

    ppo_model, ppo_checkpoint = load_ppo_policy(Path(args.ppo_checkpoint), device)
    args.temp_min = float(ppo_checkpoint["temp_min"])
    args.temp_max = float(ppo_checkpoint["temp_max"])
    args.kp_min, args.kp_max = float(ppo_checkpoint["kp_min"]), float(ppo_checkpoint["kp_max"])
    args.ki_min, args.ki_max = float(ppo_checkpoint["ki_min"]), float(ppo_checkpoint["ki_max"])
    args.kd_min, args.kd_max = float(ppo_checkpoint["kd_min"]), float(ppo_checkpoint["kd_max"])
    print(f"advisor        : mpc_hold_s={ppo_checkpoint['mpc_hold_s']:g}  "
          f"kp[{args.kp_min:g},{args.kp_max:g}] ki[{args.ki_min:g},{args.ki_max:g}] "
          f"kd[{args.kd_min:g},{args.kd_max:g}]")

    session = args.session_name or make_run_id(prefix="advisor_experiment")
    session_dir = Path(args.output_dir) / session
    session_dir.mkdir(parents=True, exist_ok=True)
    run_id = session

    if args.precondition:
        precondition_chamber(run_id, args)

    real, decisions, start_temp, aborted, abort_reason = run_real_experiment(
        run_id, ppo_model=ppo_model, ppo_checkpoint=ppo_checkpoint, device=device, args=args,
    )
    real = append_mae_column(real)
    real.to_csv(session_dir / "run_samples.csv", index=False)
    pd.DataFrame(decisions, columns=["elapsed_s", "kp", "ki", "kd"]).to_csv(session_dir / "decisions.csv", index=False)

    args.start_temp = start_temp

    pid_schedule = decisions_to_pid_schedule(decisions, float(args.duration_s), float(args.dt_s))
    warm_window, warm_terms = cold_start_seed(
        feature_names=feature_names, window_steps=window_steps, start_temp=start_temp,
        target_temp=float(args.target_temp), dt_s=float(args.dt_s),
        kp=pid_schedule[0][0], ki=pid_schedule[0][1], kd=pid_schedule[0][2],
    )
    expected = simulate_twin(
        model=twin_model, checkpoint=twin_checkpoint, feature_names=feature_names, window_steps=window_steps,
        device=device, args=args, start_temp=start_temp, target_temp=float(args.target_temp),
        duration_s=float(args.duration_s), dt_s=float(args.dt_s), pid_schedule=pid_schedule,
        warm_window=warm_window, warm_terms=warm_terms,
    )
    expected.to_csv(session_dir / "expected_trajectory.csv", index=False)

    twin_metrics, aligned = compare_trajectories(
        real["elapsed_s"].to_numpy(dtype=float), real["temp"].to_numpy(dtype=float),
        expected["elapsed_s"].to_numpy(dtype=float), expected["temp"].to_numpy(dtype=float),
        float(args.target_temp), args,
    )
    aligned.to_csv(session_dir / "twin_accuracy.csv", index=False)

    baseline_t, baseline_temp, baseline_pid, baseline_metrics, baseline_n_evaluated = compute_cem_baseline(
        model=twin_model, checkpoint=twin_checkpoint, feature_names=feature_names, window_steps=window_steps,
        device=device, args=args,
    )
    pd.DataFrame({"elapsed_s": baseline_t, "temp": baseline_temp, "temp_ref": float(args.target_temp)}).to_csv(
        session_dir / "baseline_trajectory.csv", index=False
    )

    improvement = improvement_table(
        real["elapsed_s"].to_numpy(dtype=float), real["temp"].to_numpy(dtype=float),
        baseline_t, baseline_temp, float(args.target_temp), args,
    )

    if not args.no_plot:
        plot_comparison(real, expected, baseline_t, baseline_temp, decisions, float(args.target_temp),
                        session_dir / "trajectory_comparison.png")

    report = {
        "run_id": run_id,
        "gru_checkpoint": str(Path(args.gru_checkpoint).resolve()),
        "ppo_checkpoint": str(Path(args.ppo_checkpoint).resolve()),
        "start_temp": start_temp,
        "target_temp": float(args.target_temp),
        "duration_s": float(args.duration_s),
        "mpc_hold_s": float(ppo_checkpoint["mpc_hold_s"]),
        "n_decisions": len(decisions),
        "aborted": aborted,
        "abort_reason": abort_reason,
        "baseline_pid": {"kp": baseline_pid[0], "ki": baseline_pid[1], "kd": baseline_pid[2]},
        "baseline_n_candidates_evaluated": baseline_n_evaluated,
        "twin_accuracy": twin_metrics,
        "advisor_vs_baseline": improvement,
    }
    (session_dir / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print("\n=== Digital-twin accuracy (expected vs real) ===")
    print(f"mae            : {twin_metrics['mae_temp']:.3f} deg C")
    print(f"rmse           : {twin_metrics['rmse_temp']:.3f} deg C")
    print(f"bias           : {twin_metrics['bias_temp']:+.3f} deg C")
    print(f"p90 abs error  : {twin_metrics['p90_abs_error_temp']:.3f} deg C")
    print(f"max abs error  : {twin_metrics['max_abs_error_temp']:.3f} deg C")
    print(f"final error    : {twin_metrics['final_error_temp']:+.3f} deg C")
    print(f"overshoot err  : real={fmt(twin_metrics.get('real_overshoot'))} "
          f"pred={fmt(twin_metrics.get('pred_overshoot'))} err={fmt(twin_metrics.get('err_overshoot'))}")

    print(f"\n=== Advisor (real) vs CEM baseline (kp={baseline_pid[0]:g}, ki={baseline_pid[1]:g}, "
          f"kd={baseline_pid[2]:g}) ===")
    for key, row in improvement.items():
        pct = row["improvement_percent"]
        pct_str = "n/a" if pct is None else f"{pct:+.1f}%"
        print(f"{key:10s}: advisor={fmt(row['advisor'])}  baseline={fmt(row['baseline'])}  improvement={pct_str}")

    if aborted:
        print(f"\nWARNING: experiment aborted early -- {abort_reason}")

    print(f"\nrun_samples          : {session_dir / 'run_samples.csv'}")
    print(f"decisions            : {session_dir / 'decisions.csv'}")
    print(f"expected_trajectory  : {session_dir / 'expected_trajectory.csv'}")
    print(f"baseline_trajectory  : {session_dir / 'baseline_trajectory.csv'}")
    print(f"report               : {session_dir / 'report.json'}")
    if not args.no_plot:
        print(f"plot                 : {session_dir / 'trajectory_comparison.png'}")


if __name__ == "__main__":
    main()
