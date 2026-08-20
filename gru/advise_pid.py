#!/usr/bin/env python3
"""AI advisor: recommend PID gains for a temperature setpoint using the trained
GRU plant model as a simulator, with CEM/random-shooting search over it.

    --mode suggest   One CEM search over the whole intended run. Returns a
                     triplet and its predicted whole-run outcome. Seeded cold by
                     default, or from a real run via --context-csv.

    --mode adapt     Receding-horizon control: replans on a short --mpc-horizon-s
                     every --mpc-hold-s, for the whole --duration-s.

Point --checkpoint at a rollout-trained checkpoint (gru/train_gru_rollout.py's
output), not the plain one-step gru_t1.pt.

Defaults to the ~24 -> 0 problem (experiments/run_history's baseline); override
--start-temp/--target-temp for anything else the twin has data for.

Examples:

    python -m gru.advise_pid --mode suggest --checkpoint gru/validation_rollout/promoted.pt
    python -m gru.advise_pid --mode adapt   --checkpoint gru/validation_rollout/promoted.pt \
        --duration-s 1200 --mpc-horizon-s 80 --mpc-hold-s 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import make_run_id  # noqa: E402
from deepvac.datasets import prepare_run_dataframe  # noqa: E402
from deepvac.mpc import (  # noqa: E402
    SimState,
    add_common_mpc_args,
    initialize_feature_window,
    optimize_pid_for_state,
    overshoot_array,
    rollout_constant_pid,
    run_mpc_simulation,
)
from deepvac.pid import pid_bounds  # noqa: E402

from gru.gru_common import ChamberPID, CodesysDiff, load_model, predict_delta_t1  # noqa: E402
from gru.twin_acceptance import describe_trajectory, warm_start_at  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "advisor"


def add_cost_args(ap: argparse.ArgumentParser) -> None:
    """CLI weights for horizon_cost()."""
    ap.add_argument("--w-overshoot-max", type=float, default=100.0)
    ap.add_argument("--w-overshoot-rmse", type=float, default=30.0)
    ap.add_argument("--w-abs-error", type=float, default=0.2)
    ap.add_argument("--w-final-abs-error", type=float, default=0.5)
    ap.add_argument("--w-near-std", type=float, default=2.0)
    ap.add_argument("--w-control-change", type=float, default=0.01,
                    help="Penalty for changing PID coefficients. Helps avoid noisy schedules.")


def normalized_pid_distance(a: np.ndarray, b: np.ndarray, args: argparse.Namespace) -> float:
    lo, hi = pid_bounds(args)
    denom = np.maximum(hi - lo, 1e-9)
    z = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) / denom
    return float(np.sqrt(np.mean(np.square(z))))


def horizon_cost(
    *,
    temps: np.ndarray,
    candidate_pid: np.ndarray,
    previous_pid: np.ndarray,
    start_temp: float,
    target_temp: float,
    valid: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Rollout cost: overshoot (max + RMSE), tracking error, final-error
    emphasis, and a penalty for changing PID coefficients."""
    temps = np.asarray(temps, dtype=float)
    if temps.size == 0:
        return {"cost": float(args.w_invalid), "valid": False}

    error = target_temp - temps
    abs_error = np.abs(error)
    overshoot = overshoot_array(temps, start_temp=start_temp, target_temp=target_temp)

    near_mask = abs_error <= float(args.near_band)
    near_std = float(np.std(temps[near_mask])) if np.any(near_mask) else 0.0

    overshoot_max = float(np.max(overshoot))
    overshoot_rmse = float(np.sqrt(np.mean(np.square(overshoot))))
    mae = float(np.mean(abs_error))
    final_abs_error = float(abs_error[-1])
    change_penalty = normalized_pid_distance(candidate_pid, previous_pid, args)

    cost = (
        float(args.w_overshoot_max) * overshoot_max
        + float(args.w_overshoot_rmse) * overshoot_rmse
        + float(args.w_abs_error) * mae
        + float(args.w_final_abs_error) * final_abs_error
        + float(args.w_near_std) * near_std
        + float(args.w_control_change) * change_penalty
    )
    if not valid:
        cost += float(args.w_invalid)

    return {
        "cost": float(cost),
        "valid": bool(valid),
        "horizon_mae": mae,
        "horizon_final_abs_error": final_abs_error,
        "horizon_near_std": near_std,
        "horizon_overshoot_max": overshoot_max,
        "horizon_overshoot_rmse": overshoot_rmse,
        "pid_change_norm": change_penalty,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="AI advisor: recommend or continuously adapt PID gains via CEM search over the GRU twin."
    )

    ap.add_argument("--mode", choices=["suggest", "adapt"], default="suggest")
    ap.add_argument("--checkpoint", required=True,
                    help="A rollout-trained checkpoint (gru/train_gru_rollout.py's output).")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--session-name", default=None, help="Output subfolder name. Default: generated id.")
    ap.add_argument("--no-plot", action="store_true")

    ap.add_argument(
        "--context-csv", default=None,
        help="A real run_samples.csv to seed the twin's window from, instead of the "
             "no-telemetry PID-driven estimate (--mode suggest only).",
    )
    ap.add_argument(
        "--context-offset", type=int, default=None,
        help="Row offset into --context-csv. Default: window_steps - 1 (that run's own start).",
    )
    ap.add_argument("--mpc-horizon-s", type=float, default=60.0,
                    help="Future horizon optimized at every MPC decision (--mode adapt).")
    ap.add_argument("--mpc-hold-s", type=float, default=20.0,
                    help="How long to apply a selected PID before replanning (--mode adapt).")

    add_common_mpc_args(ap)
    add_cost_args(ap)
    ap.set_defaults(start_temp=24.0, target_temp=0.0)

    return ap


def build_initial_state(
    *,
    checkpoint: dict[str, object],
    feature_names: list,
    window_steps: int,
    args: argparse.Namespace,
    verbose: bool = True,
) -> SimState:
    """Seed one SimState, from --context-csv if given, else a PID-driven cold start."""
    start_temp = float(args.start_temp)
    kp0, ki0, kd0 = float(args.kp_init), float(args.ki_init), float(args.kd_init)
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)

    pid = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    diff = CodesysDiff()
    diff.prev_value = start_temp

    if args.context_csv:
        loader_args = argparse.Namespace(min_samples=1, min_duration_s=0.0)
        df, _ = prepare_run_dataframe(Path(args.context_csv), loader_args)
        offset = args.context_offset if args.context_offset is not None else window_steps - 1
        if offset < window_steps - 1 or offset >= len(df):
            raise ValueError(
                f"--context-offset must be in [{window_steps - 1}, {len(df) - 1}] for this CSV, got {offset}"
            )
        feature_window, terms = warm_start_at(df, feature_names, window_steps, offset)
        pid.p_part = float(terms.get("temp_u_p", 0.0)) / feature_scale
        pid.i_part = float(terms.get("temp_u_i", 0.0)) / feature_scale
        pid.d_part = float(terms.get("temp_u_d", 0.0)) / feature_scale
        if kd0 != 0.0:
            diff_out = -(pid.d_part * kp0) / kd0
            diff.out = diff_out
            diff.filter_out = diff_out / 10.0
        if verbose:
            print(f"context    : warm-started from {args.context_csv} @ row {offset}")
    else:
        precondition_ref = start_temp if args.precondition_ref is None else float(args.precondition_ref)
        feature_window = initialize_feature_window(
            feature_names=feature_names, window_steps=window_steps,
            start_temp=start_temp, precondition_ref=precondition_ref,
            dt_s=float(args.dt_s), kp=kp0, ki=ki0, kd=kd0,
        )
        pid.i_part = float(args.initial_i)
        pid.d_part = float(args.initial_d)
        pid.p_part = float(args.initial_p)
        if verbose:
            print("context    : no real telemetry, PID-driven cold-start estimate")

    return SimState(
        elapsed_s=0.0, temp=start_temp, previous_temp=start_temp,
        feature_window=feature_window, pid=pid, diff=diff,
        kp=kp0, ki=ki0, kd=kd0,
    )


def run_suggest(
    *,
    model, checkpoint: dict[str, object], feature_names: list, window_steps: int,
    device: torch.device, args: argparse.Namespace, session_dir: Path,
) -> None:
    state = build_initial_state(
        checkpoint=checkpoint, feature_names=feature_names, window_steps=window_steps, args=args,
    )

    suggest_args = argparse.Namespace(**vars(args))
    suggest_args.mpc_horizon_s = float(args.duration_s)

    rng = np.random.default_rng(int(args.seed))
    decision = optimize_pid_for_state(
        state=state, model=model, checkpoint=checkpoint, feature_names=feature_names,
        device=device, args=suggest_args, rng=rng,
        predict_fn=predict_delta_t1, cost_fn=horizon_cost, decision_idx=1,
    )
    pid = np.asarray([decision["kp"], decision["ki"], decision["kd"]], dtype=float)

    horizon_steps = max(1, int(math.ceil(float(args.duration_s) / float(args.dt_s))))
    metrics, temps = rollout_constant_pid(
        initial_state=state, candidate_pid=pid, model=model, checkpoint=checkpoint,
        feature_names=feature_names, device=device, args=suggest_args,
        horizon_steps=horizon_steps, predict_fn=predict_delta_t1, cost_fn=horizon_cost,
    )

    times = np.arange(1, len(temps) + 1, dtype=float) * float(args.dt_s)
    temps_arr = np.asarray(temps, dtype=float)
    desc = describe_trajectory(times, temps_arr, float(args.target_temp), args)

    trajectory = pd.DataFrame({"elapsed_s": times, "temp": temps_arr, "temp_ref": float(args.target_temp)})
    trajectory.to_csv(session_dir / "trajectory.csv", index=False)

    report = {
        "mode": "suggest",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "start_temp": args.start_temp, "target_temp": args.target_temp, "duration_s": args.duration_s,
        "recommended_pid": {"kp": float(pid[0]), "ki": float(pid[1]), "kd": float(pid[2])},
        "n_candidates_evaluated": int(decision["n_evaluated"]),
        "valid": bool(metrics["valid"]),
        **desc,
        "cost": float(metrics["cost"]),
        "horizon_overshoot_max": float(metrics.get("horizon_overshoot_max", float("nan"))),
    }
    (session_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_suggest(times, temps_arr, args, session_dir / "trajectory.png")

    print("\n=== Recommended PID ===")
    print(f"kp={pid[0]:g} ki={pid[1]:g} kd={pid[2]:g}  "
          f"({int(decision['n_evaluated'])} candidates evaluated over a {args.duration_s:g}s horizon)")
    print("\n=== Predicted outcome ===")
    print(f"final temp     : {desc['final_temp']:.3f} deg C")
    print(f"overshoot      : {desc['overshoot']:.3f} deg C")
    print(f"tail MAE       : {desc['tail_mae']:.3f} deg C (last {args.tail_window_s:g}s)")
    print(f"time to near   : {fmt_time(desc['time_to_near_s'])} (band {args.near_band:g} deg C)")
    print(f"time to settle : {fmt_time(desc['time_to_settle_s'])} (band {args.settle_band:g} deg C)")
    print(f"\ntrajectory : {session_dir / 'trajectory.csv'}")
    print(f"report     : {session_dir / 'report.json'}")


def run_adapt(
    *,
    model, checkpoint: dict[str, object], feature_names: list, window_steps: int,
    device: torch.device, args: argparse.Namespace, session_dir: Path,
) -> None:
    if args.context_csv:
        print("WARNING: --context-csv is ignored in --mode adapt; "
              "run_mpc_simulation always cold-starts (see its own docstring).")

    trajectory, decisions, metrics = run_mpc_simulation(
        model=model, checkpoint=checkpoint, feature_names=feature_names,
        window_steps=window_steps, device=device, args=args,
        predict_fn=predict_delta_t1, cost_fn=horizon_cost,
    )

    trajectory.to_csv(session_dir / "trajectory.csv", index=False)
    decisions.to_csv(session_dir / "decisions.csv", index=False)
    report = {
        "mode": "adapt",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "start_temp": args.start_temp, "target_temp": args.target_temp,
        "duration_s": args.duration_s, "mpc_horizon_s": args.mpc_horizon_s, "mpc_hold_s": args.mpc_hold_s,
        "n_decisions": int(len(decisions)),
        "n_pid_changes": int(metrics["pid_changes"]),
        **metrics,
    }
    (session_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_adapt(trajectory, decisions, args, session_dir / "trajectory.png")

    print("\n=== Adaptive run summary ===")
    print(f"decisions       : {len(decisions)} ({int(metrics['pid_changes'])} actually changed the PID)")
    print(f"final temp      : {metrics['end_temp']:.3f} deg C (error {metrics['final_error']:+.3f})")
    print(f"overshoot (max) : {metrics['overshoot_max']:.3f} deg C")
    print(f"tail MAE        : {metrics['tail_mae']:.3f} deg C (last {args.tail_window_s:g}s)")
    print(f"time to near    : {fmt_time(metrics['time_to_near_s'])} (band {args.near_band:g} deg C)")
    print(f"time to settle  : {fmt_time(metrics['time_to_settle_s'])} (band {args.settle_band:g} deg C)")
    print(f"\ntrajectory : {session_dir / 'trajectory.csv'}")
    print(f"decisions  : {session_dir / 'decisions.csv'}")
    print(f"report     : {session_dir / 'report.json'}")


def fmt_time(value) -> str:
    return "never" if value is None else f"{value:.0f}s"


def plot_suggest(times: np.ndarray, temps: np.ndarray, args: argparse.Namespace, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, temps, label="predicted temp")
    ax.axhline(args.target_temp, color="#888888", linestyle="--", label="target")
    ax.axhspan(args.target_temp - args.near_band, args.target_temp + args.near_band,
              color="#888888", alpha=0.1, label=f"+-{args.near_band:g} deg C band")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title("Suggested PID: predicted whole-run trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_adapt(trajectory: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_temp, ax_pid) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_temp.plot(trajectory["elapsed_s"], trajectory["temp"], label="temp")
    ax_temp.axhline(args.target_temp, color="#888888", linestyle="--", label="target")
    changed = decisions[decisions["changed"]]
    for t in changed["elapsed_s"]:
        ax_temp.axvline(t, color="#d62728", alpha=0.2, linewidth=1.0)
    ax_temp.set_ylabel("Temperature (deg C)")
    ax_temp.set_title(f"Adaptive run: {int(changed.shape[0])} PID changes over {len(decisions)} decisions")
    ax_temp.grid(True, alpha=0.25)
    ax_temp.legend()

    ax_pid.step(decisions["elapsed_s"], decisions["kp"], where="post", label="kp")
    ax_pid.step(decisions["elapsed_s"], decisions["kd"], where="post", label="kd")
    ax_pid.set_xlabel("Elapsed seconds")
    ax_pid.set_ylabel("kp / kd")
    ax_pid.grid(True, alpha=0.25)
    ax_pid.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.duration_s <= 0 or args.dt_s <= 0:
        raise ValueError("--duration-s and --dt-s must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", []))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    session = args.session_name or make_run_id(prefix=f"advise_{args.mode}")
    session_dir = Path(args.output_dir) / session
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode       : {args.mode}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"candidate  : {args.start_temp:g} -> {args.target_temp:g} deg C over {args.duration_s:g}s")

    if args.mode == "suggest":
        run_suggest(model=model, checkpoint=checkpoint, feature_names=feature_names,
                    window_steps=window_steps, device=device, args=args, session_dir=session_dir)
    else:
        run_adapt(model=model, checkpoint=checkpoint, feature_names=feature_names,
                  window_steps=window_steps, device=device, args=args, session_dir=session_dir)


if __name__ == "__main__":
    main()
