#!/usr/bin/env python3
"""Continuous GRU + MPC PID scheduler for a thermal chamber.

1. Reads the current simulated state.
2. Uses the GRU as the plant model.
3. Optimizes PID coefficients with CEM/random shooting over a short horizon.
4. Applies the best PID only for a short hold interval.
5. Replans from the new state.

The receding-horizon rollout/optimizer loop is shared with lstm/mpc_lstm.py
via deepvac.mpc; this file supplies the GRU-specific prediction function and
this script's own cost-weight design (see horizon_cost below).

Example:

    python gru_mpc_continuous.py \
        --checkpoint scripts/gru/validation_t1/gru_t1.pt \
        --start-temp 27 \
        --target-temp 0 \
        --duration-s 1200 \
        --dt-s 2 \
        --mpc-horizon-s 60 \
        --mpc-hold-s 10

"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from gru.gru_common import (  # type: ignore
        DEFAULT_CHECKPOINT,
        DEFAULT_FEATURE_NAMES,
        load_model,
        predict_delta_t1,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import gru_common.py. Put this script next to gru_common.py "
        f"or add that directory to PYTHONPATH. Original error: {exc!r}"
    )

from deepvac import mpc as _mpc
from deepvac.mpc import CandidateTable, load_candidate_table  # noqa: F401  (re-exported for callers)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "mpc_pid_runs"
DEFAULT_CANDIDATE_TABLE = DEFAULT_OUTPUT_DIR / "mpc_candidate_table.csv"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Continuous GRU+MPC PID scheduler using CEM/random shooting."
    )

    _mpc.add_common_mpc_args(ap)

    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--candidate-table", default=str(DEFAULT_CANDIDATE_TABLE),
                    help="Prebuilt CSV from mp_build.py. Empty string disables history seeding.")

    # MPC horizon/hold and cost weights are model-specific tuning, not shared
    # with lstm/mpc_lstm.py's cost design.
    ap.add_argument("--mpc-horizon-s", type=float, default=80.0,
                    help="Future horizon optimized at every MPC decision.")
    ap.add_argument("--mpc-hold-s", type=float, default=20.0,
                    help="How long to apply selected PID before replanning.")
    ap.add_argument("--w-overshoot-max", type=float, default=80.0)
    ap.add_argument("--w-abs-error", type=float, default=2.0)
    ap.add_argument("--w-motion", type=float, default=20.0)
    ap.add_argument("--motion-error-scale", type=float, default=5.0,
                    help="Larger values make the controller start braking earlier.")
    ap.add_argument("--w-near-std", type=float, default=1.0)

    return ap


# -----------------------------------------------------------------------------
# Model-specific cost design.
# -----------------------------------------------------------------------------


def horizon_cost(
    *,
    temps: np.ndarray,
    candidate_pid: np.ndarray,
    previous_pid: np.ndarray,
    start_temp: float,
    target_temp: float,
    valid: bool,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """GRU MPC's cost design: overshoot + tracking error + a "motion" term
    that penalizes velocity while still far from the target (weighted down
    by closeness so it doesn't fight the final approach)."""
    temps = np.asarray(temps, dtype=float)

    if temps.size == 0:
        return {
            "cost": float(args.w_invalid),
            "valid": False,
            "horizon_mae": float("nan"),
            "horizon_near_std": float("nan"),
            "horizon_overshoot_max": float("nan"),
            "horizon_weighted_motion": float("nan"),
            "horizon_mean_abs_velocity": float("nan"),
        }

    error = float(target_temp) - temps
    abs_error = np.abs(error)

    overshoot = _mpc.overshoot_array(temps, start_temp=float(start_temp), target_temp=float(target_temp))
    overshoot_max = float(np.max(overshoot))
    mae = float(np.mean(abs_error))

    near_mask = abs_error <= float(args.near_band)
    if np.any(near_mask):
        near_std = float(np.std(temps[near_mask]))
    else:
        near_std = 0.0

    if temps.size >= 2:
        dt = max(float(args.dt_s), 1e-9)
        temp_velocity = np.diff(temps) / dt
        err_for_velocity = abs_error[:-1]
        motion_error_scale = max(float(args.motion_error_scale), 1e-9)
        closeness_weight = np.exp(-err_for_velocity / motion_error_scale)
        weighted_motion = float(np.mean(closeness_weight * np.square(temp_velocity)))
        mean_abs_velocity = float(np.mean(np.abs(temp_velocity)))
    else:
        weighted_motion = 0.0
        mean_abs_velocity = 0.0

    cost = (
        float(args.w_overshoot_max) * overshoot_max
        + float(args.w_abs_error) * mae
        + float(args.w_motion) * weighted_motion
        + float(args.w_near_std) * near_std
    )
    if not valid:
        cost += float(args.w_invalid)

    return {
        "cost": float(cost),
        "valid": bool(valid),
        "horizon_mae": mae,
        "horizon_near_std": near_std,
        "horizon_overshoot_max": overshoot_max,
        "horizon_weighted_motion": weighted_motion,
        "horizon_mean_abs_velocity": mean_abs_velocity,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"mpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    feature_names = list(checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))
    candidate_table = load_candidate_table(args)

    print("=== Continuous GRU + MPC PID scheduler ===")
    print(f"checkpoint:       {checkpoint_path}")
    print(f"device:           {device}")
    print(f"features:         {feature_names}")
    print(f"window_steps:     {window_steps}")
    print(f"scenario:         {args.start_temp} °C -> {args.target_temp} °C")
    print(f"duration/dt:      {args.duration_s}s / {args.dt_s}s")
    print(f"MPC horizon/hold: {args.mpc_horizon_s}s / {args.mpc_hold_s}s")
    print(f"optimizer:        {args.optimizer}")
    print(f"population/iters: {args.cem_population} / {args.cem_iterations}")
    print(f"candidate table:  {candidate_table.path if candidate_table is not None else 'disabled'}")
    print(
        f"PID bounds:       kp=({args.kp_min},{args.kp_max}) ki=({args.ki_min},{args.ki_max}) kd=({args.kd_min},{args.kd_max})")
    print(f"output dir:       {output_dir}")

    trajectory, decisions, metrics = _mpc.run_mpc_simulation(
        model=model,
        checkpoint=checkpoint,
        feature_names=feature_names,
        window_steps=window_steps,
        device=device,
        args=args,
        predict_fn=predict_delta_t1,
        cost_fn=horizon_cost,
        candidate_table=candidate_table,
    )

    trajectory_csv = output_dir / "mpc_trajectory.csv"
    decisions_csv = output_dir / "mpc_decisions.csv"
    summary_json = output_dir / "mpc_summary.json"

    if bool(args.save_trajectory):
        trajectory.to_csv(trajectory_csv, index=False)
    decisions.to_csv(decisions_csv, index=False)

    summary = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "feature_names": feature_names,
        "window_steps": window_steps,
        "scenario": {
            "start_temp": float(args.start_temp),
            "target_temp": float(args.target_temp),
            "duration_s": float(args.duration_s),
            "dt_s": float(args.dt_s),
            "precondition_ref": float(args.start_temp if args.precondition_ref is None else args.precondition_ref),
        },
        "mpc": {
            "optimizer": args.optimizer,
            "horizon_s": float(args.mpc_horizon_s),
            "hold_s": float(args.mpc_hold_s),
            "population": int(args.cem_population),
            "iterations": int(args.cem_iterations),
            "elite_frac": float(args.cem_elite_frac),
            "apply_margin": float(args.apply_margin),
            "candidate_table": str(candidate_table.path) if candidate_table is not None else None,
            "candidate_table_rows": int(len(candidate_table.rows)) if candidate_table is not None else 0,
            "history_candidates": int(args.history_candidates),
            "history_neighbor_pool": int(args.history_neighbor_pool),
        },
        "bounds": {
            "kp": [float(args.kp_min), float(args.kp_max)],
            "ki": [float(args.ki_min), float(args.ki_max)],
            "kd": [float(args.kd_min), float(args.kd_max)],
        },
        "cost_weights": {
            "overshoot_max": float(args.w_overshoot_max),
            "abs_error": float(args.w_abs_error),
            "motion": float(args.w_motion),
            "motion_error_scale": float(args.motion_error_scale),
            "near_std": float(args.w_near_std),
        },
        "metrics": metrics,
        "trajectory_csv": str(trajectory_csv) if bool(args.save_trajectory) else None,
        "decisions_csv": str(decisions_csv),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final metrics ===")
    for key in [
        "overshoot_max", "overshoot_rmse", "tail_mae", "tail_std",
        "final_abs_error", "end_temp", "time_to_near_s", "time_to_settle_s", "pid_changes",
    ]:
        print(f"{key:18s}: {metrics[key]}")

    print("\n=== Saved ===")
    if bool(args.save_trajectory):
        print(f"trajectory csv: {trajectory_csv}")
    print(f"decisions csv:  {decisions_csv}")
    print(f"summary json:   {summary_json}")


if __name__ == "__main__":
    main()
