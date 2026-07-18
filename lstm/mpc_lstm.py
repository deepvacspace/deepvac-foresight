#!/usr/bin/env python3
"""Continuous LSTM + MPC PID scheduler for a thermal chamber.

1. Reads the current simulated state.
2. Uses the LSTM as the plant model.
3. Optimizes PID coefficients with CEM/random shooting over a short horizon.
4. Applies the best PID only for a short hold interval.
5. Replans from the new state.

The receding-horizon rollout/optimizer loop is shared with gru/mpc_gru.py
via deepvac.mpc; this file supplies the LSTM-specific prediction function
and this script's own cost-weight design (see horizon_cost below).

Example:

    python lstm/mpc_lstm.py \
        --checkpoint scripts/lstm/validation_t1/lstm_t1.pt \
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
from typing import Dict, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from lstm.model import LSTMModel  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "Could not import LSTM/PID helpers. Run this script from the scripts "
        f"tree or add scripts/ and scripts/lstm/ to PYTHONPATH. Original error: {exc!r}"
    )

from deepvac import mpc as _mpc
from deepvac.mpc import load_candidate_table  # noqa: F401  (re-exported for callers)
from deepvac.pid import pid_bounds
from deepvac.schemas import DEFAULT_FEATURE_NAMES

DEFAULT_CHECKPOINT = SCRIPT_DIR / "validation_t1" / "lstm_t1.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "mpc_pid_runs"
DEFAULT_CANDIDATE_TABLE = DEFAULT_OUTPUT_DIR / "mpc_candidate_table.csv"


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[LSTMModel, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LSTMModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_delta_t1(
    model: LSTMModel,
    checkpoint: Dict[str, object],
    feature_window: np.ndarray,
    device: torch.device,
) -> float:
    x_scaler = checkpoint["x_scaler"]
    y_scaler = checkpoint["y_scaler"]

    n_features = feature_window.shape[-1]
    x_scaled = x_scaler.transform(
        feature_window.reshape(-1, n_features)
    ).reshape(feature_window.shape)

    xb = torch.as_tensor(x_scaled[None, :, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_scaled = model(xb).cpu().numpy()

    pred_real = y_scaler.inverse_transform(pred_scaled)
    return float(pred_real[0, 0])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Continuous LSTM+MPC PID scheduler using CEM/random shooting."
    )

    _mpc.add_common_mpc_args(ap)

    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--candidate-table", default=str(DEFAULT_CANDIDATE_TABLE),
                    help="Prebuilt CSV from mp_build.py. Empty string disables history seeding.")

    # MPC horizon/hold and cost weights are model-specific tuning, not shared
    # with gru/mpc_gru.py's cost design.
    ap.add_argument("--mpc-horizon-s", type=float, default=60.0,
                    help="Future horizon optimized at every MPC decision.")
    ap.add_argument("--mpc-hold-s", type=float, default=20.0,
                    help="How long to apply selected PID before replanning.")
    ap.add_argument("--w-overshoot-max", type=float, default=100.0)
    ap.add_argument("--w-overshoot-rmse", type=float, default=30.0)
    ap.add_argument("--w-abs-error", type=float, default=0.2)
    ap.add_argument("--w-final-abs-error", type=float, default=0.5)
    ap.add_argument("--w-near-std", type=float, default=2.0)
    ap.add_argument("--w-control-change", type=float, default=0.01,
                    help="Penalty for changing PID coefficients. Helps avoid noisy schedules.")

    return ap


# -----------------------------------------------------------------------------
# Model-specific cost design.
# -----------------------------------------------------------------------------


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
) -> Dict[str, float]:
    """LSTM MPC's cost design: overshoot (max + RMSE) + tracking error +
    final-error emphasis + a penalty for changing PID coefficients (helps
    avoid noisy schedules)."""
    temps = np.asarray(temps, dtype=float)
    error = target_temp - temps
    abs_error = np.abs(error)
    overshoot = _mpc.overshoot_array(temps, start_temp=start_temp, target_temp=target_temp)

    if temps.size == 0:
        return {"cost": float(args.w_invalid), "valid": False}

    near_mask = abs_error <= float(args.near_band)
    if np.any(near_mask):
        near_std = float(np.std(temps[near_mask]))
    else:
        near_std = 0.0

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

    print("=== Continuous LSTM + MPC PID scheduler ===")
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
            "overshoot_rmse": float(args.w_overshoot_rmse),
            "abs_error": float(args.w_abs_error),
            "final_abs_error": float(args.w_final_abs_error),
            "near_std": float(args.w_near_std),
            "control_change": float(args.w_control_change),
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
