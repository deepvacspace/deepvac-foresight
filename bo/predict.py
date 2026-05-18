#!/usr/bin/env python3
"""Load BO GP model and print the next PID values to try"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

from bo_common import parse_bounds, suggest_next_params


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Predict the next PID parameters using the current saved BO model."
    )

    ap.add_argument("--model-in", default="history/bo_gp_model.pkl")

    ap.add_argument("--kp-bounds", default="6,50")
    ap.add_argument("--ki-bounds", default="200,1000")
    ap.add_argument("--kd-bounds", default="5,20")

    ap.add_argument("--n-candidates", type=int, default=7000)
    ap.add_argument("--xi", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    model_path = Path(args.model_in)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bounds: Dict[str, Tuple[float, float]] = {
        "kp": parse_bounds(args.kp_bounds),
        "ki": parse_bounds(args.ki_bounds),
        "kd": parse_bounds(args.kd_bounds),
    }

    with model_path.open("rb") as fh:
        model = pickle.load(fh)

    required_keys = {"scaler", "gp", "n_samples", "best_mse"}
    missing = required_keys - set(model.keys())
    if missing:
        raise ValueError(f"Model missing keys: {sorted(missing)}")

    suggestion = suggest_next_params(
        model=model,
        bounds=bounds,
        n_candidates=args.n_candidates,
        xi=args.xi,
        random_state=args.seed,
    )

    print(f"Loaded model from: {model_path}")
    print(f"n_training_runs={int(model['n_samples'])}")
    print(f"best_observed_cost={float(suggestion['incumbent_mse']):.6f}")
    print()
    print("Predicted next params:")
    print(f"  kp = {float(suggestion['kp']):.6f}")
    print(f"  ki = {float(suggestion['ki']):.6f}")
    print(f"  kd = {float(suggestion['kd']):.6f}")
    print()
    print("Prediction diagnostics:")
    print(f"  predicted_cost_mean = {float(suggestion['pred_mse']):.6f}")
    print(f"  predicted_cost_std  = {float(suggestion['pred_std']):.6f}")
    print(f"  expected_improvement = {float(suggestion['expected_improvement']):.6f}")


if __name__ == "__main__":
    main()