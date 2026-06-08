#!/usr/bin/env python3
"""Shared Bayesian optimization, cost, and run-file utilities."""

from __future__ import annotations

import csv
import json
import math
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def append_rows_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def history_run_file(run_id: str, filename: str, folder_name: str = "history") -> str:
    return str(Path(folder_name) / run_id / Path(filename).name)


def load_runs_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)

    # Fallback for run-scoped storage under history/<run_id>/<filename>.
    target_name = p.name
    tables = []
    for match in Path("history").glob(f"*/{target_name}"):
        if match.is_file() and match.stat().st_size > 0:
            tables.append(pd.read_csv(match))
    if tables:
        return pd.concat(tables, ignore_index=True)

    return pd.DataFrame(columns=["run_id", "kp", "ki", "kd", "mse"])


def fit_gp_model(runs_df: pd.DataFrame) -> Dict[str, object]:
    required = ["kp", "ki", "kd", "mse"]
    missing = [c for c in required if c not in runs_df.columns]
    if missing:
        raise ValueError(f"Missing columns in runs table: {missing}")

    clean = runs_df.dropna(subset=required).copy()
    if len(clean) < 3:
        raise ValueError("At least 3 completed runs are required to fit a GP model")

    X = clean[["kp", "ki", "kd"]].to_numpy(dtype=float)
    y = clean["mse"].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(
        noise_level=1e-4,
        noise_level_bounds=(1e-8, 1e-1),
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp.fit(Xs, y)

    return {
        "scaler": scaler,
        "gp": gp,
        "n_samples": int(len(clean)),
        "best_mse": float(np.min(y)),
    }


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * _normal_cdf(z) + sigma_safe * _normal_pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def suggest_next_params(
    model: Dict[str, object],
    bounds: Dict[str, Tuple[float, float]],
    n_candidates: int = 5000,
    xi: float = 0.01,
    random_state: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)

    kp_lo, kp_hi = bounds["kp"]
    ki_lo, ki_hi = bounds["ki"]
    kd_lo, kd_hi = bounds["kd"]

    candidates = np.column_stack(
        [
            rng.uniform(kp_lo, kp_hi, size=n_candidates),
            rng.uniform(ki_lo, ki_hi, size=n_candidates),
            rng.uniform(kd_lo, kd_hi, size=n_candidates),
        ]
    )

    scaler: StandardScaler = model["scaler"]
    gp: GaussianProcessRegressor = model["gp"]

    Xs = scaler.transform(candidates)
    mu, std = gp.predict(Xs, return_std=True)

    y_best = float(model["best_mse"])
    ei = expected_improvement(mu=mu, sigma=std, y_best=y_best, xi=xi)
    idx = int(np.argmax(ei))

    return {
        "kp": float(candidates[idx, 0]),
        "ki": float(candidates[idx, 1]),
        "kd": float(candidates[idx, 2]),
        "pred_mse": float(mu[idx]),
        "pred_std": float(std[idx]),
        "expected_improvement": float(ei[idx]),
        "incumbent_mse": y_best,
    }


def save_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

def parse_bounds(text: str) -> Tuple[float, float]:
    
    if not isinstance(text, str):
        raise ValueError(f"Bounds must be a string, got: {type(text)}")

    parts = [p.strip() for p in text.split(",")]

    if len(parts) != 2:
        raise ValueError(f"Invalid bounds format: '{text}'. Expected 'low,high'")

    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        raise ValueError(f"Bounds must be numeric: '{text}'")

    if lo >= hi:
        raise ValueError(f"Invalid bounds '{text}': low must be < high")

    return lo, hi

def append_mae_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with a per-row absolute error column named 'mae'
    """
    required = {"temp", "temp_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"append_mae_column missing required columns: {sorted(missing)}")

    out = df.copy()
    out["mae"] = (out["temp_ref"].astype(float) - out["temp"].astype(float)).abs()
    return out


def compute_tail_cost(
    df: pd.DataFrame,
    entry_band: float = 2.0,
    overshoot_weight: float = 10.0,
) -> dict:
    """
    Compute tail cost after first entering a target band.

    cost = tail_mae + overshoot_weight * overshoot^2
    """
    required = {"timestamp", "temp", "temp_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_tail_cost missing required columns: {sorted(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    temp = df["temp"].astype(float).to_numpy()
    temp_ref = df["temp_ref"].astype(float).to_numpy()
    target = float(temp_ref[0])
    start_temp = float(temp[0])

    # 1 for cooling, -1 for heating
    direction = 1.0 if start_temp > target else -1.0

    abs_err = np.abs(temp - target)

    idx = np.where(abs_err <= entry_band)[0]

    if len(idx) == 0:
        return {
            "cost": 1e9,
            "tail_mae": None,
            "overshoot": None,
            "tail_start_index": None,
            "target": target,
            "start_temp": start_temp,
            "direction": direction,
        }

    start_idx = int(idx[0])
    tail_temp = temp[start_idx:]

    tail_mae = float(np.mean(np.abs(tail_temp - target)))

    dev = tail_temp - target

    if direction > 0:
        wrong_dev = np.maximum(0.0, -dev)
    else:
        wrong_dev = np.maximum(0.0, dev)

    overshoot = float(np.max(wrong_dev)) if len(wrong_dev) else 0.0

    cost = tail_mae + overshoot_weight * (overshoot ** 2)

    return {
        "cost": cost,
        "tail_mae": tail_mae,
        "overshoot": overshoot,
        "tail_start_index": start_idx,
        "target": target,
        "start_temp": start_temp,
        "direction": direction,
    }
