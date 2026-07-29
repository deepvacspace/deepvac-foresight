"""Shared fixtures for the deepvac test suite.

Two kinds of synthetic data get reused across unit and integration tests:

- A `history_root` of synthetic run_samples.csv files (write_synthetic_run),
  standing in for optimization/run_history/ -- enough real structure
  (elapsed_s, temp approaching temp_ref, constant PID, zeroed control terms)
  to exercise deepvac.datasets' loading/sequence-building/splitting without
  needing real chamber telemetry.
- A tiny, fast-to-train GRU checkpoint (tiny_gru_checkpoint) in the exact
  torch.save() shape gru/gru_common.py:load_model expects, so MPC rollout,
  checkpoint round-trip, and ONNX export tests don't need a real trained
  model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from deepvac.schemas import DEFAULT_FEATURE_NAMES


def write_synthetic_run(
    root: Path,
    run_id: str,
    *,
    n_samples: int = 200,
    start_temp: float = 25.0,
    target_temp: float = 0.0,
    kp: float = 6.0,
    ki: float = 997.0,
    kd: float = 16.0,
    dt_s: float = 1.0,
    seed: int = 0,
) -> Path:
    """Write a synthetic run_samples.csv exponentially decaying start_temp ->
    target_temp, with small deterministic noise. Returns the run's directory."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float) * dt_s
    # Simple first-order decay toward target_temp, small noise so a model has
    # something nontrivial (but easy) to fit.
    tau = max(n_samples * dt_s / 4.0, 1.0)
    temp = target_temp + (start_temp - target_temp) * np.exp(-t / tau)
    temp += rng.normal(scale=0.01, size=n_samples)

    df = pd.DataFrame({
        "timestamp": 1_700_000_000.0 + t,
        "temp": temp,
        "temp_ref": np.full(n_samples, target_temp),
        "kp": np.full(n_samples, kp),
        "ki": np.full(n_samples, ki),
        "kd": np.full(n_samples, kd),
        "temp_u": np.zeros(n_samples),
        "temp_u_p": np.zeros(n_samples),
        "temp_u_i": np.zeros(n_samples),
        "temp_u_d": np.zeros(n_samples),
    })

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "run_samples.csv", index=False)
    return run_dir


@pytest.fixture
def history_root(tmp_path: Path) -> Path:
    """A run-history directory with 4 synthetic runs -- enough for
    deepvac.datasets.split_runs' >= 3 run-level train/val/test requirement."""
    root = tmp_path / "run_history"
    scenarios = [
        ("run_0001", 25.0, 0.0, 0),
        ("run_0002", 30.0, 5.0, 1),
        ("run_0003", 20.0, -5.0, 2),
        ("run_0004", 27.0, 0.0, 3),
    ]
    for run_id, start_temp, target_temp, seed in scenarios:
        write_synthetic_run(root, run_id, start_temp=start_temp, target_temp=target_temp, seed=seed)
    return root


def build_tiny_gru_checkpoint(
    path: Path,
    *,
    window_steps: int = 5,
    hidden_dim: int = 4,
    num_layers: int = 1,
    seed: int = 0,
) -> Path:
    """Build and torch.save() a tiny, randomly-initialized GRU checkpoint in
    the exact shape gru/gru_common.py:load_model expects -- fast enough to
    construct fresh in every test that needs "a trained checkpoint" without
    depending on gru/validation_t1/gru_t1.pt or any real training run."""
    from gru.gru_common import GRUModel
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    n_features = len(DEFAULT_FEATURE_NAMES)
    model = GRUModel(input_dim=n_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.0)
    model.eval()

    rng = np.random.default_rng(seed)
    x_scaler = StandardScaler().fit(rng.normal(size=(50, n_features)))
    y_scaler = StandardScaler().fit(rng.normal(scale=0.1, size=(50, 1)))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": n_features,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": 0.0,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "window_steps": window_steps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def tiny_gru_checkpoint(tmp_path: Path) -> Path:
    return build_tiny_gru_checkpoint(tmp_path / "tiny_gru_t1.pt")
