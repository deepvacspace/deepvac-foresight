"""Integration smoke test for the GRU training pipeline
(gru/train_gru.py, sharing deepvac/datasets.py with lstm/train_lstm.py).

Trains a tiny model for 2 epochs on synthetic data. This asserts the
pipeline runs end-to-end and produces a well-formed, loadable checkpoint
plus a validation report with finite metrics -- it does NOT assert anything
about model quality (that would be flaky for a 2-epoch/4-hidden-unit model
on synthetic data). Marked slow: this is the one test in the suite that
actually trains a neural network, so it's excluded from the fast default
loop and run explicitly (`pytest -m slow`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_train_and_validate_produces_a_loadable_checkpoint(history_root: Path, tmp_path: Path):
    from gru.gru_common import load_model
    from gru.train_gru import build_arg_parser, train_and_validate

    output_dir = tmp_path / "output"
    plots_dir = tmp_path / "plots"

    args = build_arg_parser().parse_args([
        "--history-root", str(history_root),
        "--output-dir", str(output_dir),
        "--plots-dir", str(plots_dir),
        "--window-steps", "5",
        "--min-samples", "20",
        "--min-duration-s", "10",
        "--hidden-dim", "4",
        "--num-layers", "1",
        "--batch-size", "8",
        "--epochs", "2",
        "--patience", "1",
        "--train-fraction", "0.5",
        "--val-fraction", "0.25",
        "--no-mlflow",
        "--cpu",
    ])

    train_and_validate(args)

    checkpoint_path = output_dir / "gru_t1.pt"
    assert checkpoint_path.exists()

    report = json.loads((output_dir / "validation_report_t1.json").read_text())
    assert math.isfinite(report["final_metrics"]["mae_delta_t1"])
    assert math.isfinite(report["test_metrics"]["mae_delta_t1"])
    assert report["num_train_runs"] >= 1
    assert report["num_val_runs"] >= 1
    assert report["num_test_runs"] >= 1

    # The checkpoint itself must be loadable and usable for prediction --
    # this is the artifact simulate_gru.py / mpc_gru.py / package-model all
    # consume downstream.
    import torch

    model, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
    assert checkpoint["input_dim"] == 10
    assert set(("model_state_dict", "x_scaler", "y_scaler", "feature_names", "window_steps")) <= checkpoint.keys()
