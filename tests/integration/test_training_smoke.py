"""Integration smoke test for the digital-twin training pipeline
(digitaltwin/train.py, shared by both --model-family gru and lstm).

Trains a tiny model for 2 epochs on synthetic data and asserts the pipeline runs
end to end, producing a loadable checkpoint and a validation report with finite
metrics. Model quality is not asserted. Marked slow (`pytest -m slow`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_train_and_validate_produces_a_loadable_checkpoint(history_root: Path, tmp_path: Path):
    from digitaltwin.common import load_model
    from digitaltwin.train import build_arg_parser, train_and_validate

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
