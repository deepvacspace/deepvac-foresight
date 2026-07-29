"""Integration tests for the checkpoint contract shared by gru/gru_common.py,
gru/mpc_gru.py, lstm/mpc_lstm.py, and deepvac/packaging.py's ONNX export --
the torch.save() dict shape documented in DEV_GUIDE.md §6 (model_state_dict,
x_scaler/y_scaler, feature_names, window_steps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from deepvac.schemas import DEFAULT_FEATURE_NAMES
from tests.conftest import build_tiny_gru_checkpoint


def test_load_model_round_trips_a_checkpoint(tmp_path: Path):
    from gru.gru_common import load_model

    path = build_tiny_gru_checkpoint(tmp_path / "ckpt.pt")
    model, checkpoint = load_model(path, torch.device("cpu"))

    assert checkpoint["feature_names"] == DEFAULT_FEATURE_NAMES
    assert checkpoint["window_steps"] == 5
    assert not model.training  # load_model() calls .eval()


def test_predict_delta_t1_is_deterministic_across_reloads(tmp_path: Path):
    from gru.gru_common import load_model, predict_delta_t1

    path = build_tiny_gru_checkpoint(tmp_path / "ckpt.pt")
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    feature_window = rng.normal(size=(5, len(DEFAULT_FEATURE_NAMES))).astype(np.float32)

    model_a, checkpoint_a = load_model(path, device)
    delta_a = predict_delta_t1(model_a, checkpoint_a, feature_window, device)

    model_b, checkpoint_b = load_model(path, device)
    delta_b = predict_delta_t1(model_b, checkpoint_b, feature_window, device)

    assert delta_a == pytest.approx(delta_b)


def test_predict_delta_t1_reacts_to_scaler_not_just_weights(tmp_path: Path):
    """Two checkpoints with identical weights but different y_scaler stats
    must produce different real-unit predictions -- catches a regression
    where predict_delta_t1 stops applying inverse_transform."""
    from gru.gru_common import load_model, predict_delta_t1

    path_a = build_tiny_gru_checkpoint(tmp_path / "a.pt", seed=1)
    path_b = build_tiny_gru_checkpoint(tmp_path / "b.pt", seed=1)

    device = torch.device("cpu")
    model_a, checkpoint_a = load_model(path_a, device)
    model_b, checkpoint_b = load_model(path_b, device)

    # Same weights (same seed) but blow up checkpoint_b's y_scaler scale.
    checkpoint_b["y_scaler"].scale_ = checkpoint_b["y_scaler"].scale_ * 1000.0

    rng = np.random.default_rng(0)
    feature_window = rng.normal(size=(5, len(DEFAULT_FEATURE_NAMES))).astype(np.float32)

    delta_a = predict_delta_t1(model_a, checkpoint_a, feature_window, device)
    delta_b = predict_delta_t1(model_b, checkpoint_b, feature_window, device)

    assert delta_a != pytest.approx(delta_b)
