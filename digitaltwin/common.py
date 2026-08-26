from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deepvac.schemas import DEFAULT_FEATURE_NAMES  # noqa: F401
from digitaltwin.model import GRUModel, LSTMModel, MODEL_CLASSES  # noqa: F401


def _ensure_sklearn_stub() -> None:
    """Register minimal sklearn stubs so torch.load can unpickle StandardScaler
    objects from checkpoints without requiring scikit-learn to be installed.

    Both defining module paths are registered, since the one a checkpoint
    references depends on the sklearn version that saved it:
        sklearn.preprocessing._data.StandardScaler   (sklearn >= 0.24)
        sklearn.preprocessing.data.StandardScaler    (sklearn < 0.24)
    """
    if "sklearn" in sys.modules:
        return

    class _StandardScaler:
        def transform(self, X: np.ndarray) -> np.ndarray:
            return (X - self.mean_) / self.scale_

        def inverse_transform(self, X: np.ndarray) -> np.ndarray:
            return X * self.scale_ + self.mean_

    def _pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore[attr-defined]  # marks it as a package
        m.__package__ = name
        return m

    sklearn_mod = _pkg("sklearn")
    pre_mod = _pkg("sklearn.preprocessing")
    pre_mod.StandardScaler = _StandardScaler  # type: ignore[attr-defined]
    sklearn_mod.preprocessing = pre_mod  # type: ignore[attr-defined]

    # submodule where StandardScaler is actually defined (version-dependent)
    for sub in ("sklearn.preprocessing._data", "sklearn.preprocessing.data"):
        sub_mod = types.ModuleType(sub)
        sub_mod.StandardScaler = _StandardScaler  # type: ignore[attr-defined]
        sys.modules[sub] = sub_mod

    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.preprocessing"] = pre_mod


def limit(low: float, x: float, high: float) -> float:
    return max(low, min(float(x), high))


class CodesysDiff:
    """Stateful implementation of digitaltwin/codesys/diff.txt."""

    def __init__(self, dc: float = 0.995) -> None:
        self.dc = float(dc)
        self.prev_value = 0.0
        self.filter_out = 0.0
        self.out = 0.0

    def update(self, value: float) -> float:
        diff_value = float(value) - self.prev_value
        filter_in = limit(-5.0, diff_value, 5.0)
        self.filter_out = self.dc * self.filter_out + (1.0 - self.dc) * filter_in
        self.prev_value = float(value)
        self.out = 10.0 * limit(-5.0, self.filter_out, 5.0)
        return self.out


class PidCoefSelector:
    """Python equivalent of digitaltwin/codesys/pidcoefselector.txt."""

    def __init__(
        self,
        points: Sequence[tuple[float, float, float]],
        min_range: float,
        max_range: float,
    ) -> None:
        if not points:
            raise ValueError("PidCoefSelector requires at least one point.")
        self.points = [(float(p), float(i), float(d)) for p, i, d in points]
        self.points_count = len(self.points)
        self.min_range = float(min_range)
        self.range_width = (float(max_range) - float(min_range)) / float(self.points_count)
        if self.range_width == 0.0:
            raise ValueError("PidCoefSelector max_range must differ from min_range.")

    def get_coefs(self, x: float) -> tuple[float, float, float, int]:
        raw_index = math.trunc((float(x) - self.min_range) / self.range_width)
        interval_index = int(limit(0, raw_index, self.points_count))
        interval_index = min(interval_index, self.points_count - 1)
        kp, ki, kd = self.points[interval_index]
        return kp, ki, kd, interval_index


class ChamberPID:
    def __init__(
        self,
        u_min: float = -1.0,
        u_max: float = 1.0,
        pid_i_reverse_mul: float = 0.333,
    ) -> None:
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.pid_i_reverse_mul = float(pid_i_reverse_mul)

        self.i_part = 0.0
        self.p_part = 0.0
        self.d_part = 0.0

    def step(
        self,
        enable: bool,
        x_target: float,
        x_measured: float,
        p_coef: float,
        i_coef: float,
        d_coef: float,
        diff_out: float,
    ) -> tuple[float, float, float, float]:
        if not enable:
            self.p_part = 0.0
            self.i_part = 0.0
            self.d_part = 0.0
            return 0.0, self.p_part, self.i_part, self.d_part

        delta = float(x_target) - float(x_measured)

        if float(p_coef) == 0.0:
            return 0.0, self.p_part, self.i_part, self.d_part

        self.p_part = (1.0 / float(p_coef)) * delta

        effective_i_coef = float(i_coef)
        if delta * self.i_part < 0.0:
            effective_i_coef = float(i_coef) * self.pid_i_reverse_mul

        delta_edge = 1.2 * float(p_coef)

        if effective_i_coef != 0.0 and abs(delta) < delta_edge:
            self.i_part += (1.0 / float(p_coef)) * (delta * 0.1 / effective_i_coef)

        self.d_part = (1.0 / float(p_coef)) * (float(d_coef) * -float(diff_out))

        self.i_part = limit(self.u_min, self.i_part, self.u_max)
        self.d_part = limit(-0.4, self.d_part, 0.4)

        u = self.p_part + self.i_part + self.d_part
        u = limit(self.u_min, u, self.u_max)

        # Clipped for logging only.
        self.p_part = limit(self.u_min, self.p_part, self.u_max)

        return u, self.p_part, self.i_part, self.d_part


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_family: str | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    _ensure_sklearn_stub()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    family = model_family or checkpoint.get("model_family", "gru")
    model_cls = MODEL_CLASSES[family]

    model = model_cls(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        layer_norm=bool(checkpoint.get("layer_norm", False)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_delta_t1(
    model: nn.Module,
    checkpoint: dict[str, object],
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
