from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "validation_t1" / "gru_t1.pt"

DEFAULT_FEATURE_NAMES = [
    "temp",
    "temp_ref",
    "error",
    "temp_u",
    "temp_u_p",
    "temp_u_i",
    "temp_u_d",
    "kp",
    "ki",
    "kd",
]


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def limit(low: float, x: float, high: float) -> float:
    return max(low, min(float(x), high))


class CodesysDiff:
    """Stateful implementation of gru/codesys/diff.txt."""

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
    """Python equivalent of gru/codesys/pidcoefselector.txt."""

    def __init__(
        self,
        points: Sequence[Tuple[float, float, float]],
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

    def get_coefs(self, x: float) -> Tuple[float, float, float, int]:
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
    ) -> Tuple[float, float, float, float]:
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

        # For logging, same as the ST code.
        self.p_part = limit(self.u_min, self.p_part, self.u_max)

        return u, self.p_part, self.i_part, self.d_part


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[GRUModel, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GRUModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_delta_t1(
    model: GRUModel,
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
