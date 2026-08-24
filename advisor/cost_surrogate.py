"""A small MLP that predicts CEM's rollout cost for a candidate PID directly
from a state, without simulating the GRU twin step by step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURE_NAMES = [
    "temp", "current_kp", "current_ki", "current_kd",
    "candidate_kp", "candidate_ki", "candidate_kd",
]


class CostSurrogateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_cost_surrogate(
    checkpoint_path: Path, device: torch.device
) -> tuple[CostSurrogateMLP, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CostSurrogateMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_cost(
    model: CostSurrogateMLP,
    checkpoint: dict[str, object],
    temp: float,
    current_pid: np.ndarray,
    candidate_pids: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predicted cost for a batch of candidate triplets at one state.

    With checkpoint["target_normalization"] != "none", these are per-state
    relative scores: valid for ranking within one call's candidate_pids, not
    comparable across states.
    """
    x_scaler = checkpoint["x_scaler"]
    y_scaler = checkpoint["y_scaler"]
    n = len(candidate_pids)
    rows = np.column_stack([
        np.full(n, float(temp)),
        np.full(n, float(current_pid[0])),
        np.full(n, float(current_pid[1])),
        np.full(n, float(current_pid[2])),
        candidate_pids[:, 0], candidate_pids[:, 1], candidate_pids[:, 2],
    ]).astype(np.float32)

    x_scaled = x_scaler.transform(rows)
    with torch.no_grad():
        pred_scaled = model(torch.as_tensor(x_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    return np.asarray(y_scaler.inverse_transform(pred_scaled)[:, 0], dtype=float)
