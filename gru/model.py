from __future__ import annotations

import torch
import torch.nn as nn
from deepvac.models import SequenceDataset  # noqa: F401  (re-exported for callers)


class LayerNormGRUCell(nn.Module):
    """A GRU cell with LayerNorm applied to each gate's pre-activation."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.x_linear = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.h_linear = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.x_norm = nn.LayerNorm(3 * hidden_dim)
        self.h_norm = nn.LayerNorm(3 * hidden_dim)
        self.bias = nn.Parameter(torch.zeros(3 * hidden_dim))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        i_r, i_z, i_n = self.x_norm(self.x_linear(x)).chunk(3, dim=-1)
        h_r, h_z, h_n = self.h_norm(self.h_linear(h)).chunk(3, dim=-1)
        b_r, b_z, b_n = self.bias.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r + b_r)
        z = torch.sigmoid(i_z + h_z + b_z)
        n = torch.tanh(i_n + b_n + r * h_n)
        return (1.0 - z) * n + z * h


class LayerNormGRU(nn.Module):
    """Stack of LayerNormGRUCell, batch_first, matching nn.GRU's
    forward(x) -> (output, h_n) interface so GRUModel can swap between them."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LayerNormGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        h = ([h0[i] for i in range(self.num_layers)] if h0 is not None
             else [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)])

        outputs = []
        for t in range(seq_len):
            layer_in = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(layer_in, h[i])
                layer_in = h[i]
                if self.dropout is not None and i < self.num_layers - 1:
                    layer_in = self.dropout(layer_in)
            outputs.append(h[-1])

        return torch.stack(outputs, dim=1), torch.stack(h, dim=0)


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if layer_norm:
            self.gru = LayerNormGRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
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
