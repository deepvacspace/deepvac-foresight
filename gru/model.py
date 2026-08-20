from __future__ import annotations

import torch
import torch.nn as nn

from deepvac.models import SequenceDataset  # noqa: F401  (re-exported for callers)


class LayerNormGRUCell(nn.Module):
    """A GRU cell with LayerNorm on each gate's pre-activation (Ba et al. style)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = self.ln_ih(self.weight_ih(x))
        gh = self.ln_hh(self.weight_hh(h))
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h


class LayerNormGRU(nn.Module):
    """Multi-layer batch-first stack of LayerNormGRUCell. Drop-in for nn.GRU's
    forward(x) -> (output, h_n) shape."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([
            LayerNormGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if (dropout > 0 and num_layers > 1) else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        h = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(layer_input, h[i])
                layer_input = h[i]
                if self.dropout is not None and i < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
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
        self.layer_norm = bool(layer_norm)

        if self.layer_norm:
            self.gru = LayerNormGRU(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout,
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
