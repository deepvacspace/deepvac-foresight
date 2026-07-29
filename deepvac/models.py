"""Model-agnostic PyTorch dataset helper shared by the GRU and LSTM plant models.

The GRU/LSTM network architectures themselves (gru/model.py:GRUModel,
lstm/model.py:LSTMModel) are intentionally separate -- they use different
recurrent cells. SequenceDataset was a byte-identical copy in both files.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
