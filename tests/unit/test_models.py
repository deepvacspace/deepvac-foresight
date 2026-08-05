"""Unit tests for deepvac/models.py: SequenceDataset, the model-agnostic
torch Dataset wrapper shared by gru/model.py and lstm/model.py."""

from __future__ import annotations

import numpy as np
import torch
from deepvac.models import SequenceDataset


def test_len_matches_input_length():
    X = np.zeros((7, 3, 2), dtype=np.float32)
    y = np.zeros((7, 1), dtype=np.float32)
    assert len(SequenceDataset(X, y)) == 7


def test_getitem_returns_matching_tensors():
    X = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    y = np.array([[1.0], [2.0]], dtype=np.float32)
    ds = SequenceDataset(X, y)

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor) and isinstance(y0, torch.Tensor)
    assert x0.dtype == torch.float32
    np.testing.assert_array_equal(x0.numpy(), X[0])
    np.testing.assert_array_equal(y0.numpy(), y[0])


def test_works_with_dataloader_batching():
    from torch.utils.data import DataLoader

    X = np.random.default_rng(0).normal(size=(10, 4, 3)).astype(np.float32)
    y = np.random.default_rng(0).normal(size=(10, 1)).astype(np.float32)
    loader = DataLoader(SequenceDataset(X, y), batch_size=4)

    batches = list(loader)
    assert sum(len(xb) for xb, _ in batches) == 10
    assert batches[0][0].shape[1:] == (4, 3)
