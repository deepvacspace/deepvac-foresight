"""Registry mapping a model family name ("gru", "lstm", ...) to its loader,
predictor, and plant-model class. Add a new family by adding one factory
function and one `_REGISTRY` entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn as nn


class _Loader(Protocol):
    def __call__(self, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]: ...


class _Predictor(Protocol):
    def __call__(
        self, model: nn.Module, checkpoint: dict, feature_window: np.ndarray, device: torch.device
    ) -> float: ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    load_model: _Loader
    predict_delta_t1: _Predictor
    plant_model_class: type[nn.Module]


def _gru_spec() -> ModelSpec:
    from gru.gru_common import GRUModel, load_model, predict_delta_t1

    return ModelSpec(name="gru", load_model=load_model, predict_delta_t1=predict_delta_t1, plant_model_class=GRUModel)


def _lstm_spec() -> ModelSpec:
    from lstm.model import LSTMModel
    from lstm.mpc_lstm import load_model, predict_delta_t1

    return ModelSpec(
        name="lstm", load_model=load_model, predict_delta_t1=predict_delta_t1, plant_model_class=LSTMModel
    )


_REGISTRY: dict[str, Callable[[], ModelSpec]] = {
    "gru": _gru_spec,
    "lstm": _lstm_spec,
}


def registered_model_types() -> list[str]:
    return sorted(_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown model type {name!r}; registered types are {registered_model_types()}.")
    return factory()
