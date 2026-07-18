"""Canonical feature names and lightweight run/telemetry data shapes.

These document the CSV/JSON conventions used throughout the toolkit. They
are intentionally plain dataclasses (no validation library dependency) --
callers that read CSV/JSON directly with pandas are unaffected; this module
exists so the shape of a "run sample" or "PID bounds" is defined once
instead of being implicit in each script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# One-step plant-model feature vector, in column order. Shared verbatim by
# gru/gru_common.py, deepvac/datasets.py (training), and the GRU/LSTM MPC
# schedulers (deepvac/mpc.py) -- previously three independent literal copies.
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


@dataclass(frozen=True)
class PIDBounds:
    """Inclusive [min, max] bounds for one PID coefficient."""

    kp: Tuple[float, float]
    ki: Tuple[float, float]
    kd: Tuple[float, float]


@dataclass(frozen=True)
class RunSample:
    """One row of a run's `run_samples.csv` / telemetry CSV.

    `timestamp` (physical runs, optimization/) and `elapsed_s` (offline
    training/simulation, gru/ + lstm/) are alternative time bases -- most
    loaders accept either and derive the other.
    """

    timestamp: float
    temp: float
    temp_ref: float
    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class RunSummary:
    """Aggregate metrics for one completed run, as written to summary JSON."""

    run_id: str
    cost: float
    tail_mae: float
    overshoot: float
    target: float
    start_temp: float
