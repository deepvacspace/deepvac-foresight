"""Canonical feature names and lightweight run/telemetry data shapes.

These document the CSV/JSON conventions used throughout the toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass

# One-step plant-model feature vector, in column order.
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

    kp: tuple[float, float]
    ki: tuple[float, float]
    kd: tuple[float, float]


@dataclass(frozen=True)
class RunSample:
    """One row of a run's `run_samples.csv` / telemetry CSV.

    `timestamp` and `elapsed_s` are alternative time bases; most loaders accept
    either and derive the other.
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
