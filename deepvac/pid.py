"""PID coefficient bounds, banding, scheduling, and TCP readback.

Consolidates logic that used to live in utils/utils.py (band scheduling,
random selection, TCP readback -- used by optimization/) and, byte-for-byte
duplicated, in gru/mpc_gru.py and lstm/mpc_lstm.py (bounds/clipping for the
MPC candidate optimizers).
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Tuple

import numpy as np

from deepvac import protocol

PIDTriplet = Tuple[int, int, int]


def parse_bounds(text: str) -> Tuple[float, float]:
    if not isinstance(text, str):
        raise ValueError(f"Bounds must be a string, got: {type(text)}")

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid bounds format: '{text}'. Expected 'low,high'")

    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        raise ValueError(f"Bounds must be numeric: '{text}'")

    if lo >= hi:
        raise ValueError(f"Invalid bounds '{text}': low must be < high")

    return lo, hi


# -----------------------------------------------------------------------------
# MPC-style array bounds (gru/mpc_gru.py, lstm/mpc_lstm.py).
# -----------------------------------------------------------------------------


def pid_bounds(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Read (kp_min, ki_min, kd_min) / (kp_max, ki_max, kd_max) off `args`."""
    lo = np.asarray([args.kp_min, args.ki_min, args.kd_min], dtype=float)
    hi = np.asarray([args.kp_max, args.ki_max, args.kd_max], dtype=float)
    if np.any(hi <= lo):
        raise ValueError(f"Invalid PID bounds: lo={lo}, hi={hi}")
    return lo, hi


def clip_pid(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Clip a (kp, ki, kd) vector to `args`' bounds and round to the nearest int."""
    lo, hi = pid_bounds(args)
    y = np.clip(np.asarray(x, dtype=float), lo, hi)
    y = np.floor(y + 0.5)
    y = np.clip(y, lo, hi)
    return y.astype(float)


# -----------------------------------------------------------------------------
# TCP readback and band scheduling (optimization/).
# -----------------------------------------------------------------------------


def read_pid_from_tcp(row: int, args: Any) -> Dict[str, float]:
    settings = protocol.request_settings(
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    kp_key, ki_key, kd_key = protocol._pid_keys(row)
    for key in (kp_key, ki_key, kd_key):
        if key not in settings:
            raise KeyError(f"TCP settings missing expected key: {key}")

    kp = float(settings[kp_key])
    ki = float(settings[ki_key])
    kd = float(settings[kd_key])
    if not (math.isfinite(kp) and math.isfinite(ki) and math.isfinite(kd)):
        raise RuntimeError(f"PID read from TCP contains non-finite values: kp={kp}, ki={ki}, kd={kd}")

    return {"kp": kp, "ki": ki, "kd": kd}


def random_pid(args: Any, rng: Any) -> PIDTriplet:
    kp = rng.randint(args.kp_min, args.kp_max)
    ki = rng.randint(args.ki_min, args.ki_max)
    kd = rng.randint(args.kd_min, args.kd_max)
    return kp, ki, kd


def _band_range(args: Any, band: str, coef: str, bounds: str) -> int:
    attr_name = f"{band}_{coef}_{bounds}"
    band_val = getattr(args, attr_name)
    if band_val is not None:
        return int(band_val)
    return int(getattr(args, f"{coef}_{bounds}"))


def random_pid_band(args: Any, rng: Any, band: str) -> PIDTriplet:
    kp = rng.randint(_band_range(args, band, "kp", "min"), _band_range(args, band, "kp", "max"))
    ki = rng.randint(_band_range(args, band, "ki", "min"), _band_range(args, band, "ki", "max"))
    kd = rng.randint(_band_range(args, band, "kd", "min"), _band_range(args, band, "kd", "max"))
    return kp, ki, kd


def _schedule_pid(schedule: Tuple[int, ...], bands: Tuple[str, ...]) -> Dict[str, PIDTriplet]:
    expected_len = len(bands) * 3
    if len(schedule) != expected_len:
        raise ValueError(f"Each PID_SCHEDULES entry must have {expected_len} numbers, got {len(schedule)}")

    planned: Dict[str, PIDTriplet] = {}
    for idx, band in enumerate(bands):
        offset = idx * 3
        planned[band] = (
            int(schedule[offset]),
            int(schedule[offset + 1]),
            int(schedule[offset + 2]),
        )
    return planned


def plan_pid(
    run_idx: int,
    args: Any,
    rng: Any,
    pid_schedules: list[Tuple[int, ...]],
    bands: Tuple[str, ...],
) -> Tuple[Dict[str, PIDTriplet], str]:
    if pid_schedules:
        schedule_idx = (run_idx - 1) % len(pid_schedules)
        return _schedule_pid(pid_schedules[schedule_idx], bands), f"schedule[{schedule_idx}]"

    return {
        band: random_pid_band(args=args, rng=rng, band=band)
        for band in bands
    }, "random-band-ranges"
