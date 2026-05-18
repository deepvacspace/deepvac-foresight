from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Tuple

from tcp.tcp_common import _pid_keys, request_settings

PIDTriplet = Tuple[int, int, int]


def append_row_csv(path: str, row: Dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if (not out.exists()) or out.stat().st_size == 0:
        with out.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])

    if not header:
        with out.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    aligned_row = {col: row.get(col) for col in header}
    with out.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writerow(aligned_row)


def read_pid_from_tcp(row: int, args: Any) -> Dict[str, float]:
    settings = request_settings(
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    kp_key, ki_key, kd_key = _pid_keys(row)
    for key in (kp_key, ki_key, kd_key):
        if key not in settings:
            raise KeyError(f"TCP settings missing expected key: {key}")

    kp = float(settings[kp_key])
    ki = float(settings[ki_key])
    kd = float(settings[kd_key])
    if not (math.isfinite(kp) and math.isfinite(ki) and math.isfinite(kd)):
        raise RuntimeError(f"PID read from TCP contains non-finite values: kp={kp}, ki={ki}, kd={kd}")

    return {
        "kp": kp,
        "ki": ki,
        "kd": kd,
    }


def random_pid(args: Any, rng: Any) -> Tuple[int, int, int]:
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
