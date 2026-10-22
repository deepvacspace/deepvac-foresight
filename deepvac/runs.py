"""Lightweight run-history loading shared by anomalydetection and predictivemaintenance."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("temp", "temp_u")
_RUN_ID_TS = re.compile(r"_(\d{9,11})_")


def find_run_csvs(history_root: Path, telemetry_names: Sequence[str] = ("run_samples.csv",)) -> list[Path]:
    root = Path(history_root)
    if not root.exists():
        raise FileNotFoundError(f"History root does not exist: {root}")
    found: list[Path] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for name in telemetry_names:
            path = run_dir / name
            if path.exists() and path.stat().st_size > 0:
                found.append(path)
                break
    return found


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_run(csv_path: Path, *, min_samples: int = 100) -> pd.DataFrame | None:
    """Read one run into a clean, time-sorted frame, or None if unusable."""
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if df.empty or any(c not in df.columns for c in REQUIRED_COLUMNS):
        return None

    run_id = csv_path.parent.name
    for col in df.columns:
        if col != "run_id":
            df[col] = _numeric(df[col])

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        start = float(df["timestamp"].iloc[0])
        df["elapsed_s"] = df["timestamp"] - start
    elif "elapsed_s" in df.columns and df["elapsed_s"].notna().any():
        df = df.dropna(subset=["elapsed_s"]).sort_values("elapsed_s")
        match = _RUN_ID_TS.search(run_id)
        start = float(match.group(1)) if match else float("nan")
        df["timestamp"] = start + df["elapsed_s"]
    else:
        return None

    if "temp_ref" not in df.columns:
        df["temp_ref"] = _numeric(df["target_temp"]) if "target_temp" in df.columns else np.nan

    df = df.dropna(subset=["temp", "temp_u"]).reset_index(drop=True)
    if len(df) < min_samples:
        return None
    df = df.drop(columns=["run_id"], errors="ignore")
    df.insert(0, "run_id", run_id)
    return df


def iter_runs(history_root: Path, *, min_samples: int = 100) -> Iterator[pd.DataFrame]:
    """Yield every usable run, oldest first."""
    runs = [r for p in find_run_csvs(history_root) if (r := load_run(p, min_samples=min_samples)) is not None]
    runs.sort(key=lambda r: float(r["timestamp"].iloc[0]) if r["timestamp"].notna().any() else float("inf"))
    yield from runs
