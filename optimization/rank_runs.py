#!/usr/bin/env python3
"""Rank history runs metrics"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


SAMPLE_FILE_NAMES = (
    "run_samples.csv"
)

COEFF_COLS = (
    "far_kp", "far_ki", "far_kd",
    "mid_kp", "mid_ki", "mid_kd",
    "near_kp", "near_ki", "near_kd",
)


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parent
    summary_dir = root / "summary"
    ap = argparse.ArgumentParser(
        description="Write top runs by tail MAE, time to zero, cost, and overshoot."
    )
    ap.add_argument(
        "--history-root",
        default=str(Path(__file__).with_name("history")),
        help="History folder containing one subfolder per run.",
    )
    ap.add_argument(
        "--out",
        default=str(summary_dir / "history_top_runs.txt"),
        help="Text output path.",
    )
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument(
        "--tail-window-s",
        type=float,
        default=300.0,
        help="Seconds at the end of the run used to recompute tail MAE.",
    )
    ap.add_argument("--overshoot-weight", type=float, default=10.0)
    return ap


def iter_run_dirs(history_root: Path) -> Iterable[Path]:
    if not history_root.exists():
        raise FileNotFoundError(f"History root does not exist: {history_root}")

    for child in sorted(history_root.iterdir()):
        if child.is_dir():
            yield child


def first_existing(run_dir: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        candidate = run_dir / name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out) or not np.isfinite(out):
        return None
    return out


def infer_coefficients_from_samples(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if not {"kp", "ki", "kd"}.issubset(df.columns) or df.empty:
        return None

    work = df.copy()
    if "timestamp" in work.columns:
        work = work.sort_values("timestamp")

    triplets: list[tuple[float, float, float]] = []
    last: Optional[tuple[float, float, float]] = None

    for _, row in work.iterrows():
        kp = safe_float(row.get("kp"))
        ki = safe_float(row.get("ki"))
        kd = safe_float(row.get("kd"))
        if kp is None or ki is None or kd is None:
            continue

        triplet = (float(kp), float(ki), float(kd))
        if last is None or triplet != last:
            triplets.append(triplet)
            last = triplet

    if not triplets:
        return None

    while len(triplets) < 3:
        triplets.append(triplets[-1])

    far, mid, near = triplets[:3]
    return {
        "far_kp": far[0], "far_ki": far[1], "far_kd": far[2],
        "mid_kp": mid[0], "mid_ki": mid[1], "mid_kd": mid[2],
        "near_kp": near[0], "near_ki": near[1], "near_kd": near[2],
    }


def time_to_zero(df: pd.DataFrame) -> Optional[float]:
    if not {"timestamp", "temp", "temp_ref"}.issubset(df.columns) or df.empty:
        return None

    work = df.sort_values("timestamp").reset_index(drop=True).copy()
    temp = pd.to_numeric(work["temp"], errors="coerce")
    timestamp = pd.to_numeric(work["timestamp"], errors="coerce")
    temp_ref = pd.to_numeric(work["temp_ref"], errors="coerce")

    valid = temp.notna() & timestamp.notna() & temp_ref.notna()
    if not valid.any():
        return None

    temp = temp[valid].reset_index(drop=True)
    timestamp = timestamp[valid].reset_index(drop=True)
    temp_ref = temp_ref[valid].reset_index(drop=True)

    start_temp = float(temp.iloc[0])
    target = float(temp_ref.iloc[0])
    direction = 1.0 if start_temp > target else -1.0

    if direction > 0:
        reached = temp <= target
    else:
        reached = temp >= target

    idxs = np.where(reached.to_numpy())[0]
    if len(idxs) == 0:
        return None

    first_idx = int(idxs[0])
    return float(timestamp.iloc[first_idx] - timestamp.iloc[0])


def compute_cost_metrics(
    df: pd.DataFrame,
    entry_band: float,
    tail_window_s: float,
    overshoot_weight: float,
) -> Dict[str, Optional[float]]:
    if not {"timestamp", "temp", "temp_ref"}.issubset(df.columns) or df.empty:
        return {"cost": None, "tail_mae": None, "overshoot": None}

    work = df.sort_values("timestamp").reset_index(drop=True).copy()
    temp = pd.to_numeric(work["temp"], errors="coerce").to_numpy(dtype=float)
    temp_ref = pd.to_numeric(
        work["temp_ref"], errors="coerce").to_numpy(dtype=float)
    timestamp = pd.to_numeric(
        work["timestamp"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(temp) & np.isfinite(temp_ref) & np.isfinite(timestamp)
    temp = temp[valid]
    temp_ref = temp_ref[valid]
    timestamp = timestamp[valid]

    if len(temp) == 0:
        return {"cost": None, "tail_mae": None, "overshoot": None}

    target = float(temp_ref[0])
    start_temp = float(temp[0])
    direction = 1.0 if start_temp > target else -1.0
    abs_err = np.abs(temp - target)

    entry_idxs = np.where(abs_err <= entry_band)[0]
    if len(entry_idxs) == 0:
        return {"cost": 1e9, "tail_mae": None, "overshoot": None}

    start_idx = int(entry_idxs[0])
    metric_temp = temp[start_idx:]

    tail_start_time = timestamp[-1] - max(0.0, float(tail_window_s))
    tail_mask = timestamp >= tail_start_time
    if not np.any(tail_mask):
        tail_mask = np.ones_like(timestamp, dtype=bool)

    tail_temp = temp[tail_mask]
    tail_mae = float(np.mean(np.abs(tail_temp - target)))

    dev = metric_temp - target

    if direction > 0:
        wrong_dev = np.maximum(0.0, -dev)
    else:
        wrong_dev = np.maximum(0.0, dev)

    overshoot = float(np.max(wrong_dev)) if len(wrong_dev) else 0.0
    cost = float(tail_mae + overshoot_weight * (overshoot**2))

    return {"cost": cost, "tail_mae": tail_mae, "overshoot": overshoot}


def summarize_run(run_dir: Path, args: argparse.Namespace) -> Optional[Dict[str, object]]:
    sample_file = first_existing(run_dir, SAMPLE_FILE_NAMES)
    if sample_file is None:
        return None

    try:
        df = pd.read_csv(sample_file).dropna(axis=0, how="any")
    except Exception:
        return None

    if df.empty:
        return None

    coeffs = infer_coefficients_from_samples(df)
    if coeffs is None:
        return None

    run_id = run_dir.name
    if "run_id" in df.columns and pd.notna(df["run_id"].iloc[0]):
        run_id = str(df["run_id"].iloc[0])

    metrics = compute_cost_metrics(
        df,
        entry_band=args.entry_band,
        tail_window_s=args.tail_window_s,
        overshoot_weight=args.overshoot_weight,
    )

    row: Dict[str, object] = {
        "run_id": run_id,
        "sample_file": sample_file.name,
        "cost": safe_float(metrics.get("cost")),
        "tail_mae": safe_float(metrics.get("tail_mae")),
        "overshoot": safe_float(metrics.get("overshoot")),
        "time_to_cero": time_to_zero(df),
    }
    row.update(coeffs)
    return row


def coeff_text(row: pd.Series) -> str:
    return (
        "far_kp={far_kp:.0f}, far_ki={far_ki:.0f}, far_kd={far_kd:.0f}; "
        "mid_kp={mid_kp:.0f}, mid_ki={mid_ki:.0f}, mid_kd={mid_kd:.0f}; "
        "near_kp={near_kp:.0f}, near_ki={near_ki:.0f}, near_kd={near_kd:.0f}"
    ).format(**{col: float(row[col]) for col in COEFF_COLS})


def format_rank_row(rank_label: str, row: pd.Series, metric: str) -> list[str]:
    other_metrics = [
        f"{name}={float(row[name]):.6f}"
        for name in ("cost", "tail_mae", "time_to_cero", "overshoot")
        if name != metric and pd.notna(row[name])
    ]
    return [
        (
            f"{rank_label}. {row['run_id']} | {metric}={float(row[metric]):.6f} | "
            + " | ".join(other_metrics)
        ),
        f"   coeffs: {coeff_text(row)}",
    ]


def closest_metric_row(metric_df: pd.DataFrame, metric: str, target: float) -> pd.Series:
    distances = (metric_df[metric] - target).abs()
    return metric_df.loc[distances.idxmin()]


def format_section(title: str, df: pd.DataFrame, metric: str, top_n: int) -> str:
    metric_df = df.dropna(subset=[metric]).sort_values(metric, ascending=True)

    lines = [title, "-" * len(title)]
    if metric_df.empty:
        lines.append("No valid runs.")
        return "\n".join(lines)

    lines.append("Best")
    for rank, (_, row) in enumerate(metric_df.head(top_n).iterrows(), start=1):
        lines.extend(format_rank_row(str(rank), row, metric))

    lines.append("")
    lines.append("Worst")
    for rank, (_, row) in enumerate(metric_df.tail(top_n).iloc[::-1].iterrows(), start=1):
        lines.extend(format_rank_row(str(rank), row, metric))

    median_value = float(metric_df[metric].median())
    mean_value = float(metric_df[metric].mean())
    median_row = closest_metric_row(metric_df, metric, median_value)
    mean_row = closest_metric_row(metric_df, metric, mean_value)

    lines.append("")
    lines.append("Middle")
    lines.extend(format_rank_row(
        f"median closest ({median_value:.6f})", median_row, metric))
    if str(mean_row["run_id"]) != str(median_row["run_id"]):
        lines.extend(format_rank_row(
            f"average closest ({mean_value:.6f})", mean_row, metric))
    else:
        lines.append(f"average closest ({mean_value:.6f}) is the same run")

    return "\n".join(lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    history_root = Path(args.history_root)

    rows = []
    for run_dir in iter_run_dirs(history_root):
        row = summarize_run(run_dir, args)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid runs found under: {history_root}")

    df = pd.DataFrame(rows)

    sections = [
        f"Metrics computed from {len(df)} run_samples.csv. Tail MAE window: last {args.tail_window_s:g} seconds.",
        format_section(f"Tail MAE Rankings", df, "tail_mae", args.top_n),
        format_section(f"Time To Cero Rankings", df,
                       "time_to_cero", args.top_n),
        format_section(f"Cost Rankings", df, "cost", args.top_n),
        format_section(f"Overshoot Rankings", df, "overshoot", args.top_n),
    ]

    report = "\n\n".join(sections)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nSaved ranking report: {out_path}")


if __name__ == "__main__":
    main()
