#!/usr/bin/env python3
"""Summarize run samples files. Clean nans"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from deepvac.artifacts import iter_run_dirs


SAMPLE_FILE_NAMES = (
    "run_samples.csv"
)


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parent
    summary_dir = root / "summary"
    ap = argparse.ArgumentParser(
        description="List history runs with duration, samples, temperatures, and NaN counts."
    )
    ap.add_argument(
        "--history-root",
        default=str(Path(__file__).resolve().parents[1] / "experiments" / "run_history"),
        help="History folder containing one subfolder per run.",
    )
    ap.add_argument(
        "--out",
        default=str(summary_dir / "history_run_summary.txt"),
        help="Output path for the generated summary. .txt writes the printed table; .csv writes CSV.",
    )
    ap.add_argument(
        "--sort-by",
        default="run_id",
        choices=("run_id", "duration_s", "num_samples",
                 "start_temp", "last_temp", "nan_values"),
    )
    ap.add_argument("--descending", action="store_true")
    ap.add_argument(
        "--include-missing",
        action="store_true",
        help="Include subfolders that do not contain a recognized sample CSV.",
    )
    return ap


def find_sample_file(run_dir: Path) -> Optional[Path]:
    for file_name in SAMPLE_FILE_NAMES:
        candidate = run_dir / file_name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def clean_nan_rows(sample_file: Path) -> int:
    df = pd.read_csv(sample_file)
    before = len(df)
    cleaned = df.dropna(axis=0, how="any").reset_index(drop=True)
    removed = before - len(cleaned)

    if removed:
        cleaned.to_csv(sample_file, index=False)

    return removed


def summarize_run(run_dir: Path) -> Dict[str, object]:
    sample_file = find_sample_file(run_dir)
    if sample_file is None:
        return {
            "run_id": run_dir.name,
            "sample_file": "",
            "duration_s": None,
            "num_samples": 0,
            "start_temp": None,
            "last_temp": None,
            "nan_values": None,
            "status": "missing_samples",
        }

    try:
        removed_nan_rows = clean_nan_rows(sample_file)
        df = pd.read_csv(sample_file)
    except Exception as exc:
        return {
            "run_id": run_dir.name,
            "sample_file": str(sample_file),
            "duration_s": None,
            "num_samples": None,
            "start_temp": None,
            "last_temp": None,
            "nan_values": None,
            "removed_nan_rows": None,
            "status": f"read_error: {exc}",
        }

    run_id = run_dir.name
    if "run_id" in df.columns and len(df) and pd.notna(df["run_id"].iloc[0]):
        run_id = str(df["run_id"].iloc[0])

    duration_s: Optional[float] = None
    if "timestamp" in df.columns and len(df) >= 2:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            duration_s = float(ts.iloc[-1] - ts.iloc[0])

    start_temp: Optional[float] = None
    last_temp: Optional[float] = None
    if "temp" in df.columns and len(df):
        temps = pd.to_numeric(df["temp"], errors="coerce")
        valid_temps = temps.dropna()
        if len(valid_temps):
            start_temp = _safe_float(valid_temps.iloc[0])
            last_temp = _safe_float(valid_temps.iloc[-1])

    return {
        "run_id": run_id,
        "sample_file": sample_file.name,
        "duration_s": duration_s,
        "num_samples": int(len(df)),
        "start_temp": start_temp,
        "last_temp": last_temp,
        "nan_values": int(df.isna().sum().sum()),
        "removed_nan_rows": int(removed_nan_rows),
        "status": "ok",
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    history_root = Path(args.history_root)

    rows = []
    for run_dir in iter_run_dirs(history_root):
        row = summarize_run(run_dir)
        if row["status"] == "missing_samples" and not args.include_missing:
            continue
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No run folders found under: {history_root}")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        args.sort_by, ascending=not args.descending, na_position="last")

    display_cols = [
        "run_id",
        "duration_s",
        "num_samples",
        "start_temp",
        "last_temp",
        "nan_values",
        "sample_file",
    ]
    table_text = summary[display_cols].to_string(index=False)

    ok_summary = summary[summary["status"] == "ok"]
    totals = {
        "total_runs": int(len(ok_summary)),
        "total_samples": int(pd.to_numeric(ok_summary["num_samples"], errors="coerce").fillna(0).sum()),
        "total_nan_values": int(pd.to_numeric(ok_summary["nan_values"], errors="coerce").fillna(0).sum()),
        "total_removed_nan_rows": int(pd.to_numeric(ok_summary["removed_nan_rows"], errors="coerce").fillna(0).sum()),
    }
    totals_text = (
        "\n\nTotals:\n"
        f"  runs: {totals['total_runs']}\n"
        f"  samples: {totals['total_samples']}\n"
        f"  nan_values: {totals['total_nan_values']}"
    )
    totals_text += f"\n  removed_nan_rows: {totals['total_removed_nan_rows']}"

    report_text = table_text + totals_text
    print(report_text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            out = summary.copy()
            out.loc[len(out)] = {
                "run_id": "TOTAL",
                "sample_file": "",
                "duration_s": None,
                "num_samples": totals["total_samples"],
                "start_temp": None,
                "last_temp": None,
                "nan_values": totals["total_nan_values"],
                "removed_nan_rows": totals["total_removed_nan_rows"],
                "status": f"runs={totals['total_runs']}",
            }
            out.to_csv(out_path, index=False)
        else:
            out_path.write_text(report_text + "\n", encoding="utf-8")
        print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
