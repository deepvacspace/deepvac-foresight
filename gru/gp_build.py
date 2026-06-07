#!/usr/bin/env python3
"""Build a historical PID candidate table for GRU-based candidate ranking.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0] if len(SCRIPT_DIR.parents) else SCRIPT_DIR
DEFAULT_HISTORY_ROOT = ROOT / "optimization" / "run_history"
DEFAULT_MPC_ROOT = SCRIPT_DIR / "mpc_pid_runs"
DEFAULT_OUTPUT = DEFAULT_MPC_ROOT / "gru_pid_candidate_table.csv"


BANDS = ("far", "mid", "near")


def safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def infer_elapsed_s(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "elapsed_s" in df.columns:
        df["elapsed_s"] = safe_numeric(df["elapsed_s"], 0.0)
        return df.sort_values("elapsed_s").reset_index(drop=True)

    if "timestamp" in df.columns and len(df):
        df["timestamp"] = safe_numeric(df["timestamp"], 0.0)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"] = df["timestamp"] - float(df["timestamp"].iloc[0])
        return df

    df["elapsed_s"] = np.arange(len(df), dtype=float)
    return df


def classify_band(abs_error: pd.Series, far_threshold: float, near_threshold: float) -> pd.Series:
    band = pd.Series("mid", index=abs_error.index, dtype="object")
    band.loc[abs_error > float(far_threshold)] = "far"
    band.loc[abs_error <= float(near_threshold)] = "near"
    return band


def compute_temp_velocity(df: pd.DataFrame) -> pd.Series:
    dt = safe_numeric(df["elapsed_s"], 0.0).diff().replace(0.0, np.nan)
    velocity = safe_numeric(df["temp"]).diff() / dt
    return velocity.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def read_one_run(run_dir: Path, args: argparse.Namespace) -> Optional[pd.DataFrame]:
    samples_path = run_dir / args.samples_name
    metrics_path = run_dir / args.band_metrics_name
    if not samples_path.exists() or not metrics_path.exists():
        return None

    try:
        samples = infer_elapsed_s(pd.read_csv(samples_path))
        metrics = pd.read_csv(metrics_path)
    except Exception as exc:
        print(f"[SKIP] {run_dir.name}: {exc}")
        return None

    required_samples = {"temp", "temp_ref", "kp", "ki", "kd"}
    if not required_samples.issubset(samples.columns):
        return None
    if "band" not in metrics.columns or "cost" not in metrics.columns:
        return None

    run_id = str(samples["run_id"].iloc[0]) if "run_id" in samples.columns and len(samples) else run_dir.name
    samples["run_id"] = samples.get("run_id", run_id)
    samples["run_id"] = samples["run_id"].fillna(run_id).astype(str)

    samples["temp"] = safe_numeric(samples["temp"])
    samples["target_temp"] = safe_numeric(samples["temp_ref"])
    samples["error"] = samples["target_temp"] - samples["temp"]
    samples["abs_error"] = samples["error"].abs()
    samples["temp_velocity"] = compute_temp_velocity(samples)
    samples["band"] = classify_band(samples["abs_error"], args.far_threshold, args.near_threshold)

    for col in ("kp", "ki", "kd"):
        samples[col] = safe_numeric(samples[col])
        if col in metrics.columns:
            metrics[col] = safe_numeric(metrics[col])

    metric_cols = [
        "run_id", "band", "kp", "ki", "kd", "cost", "n_samples",
        "far_mae", "time_to_reach_mid_band",
        "mid_mae", "approach_speed", "weighted_approach_speed", "brake_penalty",
        "tail_mae", "jitter_std", "overshoot", "max_overshoot",
    ]
    present = [c for c in metric_cols if c in metrics.columns]
    metrics = metrics[present].copy()

    if "run_id" not in metrics.columns:
        metrics["run_id"] = run_id
    metrics["run_id"] = metrics["run_id"].fillna(run_id).astype(str)
    metrics["history_score"] = safe_numeric(metrics["cost"])

    keep_cols = [
        "run_id", "elapsed_s", "temp", "target_temp", "error", "abs_error",
        "temp_velocity", "kp", "ki", "kd", "band",
    ]
    state_rows = samples[keep_cols].copy()

    state_rows = state_rows.merge(
        metrics,
        on=["run_id", "band", "kp", "ki", "kd"],
        how="left",
        suffixes=("", "_metric"),
    )

    state_rows["history_score"] = safe_numeric(state_rows["history_score"], np.nan)
    state_rows["history_score"] = state_rows["history_score"].fillna(state_rows["abs_error"])

    # Downsample per run, but preserve beginning/end coverage.
    if args.max_samples_per_run > 0 and len(state_rows) > args.max_samples_per_run:
        idx = np.linspace(0, len(state_rows) - 1, args.max_samples_per_run).round().astype(int)
        state_rows = state_rows.iloc[np.unique(idx)].copy()

    # Current PID is the PID active at that state. Candidate PID is also kp/ki/kd.
    # The selector will use current_* to match similar situations and kp/ki/kd as candidate action.
    state_rows["current_kp"] = state_rows["kp"]
    state_rows["current_ki"] = state_rows["ki"]
    state_rows["current_kd"] = state_rows["kd"]
    state_rows["source"] = "run_history"
    return state_rows


def read_run_history(args: argparse.Namespace) -> List[pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    history_root = Path(args.history_root)
    if not history_root.exists():
        print(f"[WARN] history root not found: {history_root}")
        return rows

    for run_dir in sorted(p for p in history_root.iterdir() if p.is_dir()):
        frame = read_one_run(run_dir, args)
        if frame is not None and len(frame):
            rows.append(frame)
    return rows


def read_mpc_decisions(args: argparse.Namespace) -> List[pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    mpc_root = Path(args.mpc_root)
    if not mpc_root.exists():
        return rows

    for path in sorted(mpc_root.rglob(args.mpc_decisions_name)):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[SKIP] {path}: {exc}")
            continue

        required = {"elapsed_s", "temp", "error", "old_kp", "old_ki", "old_kd", "kp", "ki", "kd", "mpc_cost"}
        if not required.issubset(df.columns):
            continue

        out = pd.DataFrame()
        out["run_id"] = path.parent.name
        out["elapsed_s"] = safe_numeric(df["elapsed_s"], 0.0)
        out["temp"] = safe_numeric(df["temp"])
        out["error"] = safe_numeric(df["error"])
        out["abs_error"] = out["error"].abs()
        out["target_temp"] = out["temp"] + out["error"]
        out["temp_velocity"] = out["temp"].diff() / out["elapsed_s"].diff().replace(0.0, np.nan)
        out["temp_velocity"] = out["temp_velocity"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["current_kp"] = safe_numeric(df["old_kp"])
        out["current_ki"] = safe_numeric(df["old_ki"])
        out["current_kd"] = safe_numeric(df["old_kd"])
        out["kp"] = safe_numeric(df["kp"])
        out["ki"] = safe_numeric(df["ki"])
        out["kd"] = safe_numeric(df["kd"])
        out["history_score"] = safe_numeric(df["mpc_cost"])
        out["cost"] = out["history_score"]
        out["band"] = classify_band(out["abs_error"], args.far_threshold, args.near_threshold)
        out["source"] = "mpc_decision"
        rows.append(out)
    return rows


def finalize_table(frames: List[pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    columns = [
        "source", "run_id", "band",
        "elapsed_s", "temp", "target_temp", "error", "abs_error", "temp_velocity",
        "current_kp", "current_ki", "current_kd",
        "kp", "ki", "kd",
        "history_score", "cost", "n_samples",
        "far_mae", "time_to_reach_mid_band",
        "mid_mae", "approach_speed", "weighted_approach_speed", "brake_penalty",
        "tail_mae", "jitter_std", "overshoot", "max_overshoot",
    ]
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[columns].copy()

    numeric_cols = [c for c in columns if c not in {"source", "run_id", "band"}]
    for col in numeric_cols:
        out[col] = safe_numeric(out[col])

    out = out.dropna(subset=["temp", "target_temp", "error", "kp", "ki", "kd", "history_score"])

    # Optional band filtering: keep only the best rows per band to avoid a huge table.
    if args.max_rows_per_band > 0:
        kept = []
        for band in BANDS:
            b = out[out["band"] == band].sort_values(["history_score", "abs_error"])
            kept.append(b.head(args.max_rows_per_band))
        out = pd.concat(kept, ignore_index=True, sort=False) if kept else out

    # Remove exact duplicates while preserving best score.
    out = out.sort_values(["history_score", "abs_error"]).drop_duplicates(
        subset=["band", "temp", "target_temp", "current_kp", "current_ki", "current_kd", "kp", "ki", "kd"],
        keep="first",
    )

    out = out.sort_values(["band", "history_score", "abs_error"]).reset_index(drop=True)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build historical PID candidate table for GRU risk ranking.")
    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument("--mpc-root", default=str(DEFAULT_MPC_ROOT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--samples-name", default="run_samples.csv")
    ap.add_argument("--band-metrics-name", default="band_metrics.csv")
    ap.add_argument("--mpc-decisions-name", default="mpc_decisions.csv")
    ap.add_argument("--far-threshold", type=float, default=10.0)
    ap.add_argument("--near-threshold", type=float, default=3.0)
    ap.add_argument("--max-samples-per-run", type=int, default=80)
    ap.add_argument("--max-rows-per-band", type=int, default=5000)
    ap.add_argument(
        "--include-mpc-decisions",
        action="store_true",
        help="Include previous MPC decisions. Disabled by default because bad MPC runs can poison the table.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = read_run_history(args)
    if args.include_mpc_decisions:
        frames.extend(read_mpc_decisions(args))

    table = finalize_table(frames, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print("=== GRU PID candidate table ===")
    print(f"rows:   {len(table)}")
    print(f"output: {output}")
    if len(table):
        print("\nRows by band:")
        print(table["band"].value_counts().reindex(BANDS).fillna(0).astype(int).to_string())
        print("\nBest candidates preview:")
        cols = ["band", "temp", "error", "temp_velocity", "kp", "ki", "kd", "history_score", "source"]
        print(table[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
