#!/usr/bin/env python3
"""Compare run-history baseline behavior against the best AI Advisor run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepvac.artifacts import iter_run_dirs


REQUIRED_SAMPLE_COLS = {"timestamp", "temp", "temp_ref"}


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parent
    out_dir = root / "output" / "ai_advisor_improvement"

    ap = argparse.ArgumentParser(
        description=(
            "Rank run_history by tail MAE and overshoot, then compare the "
            "average run as Baseline against the best AI Advisor run."
        )
    )
    ap.add_argument(
        "--history-root",
        default=str(root / "run_history"),
        help="Folder containing run subfolders with run_samples.csv.",
    )
    ap.add_argument("--out-dir", default=str(out_dir), help="Output folder.")
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument(
        "--tail-window-s",
        type=float,
        default=600.0,
        help="Seconds at the end of each run used for tail MAE.",
    )
    ap.add_argument(
        "--overshoot-weight",
        type=float,
        default=10.0,
        help="Cost = tail_mae + overshoot_weight * overshoot^2.",
    )
    ap.add_argument("--grid-points", type=int, default=700)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    return ap


def read_samples(run_dir: Path) -> Optional[pd.DataFrame]:
    sample_path = run_dir / "run_samples.csv"
    if not sample_path.exists() or sample_path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(sample_path)
    except Exception:
        return None

    if df.empty or not REQUIRED_SAMPLE_COLS.issubset(df.columns):
        return None

    work = df.copy()
    for col in REQUIRED_SAMPLE_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=list(REQUIRED_SAMPLE_COLS)).sort_values("timestamp")
    work = work.reset_index(drop=True)
    if work.empty:
        return None

    t0 = float(work["timestamp"].iloc[0])
    work["t_rel"] = work["timestamp"].astype(float) - t0
    return work


def run_id_for(run_dir: Path, df: pd.DataFrame) -> str:
    if "run_id" in df.columns and len(df) and pd.notna(df["run_id"].iloc[0]):
        return str(df["run_id"].iloc[0])
    return run_dir.name


def safe_percent_improvement(baseline: float, advisor: float) -> float:
    if not np.isfinite(baseline) or baseline == 0.0:
        return 0.0
    return float((baseline - advisor) / abs(baseline) * 100.0)


def smooth_series(values: np.ndarray, window_fraction: float = 0.035) -> np.ndarray:
    if len(values) < 5:
        return values

    window = max(5, int(len(values) * window_fraction))
    if window % 2 == 0:
        window += 1
    if window >= len(values):
        window = len(values) - 1 if len(values) % 2 == 0 else len(values)
    if window < 3:
        return values

    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_metrics(
    df: pd.DataFrame,
    entry_band: float,
    tail_window_s: float,
    overshoot_weight: float,
) -> Dict[str, Optional[float]]:
    temp = df["temp"].to_numpy(dtype=float)
    temp_ref = df["temp_ref"].to_numpy(dtype=float)
    t_rel = df["t_rel"].to_numpy(dtype=float)

    valid = np.isfinite(temp) & np.isfinite(temp_ref) & np.isfinite(t_rel)
    temp = temp[valid]
    temp_ref = temp_ref[valid]
    t_rel = t_rel[valid]
    if len(temp) == 0:
        return {"tail_mae": None, "overshoot": None, "cost": None, "duration_s": None}

    target = float(temp_ref[0])
    start_temp = float(temp[0])
    direction = 1.0 if start_temp > target else -1.0

    tail_start_time = float(t_rel[-1]) - max(0.0, float(tail_window_s))
    tail_mask = t_rel >= tail_start_time
    if not np.any(tail_mask):
        tail_mask = np.ones_like(t_rel, dtype=bool)
    tail_mae = float(np.mean(np.abs(temp[tail_mask] - target)))

    abs_err = np.abs(temp - target)
    entry_idxs = np.where(abs_err <= entry_band)[0]
    if len(entry_idxs) == 0:
        return {
            "tail_mae": tail_mae,
            "overshoot": None,
            "cost": None,
            "duration_s": float(t_rel[-1]),
        }

    metric_temp = temp[int(entry_idxs[0]) :]
    dev = metric_temp - target
    if direction > 0:
        wrong_dev = np.maximum(0.0, -dev)
    else:
        wrong_dev = np.maximum(0.0, dev)

    overshoot = float(np.max(wrong_dev)) if len(wrong_dev) else 0.0
    cost = float(tail_mae + overshoot_weight * (overshoot**2))
    return {
        "tail_mae": tail_mae,
        "overshoot": overshoot,
        "cost": cost,
        "duration_s": float(t_rel[-1]),
    }


def load_history(args: argparse.Namespace) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    samples_by_run: Dict[str, pd.DataFrame] = {}

    for run_dir in iter_run_dirs(Path(args.history_root)):
        df = read_samples(run_dir)
        if df is None:
            continue

        run_id = run_id_for(run_dir, df)
        metrics = compute_metrics(
            df,
            entry_band=args.entry_band,
            tail_window_s=args.tail_window_s,
            overshoot_weight=args.overshoot_weight,
        )
        if metrics["tail_mae"] is None or metrics["overshoot"] is None:
            continue

        rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "tail_mae": float(metrics["tail_mae"]),
                "overshoot": float(metrics["overshoot"]),
                "cost": float(metrics["cost"]),
                "duration_s": float(metrics["duration_s"]),
            }
        )
        samples_by_run[run_id] = df

    if not rows:
        raise RuntimeError(f"No valid run_samples.csv files found under {args.history_root}")

    rankings = pd.DataFrame(rows).sort_values(["cost", "tail_mae", "overshoot"])
    return rankings.reset_index(drop=True), samples_by_run


def interpolate_runs(
    dfs: Iterable[pd.DataFrame],
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = [df for df in dfs if len(df) >= 2 and float(df["t_rel"].iloc[-1]) > 0.0]
    if not valid:
        raise ValueError("No plottable runs were found.")

    max_common_time = min(float(df["t_rel"].iloc[-1]) for df in valid)
    grid = np.linspace(0.0, max_common_time, max(2, int(grid_points)))
    temps = np.vstack(
        [
            np.interp(
                grid,
                df["t_rel"].to_numpy(dtype=float),
                df["temp"].to_numpy(dtype=float),
            )
            for df in valid
        ]
    )
    refs = np.vstack(
        [
            np.interp(
                grid,
                df["t_rel"].to_numpy(dtype=float),
                df["temp_ref"].to_numpy(dtype=float),
            )
            for df in valid
        ]
    )

    mean_temp = np.mean(temps, axis=0)
    mean_ref = np.mean(refs, axis=0)
    spread = np.std(temps, axis=0)
    return grid, mean_temp, mean_ref, spread


def metrics_from_profile(
    grid: np.ndarray,
    temp: np.ndarray,
    target: float,
    start_temp: float,
    entry_band: float,
    tail_window_s: float,
    overshoot_weight: float,
) -> dict[str, float]:
    direction = 1.0 if start_temp > target else -1.0

    tail_start_time = float(grid[-1]) - max(0.0, float(tail_window_s))
    tail_mask = grid >= tail_start_time
    if not np.any(tail_mask):
        tail_mask = np.ones_like(grid, dtype=bool)
    tail_mae = float(np.mean(np.abs(temp[tail_mask] - target)))

    entry_idxs = np.where(np.abs(temp - target) <= entry_band)[0]
    if len(entry_idxs) == 0:
        overshoot = 0.0
    else:
        dev = temp[int(entry_idxs[0]) :] - target
        wrong_dev = np.maximum(0.0, -dev) if direction > 0 else np.maximum(0.0, dev)
        overshoot = float(np.max(wrong_dev)) if len(wrong_dev) else 0.0

    return {
        "tail_mae": tail_mae,
        "overshoot": overshoot,
        "cost": float(tail_mae + overshoot_weight * (overshoot**2)),
    }


def plot_trajectory_comparison(
    out_path: Path,
    grid: np.ndarray,
    baseline_temp: np.ndarray,
    baseline_ref: np.ndarray,
    baseline_spread: np.ndarray,
    advisor_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    target = float(baseline_ref[0])
    advisor_t = advisor_df["t_rel"].to_numpy(dtype=float)
    advisor_temp = advisor_df["temp"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.fill_between(
        grid,
        baseline_temp - baseline_spread,
        baseline_temp + baseline_spread,
        color="#9aa0a6",
        alpha=0.22,
        linewidth=0,
    )
    ax.plot(grid, baseline_temp, color="#5f6368", linewidth=2.8, label="Baseline")
    ax.plot(
        advisor_t,
        smooth_series(advisor_temp),
        color="#0b8f7a",
        linewidth=2.8,
        label="AI Advisor",
    )
    ax.axhline(target, color="#202124", linewidth=1.2, alpha=0.75, label="Target")

    tail_start = max(0.0, min(float(grid[-1]), float(grid[-1]) - args.tail_window_s))
    ax.axvspan(tail_start, float(grid[-1]), color="#fbbc04", alpha=0.11)

    ax.set_title("Baseline vs AI Advisor Temperature Trajectory")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


def improvement_color(value: float) -> str:
    return "#0b8f7a" if value >= 0.0 else "#b3261e"


def reduction_label(value: float) -> str:
    if value >= 0.0:
        return f"-{value:.1f}% reduction"
    return f"+{abs(value):.1f}% increase"


def plot_improvement_dashboard(
    out_path: Path,
    rankings: pd.DataFrame,
    baseline_metrics: dict[str, float],
    advisor_row: pd.Series,
    args: argparse.Namespace,
) -> None:
    advisor_metrics = {
        "tail_mae": float(advisor_row["tail_mae"]),
        "overshoot": float(advisor_row["overshoot"]),
        "cost": float(advisor_row["cost"]),
    }
    improvements = {
        name: safe_percent_improvement(baseline_metrics[name], advisor_metrics[name])
        for name in ("tail_mae", "overshoot", "cost")
    }

    fig = plt.figure(figsize=(16, 6), facecolor="#f8f9fa")
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    fig.suptitle("AI Advisor Improvement Over Baseline", fontsize=22, fontweight="bold", y=0.97)

    metric_labels = {
        "tail_mae": "Mean Absolute Error",
        "overshoot": "Overshoot",
        "cost": "Cost",
    }
    for idx, metric in enumerate(("tail_mae", "overshoot", "cost")):
        ax = fig.add_subplot(gs[0, idx])
        vals = [baseline_metrics[metric], advisor_metrics[metric]]
        bars = ax.bar(
            ["Baseline", "AI Advisor"],
            vals,
            color=["#5f6368", "#0b8f7a"],
            width=0.6,
        )
        ax.set_title(metric_labels[metric])
        ax.set_ylim(0.0, max(vals) * 1.24 if max(vals) > 0.0 else 1.0)
        ax.grid(axis="y", alpha=0.2)
        ax.bar_label(bars, labels=[f"{v:.4g}" for v in vals], padding=4)
        ax.text(
            0.5,
            -0.22,
            reduction_label(improvements[metric]),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=improvement_color(improvements[metric]),
        )

    fig.subplots_adjust(top=0.78, left=0.07, right=0.97, bottom=0.24)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


def write_outputs(
    rankings: pd.DataFrame,
    baseline_metrics: dict[str, float],
    advisor_row: pd.Series,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(out_dir / "run_rankings.csv", index=False)

    advisor_metrics = {
        "tail_mae": float(advisor_row["tail_mae"]),
        "overshoot": float(advisor_row["overshoot"]),
        "cost": float(advisor_row["cost"]),
    }
    summary_rows = []
    for metric in ("tail_mae", "overshoot", "cost"):
        summary_rows.append(
            {
                "metric": metric,
                "baseline": baseline_metrics[metric],
                "ai_advisor": advisor_metrics[metric],
                "improvement_percent": safe_percent_improvement(
                    baseline_metrics[metric], advisor_metrics[metric]
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "baseline_vs_ai_advisor.csv", index=False)


def main() -> None:
    args = build_arg_parser().parse_args()
    rankings, samples_by_run = load_history(args)
    advisor_row = rankings.iloc[0]
    advisor_df = samples_by_run[str(advisor_row["run_id"])]

    grid, baseline_temp, baseline_ref, baseline_spread = interpolate_runs(
        samples_by_run.values(),
        grid_points=args.grid_points,
    )
    baseline_metrics = metrics_from_profile(
        grid=grid,
        temp=baseline_temp,
        target=float(baseline_ref[0]),
        start_temp=float(baseline_temp[0]),
        entry_band=args.entry_band,
        tail_window_s=args.tail_window_s,
        overshoot_weight=args.overshoot_weight,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_comparison(
        out_dir / "baseline_vs_ai_advisor_trajectory.png",
        grid,
        baseline_temp,
        baseline_ref,
        baseline_spread,
        advisor_df,
        args,
    )
    plot_improvement_dashboard(
        out_dir / "ai_advisor_improvement.png",
        rankings,
        baseline_metrics,
        advisor_row,
        args,
    )
    write_outputs(rankings, baseline_metrics, advisor_row, out_dir, args)

    print(f"Valid runs ranked: {len(rankings)}")
    print(f"AI Advisor run: {advisor_row['run_id']}")
    print(f"Tail MAE: baseline={baseline_metrics['tail_mae']:.6f}, advisor={advisor_row['tail_mae']:.6f}")
    print(f"Overshoot: baseline={baseline_metrics['overshoot']:.6f}, advisor={advisor_row['overshoot']:.6f}")
    print(f"Saved outputs: {out_dir}")


if __name__ == "__main__":
    main()
