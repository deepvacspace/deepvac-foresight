#!/usr/bin/env python3
"""Plot metrics in regular run order"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bo_common import append_mae_column, compute_tail_cost


RUN_FILE_SPECS = (
    ("run_summary.csv", "run_samples.csv"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    default_plot_dir = Path(__file__).parent / "plots"
    parser = argparse.ArgumentParser(
        description="Plot history metrics computed directly from samples in regular run order."
    )
    parser.add_argument(
        "--history-dir",
        default=str(Path(__file__).parent / "run_history"),
        help="Directory containing run summaries like run_summary.csv and run_summary.csv",
    )
    parser.add_argument(
        "--out-cost",
        default=str(default_plot_dir / "history_cost.png"),
        help="Path to save the cost plot image",
    )
    parser.add_argument(
        "--out-mae",
        default=str(default_plot_dir / "history_tail_mae.png"),
        help="Path to save the tail_mae plot image",
    )
    parser.add_argument(
        "--out-tail-std",
        default=str(default_plot_dir / "history_tail_std.png"),
        help="Path to save the tail temperature standard deviation plot image",
    )
    parser.add_argument(
        "--out-time-to-cero",
        default=str(default_plot_dir / "history_time_to_cero.png"),
        help="Path to save the time_to_cero plot image",
    )
    parser.add_argument(
        "--out-overshoot",
        default=str(default_plot_dir / "history_overshoot.png"),
        help="Path to save the overshoot plot image",
    )
    parser.add_argument(
        "--entry-band",
        type=float,
        default=2.0,
        help="Entry band used when recomputing cost from samples",
    )
    parser.add_argument(
        "--tail-window-s",
        type=float,
        default=300.0,
        help="Seconds at the end of the run used for tail MAE and tail std plots",
    )
    parser.add_argument(
        "--overshoot-weight",
        type=float,
        default=10.0,
        help="Overshoot weight used when recomputing cost from samples",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    return parser


def compute_time_to_cero(samples_df: pd.DataFrame) -> float | None:
    df = samples_df.sort_values("timestamp").reset_index(drop=True).copy()
    temp = df["temp"].astype(float).to_numpy()
    temp_ref = df["temp_ref"].astype(float).to_numpy()
    ts = df["timestamp"].astype(float).to_numpy()

    if len(df) == 0:
        return None

    target = float(temp_ref[0])
    start_temp = float(temp[0])

    if start_temp > target:
        reached_idx = np.where(temp <= target)[0]
    elif start_temp < target:
        reached_idx = np.where(temp >= target)[0]
    else:
        reached_idx = np.array([0], dtype=int)

    if len(reached_idx) == 0:
        return None

    first_idx = int(reached_idx[0])
    return float(ts[first_idx] - ts[0])


def compute_sample_metrics(samples_path: Path, args: argparse.Namespace) -> dict[str, object] | None:
    if not samples_path.exists() or samples_path.stat().st_size == 0:
        return None

    samples_df = pd.read_csv(samples_path).sort_values("timestamp").reset_index(drop=True)
    samples_df = append_mae_column(samples_df)
    cost_info = compute_tail_cost(
        samples_df,
        entry_band=args.entry_band,
        overshoot_weight=args.overshoot_weight,
    )
    target = pd.to_numeric(samples_df["temp_ref"], errors="coerce").dropna()
    target_value = float(target.iloc[0]) if len(target) else float("nan")
    timestamp = pd.to_numeric(samples_df["timestamp"], errors="coerce")
    temp = pd.to_numeric(samples_df["temp"], errors="coerce")
    tail_start_time = float(timestamp.dropna().iloc[-1]) - max(0.0, float(args.tail_window_s)) if timestamp.notna().any() else None
    tail_mask = timestamp >= tail_start_time if tail_start_time is not None else pd.Series(False, index=samples_df.index)
    if not tail_mask.any():
        tail_mask = pd.Series(True, index=samples_df.index)
    tail_temp = temp[tail_mask].dropna()
    tail_mae = None
    tail_std = None
    if len(tail_temp) and np.isfinite(target_value):
        tail_mae = float(np.mean(np.abs(tail_temp.to_numpy(dtype=float) - target_value)))
        tail_std = float(tail_temp.std(ddof=0))
    cost = cost_info["cost"]
    if tail_mae is not None and cost_info.get("overshoot") is not None:
        cost = float(tail_mae + args.overshoot_weight * (float(cost_info["overshoot"]) ** 2))

    run_id = samples_df["run_id"].iloc[0] if "run_id" in samples_df.columns and len(samples_df) else samples_path.parent.name
    return {
        "run_id": str(run_id),
        "cost": cost,
        "tail_mae": tail_mae,
        "tail_std": tail_std,
        "overshoot": cost_info["overshoot"],
        "time_to_cero": compute_time_to_cero(samples_df),
        "source_file": str(samples_path.with_name(samples_path.name.replace("_samples", "_runs"))),
        "samples_file": str(samples_path),
    }


def load_history_runs(history_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    files: list[tuple[Path, Path]] = []
    for run_name, sample_name in RUN_FILE_SPECS:
        for csv_path in history_dir.glob(f"*/{run_name}"):
            files.append((csv_path, csv_path.with_name(sample_name)))
    files.sort(key=lambda pair: str(pair[0]))

    if not files:
        expected = ", ".join(run_name for run_name, _ in RUN_FILE_SPECS)
        raise FileNotFoundError(f"No run csv files ({expected}) found under: {history_dir}")

    records: list[dict[str, object]] = []
    for csv_path, samples_path in files:
        try:
            metrics = compute_sample_metrics(samples_path, args)
        except Exception as exc:
            print(f"Skipping unreadable samples {samples_path}: {exc}")
            continue

        if metrics is None:
            print(f"Skipping missing or empty samples file: {samples_path}")
            continue

        metrics["source_file"] = str(csv_path)
        records.append(metrics)

    if not records:
        raise RuntimeError("No valid run rows were found in the history directory.")

    all_runs = pd.DataFrame(records)
    for col in ("cost", "tail_mae", "tail_std", "overshoot", "time_to_cero"):
        all_runs[col] = pd.to_numeric(all_runs[col], errors="coerce")
    all_runs = all_runs.reset_index(drop=True)
    all_runs["run_number"] = all_runs.index + 1
    return all_runs


def plot_metric_order(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    out_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))

    x = df.index + 1

    ax.plot(x, df[metric_col], marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Run order")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

    y_span = df[metric_col].max() - df[metric_col].min()
    y_pad = max(y_span * 0.01, 0.05)
    for i, row in df.iterrows():
        xi = i + 1
        yi = float(row[metric_col])
        ax.text(
            xi,
            yi + y_pad,
            str(row["run_id"]),
            rotation=55,
            ha="left",
            va="bottom",
            fontsize=7,
        )

    best_idx = int(df[metric_col].idxmin())
    worst_idx = int(df[metric_col].idxmax())
    best_row = df.loc[best_idx]
    worst_row = df.loc[worst_idx]
    fig.suptitle(
        f"History runs: {len(df)} | best={best_row[metric_col]:.4f} ({best_row['run_id']}) | "
        f"worst={worst_row[metric_col]:.4f} ({worst_row['run_id']})",
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metric_if_available(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    out_path: Path,
    show: bool,
) -> None:
    metric_df = df.dropna(subset=[metric_col]).copy()
    if metric_df.empty:
        print(f"No numeric {metric_col} values found. Skipped {metric_col} plot.")
        return

    best_idx = int(metric_df[metric_col].idxmin())
    worst_idx = int(metric_df[metric_col].idxmax())
    best_row = metric_df.loc[best_idx]
    worst_row = metric_df.loc[worst_idx]
    print(
        f"Best {metric_col}:  #{int(best_row['run_number'])} {best_row['run_id']} "
        f"= {best_row[metric_col]:.6f}"
    )
    print(
        f"Worst {metric_col}: #{int(worst_row['run_number'])} {worst_row['run_id']} "
        f"= {worst_row[metric_col]:.6f}"
    )
    plot_metric_order(
        df=metric_df.reset_index(drop=True),
        metric_col=metric_col,
        title=title,
        y_label=y_label,
        out_path=out_path,
        show=show,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    history_dir = Path(args.history_dir)

    df = load_history_runs(history_dir, args)
    print(f"Loaded {len(df)} run summaries from: {history_dir}")

    plot_metric_if_available(
        df=df,
        metric_col="cost",
        title="Cost",
        y_label="Cost",
        out_path=Path(args.out_cost),
        show=args.show,
    )
    plot_metric_if_available(
        df=df,
        metric_col="tail_mae",
        title="Tail MAE",
        y_label="Tail MAE",
        out_path=Path(args.out_mae),
        show=args.show,
    )
    plot_metric_if_available(
        df=df,
        metric_col="tail_std",
        title="Tail Temperature Std",
        y_label="Std Dev",
        out_path=Path(args.out_tail_std),
        show=args.show,
    )
    plot_metric_if_available(
        df=df,
        metric_col="time_to_cero",
        title="Time To Cero",
        y_label="Seconds",
        out_path=Path(args.out_time_to_cero),
        show=args.show,
    )
    plot_metric_if_available(
        df=df,
        metric_col="overshoot",
        title="Overshoot",
        y_label="Overshoot",
        out_path=Path(args.out_overshoot),
        show=args.show,
    )


if __name__ == "__main__":
    main()
