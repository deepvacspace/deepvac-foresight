#!/usr/bin/env python3
"""Plot trajectories and summarize metrics for best-run folders"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLS = ("tail_mae", "tail_std", "overshoot", "cost")


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parent
    experiments_root = Path(__file__).resolve().parents[1] / "experiments"
    out_dir = root / "plots"
    summary_dir = root / "summary"
    ap = argparse.ArgumentParser(
        description="Plot trajectories for best, worst, and average run groups."
    )
    ap.add_argument("--best-one-root", default=str(experiments_root / "best_one_band"))
    ap.add_argument("--best-three-root", default=str(experiments_root / "best_three_band"))
    ap.add_argument("--worst-root", default=str(experiments_root / "worst_run"))
    ap.add_argument("--average-root", default=str(experiments_root / "average_run"))
    ap.add_argument("--out-plot", default=str(out_dir /
                    "best_run_trajectories.png"))
    ap.add_argument("--out-summary",
                    default=str(summary_dir / "best_run_summary.txt"))
    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)
    ap.add_argument("--grid-points", type=int, default=600)
    ap.add_argument("--show", action="store_true")
    return ap


def iter_child_groups(root: Path, label_prefix: str) -> Iterable[tuple[str, Path]]:
    if not root.exists():
        return

    for group_dir in sorted(root.iterdir()):
        if group_dir.is_dir():
            yield f"{label_prefix}_{group_dir.name}", group_dir


def iter_best_child_group(root: Path, label: str) -> Iterable[tuple[str, Path]]:
    if not root.exists():
        return

    for group_dir in sorted(root.iterdir()):
        if group_dir.is_dir():
            yield label, group_dir
            return


def has_run_samples(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return any((child / "run_samples.csv").exists() for child in root.iterdir() if child.is_dir())


def iter_pid_groups(
    best_one_root: Path,
    best_three_root: Path,
    worst_root: Path,
    average_root: Path,
) -> Iterable[tuple[str, Path]]:
    yield from iter_child_groups(best_one_root, "one_band")
    yield from iter_best_child_group(best_three_root, "three_band_best")

    if has_run_samples(worst_root):
        yield "worst", worst_root

    if has_run_samples(average_root):
        yield "average", average_root


def read_samples(run_dir: Path) -> Optional[pd.DataFrame]:
    sample_path = run_dir / "run_samples.csv"
    if not sample_path.exists() or sample_path.stat().st_size == 0:
        return None

    df = pd.read_csv(sample_path)
    required = {"timestamp", "temp", "temp_ref"}
    if df.empty or not required.issubset(df.columns):
        return None

    work = df.copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=list(required)).sort_values(
        "timestamp").reset_index(drop=True)
    if work.empty:
        return None

    t0 = float(work["timestamp"].iloc[0])
    work["t_rel"] = work["timestamp"].astype(float) - t0
    return work


def compute_metrics(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Optional[float]]:
    temp = df["temp"].to_numpy(dtype=float)
    temp_ref = df["temp_ref"].to_numpy(dtype=float)
    t = df["t_rel"].to_numpy(dtype=float)
    target = float(temp_ref[0])
    start_temp = float(temp[0])
    direction = 1.0 if start_temp > target else -1.0

    tail_start_time = float(t[-1]) - max(0.0, float(args.tail_window_s))
    tail_mask = t >= tail_start_time
    if not np.any(tail_mask):
        tail_mask = np.ones_like(t, dtype=bool)
    tail_temp = temp[tail_mask]

    tail_mae = float(np.mean(np.abs(tail_temp - target))
                     ) if len(tail_temp) else None
    tail_std = float(np.std(tail_temp)) if len(tail_temp) else None

    abs_err = np.abs(temp - target)
    entry_idxs = np.where(abs_err <= args.entry_band)[0]
    if len(entry_idxs) == 0:
        return {
            "tail_mae": tail_mae,
            "tail_std": tail_std,
            "overshoot": None,
            "cost": None,
        }

    start_idx = int(entry_idxs[0])
    metric_temp = temp[start_idx:]
    dev = metric_temp - target

    if direction > 0:
        wrong_dev = np.maximum(0.0, -dev)
    else:
        wrong_dev = np.maximum(0.0, dev)

    overshoot = float(np.max(wrong_dev)) if len(wrong_dev) else 0.0

    cost = None
    if tail_mae is not None:
        cost = float(tail_mae + args.overshoot_weight * (overshoot**2))

    return {
        "tail_mae": tail_mae,
        "tail_std": tail_std,
        "overshoot": overshoot,
        "cost": cost,
    }


def read_coefficients(run_dir: Path) -> Optional[Dict[str, float]]:
    summary_path = run_dir / "run_summary.csv"
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return None

    try:
        summary = pd.read_csv(summary_path, nrows=1)
    except Exception:
        return None

    coeff_cols = (
        "far_kp", "far_ki", "far_kd",
        "mid_kp", "mid_ki", "mid_kd",
        "near_kp", "near_ki", "near_kd",
    )
    if not set(coeff_cols).issubset(summary.columns):
        return None

    coeffs: Dict[str, float] = {}
    for col in coeff_cols:
        value = pd.to_numeric(summary[col], errors="coerce").iloc[0]
        if pd.isna(value):
            return None
        coeffs[col] = float(value)
    return coeffs


def summarize_group(label: str, group_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    runs: List[Dict[str, object]] = []
    coeffs: Optional[Dict[str, float]] = None
    for run_dir in sorted(group_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if coeffs is None:
            coeffs = read_coefficients(run_dir)
        df = read_samples(run_dir)
        if df is None:
            continue
        metrics = compute_metrics(df, args)
        runs.append({"run_id": run_dir.name, "df": df, **metrics})

    metric_means = {
        f"mean_{metric}": float(np.mean(values)) if values else None
        for metric in METRIC_COLS
        for values in [[float(run[metric]) for run in runs if run.get(metric) is not None]]
    }

    return {
        "label": label,
        "group_dir": group_dir,
        "runs": runs,
        "coeffs": coeffs,
        **metric_means,
    }


def interpolate_group_runs(group: Dict[str, object], grid_points: int) -> tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    runs = group["runs"]  # type: ignore[assignment]
    valid_runs = [run for run in runs if len(
        run["df"]) >= 2]  # type: ignore[index]
    if not valid_runs:
        raise ValueError(f"No plottable runs for {group['label']}")

    max_common_time = min(float(run["df"]["t_rel"].iloc[-1])
                          for run in valid_runs)  # type: ignore[index]
    grid = np.linspace(0.0, max_common_time, max(2, int(grid_points)))
    interpolated = []
    for run in valid_runs:
        df = run["df"]
        interpolated.append(
            np.interp(
                grid,
                df["t_rel"].to_numpy(dtype=float),
                df["temp"].to_numpy(dtype=float),
            )
        )

    mean_temp = np.mean(np.vstack(interpolated), axis=0)
    return grid, interpolated, mean_temp


def plot_trajectories(groups: List[Dict[str, object]], args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    cmap = plt.get_cmap("tab10")

    for idx, group in enumerate(groups):
        if not group["runs"]:
            continue
        color = cmap(idx % 10)
        grid, interpolated, mean_temp = interpolate_group_runs(
            group, args.grid_points)
        for temp in interpolated:
            ax.plot(grid, temp, color=color, alpha=0.18, linewidth=1.0)
        ax.plot(grid, mean_temp, color=color,
                linewidth=2.4, label=str(group["label"]))

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_title("Best Run Temperature Trajectories")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature")
    ax.grid(True, alpha=0.3)
    ax.legend(title="PID set")
    fig.tight_layout()

    out_path = Path(args.out_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved trajectory plot: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def plot_trajectories_tail(groups: List[Dict[str, object]], args: argparse.Namespace, tail_window_s: float = 300.0) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    cmap = plt.get_cmap("tab10")

    for idx, group in enumerate(groups):
        if not group["runs"]:
            continue
        color = cmap(idx % 10)
        grid, interpolated, mean_temp = interpolate_group_runs(
            group, args.grid_points)

        # Filter to tail window
        tail_start_idx = max(0, len(grid) - int(tail_window_s /
                             (grid[-1] / len(grid)) + 0.5)) if len(grid) > 1 else 0
        grid_tail = grid[tail_start_idx:]
        mean_temp_tail = mean_temp[tail_start_idx:]
        interpolated_tail = [temp[tail_start_idx:] for temp in interpolated]

        for temp in interpolated_tail:
            ax.plot(grid_tail, temp, color=color, alpha=0.18, linewidth=1.0)
        ax.plot(grid_tail, mean_temp_tail, color=color,
                linewidth=2.4, label=str(group["label"]))

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_title(
        f"Best Run Temperature Trajectories - Last {tail_window_s:g} seconds")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature")
    ax.grid(True, alpha=0.3)
    ax.legend(title="PID set")
    fig.tight_layout()

    # Save with _tail suffix
    out_path = Path(args.out_plot)
    tail_path = out_path.parent / (out_path.stem + "_tail_300s.png")
    tail_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tail_path, dpi=220, bbox_inches="tight")
    print(f"Saved tail trajectory plot: {tail_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def format_coeffs(coeffs: object) -> str:
    if not isinstance(coeffs, dict):
        return "NA"
    return (
        "far=({far_kp:.0f}, {far_ki:.0f}, {far_kd:.0f}); "
        "mid=({mid_kp:.0f}, {mid_ki:.0f}, {mid_kd:.0f}); "
        "near=({near_kp:.0f}, {near_ki:.0f}, {near_kd:.0f})"
    ).format(**coeffs)


def write_summary(groups: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "Best Run Summary",
        "===========================",
        "",
        f"Tail window: last {args.tail_window_s:g} seconds",
        f"Entry band for overshoot/cost: {args.entry_band:g}",
        f"Cost: tail_mae + {args.overshoot_weight:g} * overshoot^2",
        "",
    ]

    for group in groups:
        runs = group["runs"]  # type: ignore[assignment]
        lines.append(str(group["label"]))
        lines.append("-" * len(str(group["label"])))
        lines.append(
            "mean metrics: "
            f"tail_mae={format_value(group.get('mean_tail_mae'))}, "
            f"tail_std={format_value(group.get('mean_tail_std'))}, "
            f"overshoot={format_value(group.get('mean_overshoot'))}, "
            f"cost={format_value(group.get('mean_cost'))}"
        )
        lines.append(f"coeffs: {format_coeffs(group.get('coeffs'))}")
        lines.append("")
        lines.append("run_id,tail_mae,tail_std,overshoot,cost")
        for run in runs:
            lines.append(
                ",".join(
                    [
                        str(run["run_id"]),
                        format_value(run.get("tail_mae")),
                        format_value(run.get("tail_std")),
                        format_value(run.get("overshoot")),
                        format_value(run.get("cost")),
                    ]
                )
            )
        lines.append("")

    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {out_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    groups = [
        summarize_group(label, group_dir, args)
        for label, group_dir in iter_pid_groups(
            Path(args.best_one_root),
            Path(args.best_three_root),
            Path(args.worst_root),
            Path(args.average_root),
        )
    ]
    groups = [group for group in groups if group["runs"]]

    if not groups:
        raise RuntimeError(
            "No best-run groups with run_samples.csv files were found.")

    plot_trajectories(groups, args)
    plot_trajectories_tail(groups, args, tail_window_s=300.0)
    write_summary(groups, args)


if __name__ == "__main__":
    main()
