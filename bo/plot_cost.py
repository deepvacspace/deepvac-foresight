import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot Bayesian optimization progress from bo_params_history.json"
    )
    ap.add_argument(
        "--json",
        default="history/bo_params_history.json",
        help="Path to bo_params_history.json",
    )
    ap.add_argument(
        "--out-dir",
        default="plots",
        help="Directory where plots will be saved",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected non-empty list in {json_path}")

    df = pd.json_normalize(data)
    df.columns = [c.strip() for c in df.columns]

    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].fillna("").astype(str)

    if "cost" not in df.columns and "mse" in df.columns:
        df["cost"] = pd.to_numeric(df["mse"], errors="coerce")
    else:
        df["cost"] = pd.to_numeric(df.get("cost"), errors="coerce")

    if "meta.n_training_runs" not in df.columns:
        raise ValueError("Column 'meta.n_training_runs' not found in JSON history.")

    df["n_training_runs"] = pd.to_numeric(df["meta.n_training_runs"], errors="coerce")

    df = df[df["run_id"].ne("")].copy()
    df = df.dropna(subset=["cost", "n_training_runs"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No evaluated runs with numeric 'cost' and 'meta.n_training_runs' found in the JSON history."
        )

    df = df.sort_values("n_training_runs").reset_index(drop=True)
    df["best_cost_so_far"] = df["cost"].cummin()

    if "prediction.expected_improvement" in df.columns:
        df["expected_improvement"] = pd.to_numeric(
            df["prediction.expected_improvement"], errors="coerce"
        )
    else:
        df["expected_improvement"] = pd.NA

    if "prediction.cost_std" in df.columns:
        df["cost_std"] = pd.to_numeric(df["prediction.cost_std"], errors="coerce")
    else:
        df["cost_std"] = pd.NA

    # -------- Plot 1: observed cost + best --------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df["n_training_runs"],
        df["cost"],
        marker="o",
        linewidth=1.5,
        label="Observed cost",
    )
    ax.plot(
        df["n_training_runs"],
        df["best_cost_so_far"],
        marker="s",
        linewidth=2.0,
        label="Best cost",
    )

    best_idx = df["cost"].idxmin()
    best_x = float(df.loc[best_idx, "n_training_runs"])
    best_cost = float(df.loc[best_idx, "cost"])

    ax.scatter([best_x], [best_cost], s=80, label=f"Best run ({best_cost:.3f})")

    ax.set_title("Bayesian Optimization Cost")
    ax.set_xlabel("Number of training runs")
    ax.set_ylabel("Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    cost_plot_path = out_dir / "bo_cost.png"
    fig.savefig(cost_plot_path, dpi=150)
    print(f"Saved cost plot to: {cost_plot_path}")

    # -------- Plot 2: expected improvement --------
    ei_df = df.dropna(subset=["expected_improvement"]).copy()
    if not ei_df.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(
            ei_df["n_training_runs"],
            ei_df["expected_improvement"],
            marker="o",
            linewidth=1.5,
            label="Expected improvement",
        )
        ax2.set_title("Bayesian Optimization Expected Improvement")
        ax2.set_xlabel("Number of training runs")
        ax2.set_ylabel("Expected Improvement")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()

        ei_plot_path = out_dir / "bo_expected_improvement.png"
        fig2.savefig(ei_plot_path, dpi=150)
        print(f"Saved expected improvement plot to: {ei_plot_path}")
    else:
        print("No prediction.expected_improvement values found; skipped EI plot.")


    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
