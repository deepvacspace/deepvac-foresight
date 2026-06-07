#!/usr/bin/env python3
"""
Batch runner for mpc_lstm.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run multiple LSTM+MPC PID simulations and compare results."
    )

    ap.add_argument(
        "--script",
        default=str(SCRIPT_DIR / "mpc_lstm.py"),
        help="Path to the single-run MPC script.",
    )
    ap.add_argument(
        "--checkpoint",
        default=str(SCRIPT_DIR / "validation_t1" / "lstm_t1.pt"),
        help="Path to the trained LSTM checkpoint.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "mpc_pid_runs_batch"),
        help="Directory where all batch runs will be saved.",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch if one scenario fails.",
    )

    return ap


# -----------------------------------------------------------------------------
# Scenario definition
# -----------------------------------------------------------------------------


def make_scenarios() -> list[dict[str, Any]]:
    """
    Define all test cases here.

    The start and target temperatures are fixed because all tests must use
    the same thermal transition. The batch compares only MPC behavior,
    optimizer settings, and cost-function weights.
    """

    base = {
        # Fixed thermal scenario.
        "start_temp": 27,
        "target_temp": 0,

        # Simulation setup.
        "duration_s": 1200,
        "dt_s": 2,

        # Optimizer setup.
        "optimizer": "cem",
        "cem_population": 256,
        "cem_iterations": 3,
        "cem_elite_frac": 0.12,

        # Default MPC setup.
        "mpc_horizon_s": 80,
        "mpc_hold_s": 20,

        # PID bounds.
        "kp_min": 1,
        "kp_max": 50,
        "ki_min": 1,
        "ki_max": 1000,
        "kd_min": 1,
        "kd_max": 20,

        # Initial PID.
        "kp_init": 6,
        "ki_init": 997,
        "kd_init": 16,

        # Cost weights.
        "w_overshoot_max": 80,
        "w_overshoot_rmse": 30,
        "w_abs_error": 2,
        "w_final_abs_error": 0.5,
        "w_near_std": 1,
        "w_control_change": 0.01,

        # Disable noisy output from the inner script.
        "print_every_decision": False,
        "print_optimizer_progress": False,
    }

    scenarios: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Group 1: different MPC horizon/hold combinations.
    # -------------------------------------------------------------------------
    for horizon_s, hold_s in [
        (40, 10),
        (60, 10),
        (80, 20),
        (120, 20),
        (180, 30),
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"h{horizon_s}_hold{hold_s}",
                "mpc_horizon_s": horizon_s,
                "mpc_hold_s": hold_s,
            }
        )
        scenarios.append(s)

    # -------------------------------------------------------------------------
    # Group 2: different tracking/stability cost settings.
    # -------------------------------------------------------------------------
    for w_abs_error, w_final_abs_error, w_near_std, w_control_change in [
        (1, 0.5, 1, 0.01),
        (2, 0.5, 1, 0.01),
        (5, 1.0, 1, 0.01),
        (2, 2.0, 1, 0.01),
        (2, 0.5, 5, 0.05),
    ]:
        s = dict(base)
        s.update(
            {
                "name": (
                    f"abs{w_abs_error}_final{w_final_abs_error}"
                    f"_std{w_near_std}_change{w_control_change}"
                ),
                "w_abs_error": w_abs_error,
                "w_final_abs_error": w_final_abs_error,
                "w_near_std": w_near_std,
                "w_control_change": w_control_change,
            }
        )
        scenarios.append(s)

    # -------------------------------------------------------------------------
    # Group 3: different overshoot penalties.
    # -------------------------------------------------------------------------
    for w_overshoot_max in [
        40,
        80,
        120,
        200,
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"overshoot{w_overshoot_max}",
                "w_overshoot_max": w_overshoot_max,
            }
        )
        scenarios.append(s)

    # -------------------------------------------------------------------------
    # Group 4: different CEM budgets.
    # -------------------------------------------------------------------------
    for population, iterations in [
        (128, 2),
        (256, 3),
        (512, 3),
        (512, 5),
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"cem{population}_iter{iterations}",
                "cem_population": population,
                "cem_iterations": iterations,
            }
        )
        scenarios.append(s)

    return scenarios


# -----------------------------------------------------------------------------
# Command construction
# -----------------------------------------------------------------------------


def cli_flag_name(key: str) -> str:
    """
    Convert Python-style parameter names to CLI flags.

    Example:
        mpc_horizon_s -> --mpc-horizon-s
        w_abs_error   -> --w-abs-error
    """

    return "--" + key.replace("_", "-")


def scenario_to_command(
    *,
    python_exe: str,
    script: str,
    checkpoint: str,
    output_dir: str,
    scenario: dict[str, Any],
) -> list[str]:
    """
    Convert one scenario dictionary into a subprocess command.
    """

    cmd = [
        python_exe,
        script,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        output_dir,
    ]

    skip_keys = {"name"}

    for key, value in scenario.items():
        if key in skip_keys:
            continue

        flag = cli_flag_name(key)

        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            else:
                cmd.append(f"--no-{flag[2:]}")
        else:
            cmd.extend([flag, str(value)])

    return cmd


# -----------------------------------------------------------------------------
# Summary handling
# -----------------------------------------------------------------------------


def find_latest_summary(output_dir: Path, before_existing: set[Path]) -> Path | None:
    """
    Find the new mpc_summary.json created by the most recent run.
    """

    summaries = set(output_dir.glob("mpc_*/mpc_summary.json"))
    new_summaries = sorted(
        summaries - before_existing,
        key=lambda p: p.stat().st_mtime,
    )

    if not new_summaries:
        return None

    return new_summaries[-1]


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten mpc_summary.json into one CSV row.
    """

    row: dict[str, Any] = {}

    row["run_id"] = summary.get("run_id")
    row["checkpoint"] = summary.get("checkpoint")
    row["trajectory_csv"] = summary.get("trajectory_csv")
    row["decisions_csv"] = summary.get("decisions_csv")

    scenario = summary.get("scenario", {})
    for key, value in scenario.items():
        row[f"scenario_{key}"] = value

    mpc = summary.get("mpc", {})
    for key, value in mpc.items():
        row[f"mpc_{key}"] = value

    bounds = summary.get("bounds", {})
    for group_name, values in bounds.items():
        if isinstance(values, list) and len(values) == 2:
            row[f"bounds_{group_name}_min"] = values[0]
            row[f"bounds_{group_name}_max"] = values[1]
        else:
            row[f"bounds_{group_name}"] = values

    cost_weights = summary.get("cost_weights", {})
    for key, value in cost_weights.items():
        row[f"cost_{key}"] = value

    metrics = summary.get("metrics", {})
    for key, value in metrics.items():
        row[f"metric_{key}"] = value

    return row


def add_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a practical score for ranking runs.

    Lower is better.

    Priority:
        1. Avoid overshoot.
        2. End close to the reference.
        3. Keep tail MAE low.
        4. Keep tail STD low.
        5. Avoid excessive PID changes.
    """

    required = [
        "metric_overshoot_max",
        "metric_final_abs_error",
        "metric_tail_mae",
        "metric_tail_std",
        "metric_pid_changes",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Could not compute selection_score. Missing columns: {missing}")
        return df

    df = df.copy()

    df["selection_score"] = (
        100.0 * df["metric_overshoot_max"]
        + 10.0 * df["metric_final_abs_error"]
        + 5.0 * df["metric_tail_mae"]
        + 2.0 * df["metric_tail_std"]
        + 0.05 * df["metric_pid_changes"]
    )

    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = make_scenarios()
    rows: list[dict[str, Any]] = []

    print("=== Batch LSTM + MPC PID runs ===")
    print(f"script:     {args.script}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output dir: {output_dir}")
    print(f"scenarios:  {len(scenarios)}")

    for idx, scenario in enumerate(scenarios, start=1):
        name = str(scenario["name"])

        print(f"\n[{idx}/{len(scenarios)}] Running scenario: {name}")

        before_existing = set(output_dir.glob("mpc_*/mpc_summary.json"))

        cmd = scenario_to_command(
            python_exe=args.python,
            script=args.script,
            checkpoint=args.checkpoint,
            output_dir=str(output_dir),
            scenario=scenario,
        )

        if args.dry_run:
            print(" ".join(cmd))
            continue

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            print(f"Scenario failed: {name}")

            if args.stop_on_error:
                raise SystemExit(result.returncode)

            continue

        summary_path = find_latest_summary(output_dir, before_existing)

        if summary_path is None:
            print(f"No summary found for scenario: {name}")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        row = flatten_summary(summary)
        row["scenario_name"] = name
        row["summary_json"] = str(summary_path)

        rows.append(row)

    if not rows:
        print("\nNo successful runs were collected.")
        return

    df = pd.DataFrame(rows)
    df = add_selection_score(df)

    sort_cols = []

    if "metric_valid" in df.columns:
        # Valid=True should come first.
        df["valid_sort"] = df["metric_valid"].apply(lambda x: 0 if bool(x) else 1)
        sort_cols.append("valid_sort")

    if "selection_score" in df.columns:
        sort_cols.append("selection_score")
    else:
        fallback_cols = [
            "metric_overshoot_max",
            "metric_tail_mae",
            "metric_tail_std",
            "metric_final_abs_error",
        ]
        sort_cols.extend([c for c in fallback_cols if c in df.columns])

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    comparison_csv = output_dir / "batch_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    print("\n=== Best runs ===")

    display_cols = [
        "scenario_name",
        "selection_score",
        "metric_valid",
        "metric_overshoot_max",
        "metric_tail_mae",
        "metric_tail_std",
        "metric_final_abs_error",
        "metric_time_to_near_s",
        "metric_time_to_settle_s",
        "metric_pid_changes",
        "summary_json",
    ]

    display_cols = [c for c in display_cols if c in df.columns]

    print(df[display_cols].head(20).to_string(index=False))

    print("\n=== Saved ===")
    print(f"comparison csv: {comparison_csv}")


if __name__ == "__main__":
    main()
