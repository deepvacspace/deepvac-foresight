#!/usr/bin/env python3
"""Batch runner for gp_gru.py ranked-GP/GRU trajectory generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run multiple GRU-ranked GP PID simulations and compare trajectories."
    )
    ap.add_argument("--script", default=str(SCRIPT_DIR / "gp_gru.py"), help="Path to gp_gru.py.")
    ap.add_argument("--checkpoint", default=str(SCRIPT_DIR / "validation_t1" / "gru_t1.pt"))
    ap.add_argument("--candidate-table", default=str(SCRIPT_DIR / "mpc_pid_runs" / "gru_pid_candidate_table.csv"))
    ap.add_argument("--output-dir", default=str(SCRIPT_DIR / "gru_ranked_pid_runs_batch"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--only", default="", help="Comma-separated scenario names to run.")
    ap.add_argument("--limit", type=int, default=0, help="Max scenarios to run. 0 means all.")
    ap.add_argument("--start-temp", type=float, default=None, help="Override start temp for all scenarios.")
    ap.add_argument("--target-temp", type=float, default=None, help="Override target temp for all scenarios.")
    ap.add_argument("--duration-s", type=float, default=None, help="Override duration for all scenarios.")
    ap.add_argument("--dt-s", type=float, default=None, help="Override dt for all scenarios.")
    return ap


def make_scenarios() -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "start_temp": 27,
        "target_temp": 0,
        "duration_s": 1200,
        "dt_s": 2,
        "horizon_s": 60,
        "hold_s": 30,
        "kp_init": 50,
        "ki_init": 1,
        "kd_init": 1,
        "kp_min": 1,
        "kp_max": 50,
        "ki_min": 1,
        "ki_max": 1000,
        "kd_min": 1,
        "kd_max": 80,
        "candidates_per_decision": 32,
        "neighbor_pool": 300,
        "include_current_candidate": True,
        "include_anchor_candidates": True,
        "max_pid_delta_frac": 1.0,
        "history_score_weight": 0.10,
        "overshoot_tolerance": 0.05,
        "motion_error_scale": 6,
        "near_band": 2,
        "settle_band": 0.5,
        "rank_motion_weight": 1,
        "rank_std_weight": 1,
        "rank_history_weight": 0.05,
        "apply_margin": 0,
        "print_every_decision": False,
    }

    scenarios: list[dict[str, Any]] = []

    for horizon_s, hold_s in [(40, 20), (60, 30), (90, 30), (120, 30), (180, 30)]:
        s = dict(base)
        s.update({"name": f"h{horizon_s}_hold{hold_s}", "horizon_s": horizon_s, "hold_s": hold_s})
        scenarios.append(s)

    for candidates, pool in [(16, 150), (32, 300), (64, 500), (96, 750)]:
        s = dict(base)
        s.update({
            "name": f"cand{candidates}_pool{pool}",
            "candidates_per_decision": candidates,
            "neighbor_pool": pool,
        })
        scenarios.append(s)

    for motion_scale, rank_motion, rank_std in [(4, 1, 1), (6, 1, 1), (10, 1, 1), (6, 2, 1), (6, 1, 3)]:
        s = dict(base)
        s.update({
            "name": f"motion_scale{motion_scale}_mw{rank_motion}_sw{rank_std}",
            "motion_error_scale": motion_scale,
            "rank_motion_weight": rank_motion,
            "rank_std_weight": rank_std,
        })
        scenarios.append(s)

    for overshoot_tolerance in [0.02, 0.05, 0.10, 0.25]:
        s = dict(base)
        s.update({
            "name": f"overshoot_tol{str(overshoot_tolerance).replace('.', 'p')}",
            "overshoot_tolerance": overshoot_tolerance,
        })
        scenarios.append(s)

    for name, kp, ki, kd in [
        ("init_working", 50, 1, 1),
        ("init_old_bad_recovery", 6, 997, 16),
        ("init_gp_far", 11, 995, 1),
        ("init_low_all", 1, 1, 1),
    ]:
        s = dict(base)
        s.update({"name": name, "kp_init": kp, "ki_init": ki, "kd_init": kd})
        scenarios.append(s)

    return scenarios


def cli_flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def scenario_to_command(
    *,
    python_exe: str,
    script: str,
    checkpoint: str,
    candidate_table: str,
    output_dir: str,
    scenario: dict[str, Any],
) -> list[str]:
    cmd = [
        python_exe,
        script,
        "--checkpoint",
        checkpoint,
        "--candidate-table",
        candidate_table,
        "--output-dir",
        output_dir,
    ]
    for key, value in scenario.items():
        if key == "name":
            continue
        flag = cli_flag_name(key)
        if isinstance(value, bool):
            cmd.append(flag if value else f"--no-{flag[2:]}")
        else:
            cmd.extend([flag, str(value)])
    return cmd


def find_latest_summary(output_dir: Path, before_existing: set[Path]) -> Path | None:
    summaries = set(output_dir.glob("gru_ranked_*/gru_ranked_summary.json"))
    new_summaries = sorted(summaries - before_existing, key=lambda p: p.stat().st_mtime)
    return None if not new_summaries else new_summaries[-1]


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": summary.get("run_id"),
        "checkpoint": summary.get("checkpoint"),
        "trajectory_csv": summary.get("trajectory_csv"),
        "decisions_csv": summary.get("decisions_csv"),
    }
    for key, value in summary.get("scenario", {}).items():
        row[f"scenario_{key}"] = value
    for key, value in summary.get("selector", {}).items():
        row[f"selector_{key}"] = value
    bounds = summary.get("bounds", {})
    for group_name, values in bounds.items():
        if isinstance(values, list) and len(values) == 2:
            row[f"bounds_{group_name}_min"] = values[0]
            row[f"bounds_{group_name}_max"] = values[1]
        else:
            row[f"bounds_{group_name}"] = values
    for key, value in summary.get("metrics", {}).items():
        row[f"metric_{key}"] = value
    return row


def add_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "metric_overshoot_max",
        "metric_final_abs_error",
        "metric_tail_mae",
        "metric_tail_std",
        "metric_pid_changes",
    ]
    if any(col not in df.columns for col in required):
        return df
    df = df.copy()
    df["selection_score"] = (
        100.0 * df["metric_overshoot_max"].astype(float)
        + 10.0 * df["metric_final_abs_error"].astype(float)
        + 5.0 * df["metric_tail_mae"].astype(float)
        + 2.0 * df["metric_tail_std"].astype(float)
        + 0.05 * df["metric_pid_changes"].astype(float)
    )
    return df


def filter_scenarios(scenarios: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    overrides = {
        "start_temp": args.start_temp,
        "target_temp": args.target_temp,
        "duration_s": args.duration_s,
        "dt_s": args.dt_s,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if clean_overrides:
        scenarios = [dict(s, **clean_overrides) for s in scenarios]

    if args.only.strip():
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        scenarios = [s for s in scenarios if str(s["name"]) in wanted]
    if args.limit > 0:
        scenarios = scenarios[: args.limit]
    return scenarios


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = filter_scenarios(make_scenarios(), args)

    print("=== Batch GRU-ranked GP runs ===")
    print(f"script:          {args.script}")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"candidate table: {args.candidate_table}")
    print(f"output dir:      {output_dir}")
    print(f"scenarios:       {len(scenarios)}")

    rows: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios, start=1):
        name = str(scenario["name"])
        print(f"\n[{idx}/{len(scenarios)}] Running scenario: {name}")
        before_existing = set(output_dir.glob("gru_ranked_*/gru_ranked_summary.json"))
        cmd = scenario_to_command(
            python_exe=args.python,
            script=args.script,
            checkpoint=args.checkpoint,
            candidate_table=args.candidate_table,
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

        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        row = flatten_summary(summary)
        row["scenario_name"] = name
        row["summary_json"] = str(summary_path)
        rows.append(row)

    if not rows:
        if args.dry_run:
            print("\nDry run complete. No simulations were executed.")
            return
        print("\nNo successful runs were collected.")
        return

    df = add_selection_score(pd.DataFrame(rows))
    sort_cols: list[str] = []
    if "metric_valid" in df.columns:
        df["valid_sort"] = df["metric_valid"].apply(lambda x: 0 if bool(x) else 1)
        sort_cols.append("valid_sort")
    if "selection_score" in df.columns:
        sort_cols.append("selection_score")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    comparison_csv = output_dir / "batch_gp_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    print("\n=== Best runs ===")
    display_cols = [
        "scenario_name",
        "selection_score",
        "metric_tail_mae",
        "metric_final_abs_error",
        "metric_overshoot_max",
        "metric_time_to_settle_s",
        "metric_pid_changes",
        "decisions_csv",
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    print(df[display_cols].head(10).to_string(index=False))
    print(f"\nSaved comparison: {comparison_csv}")


if __name__ == "__main__":
    main()
