#!/usr/bin/env python3
"""Run multiple TCP GP decision replay experiments from a batch GP comparison CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    default_batch_csv = ROOT / "gru" / "gru_ranked_pid_runs_batch" / "batch_gp_comparison.csv"
    default_script = ROOT / "optimization" / "gp_experiment.py"
    default_output = ROOT / "optimization" / "output" / "batch_gp_decision_replay_runs.csv"

    ap = argparse.ArgumentParser(
        description="Replay multiple generated GRU-ranked GP decision schedules over TCP."
    )
    ap.add_argument("--batch-csv", default=str(default_batch_csv),
                    help="CSV produced by gru/batch_gp_gru.py.")
    ap.add_argument("--script", default=str(default_script),
                    help="Path to gp_experiment.py.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--order", choices=["csv", "best"], default="best",
                    help="Replay CSV order or selection_score order.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max schedules to replay. 0 means all.")
    ap.add_argument("--start-index", type=int, default=0,
                    help="Skip this many schedules after sorting/filtering.")
    ap.add_argument("--only", default="",
                    help="Comma-separated scenario_name values to replay.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")

    ap.add_argument("--changed-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--test-duration", type=float, default=None)
    ap.add_argument("--extra-duration", type=float, default=60.0)
    ap.add_argument("--test-temp-ref", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0)

    ap.add_argument("--condition-initial", action="store_true")
    ap.add_argument("--skip-preconditioning", action="store_true")
    ap.add_argument("--heatup-temp-ref", type=float, default=25.0)
    ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0)
    ap.add_argument("--post-heatup-cooldown", type=float, default=3.0 * 60.0)

    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)
    ap.add_argument("--pid-row", type=int, default=1)
    ap.add_argument("--tcp-host", default=None)
    ap.add_argument("--tcp-port", type=int, default=None)
    ap.add_argument("--tcp-timeout", type=float, default=None)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)

    ap.add_argument("--history-root", default="run_history")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--events-csv", default="pid_events.csv")
    ap.add_argument("--all-runs-csv", default=str(default_output))

    return ap


def load_batch(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.batch_csv)
    if not path.exists():
        raise FileNotFoundError(f"Batch comparison CSV not found: {path}")
    df = pd.read_csv(path)
    if "decisions_csv" not in df.columns:
        raise ValueError("Batch CSV must contain a decisions_csv column")

    df = df.copy()
    df = df[df["decisions_csv"].notna()]
    if "scenario_name" not in df.columns:
        df["scenario_name"] = [f"schedule_{i + 1}" for i in range(len(df))]

    if args.only.strip():
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        df = df[df["scenario_name"].astype(str).isin(wanted)]

    if args.order == "best" and "selection_score" in df.columns:
        df = df.sort_values("selection_score", ascending=True)

    if args.start_index > 0:
        df = df.iloc[args.start_index :]
    if args.limit > 0:
        df = df.head(args.limit)

    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No schedules selected for replay")
    return df


def resolve_decisions_path(path_text: str, batch_csv: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidates = [
        batch_csv.parent / path,
        ROOT / path,
        ROOT / "gru" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def add_optional_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def command_for_schedule(
    *,
    args: argparse.Namespace,
    decisions_csv: Path,
) -> list[str]:
    cmd = [
        args.python,
        args.script,
        "--decisions-csv",
        str(decisions_csv),
        "--extra-duration",
        str(args.extra_duration),
        "--test-temp-ref",
        str(args.test_temp_ref),
        "--dt",
        str(args.dt),
        "--progress-every",
        str(args.progress_every),
        "--entry-band",
        str(args.entry_band),
        "--overshoot-weight",
        str(args.overshoot_weight),
        "--pid-row",
        str(args.pid_row),
        "--read-retries",
        str(args.read_retries),
        "--read-retry-delay-s",
        str(args.read_retry_delay_s),
        "--max-consecutive-failures",
        str(args.max_consecutive_failures),
        "--history-root",
        str(args.history_root),
        "--samples-csv",
        str(args.samples_csv),
        "--runs-csv",
        str(args.runs_csv),
        "--events-csv",
        str(args.events_csv),
        "--all-runs-csv",
        str(args.all_runs_csv),
    ]

    add_optional_flag(cmd, "--test-duration", args.test_duration)
    add_optional_flag(cmd, "--tcp-host", args.tcp_host)
    add_optional_flag(cmd, "--tcp-port", args.tcp_port)
    add_optional_flag(cmd, "--tcp-timeout", args.tcp_timeout)

    cmd.append("--changed-only" if args.changed_only else "--no-changed-only")
    if args.condition_initial:
        cmd.append("--condition-initial")
    if args.skip_preconditioning:
        cmd.append("--skip-preconditioning")
    cmd.extend([
        "--heatup-temp-ref",
        str(args.heatup_temp_ref),
        "--heatup-duration",
        str(args.heatup_duration),
        "--post-heatup-cooldown",
        str(args.post_heatup_cooldown),
    ])
    return cmd


def main() -> None:
    args = build_arg_parser().parse_args()
    batch_csv = Path(args.batch_csv)
    df = load_batch(args)

    print("=== Batch GP TCP experiments ===")
    print(f"batch csv: {batch_csv}")
    print(f"script:    {args.script}")
    print(f"selected:  {len(df)}")
    print(f"all runs:  {args.all_runs_csv}")

    for idx, row in df.iterrows():
        scenario_name = str(row.get("scenario_name", f"schedule_{idx + 1}"))
        decisions_csv = resolve_decisions_path(str(row["decisions_csv"]), batch_csv)
        print(f"\n[{idx + 1}/{len(df)}] Replaying {scenario_name}")
        print(f"decisions: {decisions_csv}")

        cmd = command_for_schedule(args=args, decisions_csv=decisions_csv)
        if args.dry_run:
            print(" ".join(cmd))
            continue

        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"Replay failed: {scenario_name}")
            if args.stop_on_error:
                raise SystemExit(result.returncode)

    print("\nBatch GP experiment replay complete.")


if __name__ == "__main__":
    main()
