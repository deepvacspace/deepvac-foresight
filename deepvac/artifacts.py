"""Run-id/CSV/JSON persistence, run-directory discovery, and batch-scenario
orchestration.

Consolidates utils/bo_common.py's file helpers, utils/utils.py's
append_row_csv, the iter_run_dirs() function that was copy-pasted
identically into optimization/band_bo_gp.py, rank_runs.py, list_runs.py, and
ai_advisor_improvement.py, and the ~80% of gru/batch_mpc_runs.py and
lstm/batch_mpc_runs.py that was generic batch orchestration (CLI-command
construction, summary discovery/flattening, and result ranking) rather than
model-specific scenario definitions.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def save_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def append_rows_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_row_csv(path: str, row: Dict[str, object]) -> None:
    """Like append_rows_csv, but aligns a single row to an existing file's
    header (missing columns become blank) instead of requiring an exact
    column match. Used where the row shape can grow between runs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if (not out.exists()) or out.stat().st_size == 0:
        with out.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])

    if not header:
        with out.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    aligned_row = {col: row.get(col) for col in header}
    with out.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writerow(aligned_row)


def history_run_file(run_id: str, filename: str, folder_name: str = "history") -> str:
    return str(Path(folder_name) / run_id / Path(filename).name)


def load_runs_table(path: str):
    import pandas as pd

    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)

    # Fallback for run-scoped storage under history/<run_id>/<filename>.
    target_name = p.name
    tables = []
    for match in Path("history").glob(f"*/{target_name}"):
        if match.is_file() and match.stat().st_size > 0:
            tables.append(pd.read_csv(match))
    if tables:
        return pd.concat(tables, ignore_index=True)

    return pd.DataFrame(columns=["run_id", "kp", "ki", "kd", "mse"])


def iter_run_dirs(history_root: Path) -> Iterator[Path]:
    """Yield each run's directory under a run-history root, sorted by name."""
    if not history_root.exists():
        raise FileNotFoundError(f"History root does not exist: {history_root}")

    for child in sorted(history_root.iterdir()):
        if child.is_dir():
            yield child


# -----------------------------------------------------------------------------
# Batch-scenario orchestration (gru/batch_mpc_runs.py, lstm/batch_mpc_runs.py).
#
# Each script still defines build_arg_parser() (its own --script/--checkpoint
# defaults) and make_scenarios() (its own cost-weight scenario groups, which
# genuinely differ between the GRU and LSTM MPC cost designs -- see
# gru/mpc_gru.py:horizon_cost vs lstm/mpc_lstm.py:horizon_cost). Everything
# below was byte-identical between the two files.
# -----------------------------------------------------------------------------


def cli_flag_name(key: str) -> str:
    """mpc_horizon_s -> --mpc-horizon-s"""
    return "--" + key.replace("_", "-")


def scenario_to_command(
    *,
    python_exe: str,
    script: str,
    checkpoint: str,
    output_dir: str,
    scenario: Dict[str, Any],
) -> List[str]:
    """Convert one scenario dictionary into a subprocess command."""
    cmd = [python_exe, script, "--checkpoint", checkpoint, "--output-dir", output_dir]

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


def find_latest_summary(output_dir: Path, before_existing: set) -> Optional[Path]:
    """Find the new mpc_summary.json created by the most recent run."""
    summaries = set(output_dir.glob("mpc_*/mpc_summary.json"))
    new_summaries = sorted(summaries - before_existing, key=lambda p: p.stat().st_mtime)
    if not new_summaries:
        return None
    return new_summaries[-1]


def flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten an mpc_summary.json payload into one CSV row."""
    row: Dict[str, Any] = {}

    row["run_id"] = summary.get("run_id")
    row["checkpoint"] = summary.get("checkpoint")
    row["trajectory_csv"] = summary.get("trajectory_csv")
    row["decisions_csv"] = summary.get("decisions_csv")

    for key, value in summary.get("scenario", {}).items():
        row[f"scenario_{key}"] = value

    for key, value in summary.get("mpc", {}).items():
        row[f"mpc_{key}"] = value

    for group_name, values in summary.get("bounds", {}).items():
        if isinstance(values, list) and len(values) == 2:
            row[f"bounds_{group_name}_min"] = values[0]
            row[f"bounds_{group_name}_max"] = values[1]
        else:
            row[f"bounds_{group_name}"] = values

    for key, value in summary.get("cost_weights", {}).items():
        row[f"cost_{key}"] = value

    for key, value in summary.get("metrics", {}).items():
        row[f"metric_{key}"] = value

    return row


def add_selection_score(df):
    """Add a practical, lower-is-better ranking score for comparing runs.

    Priority: avoid overshoot, end close to the reference, keep tail MAE and
    tail STD low, and avoid excessive PID changes.
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


def run_batch_scenarios(*, args: Any, scenarios: List[Dict[str, Any]], label: str) -> None:
    """Run every scenario through `args.script` as a subprocess, collect each
    run's mpc_summary.json into one comparison CSV, and print the best runs.

    `label` is only used for the console banner (e.g. "GRU", "LSTM").
    """
    import pandas as pd

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    print(f"=== Batch {label} + MPC PID runs ===")
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
