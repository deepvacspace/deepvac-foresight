#!/usr/bin/env python3
"""Replay one gru/advise_pid.py session's PID choices against the real chamber
over TCP, and log the outcome next to the advisor's own prediction.

--advisor-dir points at one advise_pid.py session folder, which holds report.json and, 
for --mode adapt runs, decisions.csv:

  --mode suggest sessions -> single recommended_pid is held for duration_s.

  --mode adapt sessions -> PID triplets are replayed at the same elapsed-time offsets. 
    Only rows with changed=True reapply.

python -m optimization.replay_advisor_decisions --advisor-dir gru/advisor/advise_adapt_1787145280_6da9899c
python -m optimization.replay_advisor_decisions --advisor-dir gru/advisor/advise_suggest_1787136883_5d73fd26 --num-tests 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import append_row_csv, append_rows_csv, history_run_file, make_run_id  
from deepvac.metrics import append_mae_column, compute_tail_cost  
from tcp.tcp_common import (  
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
)

PIDTriplet = Tuple[int, int, int]
ScheduleStep = Tuple[float, PIDTriplet]  # (elapsed_s, (kp, ki, kd))


def build_arg_parser() -> argparse.ArgumentParser:
    output_dir = Path(__file__).with_name("output")
    ap = argparse.ArgumentParser()

    ap.add_argument("--advisor-dir", required=True,
                    help="A gru/advise_pid.py session folder (contains report.json, "
                         "and decisions.csv for --mode adapt sessions).")
    ap.add_argument("--decisions-csv", default=None,
                    help="Override for {advisor-dir}/decisions.csv.")
    ap.add_argument("--report-json", default=None,
                    help="Override for {advisor-dir}/report.json.")

    ap.add_argument("--num-tests", type=int, default=1)
    ap.add_argument("--forever", action="store_true", help="Replay continuously")
    ap.add_argument("--val", action="store_true", help="Save runs with val_* ids for validation.")
    ap.add_argument("--cooldown", type=float, default=3.0 * 60.0, help="Seconds between replays")

    # --- Anchor leg ----------------------------------------------------------
    # Brings the chamber to the advisor's own start_temp before the logged
    # replay begins, so the real run starts from the state the advisor
    # actually planned against.
    ap.add_argument("--anchor-temp", type=float, default=None,
                    help="Default: report.json's start_temp.")
    ap.add_argument("--anchor-duration-s", type=float, default=5.0 * 60.0)
    ap.add_argument("--skip-anchor", action="store_true", help="Start the replay from wherever the chamber is.")

    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0)

    # --- Safety ----------------------------------------------------------------
    ap.add_argument("--safety-margin-c", type=float, default=5.0,
                    help="Abort the replay early if temp strays this far past "
                         "[min(start_temp,target_temp), max(start_temp,target_temp)].")

    # --- Cost metric (same formula tocero_3band.py logs, for comparability) ----
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4]")

    # --- TCP ---------------------------------------------------------------------
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)

    # --- Output --------------------------------------------------------------------
    ap.add_argument("--history-root", default=str(Path(__file__).with_name("advisor_replay_history")))
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--events-csv", default="pid_schedule_events.csv")
    ap.add_argument("--all-runs-csv", default=str(output_dir / "advisor_replay_all_runs.csv"))

    return ap


# -----------------------------------------------------------------------------
# Loading the advisor session
# -----------------------------------------------------------------------------


def round_half_degree(value: float) -> float:
    """Chamber setpoints are commanded as clean numbers, not arbitrary floats."""
    return float(round(float(value) * 2.0) / 2.0)


def load_advisor_session(args: argparse.Namespace) -> Tuple[dict, List[ScheduleStep]]:
    advisor_dir = Path(args.advisor_dir)
    report_path = Path(args.report_json) if args.report_json else advisor_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    mode = report.get("mode")
    if mode == "suggest":
        pid = report["recommended_pid"]
        schedule: List[ScheduleStep] = [
            (0.0, (int(round(pid["kp"])), int(round(pid["ki"])), int(round(pid["kd"])))),
        ]
    elif mode == "adapt":
        decisions_path = Path(args.decisions_csv) if args.decisions_csv else advisor_dir / "decisions.csv"
        if not decisions_path.exists():
            raise FileNotFoundError(f"decisions.csv not found: {decisions_path}")
        df = pd.read_csv(decisions_path).sort_values("elapsed_s").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No decisions found in {decisions_path}")

        schedule = []
        for i, row in df.iterrows():
            if i == 0 or bool(row.get("changed", False)):
                pid = (int(round(row["kp"])), int(round(row["ki"])), int(round(row["kd"])))
                schedule.append((float(row["elapsed_s"]), pid))
    else:
        raise ValueError(f"Unknown advisor mode in {report_path}: {mode!r}")

    if not schedule:
        raise ValueError(f"Built an empty PID schedule from {advisor_dir}")
    return report, schedule


# -----------------------------------------------------------------------------
# One replay
# -----------------------------------------------------------------------------


def run_anchor_leg(anchor_temp: float, duration_s: float, args: argparse.Namespace) -> None:
    if duration_s <= 0:
        return
    print(f"[anchor] temp_ref={anchor_temp:.3f} for {duration_s:.1f}s (no logging)")
    publish_temp_ref_job(
        temp_ref=round_half_degree(anchor_temp),
        duration_s=duration_s,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    time.sleep(duration_s)


def run_single_replay(
    run_idx: int,
    total_tests: Optional[int],
    args: argparse.Namespace,
    report: dict,
    schedule: List[ScheduleStep],
) -> None:
    run_id = make_run_id(prefix="val" if args.val else "run")
    test_label = f"{run_idx}" if total_tests is None else f"{run_idx}/{total_tests}"

    start_temp = float(args.anchor_temp) if args.anchor_temp is not None else float(report["start_temp"])
    target_temp = float(report["target_temp"])
    duration_s = float(report["duration_s"])
    safety_lo = min(start_temp, target_temp) - args.safety_margin_c
    safety_hi = max(start_temp, target_temp) + args.safety_margin_c

    print(f"[run {run_id}] starting replay {test_label}: advisor_dir={args.advisor_dir} "
          f"mode={report.get('mode')} {start_temp:.2f} -> {target_temp:.2f} deg C over {duration_s:.0f}s, "
          f"{len(schedule)} PID step(s)")

    if not args.skip_anchor:
        run_anchor_leg(start_temp, args.anchor_duration_s, args)

    events: List[Dict[str, object]] = []
    _, first_pid = schedule[0]
    current_pid = apply_pid_update(label="start", run_id=run_id, row=args.pid_row, pid=first_pid, args=args, events=events)

    # Prime the TCP state-name cache. Some controller replies appear to omit
    # names intermittently once a job is running.
    _ = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)

    publish_temp_ref_job(
        temp_ref=target_temp,
        duration_s=duration_s,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    print(f"[run {run_id}] published temp_ref job for {duration_s:.1f}s")

    next_step_idx = 1  # schedule[0] was already applied as the start PID.
    rows: List[Dict[str, float]] = []
    consecutive_read_failures = 0
    aborted_reason = ""

    t0 = time.time()
    next_progress_ts = t0 + args.progress_every if args.progress_every > 0 else float("inf")

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= duration_s:
            break

        while next_step_idx < len(schedule) and schedule[next_step_idx][0] <= elapsed:
            step_elapsed, step_pid = schedule[next_step_idx]
            current_pid = apply_pid_update(
                label=f"t={step_elapsed:.0f}s", run_id=run_id, row=args.pid_row,
                pid=step_pid, args=args, events=events,
            )
            next_step_idx += 1

        snap: Optional[Dict[str, float]] = None
        last_read_exc: Optional[Exception] = None
        for _ in range(max(1, args.read_retries + 1)):
            try:
                snap = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)
                break
            except Exception as exc:
                last_read_exc = exc
                if args.read_retry_delay_s > 0:
                    time.sleep(args.read_retry_delay_s)

        if snap is None:
            consecutive_read_failures += 1
            if last_read_exc is not None:
                print(f"[run {run_id}] read failed "
                      f"({consecutive_read_failures}/{args.max_consecutive_failures}): {last_read_exc}")
            if consecutive_read_failures >= args.max_consecutive_failures:
                raise RuntimeError(f"Too many consecutive state read failures ({consecutive_read_failures})")
            time.sleep(args.dt)
            continue

        consecutive_read_failures = 0
        temp = float(snap["temp"])

        if temp < safety_lo or temp > safety_hi:
            aborted_reason = f"temp={temp:.2f} outside [{safety_lo:.1f}, {safety_hi:.1f}] safety range"
            print(f"[run {run_id}] SAFETY ABORT: {aborted_reason}")
            break

        sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
        rows.append({
            "run_id": run_id,
            "timestamp": now,
            "schedule_step": next_step_idx - 1,
            "kp": float(current_pid[0]),
            "ki": float(current_pid[1]),
            "kd": float(current_pid[2]),
            **snap,
            "sq_error": sq_error,
        })

        if now >= next_progress_ts:
            print(f"[run {run_id}] samples={len(rows)} elapsed={elapsed:.1f}/{duration_s:.1f}s "
                  f"temp={temp:.3f} temp_ref={snap['temp_ref']:.3f} "
                  f"pid=({current_pid[0]}, {current_pid[1]}, {current_pid[2]})")
            next_progress_ts += args.progress_every

        time.sleep(args.dt)

    df_samples = pd.DataFrame(rows)
    if df_samples.empty:
        raise RuntimeError("No TCP samples were collected during replay")

    df_samples = append_mae_column(df_samples)
    start_temp_measured = float(df_samples["temp"].iloc[0])
    df_samples["start_temp"] = start_temp_measured

    cost_info = compute_tail_cost(df_samples, entry_band=args.entry_band, overshoot_weight=args.overshoot_weight)

    run_summary = {
        "run_id": run_id,
        "advisor_dir": str(Path(args.advisor_dir).resolve()),
        "advisor_mode": report.get("mode"),
        "advisor_checkpoint": report.get("checkpoint"),
        "start_ts": float(df_samples["timestamp"].iloc[0]),
        "end_ts": float(df_samples["timestamp"].iloc[-1]),
        "duration_s": float(df_samples["timestamp"].iloc[-1] - df_samples["timestamp"].iloc[0]),
        "num_samples": int(len(df_samples)),
        "start_temp": start_temp_measured,
        "temp_ref": target_temp,
        "n_schedule_steps": len(schedule),
        "n_pid_changes_applied": next_step_idx,
        "aborted": bool(aborted_reason),
        "abort_reason": aborted_reason,
        "mse": float(df_samples["sq_error"].mean()),
        "mae": float(df_samples["mae"].mean()),
        "cost": float(cost_info["cost"]),
        "tail_mae": None if cost_info["tail_mae"] is None else float(cost_info["tail_mae"]),
        "overshoot": None if cost_info["overshoot"] is None else float(cost_info["overshoot"]),
        "final_temp": float(df_samples["temp"].iloc[-1]),
        "final_error": float(target_temp - df_samples["temp"].iloc[-1]),
        "predicted_final_temp": report.get("final_temp", report.get("end_temp")),
        "predicted_overshoot": report.get("overshoot", report.get("overshoot_max")),
        "predicted_tail_mae": report.get("tail_mae"),
    }

    samples_out = history_run_file(run_id, str(Path(args.history_root) / args.samples_csv), args.history_root)
    runs_out = history_run_file(run_id, str(Path(args.history_root) / args.runs_csv), args.history_root)
    events_out = history_run_file(run_id, str(Path(args.history_root) / args.events_csv), args.history_root)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])
    append_rows_csv(events_out, [{"run_id": run_id, **e} for e in events])

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_row_csv(args.all_runs_csv, run_summary)

    print(f"[run {run_id}] run_id={run_summary['run_id']}")
    print(f"[run {run_id}] aborted={run_summary['aborted']} {run_summary['abort_reason']}")
    print(f"[run {run_id}] samples={run_summary['num_samples']}")
    print(f"[run {run_id}] cost={run_summary['cost']:.6f} (predicted advisor cost was a different metric; "
          f"see report.json)")
    print(f"[run {run_id}] final_temp={run_summary['final_temp']:.3f} "
          f"(predicted {run_summary['predicted_final_temp']})")
    print(f"[run {run_id}] overshoot={run_summary['overshoot']} "
          f"(predicted {run_summary['predicted_overshoot']})")
    print(f"[run {run_id}] samples_csv={samples_out}")
    print(f"[run {run_id}] runs_csv={runs_out}")
    print(f"[run {run_id}] events_csv={events_out}")


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.cooldown < 0:
        raise ValueError("--cooldown must be >= 0")
    if args.anchor_duration_s < 0:
        raise ValueError("--anchor-duration-s must be >= 0")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    if args.read_retries < 0:
        raise ValueError("--read-retries must be >= 0")
    if args.read_retry_delay_s < 0:
        raise ValueError("--read-retry-delay-s must be >= 0")
    if args.max_consecutive_failures <= 0:
        raise ValueError("--max-consecutive-failures must be > 0")
    if args.safety_margin_c < 0:
        raise ValueError("--safety-margin-c must be >= 0")
    if not (0 <= args.pid_row <= 4):
        raise ValueError("--pid-row must be in range [0, 4]")

    report, schedule = load_advisor_session(args)
    print(f"[setup] advisor_dir={args.advisor_dir} mode={report.get('mode')} "
          f"schedule={len(schedule)} step(s): "
          f"{[(round(t, 1), pid) for t, pid in schedule[:10]]}"
          f"{' ...' if len(schedule) > 10 else ''}")

    run_idx = 1
    while True:
        total = None if args.forever else args.num_tests
        try:
            run_single_replay(run_idx=run_idx, total_tests=total, args=args, report=report, schedule=schedule)
        except Exception as exc:
            print(f"[replay {run_idx}] FAILED: {exc}")

        run_idx += 1
        if (not args.forever) and run_idx > args.num_tests:
            break

        if args.cooldown > 0:
            print(f"Waiting {args.cooldown:.1f}s before next replay...")
            time.sleep(args.cooldown)


if __name__ == "__main__":
    main()
