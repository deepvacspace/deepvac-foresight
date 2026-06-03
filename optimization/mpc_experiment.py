#!/usr/bin/env python3
"""Replay an MPC PID decision CSV against the real chamber over TCP."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from utils.bo_common import (  
	append_mae_column,
	append_rows_csv,
	compute_tail_cost,
	history_run_file,
	make_run_id,
)
from tcp.tcp_common import (  
	DEFAULT_HOST,
	DEFAULT_PORT,
	DEFAULT_TIMEOUT,
	apply_pid_update,
	publish_temp_ref_job,
	request_temperature_states,
)
from utils.utils import append_row_csv, read_pid_from_tcp  

PIDTriplet = Tuple[int, int, int]


def build_arg_parser() -> argparse.ArgumentParser:
	default_decisions = ROOT / "gru" / "mpc_pid_runs" / "mpc_20260603_162052_acd5dfb1" / "mpc_decisions.csv"
	output_dir = Path(__file__).with_name("output")

	ap = argparse.ArgumentParser(
		description="Run a TCP experiment that replays PID updates from an MPC decisions CSV."
	)
	ap.add_argument("--decisions-csv", default=str(default_decisions))
	ap.add_argument("--changed-only", action=argparse.BooleanOptionalAction, default=False,
		help="When true, skip decision rows where changed is false.")

	ap.add_argument("--test-duration", type=float, default=None,
		help="Seconds. Defaults to last decision elapsed_s plus --extra-duration.")
	ap.add_argument("--extra-duration", type=float, default=60.0,
		help="Extra seconds after the last decision when --test-duration is omitted.")
	ap.add_argument("--test-temp-ref", type=float, default=0.0,
		help="TCP setpoint job target for the replay experiment.")
	ap.add_argument("--dt", type=float, default=1.0)
	ap.add_argument("--progress-every", type=float, default=60.0)

	ap.add_argument("--heatup-temp-ref", type=float, default=25.0,
		help="Pre-test heatup temp_ref.")
	ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0,
		help="Seconds at heatup temp_ref before the test.")
	ap.add_argument("--post-heatup-cooldown", type=float, default=3.0 * 60.0,
		help="Seconds after heatup before the test.")
	ap.add_argument("--condition-initial", action="store_true",
		help="Run heatup and post-heatup cooldown before the replay experiment.")
	ap.add_argument("--skip-preconditioning", action="store_true",
		help="Skip heatup and cooldown before the replay experiment.")

	ap.add_argument("--entry-band", type=float, default=2.0)
	ap.add_argument("--overshoot-weight", type=float, default=10.0)

	ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4]")
	ap.add_argument("--tcp-host", default=DEFAULT_HOST)
	ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
	ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
	ap.add_argument("--read-retries", type=int, default=2)
	ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
	ap.add_argument("--max-consecutive-failures", type=int, default=10)

	ap.add_argument("--history-root", default="run_history")
	ap.add_argument("--samples-csv", default="run_samples.csv")
	ap.add_argument("--runs-csv", default="run_summary.csv")
	ap.add_argument("--events-csv", default="pid_events.csv")
	ap.add_argument("--all-runs-csv", default=str(output_dir / "mpc_decision_replay_runs.csv"))

	return ap


def _parse_bool(value: object) -> bool:
	if isinstance(value, bool):
		return value
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return False
	return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_pid_schedule(decisions_csv: str, changed_only: bool) -> pd.DataFrame:
	path = Path(decisions_csv)
	if not path.exists():
		raise FileNotFoundError(f"MPC decisions CSV not found: {path}")

	df = pd.read_csv(path)
	required = {"elapsed_s", "kp", "ki", "kd"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required decision columns: {sorted(missing)}")

	df = df.copy()
	if changed_only and "changed" in df.columns:
		df = df[df["changed"].map(_parse_bool)]

	df = df.dropna(subset=["elapsed_s", "kp", "ki", "kd"])
	df["elapsed_s"] = df["elapsed_s"].astype(float)
	df = df[df["elapsed_s"] >= 0.0]
	df = df.sort_values(["elapsed_s"]).reset_index(drop=True)

	if df.empty:
		raise ValueError("No usable PID decisions found after filtering")

	for col in ("kp", "ki", "kd"):
		df[col] = df[col].astype(float).round().astype(int)

	if "decision_idx" not in df.columns:
		df["decision_idx"] = range(1, len(df) + 1)

	return df


def _read_temperature_with_retries(args: argparse.Namespace) -> Dict[str, float]:
	last_read_exc: Optional[Exception] = None
	for _ in range(max(1, args.read_retries + 1)):
		try:
			return request_temperature_states(
				host=args.tcp_host,
				port=args.tcp_port,
				timeout=args.tcp_timeout,
			)
		except Exception as exc:
			last_read_exc = exc
			if args.read_retry_delay_s > 0:
				time.sleep(args.read_retry_delay_s)

	raise RuntimeError("TCP temperature read failed") from last_read_exc


def run_single_test(args: argparse.Namespace, schedule: pd.DataFrame) -> None:
	run_id = make_run_id(prefix="mpc_replay")
	last_decision_elapsed = float(schedule["elapsed_s"].max())
	test_duration = float(args.test_duration) if args.test_duration is not None else last_decision_elapsed + float(args.extra_duration)
	temp_ref_target = float(args.test_temp_ref)

	print(f"[run {run_id}] decisions_csv={args.decisions_csv}")
	print(f"[run {run_id}] loaded {len(schedule)} PID decisions")
	print(f"[run {run_id}] test_duration={test_duration:.1f}s temp_ref={temp_ref_target:.3f}")

	condition_pretest = bool(args.condition_initial) and not args.skip_preconditioning
	if condition_pretest:
		print(
			f"[run {run_id}] preconditioning: temp_ref={args.heatup_temp_ref:.3f} "
			f"for {args.heatup_duration:.1f}s"
		)
		publish_temp_ref_job(
			temp_ref=float(args.heatup_temp_ref),
			duration_s=args.heatup_duration,
			host=args.tcp_host,
			port=args.tcp_port,
			timeout=args.tcp_timeout,
		)
		time.sleep(args.heatup_duration)

		if args.post_heatup_cooldown > 0:
			print(f"[run {run_id}] post-heatup cooldown for {args.post_heatup_cooldown:.1f}s")
			time.sleep(args.post_heatup_cooldown)
	else:
		print(f"[run {run_id}] skipping initial preconditioning and post-heatup cooldown")

	tcp_pid_before = read_pid_from_tcp(row=args.pid_row, args=args)
	print(
		f"[run {run_id}] PID before (TCP): "
		f"kp={tcp_pid_before['kp']:.3f}, ki={tcp_pid_before['ki']:.3f}, kd={tcp_pid_before['kd']:.3f}"
	)

	# Prime the state-name cache before the active test starts.
	_ = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)

	publish_temp_ref_job(
		temp_ref=temp_ref_target,
		duration_s=test_duration,
		host=args.tcp_host,
		port=args.tcp_port,
		timeout=args.tcp_timeout,
	)
	print(f"[run {run_id}] published temp_ref job for {test_duration:.1f}s")

	rows: List[Dict[str, object]] = []
	pid_events: List[Dict[str, object]] = []
	current_kp: Optional[int] = None
	current_ki: Optional[int] = None
	current_kd: Optional[int] = None
	current_decision_idx: Optional[int] = None
	current_decision_elapsed_s: Optional[float] = None
	next_decision_idx = 0
	consecutive_read_failures = 0
	abs_error_sum = 0.0

	t0 = time.time()
	next_progress_ts = t0 + args.progress_every if args.progress_every > 0 else float("inf")

	print(f"[run {run_id}] sampling via TCP every {args.dt:.3f}s")
	while True:
		now = time.time()
		elapsed = now - t0
		if elapsed >= test_duration:
			break

		while next_decision_idx < len(schedule) and elapsed >= float(schedule.iloc[next_decision_idx]["elapsed_s"]):
			decision = schedule.iloc[next_decision_idx]
			pid = (
				int(decision["kp"]),
				int(decision["ki"]),
				int(decision["kd"]),
			)
			label = f"mpc_decision_{int(decision['decision_idx'])}@{float(decision['elapsed_s']):.1f}s"
			current_kp, current_ki, current_kd = apply_pid_update(
				label=label,
				run_id=run_id,
				row=args.pid_row,
				pid=pid,
				args=args,
				events=pid_events,
			)
			pid_events[-1]["run_id"] = run_id
			pid_events[-1]["decision_idx"] = int(decision["decision_idx"])
			pid_events[-1]["decision_elapsed_s"] = float(decision["elapsed_s"])
			pid_events[-1]["applied_elapsed_s"] = float(elapsed)
			pid_events[-1]["source_changed"] = _parse_bool(decision.get("changed")) if "changed" in schedule.columns else None
			current_decision_idx = int(decision["decision_idx"])
			current_decision_elapsed_s = float(decision["elapsed_s"])
			next_decision_idx += 1

		try:
			snap = _read_temperature_with_retries(args)
			consecutive_read_failures = 0
		except Exception as exc:
			consecutive_read_failures += 1
			print(
				f"[run {run_id}] read failed "
				f"({consecutive_read_failures}/{args.max_consecutive_failures}): {exc}"
			)
			if consecutive_read_failures >= args.max_consecutive_failures:
				raise RuntimeError(
					f"Too many consecutive state read failures ({consecutive_read_failures})"
				) from exc
			time.sleep(args.dt)
			continue

		sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
		abs_error = abs(float(snap["temp_ref"]) - float(snap["temp"]))
		abs_error_sum += abs_error
		rows.append(
			{
				"run_id": run_id,
				"timestamp": now,
				"elapsed_s": elapsed,
				"decision_idx": current_decision_idx,
				"decision_elapsed_s": current_decision_elapsed_s,
				"kp": current_kp,
				"ki": current_ki,
				"kd": current_kd,
				**snap,
				"sq_error": sq_error,
			}
		)

		if now >= next_progress_ts:
			n = len(rows)
			running_mae = abs_error_sum / max(1, n)
			remaining = max(test_duration - elapsed, 0.0)
			print(
				f"[run {run_id}] samples={n} elapsed={elapsed:.1f}s "
				f"remaining={remaining:.1f}s temp={snap['temp']:.3f} "
				f"temp_ref={snap['temp_ref']:.3f} mae={running_mae:.6f} "
				f"kp={current_kp} ki={current_ki} kd={current_kd}"
			)
			next_progress_ts += args.progress_every

		time.sleep(args.dt)

	df_samples = pd.DataFrame(rows)
	if df_samples.empty:
		raise RuntimeError("No TCP samples were collected during MPC replay test")

	df_samples = append_mae_column(df_samples)
	start_temp = float(df_samples["temp"].iloc[0])
	target_temp_measured = float(df_samples["temp_ref"].iloc[0])
	df_samples["start_temp"] = start_temp

	cost_info = compute_tail_cost(
		df_samples,
		entry_band=args.entry_band,
		overshoot_weight=args.overshoot_weight,
	)

	run_summary = {
		"run_id": run_id,
		"start_ts": float(df_samples["timestamp"].iloc[0]),
		"end_ts": float(df_samples["timestamp"].iloc[-1]),
		"duration_s": float(df_samples["timestamp"].iloc[-1] - df_samples["timestamp"].iloc[0]),
		"num_samples": int(len(df_samples)),
		"start_temp": start_temp,
		"temp_ref": target_temp_measured,
		"pid_source": str(args.decisions_csv),
		"pid_schedule_rows": int(len(schedule)),
		"pid_events_applied": int(len(pid_events)),
		"final_kp": None if current_kp is None else int(current_kp),
		"final_ki": None if current_ki is None else int(current_ki),
		"final_kd": None if current_kd is None else int(current_kd),
		"mse": float(df_samples["sq_error"].mean()),
		"mae": float(df_samples["mae"].mean()),
	}
	run_summary["cost"] = float(cost_info["cost"])
	run_summary["tail_mae"] = None if cost_info["tail_mae"] is None else float(cost_info["tail_mae"])
	run_summary["overshoot"] = None if cost_info["overshoot"] is None else float(cost_info["overshoot"])

	samples_out = history_run_file(
		run_id,
		str(Path(args.history_root) / args.samples_csv),
		args.history_root,
	)
	runs_out = history_run_file(
		run_id,
		str(Path(args.history_root) / args.runs_csv),
		args.history_root,
	)
	events_out = history_run_file(
		run_id,
		str(Path(args.history_root) / args.events_csv),
		args.history_root,
	)

	append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
	append_rows_csv(runs_out, [run_summary])
	append_rows_csv(events_out, pid_events)

	Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
	append_row_csv(args.all_runs_csv, run_summary)

	print(f"[run {run_id}] run_id={run_summary['run_id']}")
	print(f"[run {run_id}] samples={run_summary['num_samples']}")
	print(f"[run {run_id}] pid_events_applied={run_summary['pid_events_applied']}")
	print(f"[run {run_id}] cost={run_summary['cost']:.6f}")
	print(f"[run {run_id}] tail_mae={run_summary['tail_mae']}")
	print(f"[run {run_id}] overshoot={run_summary['overshoot']}")
	print(f"[run {run_id}] samples_csv={samples_out}")
	print(f"[run {run_id}] runs_csv={runs_out}")
	print(f"[run {run_id}] events_csv={events_out}")
	print(f"[run {run_id}] all_runs_csv={args.all_runs_csv}")


def main() -> None:
	args = build_arg_parser().parse_args()

	if args.dt <= 0:
		raise ValueError("--dt must be > 0")
	if args.test_duration is not None and args.test_duration <= 0:
		raise ValueError("--test-duration must be > 0")
	if args.extra_duration < 0:
		raise ValueError("--extra-duration must be >= 0")
	if args.heatup_duration < 0:
		raise ValueError("--heatup-duration must be >= 0")
	if args.post_heatup_cooldown < 0:
		raise ValueError("--post-heatup-cooldown must be >= 0")
	if args.progress_every < 0:
		raise ValueError("--progress-every must be >= 0")
	if args.read_retries < 0:
		raise ValueError("--read-retries must be >= 0")
	if args.read_retry_delay_s < 0:
		raise ValueError("--read-retry-delay-s must be >= 0")
	if args.max_consecutive_failures <= 0:
		raise ValueError("--max-consecutive-failures must be > 0")
	if not (0 <= args.pid_row <= 4):
		raise ValueError("--pid-row must be in range [0, 4]")

	schedule = load_pid_schedule(args.decisions_csv, changed_only=bool(args.changed_only))
	run_single_test(args=args, schedule=schedule)


if __name__ == "__main__":
	main()
