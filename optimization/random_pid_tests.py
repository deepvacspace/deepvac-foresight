#!/usr/bin/env python3
"""Automated multi-run TCP tests."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from bo.bo_common import (
	append_mae_column,
	append_rows_csv,
	compute_tail_cost,
	history_run_file,
	make_run_id,
)
from tcp.tcp_common import (  # noqa: E402
	DEFAULT_HOST,
	DEFAULT_PORT,
	DEFAULT_TIMEOUT,
	apply_pid_update,
	publish_temp_ref_job,
	request_temperature_states,
)
from utils.utils import random_pid, read_pid_from_tcp

def build_arg_parser() -> argparse.ArgumentParser:
	ap = argparse.ArgumentParser()

	ap.add_argument("--num-tests", type=int, default=6)
	ap.add_argument("--forever", action="store_true", help="Run tests continuously")
	ap.add_argument("--seed", type=int, default=None)

	ap.add_argument("--test-duration", type=float, default=20.0 * 60.0, help="Seconds")
	ap.add_argument("--cooldown", type=float, default=5.0 * 60.0, help="Seconds between tests")
	ap.add_argument("--dt", type=float, default=1.0)
	ap.add_argument("--progress-every", type=float, default=60.0)

	ap.add_argument("--entry-band", type=float, default=2.0)
	ap.add_argument("--overshoot-weight", type=float, default=10.0)

	ap.add_argument("--cross-band-1", type=float, default=7.0)
	ap.add_argument("--cross-band-2", type=float, default=2.0)

	ap.add_argument("--temp-ref-min", type=float, default=-20.0)
	ap.add_argument("--temp-ref-max", type=float, default=30.0)

	ap.add_argument("--kp-min", type=int, default=0)
	ap.add_argument("--kp-max", type=int, default=50)
	ap.add_argument("--ki-min", type=int, default=0)
	ap.add_argument("--ki-max", type=int, default=1000)
	ap.add_argument("--kd-min", type=int, default=0)
	ap.add_argument("--kd-max", type=int, default=200)

	ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4]")

	ap.add_argument("--tcp-host", default=DEFAULT_HOST)
	ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
	ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
	ap.add_argument("--read-retries", type=int, default=2)
	ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
	ap.add_argument("--max-consecutive-failures", type=int, default=10)

	ap.add_argument("--history-root", default="history_multi")
	ap.add_argument("--samples-csv", default="run_samples.csv")
	ap.add_argument("--runs-csv", default="run_summary.csv")
	ap.add_argument("--all-runs-csv", default="history_multi/bo_all_runs.csv")

	return ap


def apply_random_pid_update(
	label: str,
	run_id: str,
	row: int,
	args: argparse.Namespace,
	rng: random.Random,
	events: list,
) -> Tuple[int, int, int]:
	return apply_pid_update(
		label=label,
		run_id=run_id,
		row=row,
		pid=random_pid(args, rng),
		args=args,
		events=events,
	)


def run_single_test(
	run_idx: int,
	total_tests: Optional[int],
	args: argparse.Namespace,
	rng: random.Random,
) -> None:
	run_id = make_run_id(prefix="run")
	test_label = f"{run_idx}" if total_tests is None else f"{run_idx}/{total_tests}"

	temp_ref_target = rng.uniform(args.temp_ref_min, args.temp_ref_max)
	temp_ref_target = round(temp_ref_target, 3)

	print(f"[run {run_id}] starting test {test_label}")
	print(f"[run {run_id}] selected temp_ref={temp_ref_target}")

	tcp_pid_before = read_pid_from_tcp(row=args.pid_row, args=args)
	print(
		f"[run {run_id}] PID before (TCP): "
		f"kp={tcp_pid_before['kp']:.3f}, ki={tcp_pid_before['ki']:.3f}, kd={tcp_pid_before['kd']:.3f}"
	)

	# Prime the TCP state-name cache before the active test starts. Some controller
	# replies appear to omit names intermittently once a job is running.
	_ = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)

	publish_temp_ref_job(
		temp_ref=temp_ref_target,
		duration_s=args.test_duration,
		host=args.tcp_host,
		port=args.tcp_port,
		timeout=args.tcp_timeout,
	)
	print(f"[run {run_id}] published temp_ref job for {args.test_duration:.1f}s")

	pid_events = []
	current_kp, current_ki, current_kd = apply_random_pid_update(
		label="start",
		run_id=run_id,
		row=args.pid_row,
		args=args,
		rng=rng,
		events=pid_events,
	)

	rows = []
	sq_error_sum = 0.0
	abs_error_sum = 0.0
	prev_abs_err: Optional[float] = None
	crossed_band_1 = False
	crossed_band_2 = False
	consecutive_read_failures = 0

	t0 = time.time()
	next_progress_ts = t0 + args.progress_every if args.progress_every > 0 else float("inf")

	print(f"[run {run_id}] sampling via TCP every {args.dt:.3f}s for {args.test_duration:.1f}s")
	while True:
		now = time.time()
		elapsed = now - t0
		if elapsed >= args.test_duration:
			break

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
				print(
					f"[run {run_id}] read failed "
					f"({consecutive_read_failures}/{args.max_consecutive_failures}): {last_read_exc}"
				)
			if consecutive_read_failures >= args.max_consecutive_failures:
				raise RuntimeError(
					f"Too many consecutive state read failures ({consecutive_read_failures})"
				) from last_read_exc
			time.sleep(args.dt)
			continue

		consecutive_read_failures = 0
		abs_err_to_target = abs(float(snap["temp"]) - temp_ref_target)

		if prev_abs_err is not None:
			if (not crossed_band_1) and (prev_abs_err > args.cross_band_1) and (abs_err_to_target <= args.cross_band_1):
				current_kp, current_ki, current_kd = apply_random_pid_update(
					label=f"cross_{args.cross_band_1:g}deg",
					run_id=run_id,
					row=args.pid_row,
					args=args,
					rng=rng,
					events=pid_events,
				)
				crossed_band_1 = True

			if (not crossed_band_2) and (prev_abs_err > args.cross_band_2) and (abs_err_to_target <= args.cross_band_2):
				current_kp, current_ki, current_kd = apply_random_pid_update(
					label=f"cross_{args.cross_band_2:g}deg",
					run_id=run_id,
					row=args.pid_row,
					args=args,
					rng=rng,
					events=pid_events,
				)
				crossed_band_2 = True

		prev_abs_err = abs_err_to_target

		sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
		sq_error_sum += sq_error
		rows.append(
			{
				"run_id": run_id,
				"timestamp": now,
				"kp": float(current_kp),
				"ki": float(current_ki),
				"kd": float(current_kd),
				**snap,
				"sq_error": sq_error,
			}
		)
		abs_error = abs(float(snap["temp_ref"]) - float(snap["temp"]))
		abs_error_sum += abs_error

		if now >= next_progress_ts:
			n = len(rows)
			running_mae = abs_error_sum / max(1, n)
			remaining = max(args.test_duration - elapsed, 0.0)
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
		raise RuntimeError("No TCP samples were collected during automated test")

	df_samples = append_mae_column(df_samples)

	start_temp = float(df_samples["temp"].iloc[0])
	target_temp_measured = float(df_samples["temp_ref"].iloc[0])
	delta_temp_measured = start_temp - target_temp_measured

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
		"kp": float(df_samples["kp"].iloc[-1]),
		"ki": float(df_samples["ki"].iloc[-1]),
		"kd": float(df_samples["kd"].iloc[-1]),
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

	append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
	append_rows_csv(runs_out, [run_summary])

	Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
	append_rows_csv(args.all_runs_csv, [run_summary])

	print(f"[run {run_id}] run_id={run_summary['run_id']}")
	print(f"[run {run_id}] samples={run_summary['num_samples']}")
	print(f"[run {run_id}] cost={run_summary['cost']:.6f}")
	print(f"[run {run_id}] tail_mae={run_summary['tail_mae']}")
	print(f"[run {run_id}] overshoot={run_summary['overshoot']}")
	print(f"[run {run_id}] start_temp={run_summary['start_temp']}")
	print(f"[run {run_id}] temp_ref={run_summary['temp_ref']}")
	print(f"[run {run_id}] samples_csv={samples_out}")
	print(f"[run {run_id}] runs_csv={runs_out}")
	print(f"[run {run_id}] all_runs_csv={args.all_runs_csv}")


def main() -> None:
	args = build_arg_parser().parse_args()

	if args.dt <= 0:
		raise ValueError("--dt must be > 0")
	if args.test_duration <= 0:
		raise ValueError("--test-duration must be > 0")
	if args.cooldown < 0:
		raise ValueError("--cooldown must be >= 0")
	if args.progress_every < 0:
		raise ValueError("--progress-every must be >= 0")
	if args.read_retries < 0:
		raise ValueError("--read-retries must be >= 0")
	if args.read_retry_delay_s < 0:
		raise ValueError("--read-retry-delay-s must be >= 0")
	if args.max_consecutive_failures <= 0:
		raise ValueError("--max-consecutive-failures must be > 0")
	if args.cross_band_2 > args.cross_band_1:
		raise ValueError("--cross-band-2 should be <= --cross-band-1")
	if args.temp_ref_min >= args.temp_ref_max:
		raise ValueError("--temp-ref-min must be < --temp-ref-max")
	if not (0 <= args.pid_row <= 4):
		raise ValueError("--pid-row must be in range [0, 4]")

	rng = random.Random(args.seed)

	run_idx = 1
	while True:
		total = None if args.forever else args.num_tests
		try:
			run_single_test(run_idx=run_idx, total_tests=total, args=args, rng=rng)
		except Exception as exc:
			print(f"[test {run_idx}] FAILED: {exc}")

		run_idx += 1
		if (not args.forever) and run_idx > args.num_tests:
			break

		if args.cooldown > 0:
			print(f"Waiting {args.cooldown:.1f}s before next test...")
			time.sleep(args.cooldown)


if __name__ == "__main__":
	main()
