#!/usr/bin/env python3
"""Automated multi-run TCP tests with TCP-driven temp_ref jobs and PID updates."""

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
from utils.utils import _band_range, append_row_csv, plan_pid, read_pid_from_tcp

PIDTriplet = Tuple[int, int, int]
BANDS = ("very_far", "far", "mid", "near", "very_near")

# (very_far_p, very_far_i, very_far_d, far_p, far_i, far_d, mid_p, mid_i, mid_d, near_p, near_i, near_d, very_near_p, very_near_i, very_near_d)
PID_SCHEDULES = [

	
]

def build_arg_parser() -> argparse.ArgumentParser:
	output_dir = Path(__file__).with_name("output")
	ap = argparse.ArgumentParser()

	ap.add_argument("--num-tests", type=int, default=15)
	ap.add_argument("--forever", action="store_true", help="Run tests continuously")
	ap.add_argument("--seed", type=int, default=None)

	ap.add_argument("--test-duration", type=float, default=20.0 * 60.0, help="Seconds")
	ap.add_argument("--cooldown", type=float, default=3.0 * 60.0, help="Seconds between tests")
	ap.add_argument("--heatup-temp-ref", type=float, default=25.0, help="Pre-test heatup temp_ref")
	ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0, help="Seconds at heatup temp_ref")
	ap.add_argument("--post-heatup-cooldown", type=float, default=3.0 * 60.0, help="Seconds after heatup before test")
	ap.add_argument("--test-temp-ref", type=float, default=0.0, help="Logged test temp_ref target")
	ap.add_argument("--dt", type=float, default=1.0)
	ap.add_argument("--progress-every", type=float, default=60.0)

	ap.add_argument("--entry-band", type=float, default=2.0)
	ap.add_argument("--overshoot-weight", type=float, default=10.0)

	ap.add_argument("--cross-band-1", type=float, default=12.0)
	ap.add_argument("--cross-band-2", type=float, default=7.0)
	ap.add_argument("--cross-band-3", type=float, default=3.0)
	ap.add_argument("--cross-band-4", type=float, default=1.0)

	ap.add_argument("--temp-ref-min", type=float, default=-20.0)
	ap.add_argument("--temp-ref-max", type=float, default=30.0)

	ap.add_argument("--kp-min", type=int, default=1)
	ap.add_argument("--kp-max", type=int, default=50)
	ap.add_argument("--ki-min", type=int, default=1)
	ap.add_argument("--ki-max", type=int, default=1000)
	ap.add_argument("--kd-min", type=int, default=1)
	ap.add_argument("--kd-max", type=int, default=200)

	# Per-band search ranges used when PID_SCHEDULES is empty.
	ap.add_argument("--very-far-kp-min", type=int, default=4)
	ap.add_argument("--very-far-kp-max", type=int, default=8)
	ap.add_argument("--very-far-ki-min", type=int, default=500)
	ap.add_argument("--very-far-ki-max", type=int, default=1000)
	ap.add_argument("--very-far-kd-min", type=int, default=0)
	ap.add_argument("--very-far-kd-max", type=int, default=25)

	ap.add_argument("--far-kp-min", type=int, default=6)
	ap.add_argument("--far-kp-max", type=int, default=10)
	ap.add_argument("--far-ki-min", type=int, default=350)
	ap.add_argument("--far-ki-max", type=int, default=800)
	ap.add_argument("--far-kd-min", type=int, default=5)
	ap.add_argument("--far-kd-max", type=int, default=35)

	ap.add_argument("--mid-kp-min", type=int, default=8)
	ap.add_argument("--mid-kp-max", type=int, default=14)
	ap.add_argument("--mid-ki-min", type=int, default=200)
	ap.add_argument("--mid-ki-max", type=int, default=600)
	ap.add_argument("--mid-kd-min", type=int, default=10)
	ap.add_argument("--mid-kd-max", type=int, default=50)

	ap.add_argument("--near-kp-min", type=int, default=10)
	ap.add_argument("--near-kp-max", type=int, default=18)
	ap.add_argument("--near-ki-min", type=int, default=80)
	ap.add_argument("--near-ki-max", type=int, default=300)
	ap.add_argument("--near-kd-min", type=int, default=25)
	ap.add_argument("--near-kd-max", type=int, default=80)

	ap.add_argument("--very-near-kp-min", type=int, default=8)
	ap.add_argument("--very-near-kp-max", type=int, default=16)
	ap.add_argument("--very-near-ki-min", type=int, default=20)
	ap.add_argument("--very-near-ki-max", type=int, default=150)
	ap.add_argument("--very-near-kd-min", type=int, default=40)
	ap.add_argument("--very-near-kd-max", type=int, default=120)

	ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4]")

	ap.add_argument("--tcp-host", default=DEFAULT_HOST)
	ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
	ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)

	ap.add_argument("--history-root", default="history5")
	ap.add_argument("--samples-csv", default="run_samples.csv")
	ap.add_argument("--runs-csv", default="run_summary.csv")
	ap.add_argument("--all-runs-csv", default=str(output_dir / "bo_all_runs.csv"))

	return ap


def run_single_test(
	run_idx: int,
	total_tests: Optional[int],
	args: argparse.Namespace,
	pid_plan: Dict[str, PIDTriplet],
	pid_source: str,
) -> None:
	run_id = make_run_id(prefix="run")
	test_label = f"{run_idx}" if total_tests is None else f"{run_idx}/{total_tests}"

	temp_ref_target = float(args.test_temp_ref)

	print(f"[run {run_id}] starting test {test_label}")
	print(
		f"[run {run_id}] preconditioning: temp_ref={args.heatup_temp_ref:.3f} "
		f"for {args.heatup_duration:.1f}s (no logging)"
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
		print(f"[run {run_id}] post-heatup cooldown for {args.post_heatup_cooldown:.1f}s (no logging)")
		time.sleep(args.post_heatup_cooldown)

	print(f"[run {run_id}] selected temp_ref={temp_ref_target}")
	print(
		f"[run {run_id}] PID plan ({pid_source}): "
		f"very_far={pid_plan['very_far']} far={pid_plan['far']} "
		f"mid={pid_plan['mid']} near={pid_plan['near']} "
		f"very_near={pid_plan['very_near']}"
	)

	tcp_pid_before = read_pid_from_tcp(row=args.pid_row, args=args)
	print(
		f"[run {run_id}] PID before (TCP): "
		f"kp={tcp_pid_before['kp']:.3f}, ki={tcp_pid_before['ki']:.3f}, kd={tcp_pid_before['kd']:.3f}"
	)

	publish_temp_ref_job(
		temp_ref=temp_ref_target,
		duration_s=args.test_duration,
		host=args.tcp_host,
		port=args.tcp_port,
		timeout=args.tcp_timeout,
	)
	print(f"[run {run_id}] published temp_ref job for {args.test_duration:.1f}s")

	pid_events = []
	current_kp, current_ki, current_kd = apply_pid_update(
		label="very_far/start",
		run_id=run_id,
		row=args.pid_row,
		pid=pid_plan["very_far"],
		args=args,
		events=pid_events,
	)

	rows = []
	sq_error_sum = 0.0
	abs_error_sum = 0.0
	prev_abs_err: Optional[float] = None
	crossed_band_1 = False
	crossed_band_2 = False
	crossed_band_3 = False
	crossed_band_4 = False

	t0 = time.time()
	next_progress_ts = t0 + args.progress_every if args.progress_every > 0 else float("inf")

	print(
		f"[run {run_id}] sampling via TCP every {args.dt:.3f}s for {args.test_duration:.1f}s"
	)
	while True:
		now = time.time()
		elapsed = now - t0
		if elapsed >= args.test_duration:
			break

		snap = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)
		abs_err_to_target = abs(float(snap["temp"]) - temp_ref_target)

		if prev_abs_err is not None:
			if (not crossed_band_1) and (prev_abs_err > args.cross_band_1) and (abs_err_to_target <= args.cross_band_1):
				current_kp, current_ki, current_kd = apply_pid_update(
					label=f"cross_{args.cross_band_1:g}deg",
					run_id=run_id,
					row=args.pid_row,
					pid=pid_plan["far"],
					args=args,
					events=pid_events,
				)
				crossed_band_1 = True

			if (not crossed_band_2) and (prev_abs_err > args.cross_band_2) and (abs_err_to_target <= args.cross_band_2):
				current_kp, current_ki, current_kd = apply_pid_update(
					label=f"cross_{args.cross_band_2:g}deg",
					run_id=run_id,
					row=args.pid_row,
					pid=pid_plan["mid"],
					args=args,
					events=pid_events,
				)
				crossed_band_2 = True

			if (not crossed_band_3) and (prev_abs_err > args.cross_band_3) and (abs_err_to_target <= args.cross_band_3):
				current_kp, current_ki, current_kd = apply_pid_update(
					label=f"cross_{args.cross_band_3:g}deg",
					run_id=run_id,
					row=args.pid_row,
					pid=pid_plan["near"],
					args=args,
					events=pid_events,
				)
				crossed_band_3 = True

			if (not crossed_band_4) and (prev_abs_err > args.cross_band_4) and (abs_err_to_target <= args.cross_band_4):
				current_kp, current_ki, current_kd = apply_pid_update(
					label=f"cross_{args.cross_band_4:g}deg",
					run_id=run_id,
					row=args.pid_row,
					pid=pid_plan["very_near"],
					args=args,
					events=pid_events,
				)
				crossed_band_4 = True

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
		"pid_source": pid_source,
		"very_far_kp": int(pid_plan["very_far"][0]),
		"very_far_ki": int(pid_plan["very_far"][1]),
		"very_far_kd": int(pid_plan["very_far"][2]),
		"far_kp": int(pid_plan["far"][0]),
		"far_ki": int(pid_plan["far"][1]),
		"far_kd": int(pid_plan["far"][2]),
		"mid_kp": int(pid_plan["mid"][0]),
		"mid_ki": int(pid_plan["mid"][1]),
		"mid_kd": int(pid_plan["mid"][2]),
		"near_kp": int(pid_plan["near"][0]),
		"near_ki": int(pid_plan["near"][1]),
		"near_kd": int(pid_plan["near"][2]),
		"very_near_kp": int(pid_plan["very_near"][0]),
		"very_near_ki": int(pid_plan["very_near"][1]),
		"very_near_kd": int(pid_plan["very_near"][2]),
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
	append_row_csv(args.all_runs_csv, run_summary)

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
	if args.heatup_duration < 0:
		raise ValueError("--heatup-duration must be >= 0")
	if args.post_heatup_cooldown < 0:
		raise ValueError("--post-heatup-cooldown must be >= 0")
	if args.progress_every < 0:
		raise ValueError("--progress-every must be >= 0")
	if not (args.cross_band_4 <= args.cross_band_3 <= args.cross_band_2 <= args.cross_band_1):
		raise ValueError("--cross-band-4 <= --cross-band-3 <= --cross-band-2 <= --cross-band-1 is required")
	if args.temp_ref_min >= args.temp_ref_max:
		raise ValueError("--temp-ref-min must be < --temp-ref-max")
	if not (0 <= args.pid_row <= 4):
		raise ValueError("--pid-row must be in range [0, 4]")

	for band in ("very_far", "far", "mid", "near", "very_near"):
		for coef in ("kp", "ki", "kd"):
			band_min = _band_range(args, band, coef, "min")
			band_max = _band_range(args, band, coef, "max")
			if band_min > band_max:
				raise ValueError(f"--{band}-{coef}-min must be <= --{band}-{coef}-max")

	rng = random.Random(args.seed)

	run_idx = 1
	while True:
		total = None if args.forever else args.num_tests
		pid_plan, pid_source = plan_pid(
			run_idx=run_idx,
			args=args,
			rng=rng,
			pid_schedules=PID_SCHEDULES,
			bands=BANDS,
		)
		try:
			run_single_test(
				run_idx=run_idx,
				total_tests=total,
				args=args,
				pid_plan=pid_plan,
				pid_source=pid_source,
			)
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
