#!/usr/bin/env python3
"""Closed-loop band BO runner.

Run a TCP test loop using three bands from band_gp_bo.py

1. Fit three band GP models from run history.
2. Suggest a far / mid / near controller.
3. Execute one full run with band-crossing PID updates.
4. Recompute band metrics, retrain all three models, and repeat.
"""

from __future__ import annotations
from tcp.tcp_common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
)
from utils.bo_common import append_rows_csv, history_run_file, make_run_id
from optimization import band_bo_gp as band_bo

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


PIDTriplet = Tuple[int, int, int]


def build_arg_parser() -> argparse.ArgumentParser:
    output_dir = Path(__file__).with_name("output")
    ap = band_bo.build_arg_parser()
    ap.description = "Closed-loop runner using band_gp_bo three-model scheduling."
    ap.set_defaults(
        telemetry_names=["run_samples.csv"],
        per_run_metrics_csv="band_metrics.csv",
        per_run_metrics_json="band_metrics.json",
        models_out=str(output_dir / "band_gp_models.pkl"),
        next_out=str(output_dir / "band_bo_next_params.json"),
        suggestions_dir=str(output_dir / "band_bo_suggestions"),
    )

    ap.add_argument("--num-runs", type=int, default=10,
                    help="Number of successful runs to complete.")
    ap.add_argument("--test-duration", type=float, default=1200.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0)
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="Seconds to wait between successful runs.")
    ap.add_argument("--heatup-temp-ref", type=float,
                    default=25.0, help="Pre-test heatup temp_ref.")
    ap.add_argument("--heatup-duration", type=float,
                    default=300.0, help="Seconds at heatup temp_ref.")
    ap.add_argument(
        "--post-heatup-cooldown",
        type=float,
        default=180.0,
        help="Seconds after heatup before the test begins.",
    )
    ap.add_argument(
        "--condition-initial",
        action="store_true",
        help="Run heatup and post-heatup cooldown before the first experiment.",
    )
    ap.add_argument("--test-temp-ref", type=float, default=0.0,
                    help="Target temp_ref for the logged test.")
    ap.add_argument("--cross-band-1", type=float, default=7.0)
    ap.add_argument("--cross-band-2", type=float, default=2.0)
    ap.add_argument("--pid-row", type=int, default=1,
                    help="Controller PID row index [0..4].")
    ap.add_argument(
        "--endpoint", default=f"opc.tcp://{DEFAULT_HOST}:{DEFAULT_PORT}")
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-temp-ref", default="ns=2;s=Testa chamber.temp_ref")
    ap.add_argument("--node-temp-raw", default="ns=2;s=Testa chamber.temp_raw")
    ap.add_argument("--node-temp-u", default="ns=2;s=Testa chamber.temp_u")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")
    ap.add_argument("--run-retry-attempts", type=int, default=3)
    ap.add_argument("--run-retry-delay", type=float, default=10.0)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--all-runs-csv", default=str(output_dir / "band_all_runs.csv"))

    return ap


def coerce_floor_int(value: float, bounds: Tuple[float, float]) -> int:
    lo, hi = bounds
    lo_int = int(math.ceil(lo))
    hi_int = int(math.floor(hi))
    if lo_int > hi_int:
        raise ValueError(f"Invalid integer bounds derived from {bounds!r}")
    return max(lo_int, min(hi_int, int(math.floor(float(value)))))


def band_bounds(args: argparse.Namespace, band: str) -> Dict[str, Tuple[float, float]]:
    return band_bo.bounds_for_band(args, band)


def floor_controller(raw_controller: Dict[str, Dict[str, float]], args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    planned: Dict[str, Dict[str, int]] = {}
    for band in band_bo.BANDS:
        bounds = band_bounds(args, band)
        triplet = raw_controller[band]
        planned[band] = {
            "kp": coerce_floor_int(triplet["kp"], bounds["kp"]),
            "ki": coerce_floor_int(triplet["ki"], bounds["ki"]),
            "kd": coerce_floor_int(triplet["kd"], bounds["kd"]),
        }
    return planned


def empty_band_training_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_id",
            "band",
            "kp",
            "ki",
            "kd",
            "cost",
            "n_samples",
        ]
    )


def refresh_band_models(args: argparse.Namespace) -> Dict[str, object]:
    try:
        table = band_bo.compute_all_run_metrics(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[WARN] no usable band training data yet: {exc}")
        table = empty_band_training_table()

    if not table.empty and "run_id" in table.columns:
        run_ids = table["run_id"].fillna("").astype(str)
        usable_rows = pd.to_numeric(table.get("cost"), errors="coerce").notna()
        val_rows = run_ids.str.startswith("val_")
        print(
            "[INFO] training table includes "
            f"{int(val_rows.sum())} val rows "
            f"({int((val_rows & usable_rows).sum())} usable)"
        )

    payload = band_bo.train_and_suggest(table, args)
    raw_controller = payload["suggested_controller"]
    planned_controller = floor_controller(raw_controller, args)

    payload = dict(payload)
    payload["raw_suggested_controller"] = raw_controller
    payload["suggested_controller"] = planned_controller
    payload["meta"] = dict(payload.get("meta", {}))
    payload["meta"]["controller_value_mode"] = "floor_int_clamped"
    payload["meta"]["raw_suggested_controller"] = raw_controller
    payload["meta"]["planned_controller"] = planned_controller
    payload["meta"]["outputs"] = dict(payload["meta"].get("outputs", {}))
    payload["meta"]["outputs"]["next_params"] = str(
        Path(args.next_out).resolve())
    if "generated_at" in payload["meta"]:
        snapshot = Path(args.suggestions_dir) / \
            f"band_bo_next_params_{payload['meta']['generated_at']}.json"
        payload["meta"]["outputs"]["snapshot"] = str(snapshot.resolve())
        Path(args.suggestions_dir).mkdir(parents=True, exist_ok=True)
        band_bo.save_json(Path(args.next_out), payload)
        band_bo.save_json(snapshot, payload)
    else:
        band_bo.save_json(Path(args.next_out), payload)

    return payload


def run_band_test(
        run_id: str,
        controller: Dict[str, Dict[str, int]],
        args: argparse.Namespace,
        pid_source: str,
        condition_pretest: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    print(f"[run {run_id}] starting test")
    if condition_pretest:
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
            print(
                f"[run {run_id}] post-heatup cooldown for {args.post_heatup_cooldown:.1f}s (no logging)")
            time.sleep(args.post_heatup_cooldown)
    else:
        print(f"[run {run_id}] skipping initial preconditioning and post-heatup cooldown")

    temp_ref_target = float(args.test_temp_ref)
    print(f"[run {run_id}] selected temp_ref={temp_ref_target}")
    print(
        f"[run {run_id}] controller source={pid_source}: "
        f"far={(controller['far']['kp'], controller['far']['ki'], controller['far']['kd'])} "
        f"mid={(controller['mid']['kp'], controller['mid']['ki'], controller['mid']['kd'])} "
        f"near={(controller['near']['kp'], controller['near']['ki'], controller['near']['kd'])}"
    )

    publish_temp_ref_job(
        temp_ref=temp_ref_target,
        duration_s=args.test_duration,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    print(f"[run {run_id}] published temp_ref job for {args.test_duration:.1f}s")

    pid_events: List[Dict[str, object]] = []
    current_pid = apply_pid_update(
        label="far/start",
        run_id=run_id,
        row=args.pid_row,
        pid=(controller["far"]["kp"], controller["far"]
             ["ki"], controller["far"]["kd"]),
        args=args,
        events=pid_events,
    )

    rows: List[Dict[str, object]] = []
    prev_abs_err: Optional[float] = None
    crossed_band_1 = False
    crossed_band_2 = False

    t0 = time.time()
    next_progress_ts = t0 + \
        args.progress_every if args.progress_every > 0 else float("inf")

    print(
        f"[run {run_id}] sampling via TCP every {args.dt:.3f}s for {args.test_duration:.1f}s")
    consecutive_read_failures = 0
    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= args.test_duration:
            break

        snap: Optional[Dict[str, float]] = None
        last_read_exc: Optional[Exception] = None
        for _ in range(max(1, args.read_retries + 1)):
            try:
                snap = request_temperature_states(
                    host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)
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
                    f"Too many consecutive state read failures ({consecutive_read_failures})."
                ) from last_read_exc
            time.sleep(args.dt)
            continue

        consecutive_read_failures = 0
        abs_err_to_target = abs(float(snap["temp"]) - temp_ref_target)

        if prev_abs_err is not None:
            if (
                    (not crossed_band_1)
                    and (prev_abs_err > args.cross_band_1)
                    and (abs_err_to_target <= args.cross_band_1)
            ):
                current_pid = apply_pid_update(
                    label=f"cross_{args.cross_band_1:g}deg",
                    run_id=run_id,
                    row=args.pid_row,
                    pid=(controller["mid"]["kp"], controller["mid"]
                         ["ki"], controller["mid"]["kd"]),
                    args=args,
                    events=pid_events,
                )
                crossed_band_1 = True

            if (
                    (not crossed_band_2)
                    and (prev_abs_err > args.cross_band_2)
                    and (abs_err_to_target <= args.cross_band_2)
            ):
                current_pid = apply_pid_update(
                    label=f"cross_{args.cross_band_2:g}deg",
                    run_id=run_id,
                    row=args.pid_row,
                    pid=(controller["near"]["kp"], controller["near"]
                         ["ki"], controller["near"]["kd"]),
                    args=args,
                    events=pid_events,
                )
                crossed_band_2 = True

        prev_abs_err = abs_err_to_target

        sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
        rows.append(
            {
                "run_id": run_id,
                "timestamp": now,
                "kp": float(current_pid[0]),
                "ki": float(current_pid[1]),
                "kd": float(current_pid[2]),
                **snap,
                "sq_error": sq_error,
            }
        )

        if now >= next_progress_ts:
            n = len(rows)
            running_mae = sum(
                abs(float(r["temp_ref"]) - float(r["temp"])) for r in rows) / max(1, n)
            remaining = max(args.test_duration - elapsed, 0.0)
            print(
                f"[run {run_id}] samples={n} elapsed={elapsed:.1f}s remaining={remaining:.1f}s "
                f"temp={snap['temp']:.3f} temp_ref={snap['temp_ref']:.3f} mae={running_mae:.6f} "
                f"kp={current_pid[0]} ki={current_pid[1]} kd={current_pid[2]}"
            )
            next_progress_ts += args.progress_every

        time.sleep(args.dt)

    df_samples = pd.DataFrame(rows)
    if df_samples.empty:
        raise RuntimeError(
            "No TCP samples were collected during automated test")

    df_samples["mae"] = (df_samples["temp_ref"].astype(
        float) - df_samples["temp"].astype(float)).abs()
    df_samples["start_temp"] = float(df_samples["temp"].iloc[0])
    start_temp = float(df_samples["start_temp"].iloc[0])
    target_temp_measured = float(df_samples["temp_ref"].iloc[0])

    run_summary: Dict[str, object] = {
        "run_id": run_id,
        "start_ts": float(df_samples["timestamp"].iloc[0]),
        "end_ts": float(df_samples["timestamp"].iloc[-1]),
        "duration_s": float(df_samples["timestamp"].iloc[-1] - df_samples["timestamp"].iloc[0]),
        "num_samples": int(len(df_samples)),
        "start_temp": start_temp,
        "temp_ref": target_temp_measured,
        "pid_source": pid_source,
        "far_kp": int(controller["far"]["kp"]),
        "far_ki": int(controller["far"]["ki"]),
        "far_kd": int(controller["far"]["kd"]),
        "mid_kp": int(controller["mid"]["kp"]),
        "mid_ki": int(controller["mid"]["ki"]),
        "mid_kd": int(controller["mid"]["kd"]),
        "near_kp": int(controller["near"]["kp"]),
        "near_ki": int(controller["near"]["ki"]),
        "near_kd": int(controller["near"]["kd"]),
        "mse": float(df_samples["sq_error"].mean()),
        "mae": float(df_samples["mae"].mean()),
    }

    samples_out = history_run_file(run_id, str(
        Path(args.history_root) / args.samples_csv), args.history_root)
    runs_out = history_run_file(run_id, str(
        Path(args.history_root) / args.runs_csv), args.history_root)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_rows_csv(args.all_runs_csv, [run_summary])

    return df_samples, run_summary


def run_with_retry(
        controller: Dict[str, Dict[str, int]],
        args: argparse.Namespace,
        pid_source: str,
        condition_pretest: bool,
) -> Tuple[str, pd.DataFrame, Dict[str, object]]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, args.run_retry_attempts + 1):
        run_id = make_run_id(prefix="run")
        try:
            print(
                f"[{run_id}] attempt {attempt}/{args.run_retry_attempts}: "
                f"far={(controller['far']['kp'], controller['far']['ki'], controller['far']['kd'])} "
                f"mid={(controller['mid']['kp'], controller['mid']['ki'], controller['mid']['kd'])} "
                f"near={(controller['near']['kp'], controller['near']['ki'], controller['near']['kd'])}"
            )
            df_samples, run_summary = run_band_test(
                run_id=run_id,
                controller=controller,
                args=args,
                pid_source=pid_source,
                condition_pretest=condition_pretest,
            )
            return run_id, df_samples, run_summary
        except Exception as exc:
            last_exc = exc
            print(f"[{run_id}] run failed on attempt {attempt}: {exc}")
            if attempt < args.run_retry_attempts:
                time.sleep(max(0.0, args.run_retry_delay))

    assert last_exc is not None
    raise RuntimeError(
        f"Run failed after {args.run_retry_attempts} attempts with the same controller.") from last_exc


def validate_args(args: argparse.Namespace) -> None:
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0")
    if args.test_duration <= 0:
        raise ValueError("--test-duration must be > 0")
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    if args.cooldown < 0:
        raise ValueError("--cooldown must be >= 0")
    if args.heatup_duration < 0:
        raise ValueError("--heatup-duration must be >= 0")
    if args.post_heatup_cooldown < 0:
        raise ValueError("--post-heatup-cooldown must be >= 0")
    if args.run_retry_attempts <= 0:
        raise ValueError("--run-retry-attempts must be > 0")
    if args.run_retry_delay < 0:
        raise ValueError("--run-retry-delay must be >= 0")
    if args.read_retries < 0:
        raise ValueError("--read-retries must be >= 0")
    if args.read_retry_delay_s < 0:
        raise ValueError("--read-retry-delay-s must be >= 0")
    if args.max_consecutive_failures <= 0:
        raise ValueError("--max-consecutive-failures must be > 0")
    if args.cross_band_2 > args.cross_band_1:
        raise ValueError("--cross-band-2 should be <= --cross-band-1")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    print("[INFO] refreshing band models from history")
    next_payload = refresh_band_models(args)

    print("[INFO] initial planned controller:")
    for band in band_bo.BANDS:
        triplet = next_payload["suggested_controller"][band]
        print(
            f"  {band:>4}: kp={triplet['kp']} ki={triplet['ki']} kd={triplet['kd']}")

    successful_runs = 0
    while successful_runs < args.num_runs:
        planned_controller = next_payload["suggested_controller"]
        pid_source = str(next_payload.get(
            "meta", {}).get("source", "band_gp_bo"))
        condition_pretest = bool(args.condition_initial) or successful_runs > 0
        run_id, df_samples, run_summary = run_with_retry(
            planned_controller, args, pid_source, condition_pretest)
        print(
            f"[run {run_id}] completed successfully ({successful_runs + 1}/{args.num_runs})")
        print(f"[run {run_id}] samples={run_summary['num_samples']}")
        print(f"[run {run_id}] mse={run_summary['mse']:.6f}")
        print(f"[run {run_id}] mae={run_summary['mae']:.6f}")

        successful_runs += 1

        print(
            f"[INFO] recomputing band metrics and retraining after run {run_id}")
        next_payload = refresh_band_models(args)
        print("[INFO] next planned controller:")
        for band in band_bo.BANDS:
            triplet = next_payload["suggested_controller"][band]
            print(
                f"  {band:>4}: kp={triplet['kp']} ki={triplet['ki']} kd={triplet['kd']}")

        if successful_runs < args.num_runs and args.cooldown > 0:
            print(f"Waiting {args.cooldown:.1f}s before next run...")
            time.sleep(args.cooldown)

    print("\n=== Closed-loop band BO summary ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Models saved:    {args.models_out}")
    print(f"Next suggestion:  {args.next_out}")


if __name__ == "__main__":
    main()
