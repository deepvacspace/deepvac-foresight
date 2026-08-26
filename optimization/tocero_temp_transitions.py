#!/usr/bin/env python3
"""Automated multi-run TCP tests that sweep target temperature and mid-run
temperature changes, instead of always cooling 25 -> 0.

Same TCP protocol, history layout, and read/retry handling as tocero_3band.py.
Two differences from it, both deliberate -- see the module-end notes for why:

  1. Every run holds ONE PID triplet for its whole duration by default (no
     far/mid/near re-apply). tocero_3band.py's own PID_SCHEDULES already used the
     same triplet for all three bands, so that machinery was already a no-op
     there. --pid-per-leg opts into a different triplet per leg instead, so gains
     and setpoint can change together mid-run -- see the module-end notes.

  2. Each run walks a short PROFILE of (target_temp, duration_s) legs -- an anchor
     leg back to a known reference temperature, then always 3 legs at other targets
     -- logged continuously as one run, so the setpoint change from one leg to the
     next is captured in-context rather than only ever appearing at a run boundary.

python -m optimization.tocero_temp_transitions --num-tests 20
python -m optimization.tocero_temp_transitions --forever --cooldown 120
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import append_row_csv, append_rows_csv, history_run_file, make_run_id
from deepvac.metrics import append_mae_column, compute_tail_cost
from deepvac.pid import random_pid
from tcp.tcp_common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
)

PIDTriplet = Tuple[int, int, int]
Leg = Tuple[float, float, Optional[PIDTriplet]]     # (target_temp, duration_s, pid_override)
Profile = List[Leg]

PID_POOL: List[PIDTriplet] = [
    (15, 1000, 0), (10, 750, 10), (10, 500, 40), (20, 750, 10), (20, 750, 40), (20, 750, 20),
    (15, 200, 20), (15, 750, 10), (10, 200, 20), (20, 1000, 10), (20, 200, 20), (15, 200, 10),
    (15, 500, 40), (20, 200, 0), (20, 1000, 40), (10, 1000, 20), (15, 200, 0), (5, 1000, 20),
    (5, 200, 0), (5, 1000, 40), (20, 500, 20), (5, 1000, 0), (10, 1000, 40), (10, 1000, 10),
    (10, 750, 0), (20, 1000, 0), (5, 750, 10), (5, 750, 40), (15, 1000, 10), (20, 500, 0),
    (5, 750, 0), (5, 200, 20), (15, 750, 40), (10, 750, 20), (20, 500, 10), (15, 1000, 20),
    (5, 200, 10), (20, 750, 0), (15, 200, 40), (20, 500, 40), (20, 200, 40), (20, 200, 10),
    (10, 750, 40), (15, 500, 10), (5, 500, 20), (5, 1000, 10), (5, 200, 40), (10, 200, 0),
    (15, 500, 0), (10, 500, 0), (15, 500, 20), (20, 1000, 20), (10, 500, 20), (10, 1000, 0),
    (5, 500, 10), (15, 750, 20), (10, 200, 10), (15, 1000, 40), (10, 200, 40), (15, 750, 0),
    (5, 750, 20), (10, 500, 10), (5, 500, 40), (5, 500, 0),
]


def build_arg_parser() -> argparse.ArgumentParser:
    output_dir = Path(__file__).with_name("output")
    ap = argparse.ArgumentParser()

    ap.add_argument("--num-tests", type=int, default=20)
    ap.add_argument("--forever", action="store_true", help="Run tests continuously")
    ap.add_argument("--val", action="store_true", help="Save runs with val_* ids for validation.")
    ap.add_argument("--seed", type=int, default=None, help="Seeds both profile generation and PID cycling order.")

    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0)
    ap.add_argument("--cooldown", type=float, default=3.0 * 60.0, help="Seconds between runs")

    # --- Anchor leg --------------------------------------------------------------
    # Every run starts here, so runs are comparable to each other and to the
    # existing 25 -> 0 history despite the profile legs going anywhere afterward.
    ap.add_argument("--anchor-temp", type=float, default=25.0)
    ap.add_argument("--anchor-duration-s", type=float, default=5.0 * 60.0)
    ap.add_argument("--skip-anchor", action="store_true", help="Start the profile from wherever the chamber is.")

    # --- Temperature range and profile generation ---------------------------------
    # Every profile is a fixed 3 legs after the anchor. Leg 1's target is drawn
    # structurally; legs 2 and 3 are uniformly random
    ap.add_argument("--temp-ref-min", type=float, default=-22.0)
    ap.add_argument("--temp-ref-max", type=float, default=30.0)
    ap.add_argument("--sweep-points", type=int, default=9,
                    help="Leg-1 targets on an even grid across the full range.")
    ap.add_argument("--step-fractions", nargs="*", type=float, default=[0.15, 0.35, 0.6, 0.9],
                    help="More leg-1 targets: anchor +- fraction*range, each direction.")
    ap.add_argument("--n-random-profiles", type=int, default=18,
                    help="Extra profiles whose leg-1 target is also drawn uniformly at random.")
    ap.add_argument("--leg-base-duration-s", type=float, default=240.0,
                    help="Minimum dwell even for a ~0-magnitude step, so there is always "
                         "meaningful time on-target, not just barely past the settle band.")
    ap.add_argument("--leg-seconds-per-degree", type=float, default=15.0,
                    help="Extra dwell time per degree C of the leg's step size, matching the "
                         "~15 s/degC settling rate measured on real 25->0 runs.")
    ap.add_argument("--min-leg-duration-s", type=float, default=240.0)
    ap.add_argument("--max-leg-duration-s", type=float, default=1200.0,
                    help="20 min ceiling -- high enough that even a full-range (52 degC) leg "
                         "is not truncated (240 + 15*52 = 1020s).")

    # --- Safety --------------------------------------------------------------------
    ap.add_argument("--safety-margin-c", type=float, default=5.0,
                    help="Abort the current leg early if temp strays this far past [temp-ref-min, temp-ref-max].")

    # --- PID -----------------------------------------------------------------------
    ap.add_argument("--random-pid", action="store_true",
                    help="Sample from --kp/ki/kd-min/max instead of cycling PID_POOL. Applies to "
                         "the run's default triplet, and to each leg's if --pid-per-leg is set.")
    ap.add_argument("--pid-per-leg", action="store_true",
                    help="Draw an independent PID for each of legs 1-3, instead of one triplet for "
                         "the whole run -- gains changing while the setpoint is also mid-transition, "
                         "which is what MPC replanning actually does and no run has captured yet. "
                         "The anchor leg always keeps the run's default PID.")
    ap.add_argument("--kp-min", type=int, default=1)
    ap.add_argument("--kp-max", type=int, default=50)
    ap.add_argument("--ki-min", type=int, default=1)
    ap.add_argument("--ki-max", type=int, default=1000)
    ap.add_argument("--kd-min", type=int, default=1)
    ap.add_argument("--kd-max", type=int, default=200)
    ap.add_argument("--pid-row", type=int, default=1, help="Controller PID row index [0..4]")

    # --- Cost metric (computed on the final leg only; see compute_final_leg_cost) ---
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    # --- TCP -------------------------------------------------------------------------
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)

    # --- Output ------------------------------------------------------------------------
    ap.add_argument("--history-root", default=str(Path(__file__).with_name("temp_sweep_history")),
                    help="Kept separate from run_history/heatup_run_history so the existing "
                         "25 -> 0 baseline stays uncontaminated; combine roots at training time.")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--events-csv", default="temp_ref_events.csv")
    ap.add_argument("--all-runs-csv", default=str(output_dir / "temp_sweep_all_runs.csv"))

    return ap


# -----------------------------------------------------------------------------
# Profile generation
# -----------------------------------------------------------------------------


def leg_duration_s(delta_c: float, args: argparse.Namespace) -> float:
    """Dwell time scaled to the size of the step, so a 2 deg nudge doesn't get the
    same time budget as a 40 deg swing across most of the valid range."""
    dur = args.leg_base_duration_s + args.leg_seconds_per_degree * abs(delta_c)
    return float(min(max(dur, args.min_leg_duration_s), args.max_leg_duration_s))


def round_half_degree(value: float) -> float:
    """Chamber setpoints are commanded as clean numbers, not arbitrary floats."""
    return float(round(float(value) * 2.0) / 2.0)


def clip_target(value: float, args: argparse.Namespace) -> float:
    rounded = round_half_degree(value)
    return float(min(max(rounded, args.temp_ref_min), args.temp_ref_max))


def draw_leg_pid(args: argparse.Namespace, rng: random.Random) -> Optional[PIDTriplet]:
    """One leg's independent PID under --pid-per-leg, else None (use the run default)."""
    if not args.pid_per_leg:
        return None
    if args.random_pid:
        return random_pid(args, rng)
    return PID_POOL[rng.randrange(len(PID_POOL))]


def build_profiles(args: argparse.Namespace, rng: random.Random) -> List[Profile]:
    """The pool of leg sequences a run can be assigned, cycled by run
    index the same way PID_POOL is.

    Every profile is exactly 3 legs: anchor -> leg1 -> leg2 -> leg3. 
    """
    lo, hi = float(args.temp_ref_min), float(args.temp_ref_max)
    span = hi - lo
    if span <= 0:
        raise ValueError("--temp-ref-min must be < --temp-ref-max")

    leg1_targets: List[float] = [
        clip_target(float(t), args) for t in np.linspace(lo, hi, max(2, int(args.sweep_points)))
    ]
    for frac in args.step_fractions:
        step = float(frac) * span
        for sign in (1.0, -1.0):
            b = clip_target(args.anchor_temp + sign * step, args)
            if abs(b - args.anchor_temp) > 1e-6:
                leg1_targets.append(b)
    for _ in range(max(0, int(args.n_random_profiles))):
        leg1_targets.append(clip_target(rng.uniform(lo, hi), args))

    profiles: List[Profile] = []
    for leg1 in leg1_targets:
        leg2 = clip_target(rng.uniform(lo, hi), args)
        leg3 = clip_target(rng.uniform(lo, hi), args)
        profiles.append([
            (leg1, leg_duration_s(leg1 - args.anchor_temp, args), draw_leg_pid(args, rng)),
            (leg2, leg_duration_s(leg2 - leg1, args), draw_leg_pid(args, rng)),
            (leg3, leg_duration_s(leg3 - leg2, args), draw_leg_pid(args, rng)),
        ])

    if not profiles:
        raise RuntimeError("Profile generation produced an empty pool; check --sweep-points/--step-fractions.")

    rng.shuffle(profiles)
    return profiles


def plan_pid_for_run(run_idx: int, args: argparse.Namespace, rng: random.Random) -> Tuple[PIDTriplet, str]:
    if args.random_pid:
        return random_pid(args, rng), "random"
    idx = (run_idx - 1) % len(PID_POOL)
    return PID_POOL[idx], f"pool[{idx}]"


# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------


def sample_leg(
    *,
    run_id: str,
    leg_label: str,
    target_temp: float,
    duration_s: float,
    kp: float,
    ki: float,
    kd: float,
    rows: List[Dict[str, float]],
    args: argparse.Namespace,
    consecutive_read_failures: int,
) -> Tuple[int, bool, str]:
    """Sample one leg's duration into `rows` in place. Returns the updated failure
    streak, whether the leg was aborted early on a safety violation, and why."""
    publish_temp_ref_job(
        temp_ref=target_temp,
        duration_s=duration_s,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )

    safety_lo = args.temp_ref_min - args.safety_margin_c
    safety_hi = args.temp_ref_max + args.safety_margin_c

    t0 = time.time()
    next_progress_ts = t0 + args.progress_every if args.progress_every > 0 else float("inf")

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= duration_s:
            return consecutive_read_failures, False, ""

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
                print(f"[run {run_id}] [{leg_label}] read failed "
                      f"({consecutive_read_failures}/{args.max_consecutive_failures}): {last_read_exc}")
            if consecutive_read_failures >= args.max_consecutive_failures:
                raise RuntimeError(f"Too many consecutive state read failures ({consecutive_read_failures})")
            time.sleep(args.dt)
            continue

        consecutive_read_failures = 0
        temp = float(snap["temp"])

        if temp < safety_lo or temp > safety_hi:
            reason = f"temp={temp:.2f} outside [{safety_lo:.1f}, {safety_hi:.1f}] safety range"
            print(f"[run {run_id}] [{leg_label}] SAFETY ABORT: {reason}")
            return consecutive_read_failures, True, reason

        sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
        rows.append({
            "run_id": run_id,
            "leg": leg_label,
            "timestamp": now,
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            **snap,
            "sq_error": sq_error,
        })

        if now >= next_progress_ts:
            print(f"[run {run_id}] [{leg_label}] samples={len(rows)} elapsed={elapsed:.1f}/{duration_s:.1f}s "
                  f"temp={temp:.3f} temp_ref={snap['temp_ref']:.3f}")
            next_progress_ts += args.progress_every

        time.sleep(args.dt)


def compute_final_leg_cost(df_samples: pd.DataFrame, final_target: float, args: argparse.Namespace) -> dict:
    """compute_tail_cost assumes one constant temp_ref; restrict it to the final
    leg's own rows so a multi-leg run's cost still means "how well did it end up
    where the last command said to go", not a nonsensical blend across targets."""
    final_leg_df = df_samples[df_samples["temp_ref"] == final_target]
    if final_leg_df.empty:
        final_leg_df = df_samples
    return compute_tail_cost(final_leg_df, entry_band=args.entry_band, overshoot_weight=args.overshoot_weight)


def run_single_test(
    run_idx: int,
    total_tests: Optional[int],
    args: argparse.Namespace,
    profile: Profile,
    pid: PIDTriplet,
    pid_source: str,
) -> None:
    run_id = make_run_id(prefix="val" if args.val else "run")
    test_label = f"{run_idx}" if total_tests is None else f"{run_idx}/{total_tests}"
    kp, ki, kd = pid

    legs: List[Leg] = list(profile)
    if not args.skip_anchor:
        legs = [(round_half_degree(args.anchor_temp), float(args.anchor_duration_s), None), *legs]
    if not legs:
        raise ValueError("Run has no legs: profile is empty and --skip-anchor was set")

    print(f"[run {run_id}] starting test {test_label}: pid={pid} ({pid_source}), "
          f"{len(legs)} leg(s) -> {[t for t, _, _ in legs]}")

    current_pid = apply_pid_update(label="start", run_id=run_id, row=args.pid_row, pid=pid, args=args, events=[])

    # Prime the TCP state-name cache. Some controller replies appear to omit
    # names intermittently once a job is running.
    _ = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)

    rows: List[Dict[str, float]] = []
    events: List[Dict[str, object]] = []
    consecutive_read_failures = 0
    aborted_reason = ""

    for leg_idx, (target_temp, duration_s, leg_pid) in enumerate(legs):
        leg_label = "anchor" if (leg_idx == 0 and not args.skip_anchor) else f"leg{leg_idx}"

        if leg_pid is not None and leg_pid != current_pid:
            current_pid = apply_pid_update(
                label=f"{leg_label}/pid", run_id=run_id, row=args.pid_row,
                pid=leg_pid, args=args, events=[],
            )

        leg_start_elapsed = time.time()
        events.append({
            "run_id": run_id, "leg": leg_label, "leg_idx": leg_idx,
            "target_temp": target_temp, "planned_duration_s": duration_s,
            "kp": current_pid[0], "ki": current_pid[1], "kd": current_pid[2],
            "timestamp": leg_start_elapsed,
        })
        print(f"[run {run_id}] [{leg_label}] target={target_temp:.2f} for {duration_s:.0f}s pid={current_pid}")

        consecutive_read_failures, aborted, reason = sample_leg(
            run_id=run_id, leg_label=leg_label, target_temp=target_temp, duration_s=duration_s,
            kp=current_pid[0], ki=current_pid[1], kd=current_pid[2],
            rows=rows, args=args, consecutive_read_failures=consecutive_read_failures,
        )
        if aborted:
            aborted_reason = reason
            break

    df_samples = pd.DataFrame(rows)
    if df_samples.empty:
        raise RuntimeError("No TCP samples were collected during automated test")

    df_samples = append_mae_column(df_samples)
    start_temp = float(df_samples["temp"].iloc[0])
    df_samples["start_temp"] = start_temp

    final_target = float(legs[len(events) - 1][0]) if aborted_reason else float(legs[-1][0])
    cost_info = compute_final_leg_cost(df_samples, final_target, args)

    run_summary = {
        "run_id": run_id,
        "start_ts": float(df_samples["timestamp"].iloc[0]),
        "end_ts": float(df_samples["timestamp"].iloc[-1]),
        "duration_s": float(df_samples["timestamp"].iloc[-1] - df_samples["timestamp"].iloc[0]),
        "num_samples": int(len(df_samples)),
        "start_temp": start_temp,
        "temp_ref": final_target,
        "pid_source": pid_source,
        "far_kp": int(kp), "far_ki": int(ki), "far_kd": int(kd),
        "mid_kp": int(kp), "mid_ki": int(ki), "mid_kd": int(kd),
        "near_kp": int(kp), "near_ki": int(ki), "near_kd": int(kd),
        "n_legs": len(legs),
        "legs": "; ".join(f"{t:g}@{d:g}s" for t, d, _ in legs),
        "aborted": bool(aborted_reason),
        "abort_reason": aborted_reason,
        "mse": float(df_samples["sq_error"].mean()),
        "mae": float(df_samples["mae"].mean()),
    }
    run_summary["cost"] = float(cost_info["cost"])
    run_summary["tail_mae"] = None if cost_info["tail_mae"] is None else float(cost_info["tail_mae"])
    run_summary["overshoot"] = None if cost_info["overshoot"] is None else float(cost_info["overshoot"])

    samples_out = history_run_file(run_id, str(Path(args.history_root) / args.samples_csv), args.history_root)
    runs_out = history_run_file(run_id, str(Path(args.history_root) / args.runs_csv), args.history_root)
    events_out = history_run_file(run_id, str(Path(args.history_root) / args.events_csv), args.history_root)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])
    append_rows_csv(events_out, events)

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_row_csv(args.all_runs_csv, run_summary)

    print(f"[run {run_id}] run_id={run_summary['run_id']}")
    print(f"[run {run_id}] legs={run_summary['legs']}")
    print(f"[run {run_id}] aborted={run_summary['aborted']} {run_summary['abort_reason']}")
    print(f"[run {run_id}] samples={run_summary['num_samples']}")
    print(f"[run {run_id}] cost={run_summary['cost']:.6f}")
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
    if args.temp_ref_min >= args.temp_ref_max:
        raise ValueError("--temp-ref-min must be < --temp-ref-max")
    if not (args.temp_ref_min <= args.anchor_temp <= args.temp_ref_max):
        raise ValueError("--anchor-temp must be within [--temp-ref-min, --temp-ref-max]")
    if not (0 <= args.pid_row <= 4):
        raise ValueError("--pid-row must be in range [0, 4]")
    if args.safety_margin_c < 0:
        raise ValueError("--safety-margin-c must be >= 0")

    rng = random.Random(args.seed)
    profiles = build_profiles(args, rng)
    print(f"[setup] {len(profiles)} temperature profiles, {len(PID_POOL)} PID triplets in the pool")
    print(f"[setup] valid range [{args.temp_ref_min:g}, {args.temp_ref_max:g}] deg C, "
          f"safety abort outside [{args.temp_ref_min - args.safety_margin_c:g}, "
          f"{args.temp_ref_max + args.safety_margin_c:g}] deg C")

    run_idx = 1
    while True:
        total = None if args.forever else args.num_tests
        profile = profiles[(run_idx - 1) % len(profiles)]
        pid, pid_source = plan_pid_for_run(run_idx, args, rng)

        try:
            run_single_test(
                run_idx=run_idx, total_tests=total, args=args,
                profile=profile, pid=pid, pid_source=pid_source,
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