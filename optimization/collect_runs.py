#!/usr/bin/env python3
"""Replicated, start-gated chamber runs driven by a time-indexed PID profile.

Each candidate configuration is a profile of K knots spread over the run, and the
controller value is recomputed every --update-interval-s by interpolating that
profile in time:

    knots:   t=0s (kp0,ki0,kd0)   t=200s (kp1,ki1,kd1)   t=400s (kp2,ki2,kd2) ...
    applied: every 3 s, interpolated between the surrounding knots

--knots 1 is a single fixed triplet for the whole run. --profile-mode linear ramps
between knots; step jumps at each knot.

Every phase -- heatup, soak and test -- is sampled and written with a `phase`
column. The test start is gated on both temperature (--start-temp-tol) and drift
rate (--start-rate-tol) so repeats share an initial condition, and --repeats runs
of each configuration are interleaved rather than run back to back.

Nothing is scored here. Run benchmarks/settling_metrics.py over --history-root
afterwards.

SAFETY: writes PID values and temp_ref jobs to a real chamber over TCP with no
independent interlock. --dry-run exercises the whole plan without writing.

    python -m optimization.collect_runs --knots 4 --n-configs 6 --repeats 4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import append_row_csv, append_rows_csv, history_run_file, make_run_id, save_json  
from deepvac.pid import parse_bounds, read_pid_from_tcp  
from tcp.tcp_common import (  
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
    write_pid_row,
)

OUTPUT_DIR = ROOT / "experiments" / "output"

PHASE_HEATUP = "heatup"
PHASE_SOAK = "soak"
PHASE_TEST = "test"

PIDTriplet = Tuple[int, int, int]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Collect replicated, start-gated chamber runs driven by time-indexed PID profiles.",
    )

    # --- The PID profile -------------------------------------------------------
    ap.add_argument(
        "--knots",
        type=int,
        default=4,
        help="PID triplets spread over the run. 1 = a single fixed triplet the whole way.",
    )
    ap.add_argument(
        "--update-interval-s",
        type=float,
        default=3.0,
        help="How often the profile is re-evaluated and written to the chamber.",
    )
    ap.add_argument(
        "--profile-span-s",
        type=float,
        default=600.0,
        help="Knots are spread evenly from test start to here. The last triplet then holds.",
    )
    ap.add_argument(
        "--profile-mode",
        choices=["linear", "step"],
        default="linear",
        help="linear ramps between knots in 3 s increments; step jumps at each knot.",
    )
    ap.add_argument(
        "--verify-writes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read back every PID write. Costs an extra round trip per update.",
    )

    # --- What to run -----------------------------------------------------------
    ap.add_argument("--plan-file", default=None,
                    help="Load an existing plan JSON instead of generating one. Enables resume.")
    ap.add_argument("--n-configs", type=int, default=6,
                    help="Distinct PID profiles to generate and try.")
    ap.add_argument("--repeats", type=int, default=4,
                    help="Runs per profile.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--kp-bounds", default="1,20")
    ap.add_argument("--ki-bounds", default="1,1000")
    ap.add_argument("--kd-bounds", default="1,50")
    ap.add_argument("--seed-profiles", default=None,
                    help="JSON file of explicit profiles to include verbatim, e.g. a known baseline.")

    # --- Run shape -------------------------------------------------------------
    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--test-duration", type=float, default=20.0 * 60.0)
    ap.add_argument("--dt", type=float, default=1.0, help="Sampling period, all phases.")
    ap.add_argument("--progress-every", type=float, default=60.0)

    ap.add_argument("--heatup-temp-ref", type=float, default=25.0)
    ap.add_argument("--heatup-duration", type=float, default=5.0 * 60.0)
    ap.add_argument("--heatup-pid", default=None,
                    help="'kp,ki,kd' applied during heatup. Default: leave the controller's PID alone.")

    # --- Start gate ------------------------------------------------------------
    ap.add_argument("--start-temp-nominal", type=float, default=24.0,
                    help="Temperature the test must start from.")
    ap.add_argument("--start-temp-tol", type=float, default=0.3)
    ap.add_argument("--start-rate-tol", type=float, default=0.01,
                    help="Max |dT/dt| in C/s at test start.")
    ap.add_argument("--start-rate-window-s", type=float, default=30.0)
    ap.add_argument("--soak-timeout-s", type=float, default=15.0 * 60.0,
                    help="Start anyway after this long, recording gate_satisfied=False.")
    ap.add_argument("--soak-min-s", type=float, default=30.0)

    # --- Safety ----------------------------------------------------------------
    ap.add_argument("--abort-below", type=float, default=-25.0)
    ap.add_argument("--abort-above", type=float, default=40.0)
    ap.add_argument("--safe-pid", default=None,
                    help="'kp,ki,kd' restored on abort or completion. Default: whatever was read at startup.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the schedule without writing temp_ref or PID to the chamber.")

    ap.add_argument("--pid-row", type=int, default=1)
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)

    ap.add_argument("--history-root", default="run_history_profiles")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--all-runs-csv", default=str(OUTPUT_DIR / "profile_all_runs.csv"))
    ap.add_argument("--plan-out", default=str(OUTPUT_DIR / "profile_plan.json"))

    return ap


# -----------------------------------------------------------------------------
# PID profile
# -----------------------------------------------------------------------------


def knot_times(n_knots: int, span_s: float) -> List[float]:
    if n_knots <= 1:
        return [0.0]
    return [round(span_s * i / (n_knots - 1), 3) for i in range(n_knots)]


def profile_pid_at(profile: Dict[str, object], t: float, mode: str) -> PIDTriplet:
    """The triplet the profile calls for at t seconds into the test.

    Before the first knot the first triplet applies; after the last knot the last
    triplet holds for the rest of the run.
    """
    times = [float(x) for x in profile["times"]]
    gains = [[float(v) for v in g] for g in profile["gains"]]

    if len(times) == 1 or t <= times[0]:
        chosen = gains[0]
    elif t >= times[-1]:
        chosen = gains[-1]
    else:
        idx = int(np.searchsorted(times, t, side="right")) - 1
        idx = max(0, min(idx, len(times) - 2))
        if mode == "step":
            chosen = gains[idx]
        else:
            span = max(times[idx + 1] - times[idx], 1e-9)
            frac = (t - times[idx]) / span
            chosen = [a + frac * (b - a) for a, b in zip(gains[idx], gains[idx + 1])]

    return int(round(chosen[0])), int(round(chosen[1])), int(round(chosen[2]))


def sobol_or_uniform(n: int, dim: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc

        return qmc.Sobol(d=dim, scramble=True, seed=seed).random(n)
    except Exception:
        return np.random.default_rng(seed).random((n, dim))


def generate_plan(args: argparse.Namespace) -> Dict[str, object]:
    """n_configs profiles x repeats runs, interleaved so repeats spread over time."""
    n_knots = max(1, int(args.knots))
    kp_lo, kp_hi = parse_bounds(args.kp_bounds)
    ki_lo, ki_hi = parse_bounds(args.ki_bounds)
    kd_lo, kd_hi = parse_bounds(args.kd_bounds)
    times = knot_times(n_knots, float(args.profile_span_s))

    configs: List[Dict[str, object]] = []

    if args.seed_profiles:
        for entry in json.loads(Path(args.seed_profiles).read_text(encoding="utf-8")):
            configs.append({
                "config_id": f"seed{len(configs):02d}",
                "times": entry.get("times", knot_times(len(entry["gains"]), float(args.profile_span_s))),
                "gains": entry["gains"],
                "source": "seed-file",
            })

    n_generated = max(0, int(args.n_configs))
    if n_generated:
        samples = sobol_or_uniform(n_generated, 3 * n_knots, args.seed)
        for i, row in enumerate(samples):
            gains = [
                [
                    int(round(kp_lo + row[3 * k + 0] * (kp_hi - kp_lo))),
                    int(round(ki_lo + row[3 * k + 1] * (ki_hi - ki_lo))),
                    int(round(kd_lo + row[3 * k + 2] * (kd_hi - kd_lo))),
                ]
                for k in range(n_knots)
            ]
            configs.append({
                "config_id": f"k{n_knots}_{i:02d}",
                "times": times,
                "gains": gains,
                "source": "generated",
            })

    if not configs:
        raise ValueError("Plan is empty; set --n-configs > 0 or pass --seed-profiles.")

    # Interleave: every profile gets repeat 1 before any gets repeat 2, reshuffled
    # each round.
    rng = random.Random(args.seed)
    runs: List[Dict[str, object]] = []
    for repeat in range(1, int(args.repeats) + 1):
        order = list(range(len(configs)))
        rng.shuffle(order)
        for idx in order:
            runs.append({
                "index": len(runs),
                "config_id": configs[idx]["config_id"],
                "repeat": repeat,
                "status": "pending",
                "run_id": None,
            })

    return {
        "created_at": int(time.time()),
        "knots": n_knots,
        "profile_mode": args.profile_mode,
        "update_interval_s": float(args.update_interval_s),
        "profile_span_s": float(args.profile_span_s),
        "target_temp": float(args.target_temp),
        "repeats": int(args.repeats),
        "configs": configs,
        "runs": runs,
    }


def save_plan(plan: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Chamber I/O
# -----------------------------------------------------------------------------


def read_snapshot(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    last_exc: Optional[Exception] = None
    for _ in range(max(1, args.read_retries + 1)):
        try:
            return request_temperature_states(
                host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout
            )
        except Exception as exc:
            last_exc = exc
            if args.read_retry_delay_s > 0:
                time.sleep(args.read_retry_delay_s)
    if last_exc is not None:
        print(f"[read] failed after {args.read_retries + 1} attempts: {last_exc}")
    return None


def write_pid(args: argparse.Namespace, *, label: str, run_id: str,
              pid: Sequence[int], events: List[Dict[str, object]],
              verbose: bool = True) -> PIDTriplet:
    triplet = (int(pid[0]), int(pid[1]), int(pid[2]))

    if args.dry_run:
        if verbose:
            print(f"[run {run_id}] DRY-RUN PID ({label}): kp={triplet[0]}, ki={triplet[1]}, kd={triplet[2]}")
        events.append({"event": label, "kp": triplet[0], "ki": triplet[1],
                       "kd": triplet[2], "timestamp": time.time()})
        return triplet

    if args.verify_writes:
        return apply_pid_update(label=label, run_id=run_id, row=args.pid_row,
                                pid=triplet, args=args, events=events)

    # Unverified fast path: one round trip instead of two.
    write_pid_row(row=args.pid_row, kp=triplet[0], ki=triplet[1], kd=triplet[2],
                  host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)
    events.append({"event": label, "kp": triplet[0], "ki": triplet[1],
                   "kd": triplet[2], "timestamp": time.time()})
    return triplet


def publish_job(args: argparse.Namespace, *, temp_ref: float, duration_s: float) -> None:
    if args.dry_run:
        print(f"[dry-run] would publish temp_ref={temp_ref:.3f} for {duration_s:.1f}s")
        return
    publish_temp_ref_job(temp_ref=float(temp_ref), duration_s=duration_s,
                         host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)


def parse_triplet(text: str) -> PIDTriplet:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'kp,ki,kd', got {text!r}")
    return int(round(float(parts[0]))), int(round(float(parts[1]))), int(round(float(parts[2])))


class Aborted(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Sampling loop
# -----------------------------------------------------------------------------


class Sampler:
    """Continuous telemetry sampling loop, shared by every phase of a run."""

    def __init__(self, args: argparse.Namespace, run_id: str) -> None:
        self.args = args
        self.run_id = run_id
        self.rows: List[Dict[str, object]] = []
        self.events: List[Dict[str, object]] = []
        self.consecutive_failures = 0
        self.t0 = time.time()
        self.next_sample = self.t0

    def elapsed(self) -> float:
        return time.time() - self.t0

    def sample(self, phase: str, pid: PIDTriplet,
               extra: Optional[Dict[str, object]] = None) -> Optional[Dict[str, float]]:
        snap = read_snapshot(self.args)
        now = time.time()

        if snap is None:
            self.consecutive_failures += 1
            print(f"[run {self.run_id}] read failed "
                  f"({self.consecutive_failures}/{self.args.max_consecutive_failures})")
            if self.consecutive_failures >= self.args.max_consecutive_failures:
                raise Aborted(f"Too many consecutive state read failures ({self.consecutive_failures})")
            return None

        self.consecutive_failures = 0
        temp = float(snap["temp"])
        if temp < self.args.abort_below or temp > self.args.abort_above:
            raise Aborted(f"Temperature {temp:.2f} C outside the safety envelope "
                          f"[{self.args.abort_below:g}, {self.args.abort_above:g}]")

        row: Dict[str, object] = {
            "run_id": self.run_id,
            "timestamp": now,
            "elapsed_s": now - self.t0,
            "phase": phase,
            "kp": float(pid[0]), "ki": float(pid[1]), "kd": float(pid[2]),
            **snap,
            "sq_error": float((snap["temp_ref"] - snap["temp"]) ** 2),
        }
        if extra:
            row.update(extra)
        self.rows.append(row)
        return snap

    def wait_next(self) -> None:
        self.next_sample += self.args.dt
        sleep_s = self.next_sample - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            missed = int((-sleep_s) // self.args.dt) + 1
            self.next_sample += missed * self.args.dt

    def recent_rate(self, window_s: float) -> Optional[float]:
        """dT/dt over the trailing window, in C/s."""
        if len(self.rows) < 3:
            return None
        recent = [r for r in self.rows if r["timestamp"] >= time.time() - window_s]
        if len(recent) < 3:
            return None
        t = np.asarray([float(r["timestamp"]) for r in recent])
        y = np.asarray([float(r["temp"]) for r in recent])
        if t[-1] - t[0] < 1e-6:
            return None
        return float(np.polyfit(t - t[0], y, 1)[0])


def run_phase_heatup(sampler: Sampler, args: argparse.Namespace, pid: PIDTriplet) -> PIDTriplet:
    """Drive to the heatup setpoint, logging throughout."""
    run_id = sampler.run_id
    if args.heatup_pid:
        pid = write_pid(args, label=f"{PHASE_HEATUP}/pid", run_id=run_id,
                        pid=parse_triplet(args.heatup_pid), events=sampler.events)

    publish_job(args, temp_ref=args.heatup_temp_ref, duration_s=args.heatup_duration)
    print(f"[run {run_id}] {PHASE_HEATUP}: temp_ref={args.heatup_temp_ref:.2f} "
          f"for {args.heatup_duration:.0f}s (logged)")

    deadline = time.time() + args.heatup_duration
    next_progress = time.time() + args.progress_every if args.progress_every > 0 else float("inf")
    while time.time() < deadline:
        snap = sampler.sample(PHASE_HEATUP, pid)
        if snap is not None and time.time() >= next_progress:
            print(f"[run {run_id}] {PHASE_HEATUP} t={sampler.elapsed():.0f}s temp={snap['temp']:.3f}")
            next_progress += args.progress_every
        sampler.wait_next()

    return pid


def run_phase_soak(sampler: Sampler, args: argparse.Namespace, pid: PIDTriplet) -> Dict[str, object]:
    """Drift with no active job until the start gate is satisfied, logging throughout.

    The gate covers both temperature and drift rate.
    """
    run_id = sampler.run_id
    print(f"[run {run_id}] {PHASE_SOAK}: waiting for {args.start_temp_nominal:.2f}"
          f"+-{args.start_temp_tol:g} C and |dT/dt| <= {args.start_rate_tol:g} C/s "
          f"(timeout {args.soak_timeout_s:.0f}s)")

    started = time.time()
    next_progress = started + args.progress_every if args.progress_every > 0 else float("inf")
    gate_satisfied = False

    while True:
        waited = time.time() - started
        snap = sampler.sample(PHASE_SOAK, pid)

        if snap is not None and waited >= args.soak_min_s:
            temp = float(snap["temp"])
            rate = sampler.recent_rate(args.start_rate_window_s)
            if abs(temp - args.start_temp_nominal) <= args.start_temp_tol and \
                    rate is not None and abs(rate) <= args.start_rate_tol:
                gate_satisfied = True
                print(f"[run {run_id}] {PHASE_SOAK}: gate satisfied at t={sampler.elapsed():.0f}s "
                      f"temp={temp:.3f} rate={rate:+.4f} C/s")
                break

        if waited >= args.soak_timeout_s:
            temp = float(snap["temp"]) if snap else float("nan")
            print(f"[run {run_id}] {PHASE_SOAK}: TIMEOUT after {waited:.0f}s at temp={temp:.3f}; "
                  f"starting anyway with gate_satisfied=False")
            break

        if snap is not None and time.time() >= next_progress:
            rate = sampler.recent_rate(args.start_rate_window_s)
            rate_s = "n/a" if rate is None else f"{rate:+.4f}"
            print(f"[run {run_id}] {PHASE_SOAK} t={sampler.elapsed():.0f}s "
                  f"temp={snap['temp']:.3f} rate={rate_s}")
            next_progress += args.progress_every

        sampler.wait_next()

    last = sampler.rows[-1] if sampler.rows else {}
    return {
        "gate_satisfied": gate_satisfied,
        "soak_seconds": time.time() - started,
        "start_temp": float(last.get("temp", float("nan"))),
        "start_rate": sampler.recent_rate(args.start_rate_window_s),
    }


def run_phase_test(sampler: Sampler, args: argparse.Namespace,
                   config: Dict[str, object]) -> Tuple[PIDTriplet, int]:
    """Drive to the target, re-evaluating the time profile every --update-interval-s."""
    run_id = sampler.run_id
    target = float(args.target_temp)

    pid = write_pid(args, label=f"{PHASE_TEST}/t0", run_id=run_id,
                    pid=profile_pid_at(config, 0.0, args.profile_mode), events=sampler.events)
    publish_job(args, temp_ref=target, duration_s=args.test_duration)
    print(f"[run {run_id}] {PHASE_TEST}: temp_ref={target:.2f} for {args.test_duration:.0f}s, "
          f"{len(config['gains'])} knot(s), update every {args.update_interval_s:g}s")

    test_t0 = time.time()
    deadline = test_t0 + args.test_duration
    next_update = test_t0 + args.update_interval_s
    next_progress = test_t0 + args.progress_every if args.progress_every > 0 else float("inf")
    n_writes = 1

    while time.time() < deadline:
        now = time.time()
        profile_t = now - test_t0

        if now >= next_update:
            wanted = profile_pid_at(config, profile_t, args.profile_mode)
            # Only write when the rounded triplet actually moved.
            if wanted != pid:
                pid = write_pid(args, label=f"{PHASE_TEST}/t{profile_t:.0f}", run_id=run_id,
                                pid=wanted, events=sampler.events, verbose=False)
                n_writes += 1
            next_update += args.update_interval_s
            behind = time.time() - next_update
            if behind > 0:
                next_update += (int(behind // args.update_interval_s) + 1) * args.update_interval_s

        snap = sampler.sample(PHASE_TEST, pid, {"profile_t": profile_t})
        if snap is not None and time.time() >= next_progress:
            print(f"[run {run_id}] {PHASE_TEST} t={profile_t:.0f}s temp={snap['temp']:.3f} "
                  f"err={abs(float(snap['temp']) - target):.3f} pid={pid} writes={n_writes}")
            next_progress += args.progress_every

        sampler.wait_next()

    return pid, n_writes


def execute_run(args: argparse.Namespace, config: Dict[str, object], repeat: int,
                initial_pid: PIDTriplet) -> Dict[str, object]:
    run_id = make_run_id(prefix="prof")
    sampler = Sampler(args, run_id)

    print(f"\n[run {run_id}] config={config['config_id']} repeat={repeat} "
          f"knots={config['gains']} at t={config['times']}")

    pid = run_phase_heatup(sampler, args, initial_pid)
    gate = run_phase_soak(sampler, args, pid)
    final_pid, n_writes = run_phase_test(sampler, args, config)

    df = pd.DataFrame(sampler.rows)
    if df.empty:
        raise Aborted("No samples collected")

    summary: Dict[str, object] = {
        "run_id": run_id,
        # settling_metrics.py groups repeats by this.
        "config_id": config["config_id"],
        "repeat": int(repeat),
        "n_knots": len(config["gains"]),
        "profile_mode": args.profile_mode,
        "update_interval_s": float(args.update_interval_s),
        "profile_gains": json.dumps(config["gains"], separators=(",", ":")),
        "profile_times": json.dumps(config["times"], separators=(",", ":")),
        "n_pid_writes": int(n_writes),
        "target_temp": float(args.target_temp),
        "start_ts": float(df["timestamp"].iloc[0]),
        "end_ts": float(df["timestamp"].iloc[-1]),
        "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
        "n_samples": int(len(df)),
        "final_pid": json.dumps(list(final_pid), separators=(",", ":")),
        **{k: gate[k] for k in ("gate_satisfied", "soak_seconds", "start_temp", "start_rate")},
        **{f"n_samples_{p}": int((df["phase"] == p).sum()) for p in (PHASE_HEATUP, PHASE_SOAK, PHASE_TEST)},
    }

    samples_out = history_run_file(run_id, str(Path(args.history_root) / args.samples_csv), args.history_root)
    runs_out = history_run_file(run_id, str(Path(args.history_root) / args.runs_csv), args.history_root)
    append_rows_csv(samples_out, df.to_dict(orient="records"))
    append_rows_csv(runs_out, [summary])
    save_json(Path(samples_out).with_name("pid_events.json"),
              {"run_id": run_id, "config": config, "events": sampler.events})

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_row_csv(args.all_runs_csv, summary)

    print(f"[run {run_id}] done: {summary['n_samples']} samples "
          f"(heatup {summary['n_samples_heatup']}, soak {summary['n_samples_soak']}, "
          f"test {summary['n_samples_test']}), {n_writes} PID writes, "
          f"gate_satisfied={summary['gate_satisfied']}")
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.test_duration <= 0:
        raise ValueError("--test-duration must be > 0")
    if args.heatup_duration < 0:
        raise ValueError("--heatup-duration must be >= 0")
    if args.knots < 1:
        raise ValueError("--knots must be >= 1")
    if args.update_interval_s <= 0:
        raise ValueError("--update-interval-s must be > 0")
    if args.profile_span_s <= 0:
        raise ValueError("--profile-span-s must be > 0")
    if args.knots > 1 and args.profile_span_s > args.test_duration:
        raise ValueError("--profile-span-s must be <= --test-duration, or later knots never apply")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.n_configs < 0:
        raise ValueError("--n-configs must be >= 0")
    if args.start_temp_tol <= 0:
        raise ValueError("--start-temp-tol must be > 0")
    if args.start_rate_tol <= 0:
        raise ValueError("--start-rate-tol must be > 0")
    if args.soak_timeout_s < args.soak_min_s:
        raise ValueError("--soak-timeout-s must be >= --soak-min-s")
    if args.abort_below >= args.abort_above:
        raise ValueError("--abort-below must be < --abort-above")
    if not (0 <= args.pid_row <= 4):
        raise ValueError("--pid-row must be in range [0, 4]")
    if not (args.abort_below < args.target_temp < args.abort_above):
        raise ValueError("--target-temp must lie inside the safety envelope")
    if not (args.abort_below < args.heatup_temp_ref < args.abort_above):
        raise ValueError("--heatup-temp-ref must lie inside the safety envelope")
    for name in ("kp_bounds", "ki_bounds", "kd_bounds"):
        parse_bounds(getattr(args, name))


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    plan_path = Path(args.plan_file) if args.plan_file else Path(args.plan_out)
    if args.plan_file and plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        done = sum(1 for r in plan["runs"] if r["status"] == "done")
        print(f"[plan] resuming {plan_path} ({done}/{len(plan['runs'])} already done)")
    else:
        plan = generate_plan(args)
        save_plan(plan, plan_path)
        print(f"[plan] wrote {plan_path}: {len(plan['configs'])} profiles "
              f"x {plan['repeats']} repeats = {len(plan['runs'])} runs")

    by_id = {c["config_id"]: c for c in plan["configs"]}
    pending = [r for r in plan["runs"] if r["status"] != "done"]

    per_run_s = args.heatup_duration + args.soak_min_s + args.test_duration
    updates = int(args.test_duration / args.update_interval_s)
    print(f"[plan] {args.knots} knot(s), {args.profile_mode} interpolation, "
          f"up to {updates} profile updates per run")
    if args.verify_writes and updates > 200:
        print(f"[plan] NOTE {updates} updates/run with --verify-writes costs two TCP round "
              f"trips each; pass --no-verify-writes if the {args.update_interval_s:g}s "
              f"interval starts slipping.")
    print(f"[plan] {len(pending)} runs remaining, "
          f"at least {len(pending) * per_run_s / 3600.0:.1f} h of chamber time")

    if args.dry_run:
        print("[plan] DRY-RUN: nothing will be written to the chamber")

    try:
        startup_pid = read_pid_from_tcp(row=args.pid_row, args=args)
        initial_pid = (int(round(startup_pid["kp"])), int(round(startup_pid["ki"])),
                       int(round(startup_pid["kd"])))
    except Exception as exc:
        if not args.dry_run:
            raise
        print(f"[dry-run] could not read PID ({exc}); assuming (10, 500, 20)")
        initial_pid = (10, 500, 20)
    safe_pid = parse_triplet(args.safe_pid) if args.safe_pid else initial_pid
    print(f"[plan] safe PID for abort/completion: {safe_pid}")

    completed = 0
    try:
        for entry in pending:
            config = by_id[entry["config_id"]]
            try:
                summary = execute_run(args, config, int(entry["repeat"]), initial_pid)
                entry["status"] = "done"
                entry["run_id"] = summary["run_id"]
                completed += 1
            except Aborted as exc:
                entry["status"] = "aborted"
                entry["error"] = str(exc)
                print(f"[plan] run aborted: {exc}")
                raise
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                print(f"[plan] run failed, continuing: {exc}")
            finally:
                save_plan(plan, plan_path)
    finally:
        events: List[Dict[str, object]] = []
        try:
            write_pid(args, label="restore/safe", run_id="plan", pid=safe_pid, events=events)
            publish_job(args, temp_ref=args.heatup_temp_ref, duration_s=0.0)
        except Exception as exc:
            print(f"[plan] WARNING could not restore safe state: {exc}")

        save_plan(plan, plan_path)
        remaining = sum(1 for r in plan["runs"] if r["status"] != "done")
        print(f"\n[plan] {completed} runs completed this session, {remaining} still pending")
        print(f"[plan] plan file : {plan_path}")
        print(f"[plan] telemetry : {args.history_root}")
        print(f"[plan] next      : python -m benchmarks.settling_metrics "
              f"--history-root {args.history_root}")


if __name__ == "__main__":
    main()
