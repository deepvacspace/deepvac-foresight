#!/usr/bin/env python3
"""Passively listens for a live TCP connection to the chamber and records
whatever experiment is currently running, without sending any commands.

Polls request_temperature_states() every --poll-interval-s:
  - While no connection is detected yet: keeps retrying quietly
  - The first successful read prints a "chamber connected" message and starts
    recording every successful sample, refreshing a CSV + trajectory plot on
    disk every --plot-every-s.
  - Once recording has started, if --disconnect-grace-s seconds pass without
    a successful read, the run is considered finished: a final CSV + plot are
    saved and the script exits gracefully.

Example:
    python -m optimization.listener --tcp-host 172.0.30.10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import make_run_id
from deepvac.metrics import append_mae_column
from tcp.tcp_common import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT, request_temperature_states

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)

    ap.add_argument("--poll-interval-s", type=float, default=1.0, help="Seconds between read attempts.")
    ap.add_argument("--disconnect-grace-s", type=float, default=10.0,
                    help="Seconds without a successful read, once recording has started, before the "
                         "run is considered finished.")

    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--session-name", default=None, help="Output subfolder name. Default: generated id.")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-every-s", type=float, default=30.0,
                    help="Re-save CSV + plot at this interval while recording, so progress can be "
                         "inspected mid-run. 0 disables progress snapshots (only the final save happens).")
    ap.add_argument("--progress-every-s", type=float, default=60.0,
                    help="Print a status line at this interval while recording. 0 disables.")
    ap.add_argument("--waiting-progress-every-s", type=float, default=30.0,
                    help="Print a 'still waiting' line at this interval before a connection is found. "
                         "0 disables.")

    return ap


def try_read(args: argparse.Namespace) -> dict[str, float] | None:
    """One best-effort read. Returns None if the request failed or any value
    in the snapshot is non-finite."""
    try:
        snap = request_temperature_states(host=args.tcp_host, port=args.tcp_port, timeout=args.tcp_timeout)
    except Exception:
        return None
    if any(not math.isfinite(v) for v in snap.values()):
        return None
    return snap


def plot_trajectory(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_temp, ax_pid) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_temp.plot(df["elapsed_s"], df["temp"], label="temp", color="#1f77b4", linewidth=2)
    ax_temp.plot(df["elapsed_s"], df["temp_ref"], label="temp_ref", color="#888888",
                linestyle="--", linewidth=1.5)
    ax_temp.set_ylabel("Temperature (deg C)")
    ax_temp.set_title("Chamber trajectory (listener recording)")
    ax_temp.grid(True, alpha=0.25)
    ax_temp.legend()

    ax_pid.step(df["elapsed_s"], df["kp"], where="post", label="kp")
    ax_pid.step(df["elapsed_s"], df["ki"], where="post", label="ki")
    ax_pid.step(df["elapsed_s"], df["kd"], where="post", label="kd")
    ax_pid.set_xlabel("Elapsed seconds")
    ax_pid.set_ylabel("PID gains")
    ax_pid.grid(True, alpha=0.25)
    ax_pid.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_progress(rows: list[dict[str, float]], session_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = append_mae_column(df)
    df.to_csv(session_dir / "run_samples.csv", index=False)
    if not args.no_plot:
        plot_trajectory(df, session_dir / "trajectory.png")
    return df


def wait_for_connection(args: argparse.Namespace) -> dict[str, float]:
    """Blocks until a successful chamber read, retrying at --poll-interval-s."""
    attempts = 0
    next_progress = time.time() + args.waiting_progress_every_s if args.waiting_progress_every_s > 0 else float("inf")
    while True:
        snap = try_read(args)
        attempts += 1
        if snap is not None:
            return snap
        now = time.time()
        if now >= next_progress:
            print(f"[listener] still waiting for a chamber connection ({attempts} attempts so far)...")
            next_progress = now + args.waiting_progress_every_s
        time.sleep(args.poll_interval_s)


def record_run(
    first_snap: dict[str, float], run_id: str, session_dir: Path, args: argparse.Namespace,
) -> list[dict[str, float]]:
    """Records samples until --disconnect-grace-s passes without a successful read
    (or the caller is interrupted), snapshotting CSV+plot every --plot-every-s."""
    t0 = time.time()
    last_success = t0
    stale_notice_sent = False
    rows: list[dict[str, float]] = [{"run_id": run_id, "timestamp": t0, "elapsed_s": 0.0, **first_snap}]

    next_plot = t0 + args.plot_every_s if args.plot_every_s > 0 else float("inf")
    next_progress = t0 + args.progress_every_s if args.progress_every_s > 0 else float("inf")

    while True:
        time.sleep(args.poll_interval_s)
        now = time.time()
        snap = try_read(args)

        if snap is not None:
            last_success = now
            stale_notice_sent = False
            rows.append({"run_id": run_id, "timestamp": now, "elapsed_s": now - t0, **snap})
        else:
            if not stale_notice_sent:
                print(f"[listener] read failed or returned non-finite values -- "
                      f"starting {args.disconnect_grace_s:g}s disconnect grace timer")
                stale_notice_sent = True
            gap = now - last_success
            if gap >= args.disconnect_grace_s:
                print(f"[listener] no usable connection for {gap:.1f}s -- ending recording "
                      f"({len(rows)} samples over {now - t0:.1f}s)")
                break

        if now >= next_progress:
            last = rows[-1]
            print(f"[listener] samples={len(rows)} elapsed={now - t0:.1f}s "
                  f"temp={last['temp']:.3f} temp_ref={last['temp_ref']:.3f} "
                  f"pid=({last['kp']:g}, {last['ki']:g}, {last['kd']:g})")
            next_progress = now + args.progress_every_s

        if now >= next_plot:
            save_progress(rows, session_dir, args)
            next_plot = now + args.plot_every_s

    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.poll_interval_s <= 0:
        raise ValueError("--poll-interval-s must be > 0")
    if args.disconnect_grace_s <= 0:
        raise ValueError("--disconnect-grace-s must be > 0")

    session = args.session_name or make_run_id(prefix="listener")
    session_dir = Path(args.output_dir) / session
    session_dir.mkdir(parents=True, exist_ok=True)
    run_id = session

    print(f"[listener] host={args.tcp_host}:{args.tcp_port} run_id={run_id}")
    print(f"[listener] waiting for a chamber connection (polling every {args.poll_interval_s:g}s, "
          f"read-only -- no commands will be sent)...")

    rows: list[dict[str, float]] = []
    try:
        first_snap = wait_for_connection(args)
        print("[listener] chamber connection detected -- recording started")
        rows = record_run(first_snap, run_id, session_dir, args)
    except KeyboardInterrupt:
        print("\n[listener] interrupted -- saving whatever was recorded and exiting")

    if not rows:
        print("[listener] no samples recorded -- nothing to save")
        return

    df = save_progress(rows, session_dir, args)
    print(f"[listener] samples={len(df)} duration={df['elapsed_s'].iloc[-1]:.1f}s")
    print(f"[listener] run_samples : {session_dir / 'run_samples.csv'}")
    if not args.no_plot:
        print(f"[listener] plot        : {session_dir / 'trajectory.png'}")


if __name__ == "__main__":
    main()
