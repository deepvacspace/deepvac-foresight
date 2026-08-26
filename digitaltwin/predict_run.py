#!/usr/bin/env python3
"""Predict how a candidate PID would run, without running it.

No chamber connection and no run history are required -- only a trained checkpoint
and the candidate you want to try:

    python -m digitaltwin.predict_run --kp 6 --ki 997 --kd 16

That holds one triplet for the whole run and seeds the model's starting window with
digitaltwin.twin_acceptance.pid_driven_start: a settled chamber's temperature held
constant while a real ChamberPID/CodesysDiff is stepped forward, so the seeded
control terms reflect how long the error has actually been sitting rather than
assuming a controller that has produced exactly zero output the whole time.
Validated on 8 held-out historical runs at ~1.3 degC median whole-run MAE, versus
~4.9 degC for the flat zero-control window and ~0.8 degC for a real warm start
(see that function's docstring, and digitaltwin/twin_acceptance.py --mode replay
--start-mode cold/warm).

If a real run_samples.csv is available -- your own past run, or any run recorded
near the same starting condition -- pass --context-csv to warm-start from it
instead. That is strictly more accurate; a warm start from a genuinely unrelated
run is not, and can be worse than the no-telemetry estimate (see
digitaltwin/twin_acceptance.py's module docstring for the measurement).

--model-family only picks the default --checkpoint when --checkpoint isn't
passed explicitly; the model actually used is always whichever family the
checkpoint itself is stamped with.

Example, warm-started from a real run:

    python -m digitaltwin.predict_run --kp 6 --ki 997 --kd 16 \
        --context-csv experiments/run_history/run_1776723160_994c5aad/run_samples.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import make_run_id
from deepvac.datasets import prepare_run_dataframe

from digitaltwin.common import load_model
from digitaltwin.model import MODEL_CLASSES
from digitaltwin.twin_acceptance import (
    describe_trajectory,
    simulate_twin,
    trajectory_cost,
    warm_start_at,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Predict a whole-run trajectory for a candidate PID, no chamber run needed."
    )

    ap.add_argument("--model-family", choices=sorted(MODEL_CLASSES), default="gru",
                    help="Only used to pick the default --checkpoint below.")
    ap.add_argument("--checkpoint", default=None,
                    help="Default: <script-dir>/<model-family>/validation_t1/<model-family>_t1.pt.")
    ap.add_argument("--output-dir", default=None,
                    help="Default: <script-dir>/<model-family>/predict_run.")
    ap.add_argument("--session-name", default=None, help="Output subfolder name. Default: generated id.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--window-steps", type=int, default=60, help="Fallback only if the checkpoint lacks it.")
    ap.add_argument("--no-plot", action="store_true")

    ap.add_argument("--kp", type=float, required=True)
    ap.add_argument("--ki", type=float, required=True)
    ap.add_argument("--kd", type=float, required=True)
    ap.add_argument("--start-temp", type=float, default=25.0)
    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=1200.0)
    ap.add_argument("--dt-s", type=float, default=2.0)

    ap.add_argument(
        "--context-csv", default=None,
        help="A real run_samples.csv to warm-start from instead of the no-telemetry estimate.",
    )
    ap.add_argument(
        "--context-offset", type=int, default=None,
        help="Row offset into --context-csv to seed from. Default: window_steps - 1 (that run's own start).",
    )

    ap.add_argument("--pid-period-s", type=float, default=0.1)
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--max-abs-temp", type=float, default=100.0)

    ap.add_argument("--near-band", type=float, default=2.0)
    ap.add_argument("--settle-band", type=float, default=0.5)
    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    return ap


def load_context(args: argparse.Namespace, feature_names: list, window_steps: int):
    """Real warm-start window/terms from --context-csv, or None to fall back to the
    no-telemetry PID-driven estimate."""
    if not args.context_csv:
        return None, None, ("no real telemetry: cold PID-driven estimate "
                            "(~1.3 degC median whole-run MAE on held-out validation)")

    loader_args = argparse.Namespace(min_samples=1, min_duration_s=0.0)
    df, _ = prepare_run_dataframe(Path(args.context_csv), loader_args)
    offset = args.context_offset if args.context_offset is not None else window_steps - 1
    if offset < window_steps - 1 or offset >= len(df):
        raise ValueError(
            f"--context-offset must be in [{window_steps - 1}, {len(df) - 1}] for this CSV, got {offset}"
        )
    window, terms = warm_start_at(df, feature_names, window_steps, offset)
    note = (f"warm-started from real telemetry: {args.context_csv} @ row {offset} "
            "(~0.35-0.8 degC median whole-run MAE when representative of the actual "
            "starting condition; worse if the context is from an unrelated situation)")
    return window, terms, note


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.checkpoint is None:
        args.checkpoint = str(
            SCRIPT_DIR / args.model_family / "validation_t1" / f"{args.model_family}_t1.pt"
        )
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / args.model_family / "predict_run")

    if args.duration_s <= 0 or args.dt_s <= 0:
        raise ValueError("--duration-s and --dt-s must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", []))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    warm_window, warm_terms, context_note = load_context(args, feature_names, window_steps)

    print(f"checkpoint : {args.checkpoint}")
    print(f"candidate  : kp={args.kp:g} ki={args.ki:g} kd={args.kd:g}, "
          f"{args.start_temp:g} -> {args.target_temp:g} deg C over {args.duration_s:g}s")
    print(f"context    : {context_note}")

    pred = simulate_twin(
        model=model, checkpoint=checkpoint, feature_names=feature_names,
        window_steps=window_steps, device=device, args=args,
        start_temp=float(args.start_temp), target_temp=float(args.target_temp),
        duration_s=float(args.duration_s), dt_s=float(args.dt_s),
        pid_schedule=[(float(args.kp), float(args.ki), float(args.kd))],
        warm_window=warm_window, warm_terms=warm_terms,
    )

    times = pred["elapsed_s"].to_numpy(dtype=float)
    temps = pred["temp"].to_numpy(dtype=float)
    diverged = not bool(pred["valid"].all())
    if diverged:
        first_bad = pred.loc[~pred["valid"], "elapsed_s"].iloc[0]
        print(f"\nWARNING: prediction diverged at t={first_bad:.0f}s "
              f"({100.0 * pred['valid'].mean():.0f}% of the run stayed within --max-abs-temp)")

    desc = describe_trajectory(times, temps, float(args.target_temp), args)
    cost = trajectory_cost(times, temps, float(args.target_temp), args)

    session = args.session_name or make_run_id(prefix="predict")
    session_dir = Path(args.output_dir) / session
    session_dir.mkdir(parents=True, exist_ok=True)

    pred.to_csv(session_dir / "trajectory.csv", index=False)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate": {"kp": args.kp, "ki": args.ki, "kd": args.kd},
        "start_temp": args.start_temp,
        "target_temp": args.target_temp,
        "duration_s": args.duration_s,
        "dt_s": args.dt_s,
        "context": context_note,
        "diverged": diverged,
        **desc,
        "cost": cost,
    }
    (session_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_prediction(times, temps, args, session_dir / "trajectory.png")

    print("\n=== Predicted outcome ===")
    print(f"final temp       : {desc['final_temp']:.3f} deg C")
    print(f"overshoot        : {desc['overshoot']:.3f} deg C")
    print(f"tail MAE         : {desc['tail_mae']:.3f} deg C (last {args.tail_window_s:g}s)")
    print(f"time to near     : {fmt_time(desc['time_to_near_s'])} (band {args.near_band:g} deg C)")
    print(f"time to settle   : {fmt_time(desc['time_to_settle_s'])} (band {args.settle_band:g} deg C)")
    print(f"cost             : {'n/a (never entered the entry band)' if cost is None else f'{cost:.3f}'}")
    print(f"\ntrajectory : {session_dir / 'trajectory.csv'}")
    print(f"report     : {session_dir / 'report.json'}")


def fmt_time(value) -> str:
    return "never" if value is None else f"{value:.0f}s"


def plot_prediction(times: np.ndarray, temps: np.ndarray, args: argparse.Namespace, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, temps, label="predicted temp")
    ax.axhline(args.target_temp, color="#888888", linestyle="--", label="target")
    ax.axhspan(args.target_temp - args.near_band, args.target_temp + args.near_band,
              color="#888888", alpha=0.1, label=f"+-{args.near_band:g} deg C band")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(f"Predicted run: kp={args.kp:g} ki={args.ki:g} kd={args.kd:g}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
