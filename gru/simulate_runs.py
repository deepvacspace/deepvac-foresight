#!/usr/bin/env python3
"""Simulate PID tests with GRU + CODESYS PID simulation.

python simulate_runs.py --n-candidates 1000 

python simulate_runs.py --candidate-mode grid \
    --kp-values 4,5,6,7,8 \
    --ki-values 700,850,966,997,1050 \
    --kd-values 0,1,6,11,16,21 
"""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
import math
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from gru.gru_common import (
        DEFAULT_CHECKPOINT,
        DEFAULT_FEATURE_NAMES,
        GRUModel,
        ChamberPID,
        CodesysDiff,
        load_model,
        predict_delta_t1,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import gru_common.py" +
        repr(exc)
    )


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pid_candidates"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Rank PID triplets using GRU + CODESYS PID simulation."
    )

    # Scenario.
    ap.add_argument("--start-temp", type=float, default=27.0)
    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=1200.0)
    ap.add_argument("--dt-s", type=float, default=2.0,
                    help="Synthetic logging/plant step used by the GRU.")
    ap.add_argument("--precondition-ref", type=float, default=None,
                    help="Reference used in the synthetic warmup window. Default: start-temp.")

    # Model/runtime.
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--window-steps", type=int, default=60,
                    help="Fallback only if checkpoint lacks window_steps.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Candidate generation.
    ap.add_argument("--candidate-mode",
                    choices=["random", "grid"], default="random")
    ap.add_argument("--n-candidates", type=int, default=1000,
                    help="Number of random candidates. Ignored for grid mode.")
    ap.add_argument("--integer-candidates", action=argparse.BooleanOptionalAction, default=True,
                    help="Round/floor candidate PID values to integers, matching your PLC workflow.")

    ap.add_argument("--kp-min", type=float, default=1.0)    # 6         min 1
    ap.add_argument("--kp-max", type=float, default=50.0)   # 50        max 50
    ap.add_argument("--ki-min", type=float, default=1.0)  # 200       min 1
    ap.add_argument("--ki-max", type=float,
                    default=1000.0)  # 1000     max 1000
    ap.add_argument("--kd-min", type=float, default=1.0)    # 5         min 1
    ap.add_argument("--kd-max", type=float, default=20.0)   # 20        max 20

    ap.add_argument("--kp-values", default=None,
                    help="Comma-separated kp values for grid mode.")
    ap.add_argument("--ki-values", default=None,
                    help="Comma-separated ki values for grid mode.")
    ap.add_argument("--kd-values", default=None,
                    help="Comma-separated kd values for grid mode.")

    # CODESYS PID settings.
    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--pid-period-s", type=float, default=0.1)

    # Initial PID state.
    ap.add_argument("--initial-i", type=float, default=0.0,
                    help="Initial normalized I state, not percent. Usually 0 for a fresh synthetic test.")
    ap.add_argument("--initial-d", type=float, default=0.0,
                    help="Initial normalized D state, not percent.")
    ap.add_argument("--initial-p", type=float, default=0.0,
                    help="Initial normalized P state, not percent.")

    # Ranking metrics.
    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--near-band", type=float, default=2.0)
    ap.add_argument("--settle-band", type=float, default=0.5)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--validation-top-n", type=int, default=10,
                    help="Validate this many initially top-ranked candidates before final ranking.")
    ap.add_argument("--validation-repeats", type=int, default=5,
                    help="Number of repeat simulations for each validation candidate.")

    ap.add_argument("--w-tail-mae", type=float, default=1.0)
    ap.add_argument("--w-overshoot-rmse", type=float, default=10.0)
    ap.add_argument("--w-overshoot-max", type=float, default=10.0)
    ap.add_argument("--w-tail-std", type=float, default=0.5)
    ap.add_argument("--w-final-error", type=float, default=0.5)
    ap.add_argument("--w-time-to-near", type=float, default=0.001)
    ap.add_argument("--w-invalid", type=float, default=1_000_000.0)

    # Filtering/safety.
    ap.add_argument("--max-abs-temp", type=float, default=100.0,
                    help="Mark candidate invalid if simulated abs(temp) exceeds this value.")
    ap.add_argument("--save-top-trajectories", type=int, default=10,
                    help="Save trajectory rows for top N candidates. Use 0 to disable.")

    return ap


def parse_values(text: str | None, name: str) -> List[float]:
    if text is None or not str(text).strip():
        raise ValueError(f"--{name}-values is required for grid mode")
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"--{name}-values produced no values")
    return values


def maybe_int(v: float, integer: bool) -> float:
    if integer:
        return float(int(math.floor(v)))
    return float(v)


def generate_candidates(args: argparse.Namespace) -> pd.DataFrame:
    rng = random.Random(args.seed)
    rows: List[Dict[str, float]] = []

    if args.candidate_mode == "grid":
        kp_values = parse_values(args.kp_values, "kp")
        ki_values = parse_values(args.ki_values, "ki")
        kd_values = parse_values(args.kd_values, "kd")
        for kp, ki, kd in itertools.product(kp_values, ki_values, kd_values):
            rows.append({
                "kp": maybe_int(kp, args.integer_candidates),
                "ki": maybe_int(ki, args.integer_candidates),
                "kd": maybe_int(kd, args.integer_candidates),
            })
    else:
        seen = set()
        attempts = 0
        max_attempts = max(args.n_candidates * 20, 1000)
        while len(rows) < int(args.n_candidates) and attempts < max_attempts:
            attempts += 1
            kp = maybe_int(rng.uniform(args.kp_min, args.kp_max),
                           args.integer_candidates)
            ki = maybe_int(rng.uniform(args.ki_min, args.ki_max),
                           args.integer_candidates)
            kd = maybe_int(rng.uniform(args.kd_min, args.kd_max),
                           args.integer_candidates)
            key = (kp, ki, kd)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"kp": kp, "ki": ki, "kd": kd})

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["kp", "ki", "kd"]).reset_index(drop=True)
    df.insert(0, "candidate_id", np.arange(len(df), dtype=int))
    if df.empty:
        raise RuntimeError("No PID candidates generated.")
    return df


def make_feature_row(
    feature_names: Sequence[str],
    *,
    temp: float,
    temp_ref: float,
    previous_temp: float,
    dt_s: float,
    u: float,
    u_p: float,
    u_i: float,
    u_d: float,
    kp: float,
    ki: float,
    kd: float,
) -> np.ndarray:
    error = float(temp_ref) - float(temp)
    dt_safe = max(float(dt_s), 1e-9)
    values = {
        "temp": float(temp),
        "temp_ref": float(temp_ref),
        "error": float(error),
        "abs_error": abs(float(error)),
        "dt_s": float(dt_s),
        "temp_velocity": (float(temp) - float(previous_temp)) / dt_safe,
        "error_velocity": 0.0,  # set below if feature exists
        "temp_u": float(u),
        "temp_u_p": float(u_p),
        "temp_u_i": float(u_i),
        "temp_u_d": float(u_d),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
    }
    # Approximate error velocity using temp velocity when temp_ref is constant.
    values["error_velocity"] = -values["temp_velocity"]
    return np.asarray([float(values.get(name, 0.0)) for name in feature_names], dtype=np.float32)


def initialize_feature_window(
    feature_names: Sequence[str],
    window_steps: int,
    start_temp: float,
    precondition_ref: float,
    dt_s: float,
    kp: float,
    ki: float,
    kd: float,
) -> np.ndarray:
    rows = []
    for _ in range(window_steps):
        rows.append(make_feature_row(
            feature_names,
            temp=start_temp,
            temp_ref=precondition_ref,
            previous_temp=start_temp,
            dt_s=dt_s,
            u=0.0,
            u_p=0.0,
            u_i=0.0,
            u_d=0.0,
            kp=kp,
            ki=ki,
            kd=kd,
        ))
    return np.vstack(rows).astype(np.float32)


def run_pid_substeps(
    *,
    pid: ChamberPID,
    diff: CodesysDiff,
    temp_start: float,
    temp_end: float,
    temp_ref: float,
    kp: float,
    ki: float,
    kd: float,
    dt_s: float,
    period_s: float,
    feature_scale: float,
    temp_mode: str,
) -> Dict[str, float]:
    period = max(float(period_s), 1e-6)
    dt = max(float(dt_s), period)
    n_substeps = max(1, int(round(dt / period)))

    last = {
        "u": 0.0,
        "u_p": 0.0,
        "u_i": 0.0,
        "u_d": 0.0,
        "u_norm": 0.0,
        "u_p_norm": 0.0,
        "u_i_norm": 0.0,
        "u_d_norm": 0.0,
        "diff_out": diff.out,
        "n_substeps": n_substeps,
    }

    for k in range(1, n_substeps + 1):
        alpha = k / n_substeps
        if temp_mode == "hold":
            temp_sub = float(temp_start)
        elif temp_mode == "linear":
            temp_sub = float(temp_start) + alpha * \
                (float(temp_end) - float(temp_start))
        else:
            raise ValueError(f"Unknown temp_mode: {temp_mode}")

        diff_out = diff.update(temp_sub)
        u, u_p, u_i, u_d = pid.step(
            enable=True,
            x_target=temp_ref,
            x_measured=temp_sub,
            p_coef=kp,
            i_coef=ki,
            d_coef=kd,
            diff_out=diff_out,
        )
        last = {
            "u": u * feature_scale,
            "u_p": u_p * feature_scale,
            "u_i": u_i * feature_scale,
            "u_d": u_d * feature_scale,
            "u_norm": u,
            "u_p_norm": u_p,
            "u_i_norm": u_i,
            "u_d_norm": u_d,
            "diff_out": diff_out,
            "n_substeps": n_substeps,
        }
    return last


def predict_next(
    *,
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_window: np.ndarray,
    feature_names: Sequence[str],
    device: torch.device,
    temp: float,
    previous_temp: float,
    temp_ref: float,
    dt_s: float,
    terms: Dict[str, float],
    kp: float,
    ki: float,
    kd: float,
) -> Tuple[float, float, np.ndarray]:
    local_window = feature_window.copy()
    local_window[-1, :] = make_feature_row(
        feature_names,
        temp=temp,
        temp_ref=temp_ref,
        previous_temp=previous_temp,
        dt_s=dt_s,
        u=terms["u"],
        u_p=terms["u_p"],
        u_i=terms["u_i"],
        u_d=terms["u_d"],
        kp=kp,
        ki=ki,
        kd=kd,
    )
    delta = predict_delta_t1(model, checkpoint, local_window, device)
    return float(temp) + float(delta), float(delta), local_window


def simulate_candidate(
    *,
    candidate_id: int,
    kp: float,
    ki: float,
    kd: float,
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    window_steps: int,
    args: argparse.Namespace,
    device: torch.device,
    save_trajectory: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame | None]:
    start_temp = float(args.start_temp)
    target_temp = float(args.target_temp)
    duration_s = float(args.duration_s)
    dt_s = float(args.dt_s)
    n_steps = max(1, int(math.ceil(duration_s / dt_s)))
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)
    precondition_ref = start_temp if args.precondition_ref is None else float(
        args.precondition_ref)

    pid = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    pid.p_part = float(args.initial_p)
    pid.i_part = float(args.initial_i)
    pid.d_part = float(args.initial_d)

    diff = CodesysDiff()
    diff.prev_value = start_temp
    diff.filter_out = 0.0
    diff.out = 0.0

    feature_window = initialize_feature_window(
        feature_names=feature_names,
        window_steps=window_steps,
        start_temp=start_temp,
        precondition_ref=precondition_ref,
        dt_s=dt_s,
        kp=kp,
        ki=ki,
        kd=kd,
    )

    current_temp = start_temp
    previous_temp = start_temp
    rows: List[Dict[str, float]] = []
    temps: List[float] = []
    times: List[float] = []
    controls: List[float] = []

    valid = True
    invalid_reason = ""

    for step in range(n_steps):
        terms = run_pid_substeps(
            pid=pid, diff=diff, temp_start=current_temp, temp_end=current_temp,
            temp_ref=target_temp, kp=kp, ki=ki, kd=kd,
            dt_s=dt_s, period_s=args.pid_period_s,
            feature_scale=feature_scale, temp_mode="hold",
        )
        next_temp, pred_delta, pred_window = predict_next(
            model=model, checkpoint=checkpoint, feature_window=feature_window,
            feature_names=feature_names, device=device, temp=current_temp,
            previous_temp=previous_temp, temp_ref=target_temp, dt_s=dt_s,
            terms=terms, kp=kp, ki=ki, kd=kd,
        )

        if not np.isfinite(next_temp) or abs(next_temp) > float(args.max_abs_temp):
            valid = False
            invalid_reason = f"temperature became invalid at step {step}: {next_temp}"
            # Keep a finite placeholder so metrics don't crash.
            next_temp = float(np.nan_to_num(next_temp, nan=args.max_abs_temp,
                              posinf=args.max_abs_temp, neginf=-args.max_abs_temp))

        temps.append(float(next_temp))
        times.append(float((step + 1) * dt_s))
        controls.append(float(terms["u"]))

        if save_trajectory:
            rows.append({
                "candidate_id": candidate_id,
                "step": step + 1,
                "elapsed_s": (step + 1) * dt_s,
                "temp": next_temp,
                "temp_ref": target_temp,
                "error": target_temp - next_temp,
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "u": terms["u"],
                "u_p": terms["u_p"],
                "u_i": terms["u_i"],
                "u_d": terms["u_d"],
                "diff_out": terms["diff_out"],
                "pred_delta": pred_delta,
            })

        next_feature = make_feature_row(
            feature_names,
            temp=next_temp,
            temp_ref=target_temp,
            previous_temp=current_temp,
            dt_s=dt_s,
            u=terms["u"],
            u_p=terms["u_p"],
            u_i=terms["u_i"],
            u_d=terms["u_d"],
            kp=kp,
            ki=ki,
            kd=kd,
        )
        feature_window = np.roll(pred_window, shift=-1, axis=0)
        feature_window[-1, :] = next_feature
        previous_temp = current_temp
        current_temp = next_temp

    metrics = compute_metrics(
        candidate_id=candidate_id,
        kp=kp,
        ki=ki,
        kd=kd,
        times=np.asarray(times, dtype=float),
        temps=np.asarray(temps, dtype=float),
        controls=np.asarray(controls, dtype=float),
        target_temp=target_temp,
        start_temp=start_temp,
        valid=valid,
        invalid_reason=invalid_reason,
        args=args,
    )
    traj_df = pd.DataFrame(rows) if save_trajectory else None
    return metrics, traj_df


def compute_metrics(
    *,
    candidate_id: int,
    kp: float,
    ki: float,
    kd: float,
    times: np.ndarray,
    temps: np.ndarray,
    controls: np.ndarray,
    target_temp: float,
    start_temp: float,
    valid: bool,
    invalid_reason: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    error = target_temp - temps
    abs_error = np.abs(error)

    tail_start = max(0.0, float(args.duration_s) - float(args.tail_window_s))
    tail_mask = times >= tail_start
    if not np.any(tail_mask):
        tail_mask = np.ones_like(times, dtype=bool)

    tail_abs_error = abs_error[tail_mask]
    tail_error = error[tail_mask]
    tail_temp = temps[tail_mask]

    if target_temp <= start_temp:
        # Cooling: overshoot means going below target.
        overshoot = np.maximum(target_temp - temps, 0.0)
    else:
        # Heating: overshoot means going above target.
        overshoot = np.maximum(temps - target_temp, 0.0)

    near_idx = np.where(abs_error <= float(args.near_band))[0]
    settle_idx = np.where(abs_error <= float(args.settle_band))[0]
    time_to_near = float(times[int(near_idx[0])]) if len(
        near_idx) else float(args.duration_s) + 999.0
    time_to_settle = float(times[int(settle_idx[0])]) if len(
        settle_idx) else float(args.duration_s) + 999.0

    tail_mae = float(np.mean(tail_abs_error))
    tail_bias = float(np.mean(tail_error))
    tail_std = float(np.std(tail_temp))
    overshoot_max = float(np.max(overshoot))
    overshoot_rmse = float(np.sqrt(np.mean(np.square(overshoot))))
    final_error = float(error[-1])
    final_abs_error = abs(final_error)
    cost = (
        float(args.w_tail_mae) * tail_mae
        + float(args.w_overshoot_max) * overshoot_max
        + float(args.w_tail_std) * tail_std
        + float(args.w_final_error) * final_abs_error
    )
    if not valid:
        cost += float(args.w_invalid)

    return {
        "candidate_id": int(candidate_id),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "cost": float(cost),
        "valid": bool(valid),
        "invalid_reason": invalid_reason,
        "tail_mae": tail_mae,
        "tail_bias": tail_bias,
        "tail_std": tail_std,
        "overshoot_max": overshoot_max,
        "overshoot_rmse": overshoot_rmse,
        "final_error": final_error,
        "final_abs_error": final_abs_error,
        "time_to_near_s": time_to_near,
        "time_to_settle_s": time_to_settle,
        "mae_full": float(np.mean(abs_error)),
        "min_temp": float(np.min(temps)),
        "max_temp": float(np.max(temps)),
        "end_temp": float(temps[-1]),
    }


def print_top(ranking: pd.DataFrame, top_n: int) -> None:
    cols = [
        "rank", "kp", "ki", "kd", "cost", "tail_mae", "overshoot_max",
        "tail_std", "final_abs_error", "end_temp",
    ]
    print("\n=== Top candidate PID tests ===")
    print(ranking.head(min(10, top_n))[cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))


def rank_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    ranking = rows.drop(columns=["rank"], errors="ignore").sort_values(
        ["valid", "cost", "tail_mae", "overshoot_max"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def aggregate_validation_metrics(validation_rows: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: List[Dict[str, object]] = []
    exclude_from_mean = {"validation_repeat", "initial_rank"}

    for candidate_id, group in validation_rows.groupby("candidate_id", sort=False):
        aggregate: Dict[str, object] = {
            "candidate_id": int(candidate_id),
            "validation_runs": int(len(group)),
            "initial_rank": int(group["initial_rank"].iloc[0]),
            "validated": True,
            "valid": bool(group["valid"].all()),
        }

        invalid_reasons = [
            str(reason)
            for reason in group.get("invalid_reason", pd.Series(dtype=object)).dropna().unique()
            if str(reason)
        ]
        aggregate["invalid_reason"] = "; ".join(invalid_reasons)

        for column in group.columns:
            if column in aggregate or column in exclude_from_mean:
                continue
            if column == "valid" or column == "invalid_reason":
                continue
            if pd.api.types.is_numeric_dtype(group[column]):
                aggregate[column] = float(group[column].mean())

        aggregate["candidate_id"] = int(candidate_id)
        grouped_rows.append(aggregate)

    return pd.DataFrame(grouped_rows)


def validate_top_candidates(
    *,
    initial_ranking: pd.DataFrame,
    args: argparse.Namespace,
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    window_steps: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_n = max(0, int(args.validation_top_n))
    repeats = max(0, int(args.validation_repeats))

    if top_n == 0 or repeats == 0 or initial_ranking.empty:
        final_ranking = initial_ranking.copy()
        final_ranking["initial_rank"] = final_ranking["rank"]
        final_ranking["validation_runs"] = 0
        final_ranking["validated"] = False
        return final_ranking, pd.DataFrame()

    top_candidates = initial_ranking.head(top_n).copy()
    validation_rows: List[Dict[str, object]] = []

    print(
        f"validating top {len(top_candidates)} candidates "
        f"{repeats} times each before final ranking"
    )
    for _, top_row in top_candidates.iterrows():
        for repeat_idx in range(repeats):
            metrics, _ = simulate_candidate(
                candidate_id=int(top_row["candidate_id"]),
                kp=float(top_row["kp"]),
                ki=float(top_row["ki"]),
                kd=float(top_row["kd"]),
                model=model,
                checkpoint=checkpoint,
                feature_names=feature_names,
                window_steps=window_steps,
                args=args,
                device=device,
                save_trajectory=False,
            )
            metrics["initial_rank"] = int(top_row["rank"])
            metrics["validation_repeat"] = int(repeat_idx + 1)
            validation_rows.append(metrics)

    validation_detail = pd.DataFrame(validation_rows)
    validation_aggregate = aggregate_validation_metrics(validation_detail)

    top_candidate_ids = set(top_candidates["candidate_id"].astype(int).tolist())
    non_validated = initial_ranking[
        ~initial_ranking["candidate_id"].astype(int).isin(top_candidate_ids)
    ].copy()
    non_validated["initial_rank"] = non_validated["rank"]
    non_validated["validation_runs"] = 0
    non_validated["validated"] = False

    combined = pd.concat(
        [
            validation_aggregate,
            non_validated.drop(columns=["rank"], errors="ignore"),
        ],
        ignore_index=True,
        sort=False,
    )
    final_ranking = rank_candidates(combined)
    return final_ranking, validation_detail


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    feature_names = list(checkpoint.get(
        "feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    candidates = generate_candidates(args)
    n_candidates = len(candidates)

    print("=== Synthetic PID test planner ===")
    print(f"checkpoint:      {checkpoint_path}")
    print(f"device:          {device}")
    print(f"features:        {feature_names}")
    print(f"window_steps:    {window_steps}")
    print(
        f"scenario:        start={args.start_temp} °C -> target={args.target_temp} °C")
    print(f"duration/dt:     {args.duration_s}s / {args.dt_s}s")
    print(f"candidate mode:  {args.candidate_mode}")
    print(f"candidates:      {n_candidates}")
    print("PID temp mode:   hold")
    print(f"run id:          {run_id}")
    print(f"output dir:      {output_dir}")

    rows: List[Dict[str, float]] = []

    for idx, row in candidates.iterrows():
        if (idx + 1) % 50 == 0 or (idx + 1) == n_candidates:
            print(f"simulated {idx + 1}/{n_candidates} candidates")
        metrics, _ = simulate_candidate(
            candidate_id=int(row["candidate_id"]),
            kp=float(row["kp"]),
            ki=float(row["ki"]),
            kd=float(row["kd"]),
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            window_steps=window_steps,
            args=args,
            device=device,
            save_trajectory=False,
        )
        rows.append(metrics)

    initial_ranking = rank_candidates(pd.DataFrame(rows))
    ranking, validation_detail = validate_top_candidates(
        initial_ranking=initial_ranking,
        args=args,
        model=model,
        checkpoint=checkpoint,
        feature_names=feature_names,
        window_steps=window_steps,
        device=device,
    )

    public_ranking = ranking.drop(
        columns=[
            "control_abs_mean",
            "max_abs_error_full",
            "p90_abs_error_full",
            "invalid_reason",
            "valid",
        ],
        errors="ignore",
    )

    ranking_csv = output_dir / "planned_pid_candidate_ranking.csv"
    public_ranking.to_csv(ranking_csv, index=False)

    validation_csv = None
    if not validation_detail.empty:
        validation_csv = output_dir / "planned_pid_top_validation_runs.csv"
        validation_detail.to_csv(validation_csv, index=False)

    top_traj_csv = None
    if int(args.save_top_trajectories) > 0:
        traj_rows: List[pd.DataFrame] = []
        for _, top_row in ranking.head(int(args.save_top_trajectories)).iterrows():
            _, traj_df = simulate_candidate(
                candidate_id=int(top_row["candidate_id"]),
                kp=float(top_row["kp"]),
                ki=float(top_row["ki"]),
                kd=float(top_row["kd"]),
                model=model,
                checkpoint=checkpoint,
                feature_names=feature_names,
                window_steps=window_steps,
                args=args,
                device=device,
                save_trajectory=True,
            )
            if traj_df is not None and not traj_df.empty:
                traj_df.insert(0, "rank", int(top_row["rank"]))
                traj_rows.append(traj_df)
        if traj_rows:
            top_traj = pd.concat(traj_rows, ignore_index=True)
            top_traj_csv = output_dir / "planned_pid_top_trajectories.csv"
            top_traj.to_csv(top_traj_csv, index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "feature_names": feature_names,
        "window_steps": window_steps,
        "scenario": {
            "start_temp": float(args.start_temp),
            "target_temp": float(args.target_temp),
            "duration_s": float(args.duration_s),
            "dt_s": float(args.dt_s),
            "precondition_ref": float(args.start_temp if args.precondition_ref is None else args.precondition_ref),
        },
        "candidate_mode": args.candidate_mode,
        "n_candidates": int(n_candidates),
        "validation": {
            "top_n": int(args.validation_top_n),
            "repeats": int(args.validation_repeats),
            "runs_csv": str(validation_csv) if validation_csv else None,
        },
        "bounds": {
            "kp": [float(args.kp_min), float(args.kp_max)],
            "ki": [float(args.ki_min), float(args.ki_max)],
            "kd": [float(args.kd_min), float(args.kd_max)],
        },
        "run_id": run_id,
        "cost_weights": {
            "tail_mae": float(args.w_tail_mae),
            "overshoot_max": float(args.w_overshoot_max),
            "tail_std": float(args.w_tail_std),
            "final_error": float(args.w_final_error),
        },
        "ranking_csv": str(ranking_csv),
        "validation_csv": str(validation_csv) if validation_csv else None,
        "top_trajectories_csv": str(top_traj_csv) if top_traj_csv else None,
        "top_candidates": public_ranking.head(10).to_dict(orient="records"),
    }
    summary_json = output_dir / "planned_pid_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_top(ranking, int(args.top_n))
    print("\n=== Saved ===")
    print(f"ranking csv:          {ranking_csv}")
    if validation_csv:
        print(f"validation csv:       {validation_csv}")
    if top_traj_csv:
        print(f"top trajectories csv: {top_traj_csv}")
    print(f"summary json:         {summary_json}")


if __name__ == "__main__":
    main()
