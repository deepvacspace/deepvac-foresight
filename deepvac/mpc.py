"""Model-predictive-control rollout and CEM/random-shooting optimizer.

gru/mpc_gru.py and lstm/mpc_lstm.py implemented this receding-horizon MPC
loop independently, byte-for-byte identical apart from two seams: how the
next temperature delta is predicted (predict_fn: GRU vs LSTM forward pass)
and how a candidate rollout's cost is scored (cost_fn: each model script
tunes its own cost-weight design -- GRU's includes a "motion" smoothness
term, LSTM's includes a control-change penalty -- these are genuinely
different, not duplicated, so they stay local to each mpc_*.py and are
passed in here rather than merged).

predict_fn signature:  (model, checkpoint, feature_window, device) -> float
cost_fn signature:     (*, temps, candidate_pid, previous_pid, start_temp,
                         target_temp, valid, args) -> Dict[str, float]
                        (must include a "cost" key; candidate_pid/previous_pid
                        may be ignored by cost designs that don't need them)
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# ChamberPID/CodesysDiff are the physical PID/differentiator emulation used
# by both the GRU and LSTM MPC schedulers; they live in gru.gru_common
# because that's where they were first extracted, not because they are
# GRU-specific. Imported here rather than re-implemented.
from gru.gru_common import ChamberPID, CodesysDiff

from deepvac.pid import clip_pid, pid_bounds

PredictFn = Callable[[Any, dict[str, object], np.ndarray, torch.device], float]
CostFn = Callable[..., dict[str, float]]


@dataclass
class SimState:
    elapsed_s: float
    temp: float
    previous_temp: float
    feature_window: np.ndarray
    pid: Any  # gru.gru_common.ChamberPID
    diff: Any  # gru.gru_common.CodesysDiff
    kp: float
    ki: float
    kd: float


@dataclass
class CandidateTable:
    path: Path
    features: np.ndarray
    pids: np.ndarray
    history_score: np.ndarray
    rows: pd.DataFrame


def add_common_mpc_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared verbatim by mpc_gru.py and mpc_lstm.py's parsers. Each
    script still builds its own parser (checkpoint defaults, horizon/hold
    defaults, and cost weights differ) and calls this to add the rest."""
    ap.add_argument("--start-temp", type=float, default=27.0)
    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=1200.0)
    ap.add_argument("--dt-s", type=float, default=2.0, help="Model/logging step in seconds.")
    ap.add_argument("--precondition-ref", type=float, default=None,
                    help="Reference used in the synthetic warmup window. Default: start-temp.")

    ap.add_argument("--window-steps", type=int, default=60,
                    help="Fallback only if checkpoint lacks window_steps.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--kp-min", type=float, default=1.0)
    ap.add_argument("--kp-max", type=float, default=50.0)
    ap.add_argument("--ki-min", type=float, default=1.0)
    ap.add_argument("--ki-max", type=float, default=1000.0)
    ap.add_argument("--kd-min", type=float, default=1.0)
    ap.add_argument("--kd-max", type=float, default=20.0)
    ap.add_argument("--kp-init", type=float, default=6.0)
    ap.add_argument("--ki-init", type=float, default=997.0)
    ap.add_argument("--kd-init", type=float, default=16.0)
    ap.add_argument("--initial-i", type=float, default=0.0, help="Initial normalized I state, not percent.")
    ap.add_argument("--initial-d", type=float, default=0.0, help="Initial normalized D state, not percent.")
    ap.add_argument("--initial-p", type=float, default=0.0, help="Initial normalized P state, not percent.")

    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--pid-period-s", type=float, default=0.1)

    ap.add_argument("--optimizer", choices=["cem", "random"], default="cem")
    ap.add_argument("--cem-population", type=int, default=256)
    ap.add_argument("--cem-elite-frac", type=float, default=0.12)
    ap.add_argument("--cem-iterations", type=int, default=3)
    ap.add_argument("--cem-min-std-frac", type=float, default=0.03,
                    help="Minimum CEM std as fraction of each PID bound range.")
    ap.add_argument("--include-current-candidate", action=argparse.BooleanOptionalAction, default=True,
                    help="Always evaluate keeping the current PID as one candidate.")
    ap.add_argument("--apply-margin", type=float, default=0.0,
                    help="Only change PID if best_cost < current_pid_cost - margin.")
    ap.add_argument("--max-pid-delta-frac", type=float, default=1.0,
                    help="Limit one MPC update to this fraction of each PID range. 1.0 disables the limit.")

    ap.add_argument("--history-candidates", type=int, default=24,
                    help="Number of nearest historical PID triplets to inject into each optimizer population.")
    ap.add_argument("--history-neighbor-pool", type=int, default=250,
                    help="Nearest historical states considered before ranking by historical score.")
    ap.add_argument("--history-score-weight", type=float, default=0.15,
                    help="Weight of normalized historical cost when ranking nearest history candidates.")
    ap.add_argument("--history-temp-scale", type=float, default=10.0)
    ap.add_argument("--history-error-scale", type=float, default=10.0)
    ap.add_argument("--history-time-scale", type=float, default=600.0)
    ap.add_argument("--history-velocity-scale", type=float, default=0.2)

    # NOTE: --w-near-std is intentionally NOT here -- gru's and lstm's default
    # differ (1.0 vs 2.0), so each script's build_arg_parser() adds it itself.
    ap.add_argument("--w-invalid", type=float, default=1_000_000.0)

    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--near-band", type=float, default=2.0)
    ap.add_argument("--settle-band", type=float, default=0.5)
    ap.add_argument("--max-abs-temp", type=float, default=100.0)

    ap.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print-every-decision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print-optimizer-progress", action=argparse.BooleanOptionalAction, default=True,
                    help="Print progress while evaluating candidates inside each MPC decision.")
    ap.add_argument("--optimizer-progress-every", type=int, default=64,
                    help="Print optimizer progress after this many candidates. Use 0 for iteration-only logs.")


# -----------------------------------------------------------------------------
# Feature construction and plant step
# -----------------------------------------------------------------------------


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
        "error_velocity": -(float(temp) - float(previous_temp)) / dt_safe,
        "temp_u": float(u),
        "temp_u_p": float(u_p),
        "temp_u_i": float(u_i),
        "temp_u_d": float(u_d),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
    }
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
    pid: Any,
    diff: Any,
    temp_start: float,
    temp_end: float,
    temp_ref: float,
    kp: float,
    ki: float,
    kd: float,
    dt_s: float,
    period_s: float,
    feature_scale: float,
    temp_mode: str = "hold",
) -> dict[str, float]:
    period = max(float(period_s), 1e-6)
    dt = max(float(dt_s), period)
    n_substeps = max(1, int(round(dt / period)))

    last = {
        "u": 0.0, "u_p": 0.0, "u_i": 0.0, "u_d": 0.0,
        "u_norm": 0.0, "u_p_norm": 0.0, "u_i_norm": 0.0, "u_d_norm": 0.0,
        "diff_out": diff.out, "n_substeps": n_substeps,
    }

    for k in range(1, n_substeps + 1):
        alpha = k / n_substeps
        if temp_mode == "hold":
            temp_sub = float(temp_start)
        elif temp_mode == "linear":
            temp_sub = float(temp_start) + alpha * (float(temp_end) - float(temp_start))
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
    predict_fn: PredictFn,
    model: Any,
    checkpoint: dict[str, object],
    feature_window: np.ndarray,
    feature_names: Sequence[str],
    device: torch.device,
    temp: float,
    previous_temp: float,
    temp_ref: float,
    dt_s: float,
    terms: dict[str, float],
    kp: float,
    ki: float,
    kd: float,
) -> tuple[float, float, np.ndarray]:
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
    delta = predict_fn(model, checkpoint, local_window, device)
    return float(temp) + float(delta), float(delta), local_window


def step_state(
    *,
    state: SimState,
    model: Any,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    target_temp: float,
    dt_s: float,
    pid_period_s: float,
    feature_scale: float,
    max_abs_temp: float,
    predict_fn: PredictFn,
) -> tuple[SimState, dict[str, float], bool, str]:
    terms = run_pid_substeps(
        pid=state.pid,
        diff=state.diff,
        temp_start=state.temp,
        temp_end=state.temp,
        temp_ref=target_temp,
        kp=state.kp,
        ki=state.ki,
        kd=state.kd,
        dt_s=dt_s,
        period_s=pid_period_s,
        feature_scale=feature_scale,
        temp_mode="hold",
    )

    next_temp, pred_delta, pred_window = predict_next(
        predict_fn=predict_fn,
        model=model,
        checkpoint=checkpoint,
        feature_window=state.feature_window,
        feature_names=feature_names,
        device=device,
        temp=state.temp,
        previous_temp=state.previous_temp,
        temp_ref=target_temp,
        dt_s=dt_s,
        terms=terms,
        kp=state.kp,
        ki=state.ki,
        kd=state.kd,
    )

    valid = True
    invalid_reason = ""
    if not np.isfinite(next_temp) or abs(next_temp) > float(max_abs_temp):
        valid = False
        invalid_reason = f"invalid predicted temp: {next_temp}"
        next_temp = float(np.nan_to_num(
            next_temp,
            nan=max_abs_temp,
            posinf=max_abs_temp,
            neginf=-max_abs_temp,
        ))

    next_feature = make_feature_row(
        feature_names,
        temp=next_temp,
        temp_ref=target_temp,
        previous_temp=state.temp,
        dt_s=dt_s,
        u=terms["u"],
        u_p=terms["u_p"],
        u_i=terms["u_i"],
        u_d=terms["u_d"],
        kp=state.kp,
        ki=state.ki,
        kd=state.kd,
    )
    next_window = np.roll(pred_window, shift=-1, axis=0)
    next_window[-1, :] = next_feature

    new_state = SimState(
        elapsed_s=state.elapsed_s + dt_s,
        temp=next_temp,
        previous_temp=state.temp,
        feature_window=next_window,
        pid=state.pid,
        diff=state.diff,
        kp=state.kp,
        ki=state.ki,
        kd=state.kd,
    )

    info = dict(terms)
    info["pred_delta"] = float(pred_delta)
    return new_state, info, valid, invalid_reason


# -----------------------------------------------------------------------------
# History-seeded candidates
# -----------------------------------------------------------------------------


def candidate_feature_vector(
    *,
    temp: float,
    target_temp: float,
    elapsed_s: float,
    temp_velocity: float,
    current_pid: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    lo, hi = pid_bounds(args)
    pid_denom = np.maximum(hi - lo, 1e-9)
    error = float(target_temp) - float(temp)
    return np.asarray([
        float(temp) / max(float(args.history_temp_scale), 1e-9),
        error / max(float(args.history_error_scale), 1e-9),
        abs(error) / max(float(args.history_error_scale), 1e-9),
        float(elapsed_s) / max(float(args.history_time_scale), 1e-9),
        float(temp_velocity) / max(float(args.history_velocity_scale), 1e-9),
        (float(current_pid[0]) - lo[0]) / pid_denom[0],
        (float(current_pid[1]) - lo[1]) / pid_denom[1],
        (float(current_pid[2]) - lo[2]) / pid_denom[2],
    ], dtype=float)


def load_candidate_table(args: argparse.Namespace) -> CandidateTable | None:
    table_arg = str(getattr(args, "candidate_table", "") or "").strip()
    if not table_arg or int(getattr(args, "history_candidates", 0)) <= 0:
        return None

    path = Path(table_arg)
    if not path.exists():
        print(f"[history] candidate table not found, history seeding disabled: {path}")
        return None

    df = pd.read_csv(path)
    required = {
        "temp", "target_temp", "error", "abs_error", "elapsed_s", "temp_velocity",
        "current_kp", "current_ki", "current_kd", "kp", "ki", "kd", "history_score",
    }
    missing = required.difference(df.columns)
    if missing:
        print(f"[history] candidate table missing columns {sorted(missing)}, history seeding disabled: {path}")
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).reset_index(drop=True)
    if df.empty:
        print(f"[history] candidate table has no usable rows, history seeding disabled: {path}")
        return None

    lo, hi = pid_bounds(args)
    pids = df[["kp", "ki", "kd"]].to_numpy(dtype=float)
    keep = np.all((pids >= lo) & (pids <= hi), axis=1)
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        print(f"[history] candidate table has no rows inside current PID bounds, history seeding disabled: {path}")
        return None

    features = []
    for row in df.itertuples(index=False):
        current_pid = np.asarray([row.current_kp, row.current_ki, row.current_kd], dtype=float)
        features.append(candidate_feature_vector(
            temp=float(row.temp),
            target_temp=float(row.target_temp),
            elapsed_s=float(row.elapsed_s),
            temp_velocity=float(row.temp_velocity),
            current_pid=current_pid,
            args=args,
        ))

    table = CandidateTable(
        path=path,
        features=np.vstack(features),
        pids=df[["kp", "ki", "kd"]].to_numpy(dtype=float),
        history_score=df["history_score"].to_numpy(dtype=float),
        rows=df,
    )
    print(f"[history] loaded {len(df)} candidate rows from {path}")
    return table


def select_history_candidates(
    *,
    state: SimState,
    args: argparse.Namespace,
    candidate_table: CandidateTable | None,
) -> np.ndarray:
    if candidate_table is None or int(args.history_candidates) <= 0:
        return np.empty((0, 3), dtype=float)

    current_pid = np.asarray([state.kp, state.ki, state.kd], dtype=float)
    temp_velocity = (float(state.temp) - float(state.previous_temp)) / max(float(args.dt_s), 1e-9)
    query = candidate_feature_vector(
        temp=float(state.temp),
        target_temp=float(args.target_temp),
        elapsed_s=float(state.elapsed_s),
        temp_velocity=temp_velocity,
        current_pid=current_pid,
        args=args,
    )

    diff = candidate_table.features - query.reshape(1, -1)
    distance = np.sqrt(np.mean(np.square(diff), axis=1))
    pool_n = min(len(distance), max(int(args.history_candidates), int(args.history_neighbor_pool)))
    if pool_n <= 0:
        return np.empty((0, 3), dtype=float)

    pool_idx = np.argpartition(distance, pool_n - 1)[:pool_n]
    scores = candidate_table.history_score[pool_idx]
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size:
        score_min = float(np.min(finite_scores))
        score_p90 = float(np.percentile(finite_scores, 90))
        denom = max(score_p90 - score_min, 1e-9)
        score_norm = np.clip((scores - score_min) / denom, 0.0, 10.0)
    else:
        score_norm = np.zeros_like(scores, dtype=float)

    rank = distance[pool_idx] + float(args.history_score_weight) * score_norm
    ordered = pool_idx[np.argsort(rank)]

    selected: list[np.ndarray] = []
    seen = set()
    for idx in ordered:
        pid = clip_pid(candidate_table.pids[int(idx)], args)
        key = tuple(float(x) for x in pid)
        if key in seen:
            continue
        seen.add(key)
        selected.append(pid)
        if len(selected) >= int(args.history_candidates):
            break

    if not selected:
        return np.empty((0, 3), dtype=float)
    return np.vstack(selected).astype(float)


def inject_seed_candidates(
    samples: np.ndarray,
    seed_candidates: np.ndarray,
    *,
    start_idx: int,
) -> int:
    if seed_candidates.size == 0 or samples.size == 0:
        return 0
    n_room = max(0, len(samples) - int(start_idx))
    n = min(n_room, len(seed_candidates))
    if n <= 0:
        return 0
    samples[int(start_idx):int(start_idx) + n, :] = seed_candidates[:n, :]
    return n


# -----------------------------------------------------------------------------
# MPC objective and optimizer
# -----------------------------------------------------------------------------


def overshoot_array(temps: np.ndarray, *, start_temp: float, target_temp: float) -> np.ndarray:
    if target_temp <= start_temp:
        return np.maximum(target_temp - temps, 0.0)  # cooling: below target
    return np.maximum(temps - target_temp, 0.0)      # heating: above target


def rollout_constant_pid(
    *,
    initial_state: SimState,
    candidate_pid: np.ndarray,
    model: Any,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    horizon_steps: int,
    predict_fn: PredictFn,
    cost_fn: CostFn,
) -> tuple[dict[str, float], list[float]]:
    # Important: deep-copy PID and diff so horizon rollouts do not mutate the real simulation state.
    state = copy.deepcopy(initial_state)
    pid_vec = clip_pid(candidate_pid, args)
    state.kp, state.ki, state.kd = float(pid_vec[0]), float(pid_vec[1]), float(pid_vec[2])

    temps: list[float] = []
    valid = True
    for _ in range(horizon_steps):
        state, _, step_valid, _ = step_state(
            state=state,
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            device=device,
            target_temp=float(args.target_temp),
            dt_s=float(args.dt_s),
            pid_period_s=float(args.pid_period_s),
            feature_scale=max(abs(float(args.control_feature_scale)), 1e-9),
            max_abs_temp=float(args.max_abs_temp),
            predict_fn=predict_fn,
        )
        temps.append(float(state.temp))
        valid = valid and step_valid
        if not step_valid:
            break

    previous_pid = np.asarray([initial_state.kp, initial_state.ki, initial_state.kd], dtype=float)
    metrics = cost_fn(
        temps=np.asarray(temps, dtype=float),
        candidate_pid=pid_vec,
        previous_pid=previous_pid,
        start_temp=float(args.start_temp),
        target_temp=float(args.target_temp),
        valid=valid,
        args=args,
    )
    metrics["kp"] = float(pid_vec[0])
    metrics["ki"] = float(pid_vec[1])
    metrics["kd"] = float(pid_vec[2])
    return metrics, temps


def sample_candidates_random(
    rng: np.random.Generator,
    n: int,
    args: argparse.Namespace,
    center: np.ndarray | None = None,
) -> np.ndarray:
    lo, hi = pid_bounds(args)
    if center is None or float(args.max_pid_delta_frac) >= 1.0:
        c_lo, c_hi = lo, hi
    else:
        full_range = hi - lo
        radius = np.maximum(float(args.max_pid_delta_frac) * full_range, 1e-9)
        c_lo = np.maximum(lo, center - radius)
        c_hi = np.minimum(hi, center + radius)
    samples = rng.uniform(c_lo, c_hi, size=(n, 3))
    samples = np.floor(samples + 0.5)
    return np.clip(samples, lo, hi)


def optimize_pid_for_state(
    *,
    state: SimState,
    model: Any,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    rng: np.random.Generator,
    predict_fn: PredictFn,
    cost_fn: CostFn,
    candidate_table: CandidateTable | None = None,
    decision_idx: int | None = None,
) -> dict[str, float]:
    lo, hi = pid_bounds(args)
    current_pid = np.asarray([state.kp, state.ki, state.kd], dtype=float)
    horizon_steps = max(1, int(math.ceil(float(args.mpc_horizon_s) / float(args.dt_s))))
    population = max(4, int(args.cem_population))
    progress_enabled = bool(getattr(args, "print_optimizer_progress", True))
    progress_every = max(0, int(getattr(args, "optimizer_progress_every", 64)))
    decision_label = f"{int(decision_idx):03d}" if decision_idx is not None else "---"
    history_candidates = select_history_candidates(
        state=state,
        args=args,
        candidate_table=candidate_table,
    )

    # Current PID baseline, used both for comparison and safety gating.
    progress_t0 = time.perf_counter()
    if progress_enabled:
        print(
            f"[decision {decision_label}] optimizing "
            f"t={state.elapsed_s:.1f}s temp={state.temp:.4f} "
            f"current_pid=({current_pid[0]:.0f}, {current_pid[1]:.0f}, {current_pid[2]:.0f}) "
            f"horizon_steps={horizon_steps} "
            f"history_seeds={len(history_candidates)}",
            flush=True,
        )
    current_metrics, _ = rollout_constant_pid(
        initial_state=state,
        candidate_pid=current_pid,
        model=model,
        checkpoint=checkpoint,
        feature_names=feature_names,
        device=device,
        args=args,
        horizon_steps=horizon_steps,
        predict_fn=predict_fn,
        cost_fn=cost_fn,
    )
    current_cost = float(current_metrics["cost"])
    if progress_enabled:
        print(f"[decision {decision_label}] baseline cost={current_cost:.4f}", flush=True)

    evaluated_rows: list[dict[str, float]] = []

    if args.optimizer == "random":
        samples = sample_candidates_random(rng, population, args, center=current_pid)
        seed_start = 0
        if bool(args.include_current_candidate):
            samples[0, :] = current_pid
            seed_start = 1
        inject_seed_candidates(samples, history_candidates, start_idx=seed_start)
        best_cost_so_far = current_cost
        best_pid_so_far = current_pid.copy()
        for i, sample in enumerate(samples):
            metrics, _ = rollout_constant_pid(
                initial_state=state,
                candidate_pid=sample,
                model=model,
                checkpoint=checkpoint,
                feature_names=feature_names,
                device=device,
                args=args,
                horizon_steps=horizon_steps,
                predict_fn=predict_fn,
                cost_fn=cost_fn,
            )
            metrics["sample_idx"] = i
            evaluated_rows.append(metrics)
            if float(metrics["cost"]) < best_cost_so_far:
                best_cost_so_far = float(metrics["cost"])
                best_pid_so_far = np.asarray([metrics["kp"], metrics["ki"], metrics["kd"]], dtype=float)
            if progress_enabled and progress_every > 0 and ((i + 1) % progress_every == 0 or i + 1 == len(samples)):
                elapsed_ms = 1000.0 * (time.perf_counter() - progress_t0)
                print(
                    f"[decision {decision_label}] random {i + 1}/{len(samples)} "
                    f"best={best_cost_so_far:.4f} "
                    f"pid=({best_pid_so_far[0]:.0f}, {best_pid_so_far[1]:.0f}, {best_pid_so_far[2]:.0f}) "
                    f"elapsed={elapsed_ms:.0f}ms",
                    flush=True,
                )
    else:
        # CEM initialization around the current PID, but broad enough to explore.
        mean = np.clip(current_pid, lo, hi)
        std = 0.40 * (hi - lo)
        min_std = np.maximum(float(args.cem_min_std_frac) * (hi - lo), 1e-9)
        elite_n = max(2, int(round(population * float(args.cem_elite_frac))))
        total_iters = max(1, int(args.cem_iterations))

        for cem_iter in range(total_iters):
            iter_t0 = time.perf_counter()
            samples = rng.normal(loc=mean, scale=std, size=(population, 3))
            samples = np.clip(samples, lo, hi)
            samples = np.floor(samples + 0.5)
            samples = np.clip(samples, lo, hi)
            seed_start = 0
            if bool(args.include_current_candidate):
                samples[0, :] = current_pid
                seed_start = 1
            inject_seed_candidates(samples, history_candidates, start_idx=seed_start)

            iter_rows: list[dict[str, float]] = []
            best_cost_so_far = float("inf")
            best_pid_so_far = current_pid.copy()
            for i, sample in enumerate(samples):
                metrics, _ = rollout_constant_pid(
                    initial_state=state,
                    candidate_pid=sample,
                    model=model,
                    checkpoint=checkpoint,
                    feature_names=feature_names,
                    device=device,
                    args=args,
                    horizon_steps=horizon_steps,
                    predict_fn=predict_fn,
                    cost_fn=cost_fn,
                )
                metrics["sample_idx"] = i
                metrics["cem_iter"] = cem_iter + 1
                iter_rows.append(metrics)
                if float(metrics["cost"]) < best_cost_so_far:
                    best_cost_so_far = float(metrics["cost"])
                    best_pid_so_far = np.asarray([metrics["kp"], metrics["ki"], metrics["kd"]], dtype=float)
                if progress_enabled and progress_every > 0 and ((i + 1) % progress_every == 0 or i + 1 == len(samples)):
                    elapsed_ms = 1000.0 * (time.perf_counter() - progress_t0)
                    print(
                        f"[decision {decision_label}] cem {cem_iter + 1}/{total_iters} "
                        f"{i + 1}/{len(samples)} best={best_cost_so_far:.4f} "
                        f"pid=({best_pid_so_far[0]:.0f}, {best_pid_so_far[1]:.0f}, {best_pid_so_far[2]:.0f}) "
                        f"elapsed={elapsed_ms:.0f}ms",
                        flush=True,
                    )

            iter_rows.sort(key=lambda r: float(r["cost"]))
            elites = np.asarray([[r["kp"], r["ki"], r["kd"]] for r in iter_rows[:elite_n]], dtype=float)
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), min_std)
            evaluated_rows.extend(iter_rows)
            if progress_enabled:
                iter_ms = 1000.0 * (time.perf_counter() - iter_t0)
                iter_best = iter_rows[0]
                print(
                    f"[decision {decision_label}] cem {cem_iter + 1}/{total_iters} done "
                    f"best={float(iter_best['cost']):.4f} "
                    f"pid=({float(iter_best['kp']):.0f}, {float(iter_best['ki']):.0f}, {float(iter_best['kd']):.0f}) "
                    f"next_mean=({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}) "
                    f"iter={iter_ms:.0f}ms",
                    flush=True,
                )

    evaluated_rows.sort(key=lambda r: float(r["cost"]))
    best = dict(evaluated_rows[0])
    best_pid = np.asarray([best["kp"], best["ki"], best["kd"]], dtype=float)

    # Safety gate: keep current PID unless improvement is meaningful.
    if float(best["cost"]) >= current_cost - float(args.apply_margin):
        best = dict(current_metrics)
        best["kp"] = float(current_pid[0])
        best["ki"] = float(current_pid[1])
        best["kd"] = float(current_pid[2])
        best["changed"] = False
    else:
        best["changed"] = bool(np.linalg.norm(best_pid - current_pid) > 1e-9)

    best["current_cost"] = current_cost
    best["n_evaluated"] = int(len(evaluated_rows))
    best["horizon_steps"] = int(horizon_steps)
    best["n_history_candidates"] = int(len(history_candidates))
    if progress_enabled:
        total_ms = 1000.0 * (time.perf_counter() - progress_t0)
        print(
            f"[decision {decision_label}] selected "
            f"pid=({float(best['kp']):.0f}, {float(best['ki']):.0f}, {float(best['kd']):.0f}) "
            f"cost={float(best['cost']):.4f} changed={bool(best['changed'])} "
            f"evaluated={int(best['n_evaluated'])} total={total_ms:.0f}ms",
            flush=True,
        )
    return best


# -----------------------------------------------------------------------------
# Full receding-horizon simulation
# -----------------------------------------------------------------------------


def compute_final_metrics(
    *,
    trajectory: pd.DataFrame,
    args: argparse.Namespace,
    valid: bool,
    invalid_reason: str,
) -> dict[str, float | bool | str]:
    times = trajectory["elapsed_s"].to_numpy(dtype=float)
    temps = trajectory["temp"].to_numpy(dtype=float)
    target_temp = float(args.target_temp)
    start_temp = float(args.start_temp)

    error = target_temp - temps
    abs_error = np.abs(error)
    overshoot = overshoot_array(temps, start_temp=start_temp, target_temp=target_temp)

    tail_start = max(0.0, float(args.duration_s) - float(args.tail_window_s))
    tail_mask = times >= tail_start
    if not np.any(tail_mask):
        tail_mask = np.ones_like(times, dtype=bool)

    near_idx = np.where(abs_error <= float(args.near_band))[0]
    settle_idx = np.where(abs_error <= float(args.settle_band))[0]

    return {
        "valid": bool(valid),
        "invalid_reason": invalid_reason,
        "mae_full": float(np.mean(abs_error)),
        "tail_mae": float(np.mean(abs_error[tail_mask])),
        "tail_bias": float(np.mean(error[tail_mask])),
        "tail_std": float(np.std(temps[tail_mask])),
        "overshoot_max": float(np.max(overshoot)),
        "overshoot_rmse": float(np.sqrt(np.mean(np.square(overshoot)))),
        "final_error": float(error[-1]),
        "final_abs_error": float(abs(error[-1])),
        "end_temp": float(temps[-1]),
        "min_temp": float(np.min(temps)),
        "max_temp": float(np.max(temps)),
        "time_to_near_s": float(times[int(near_idx[0])]) if len(near_idx) else float(args.duration_s) + 999.0,
        "time_to_settle_s": float(times[int(settle_idx[0])]) if len(settle_idx) else float(args.duration_s) + 999.0,
        "pid_changes": int(np.sum(
            trajectory[["kp", "ki", "kd"]].diff().abs().sum(axis=1).fillna(0.0).to_numpy() > 1e-9
        )),
    }


def run_mpc_simulation(
    *,
    model: Any,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    window_steps: int,
    device: torch.device,
    args: argparse.Namespace,
    predict_fn: PredictFn,
    cost_fn: CostFn,
    candidate_table: CandidateTable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(int(args.seed))

    start_temp = float(args.start_temp)
    precondition_ref = start_temp if args.precondition_ref is None else float(args.precondition_ref)
    dt_s = float(args.dt_s)
    total_steps = max(1, int(math.ceil(float(args.duration_s) / dt_s)))
    hold_steps = max(1, int(math.ceil(float(args.mpc_hold_s) / dt_s)))

    feature_window = initialize_feature_window(
        feature_names=feature_names,
        window_steps=window_steps,
        start_temp=start_temp,
        precondition_ref=precondition_ref,
        dt_s=dt_s,
        kp=float(args.kp_init),
        ki=float(args.ki_init),
        kd=float(args.kd_init),
    )

    pid = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    pid.p_part = float(args.initial_p)
    pid.i_part = float(args.initial_i)
    pid.d_part = float(args.initial_d)

    diff = CodesysDiff()
    diff.prev_value = start_temp
    diff.filter_out = 0.0
    diff.out = 0.0

    state = SimState(
        elapsed_s=0.0,
        temp=start_temp,
        previous_temp=start_temp,
        feature_window=feature_window,
        pid=pid,
        diff=diff,
        kp=float(args.kp_init),
        ki=float(args.ki_init),
        kd=float(args.kd_init),
    )

    trajectory_rows: list[dict[str, float | int | bool | str]] = []
    decision_rows: list[dict[str, float | int | bool | str]] = []
    valid = True
    invalid_reason = ""
    step = 0
    decision_idx = 0

    while step < total_steps:
        decision_idx += 1
        t0 = time.perf_counter()
        decision = optimize_pid_for_state(
            state=state,
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            device=device,
            args=args,
            rng=rng,
            predict_fn=predict_fn,
            cost_fn=cost_fn,
            candidate_table=candidate_table,
            decision_idx=decision_idx,
        )
        optimize_ms = 1000.0 * (time.perf_counter() - t0)

        old_pid = np.asarray([state.kp, state.ki, state.kd], dtype=float)
        new_pid = clip_pid(np.asarray([decision["kp"], decision["ki"], decision["kd"]], dtype=float), args)
        state.kp, state.ki, state.kd = float(new_pid[0]), float(new_pid[1]), float(new_pid[2])

        decision_row: dict[str, float | int | bool | str] = {
            "decision_idx": decision_idx,
            "elapsed_s": float(state.elapsed_s),
            "temp": float(state.temp),
            "error": float(args.target_temp - state.temp),
            "old_kp": float(old_pid[0]),
            "old_ki": float(old_pid[1]),
            "old_kd": float(old_pid[2]),
            "kp": float(state.kp),
            "ki": float(state.ki),
            "kd": float(state.kd),
            "changed": bool(decision.get("changed", False)),
            "mpc_cost": float(decision["cost"]),
            "current_pid_cost": float(decision["current_cost"]),
            "n_evaluated": int(decision["n_evaluated"]),
            "n_history_candidates": int(decision.get("n_history_candidates", 0)),
            "horizon_steps": int(decision["horizon_steps"]),
            "optimize_ms": float(optimize_ms),
        }
        # Only include horizon_* / pid_change_norm keys this cost_fn actually
        # produced, so each model's decisions CSV keeps its original column
        # set instead of being padded with always-NaN columns from the
        # other model's cost design.
        for optional_key in (
            "horizon_overshoot_max",
            "horizon_overshoot_rmse",
            "horizon_mae",
            "horizon_final_abs_error",
            "horizon_weighted_motion",
            "horizon_mean_abs_velocity",
            "horizon_near_std",
            "pid_change_norm",
        ):
            if optional_key in decision:
                decision_row[optional_key] = float(decision[optional_key])
        decision_rows.append(decision_row)

        if bool(args.print_every_decision):
            print(
                f"[decision {decision_idx:03d}] "
                f"t={state.elapsed_s:7.1f}s temp={state.temp:8.4f} "
                f"pid=({state.kp:7.3f}, {state.ki:8.3f}, {state.kd:7.3f}) "
                f"cost={float(decision['cost']):9.4f} "
                f"os={float(decision.get('horizon_overshoot_max', 0.0)):7.4f} "
                f"eval={int(decision['n_evaluated'])} time={optimize_ms:7.1f}ms"
            )

        for _ in range(hold_steps):
            if step >= total_steps:
                break
            state, info, step_valid, reason = step_state(
                state=state,
                model=model,
                checkpoint=checkpoint,
                feature_names=feature_names,
                device=device,
                target_temp=float(args.target_temp),
                dt_s=dt_s,
                pid_period_s=float(args.pid_period_s),
                feature_scale=max(abs(float(args.control_feature_scale)), 1e-9),
                max_abs_temp=float(args.max_abs_temp),
                predict_fn=predict_fn,
            )
            valid = valid and step_valid
            if reason and not invalid_reason:
                invalid_reason = reason

            trajectory_rows.append({
                "step": step + 1,
                "decision_idx": decision_idx,
                "elapsed_s": float(state.elapsed_s),
                "temp": float(state.temp),
                "temp_ref": float(args.target_temp),
                "error": float(args.target_temp - state.temp),
                "abs_error": float(abs(args.target_temp - state.temp)),
                "kp": float(state.kp),
                "ki": float(state.ki),
                "kd": float(state.kd),
                "u": float(info["u"]),
                "u_p": float(info["u_p"]),
                "u_i": float(info["u_i"]),
                "u_d": float(info["u_d"]),
                "diff_out": float(info["diff_out"]),
                "pred_delta": float(info["pred_delta"]),
                "valid": bool(step_valid),
            })
            step += 1

    trajectory = pd.DataFrame(trajectory_rows)
    decisions = pd.DataFrame(decision_rows)
    metrics = compute_final_metrics(
        trajectory=trajectory,
        args=args,
        valid=valid,
        invalid_reason=invalid_reason,
    )
    return trajectory, decisions, metrics
