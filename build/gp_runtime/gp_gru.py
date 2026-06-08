#!/usr/bin/env python3
"""GRU-ranked PID scheduler for a thermal chamber.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from gru_common import (  # type: ignore
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
        "Could not import gru_common.py. Put this script next to gru_common.py "
        f"or add that directory to PYTHONPATH. Original error: {exc!r}"
    )


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gru_ranked_pid_runs"
DEFAULT_CANDIDATE_TABLE = SCRIPT_DIR / "mpc_pid_runs" / "gru_pid_candidate_table.csv"


@dataclass
class SimState:
    elapsed_s: float
    temp: float
    previous_temp: float
    feature_window: np.ndarray
    pid: ChamberPID
    diff: CodesysDiff
    kp: float
    ki: float
    kd: float


@dataclass
class CandidateTable:
    path: Path
    rows: pd.DataFrame
    features: np.ndarray
    pids: np.ndarray
    history_score: np.ndarray


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GRU-ranked PID scheduler using historical/GP candidates.")

    # Scenario.
    ap.add_argument("--start-temp", type=float, default=27.0)
    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=1200.0)
    ap.add_argument("--dt-s", type=float, default=2.0)
    ap.add_argument("--precondition-ref", type=float, default=None)

    # Model/runtime.
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--window-steps", type=int, default=60)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Initial PID and bounds.
    ap.add_argument("--kp-init", type=float, default=50.0)
    ap.add_argument("--ki-init", type=float, default=1.0)
    ap.add_argument("--kd-init", type=float, default=1.0)
    ap.add_argument("--kp-min", type=float, default=1.0)
    ap.add_argument("--kp-max", type=float, default=50.0)
    ap.add_argument("--ki-min", type=float, default=1.0)
    ap.add_argument("--ki-max", type=float, default=1000.0)
    ap.add_argument("--kd-min", type=float, default=1.0)
    ap.add_argument("--kd-max", type=float, default=80.0)
    ap.add_argument("--initial-i", type=float, default=0.0)
    ap.add_argument("--initial-d", type=float, default=0.0)
    ap.add_argument("--initial-p", type=float, default=0.0)

    # CODESYS PID settings.
    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--pid-period-s", type=float, default=0.1)

    # Band/candidate settings.
    ap.add_argument("--far-threshold", type=float, default=10.0)
    ap.add_argument("--near-threshold", type=float, default=3.0)
    ap.add_argument("--candidate-table", default=str(DEFAULT_CANDIDATE_TABLE))
    ap.add_argument("--candidates-per-decision", type=int, default=32)
    ap.add_argument("--neighbor-pool", type=int, default=300)
    ap.add_argument("--include-current-candidate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-anchor-candidates", action=argparse.BooleanOptionalAction, default=True,
                    help="Always evaluate a small set of broad PID anchors so nearest-history lookup cannot get stuck.")
    ap.add_argument("--max-pid-delta-frac", type=float, default=1.0,
                    help="Reject candidate PID changes larger than this fraction of each PID range. Use 1.0 to disable.")
    ap.add_argument("--history-score-weight", type=float, default=0.10)

    # Feature scales for nearest candidate lookup.
    ap.add_argument("--history-temp-scale", type=float, default=10.0)
    ap.add_argument("--history-error-scale", type=float, default=10.0)
    ap.add_argument("--history-time-scale", type=float, default=600.0)
    ap.add_argument("--history-velocity-scale", type=float, default=0.2)

    # GRU forecast/ranking settings.
    ap.add_argument("--horizon-s", type=float, default=60.0)
    ap.add_argument("--hold-s", type=float, default=30.0)
    ap.add_argument("--overshoot-tolerance", type=float, default=0.05,
                    help="Predicted overshoot above this value is unsafe. Units are degrees C.")
    ap.add_argument("--motion-error-scale", type=float, default=6.0,
                    help="Larger values make motion penalty start farther from target.")
    ap.add_argument("--near-band", type=float, default=2.0)
    ap.add_argument("--settle-band", type=float, default=0.5)
    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--max-abs-temp", type=float, default=100.0)

    # Selection weights inside lexicographic ranking.
    ap.add_argument("--rank-motion-weight", type=float, default=1.0)
    ap.add_argument("--rank-std-weight", type=float, default=1.0)
    ap.add_argument("--rank-history-weight", type=float, default=0.05)
    ap.add_argument("--apply-margin", type=float, default=0.0,
                    help="Keep current PID unless selected candidate improves rank score by at least this amount.")

    # Output/logging.
    ap.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print-every-decision", action=argparse.BooleanOptionalAction, default=True)

    return ap


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def pid_bounds(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray([args.kp_min, args.ki_min, args.kd_min], dtype=float)
    hi = np.asarray([args.kp_max, args.ki_max, args.kd_max], dtype=float)
    if np.any(hi <= lo):
        raise ValueError(f"Invalid PID bounds: lo={lo}, hi={hi}")
    return lo, hi


def clip_pid(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    lo, hi = pid_bounds(args)
    y = np.clip(np.asarray(x, dtype=float), lo, hi)
    y = np.floor(y + 0.5)
    return np.clip(y, lo, hi).astype(float)


def current_band(temp: float, target_temp: float, args: argparse.Namespace) -> str:
    abs_error = abs(float(target_temp) - float(temp))
    if abs_error > float(args.far_threshold):
        return "far"
    if abs_error <= float(args.near_threshold):
        return "near"
    return "mid"


def overshoot_array(temps: np.ndarray, *, start_temp: float, target_temp: float) -> np.ndarray:
    temps = np.asarray(temps, dtype=float)
    if target_temp <= start_temp:
        return np.maximum(float(target_temp) - temps, 0.0)  # cooling: below target
    return np.maximum(temps - float(target_temp), 0.0)      # heating: above target


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


def normalized_history_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores)
    lo = float(np.min(finite))
    hi = float(np.percentile(finite, 90))
    denom = max(hi - lo, 1e-9)
    return np.clip((scores - lo) / denom, 0.0, 10.0)


# -----------------------------------------------------------------------------
# Candidate table
# -----------------------------------------------------------------------------


def load_candidate_table(args: argparse.Namespace) -> CandidateTable:
    path = Path(str(args.candidate_table))
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate table not found: {path}\n"
            "Build it first with build_gru_pid_candidate_table.py"
        )

    df = pd.read_csv(path)
    required = {
        "band", "temp", "target_temp", "error", "abs_error", "elapsed_s", "temp_velocity",
        "current_kp", "current_ki", "current_kd", "kp", "ki", "kd", "history_score",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")

    for col in required.difference({"band"}):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required.difference({"band"}))).reset_index(drop=True)

    lo, hi = pid_bounds(args)
    pids = df[["kp", "ki", "kd"]].to_numpy(dtype=float)
    keep = np.all((pids >= lo) & (pids <= hi), axis=1)
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        raise ValueError("Candidate table has no rows inside current PID bounds.")

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

    return CandidateTable(
        path=path,
        rows=df,
        features=np.vstack(features),
        pids=df[["kp", "ki", "kd"]].to_numpy(dtype=float),
        history_score=df["history_score"].to_numpy(dtype=float),
    )


def select_candidate_pids(
    *,
    state: SimState,
    candidate_table: CandidateTable,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    band = current_band(state.temp, args.target_temp, args)
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

    rows = candidate_table.rows
    band_mask = rows["band"].astype(str).to_numpy() == band
    candidate_idx = np.where(band_mask)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(len(rows))

    diff = candidate_table.features[candidate_idx] - query.reshape(1, -1)
    distance = np.sqrt(np.mean(np.square(diff), axis=1))

    pool_n = min(len(candidate_idx), max(int(args.neighbor_pool), int(args.candidates_per_decision)))
    if pool_n <= 0:
        return []

    local_pool = np.argpartition(distance, pool_n - 1)[:pool_n]
    pool_idx = candidate_idx[local_pool]
    pool_distance = distance[local_pool]
    score_norm = normalized_history_scores(candidate_table.history_score[pool_idx])
    rank = pool_distance + float(args.history_score_weight) * score_norm
    ordered = pool_idx[np.argsort(rank)]

    lo, hi = pid_bounds(args)
    full_range = hi - lo
    max_delta = float(args.max_pid_delta_frac) * full_range

    selected: List[Dict[str, float]] = []
    seen = set()

    def append_candidate(
        pid_like: Sequence[float],
        *,
        history_score: float,
        source: str,
        ignore_delta_limit: bool = False,
    ) -> None:
        pid = clip_pid(np.asarray(pid_like, dtype=float), args)
        if not ignore_delta_limit and float(args.max_pid_delta_frac) < 1.0:
            if np.any(np.abs(pid - current_pid) > max_delta):
                return
        key = tuple(float(x) for x in pid)
        if key in seen:
            return
        selected.append({
            "kp": float(pid[0]),
            "ki": float(pid[1]),
            "kd": float(pid[2]),
            "history_score": float(history_score),
            "source": source,
            "lookup_band": band,
        })
        seen.add(key)

    if bool(args.include_current_candidate):
        append_candidate(current_pid, history_score=0.0, source="current_pid", ignore_delta_limit=True)

    if bool(args.include_anchor_candidates):
        lo, hi = pid_bounds(args)
        mid = 0.5 * (lo + hi)
        anchor_pids = [
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], mid[2]],
            [mid[0], lo[1], lo[2]],
            [mid[0], mid[1], lo[2]],
            [lo[0], lo[1], lo[2]],
        ]
        for anchor_pid in anchor_pids:
            append_candidate(
                anchor_pid,
                history_score=0.0,
                source="anchor_pid",
                ignore_delta_limit=True,
            )

    for idx in ordered:
        row = rows.iloc[int(idx)]
        append_candidate(
            candidate_table.pids[int(idx)],
            history_score=float(row.get("history_score", np.nan)),
            source=str(row.get("source", "candidate_table")),
        )
        if len(selected) >= int(args.candidates_per_decision):
            break

    return selected


# -----------------------------------------------------------------------------
# GRU feature construction and plant step
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
    pid: ChamberPID,
    diff: CodesysDiff,
    temp_start: float,
    temp_ref: float,
    kp: float,
    ki: float,
    kd: float,
    dt_s: float,
    period_s: float,
    feature_scale: float,
) -> Dict[str, float]:
    period = max(float(period_s), 1e-6)
    dt = max(float(dt_s), period)
    n_substeps = max(1, int(round(dt / period)))

    last = {
        "u": 0.0, "u_p": 0.0, "u_i": 0.0, "u_d": 0.0,
        "u_norm": 0.0, "u_p_norm": 0.0, "u_i_norm": 0.0, "u_d_norm": 0.0,
        "diff_out": diff.out, "n_substeps": n_substeps,
    }

    for _ in range(n_substeps):
        diff_out = diff.update(float(temp_start))
        u, u_p, u_i, u_d = pid.step(
            enable=True,
            x_target=float(temp_ref),
            x_measured=float(temp_start),
            p_coef=float(kp),
            i_coef=float(ki),
            d_coef=float(kd),
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


def step_state(
    *,
    state: SimState,
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    target_temp: float,
    dt_s: float,
    pid_period_s: float,
    feature_scale: float,
    max_abs_temp: float,
) -> Tuple[SimState, Dict[str, float], bool, str]:
    terms = run_pid_substeps(
        pid=state.pid,
        diff=state.diff,
        temp_start=state.temp,
        temp_ref=target_temp,
        kp=state.kp,
        ki=state.ki,
        kd=state.kd,
        dt_s=dt_s,
        period_s=pid_period_s,
        feature_scale=feature_scale,
    )

    next_temp, pred_delta, pred_window = predict_next(
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
        next_temp = float(np.nan_to_num(next_temp, nan=max_abs_temp, posinf=max_abs_temp, neginf=-max_abs_temp))

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
# Forecasting and ranking
# -----------------------------------------------------------------------------


def horizon_metrics(
    *,
    temps: np.ndarray,
    start_temp: float,
    target_temp: float,
    valid: bool,
    args: argparse.Namespace,
) -> Dict[str, float | bool]:
    temps = np.asarray(temps, dtype=float)
    if temps.size == 0:
        return {
            "valid": False,
            "horizon_mae": float("inf"),
            "horizon_near_std": float("inf"),
            "horizon_overshoot_max": float("inf"),
            "horizon_weighted_motion": float("inf"),
            "horizon_mean_abs_velocity": float("inf"),
            "horizon_final_abs_error": float("inf"),
        }

    error = float(target_temp) - temps
    abs_error = np.abs(error)
    overshoot = overshoot_array(temps, start_temp=start_temp, target_temp=target_temp)

    near_mask = abs_error <= float(args.near_band)
    near_std = float(np.std(temps[near_mask])) if np.any(near_mask) else 0.0

    if temps.size >= 2:
        velocity = np.diff(temps) / max(float(args.dt_s), 1e-9)
        err_for_velocity = abs_error[:-1]
        closeness_weight = np.exp(-err_for_velocity / max(float(args.motion_error_scale), 1e-9))
        weighted_motion = float(np.mean(closeness_weight * np.square(velocity)))
        mean_abs_velocity = float(np.mean(np.abs(velocity)))
    else:
        weighted_motion = 0.0
        mean_abs_velocity = 0.0

    return {
        "valid": bool(valid),
        "horizon_mae": float(np.mean(abs_error)),
        "horizon_near_std": near_std,
        "horizon_overshoot_max": float(np.max(overshoot)),
        "horizon_weighted_motion": weighted_motion,
        "horizon_mean_abs_velocity": mean_abs_velocity,
        "horizon_final_abs_error": float(abs_error[-1]),
    }


def rollout_candidate_pid(
    *,
    initial_state: SimState,
    candidate: Dict[str, float],
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    horizon_steps: int,
) -> Tuple[Dict[str, float | bool], List[float]]:
    state = copy.deepcopy(initial_state)
    pid_vec = clip_pid(np.asarray([candidate["kp"], candidate["ki"], candidate["kd"]], dtype=float), args)
    state.kp, state.ki, state.kd = float(pid_vec[0]), float(pid_vec[1]), float(pid_vec[2])

    temps: List[float] = []
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
        )
        temps.append(float(state.temp))
        valid = valid and step_valid
        if not step_valid:
            break

    metrics = horizon_metrics(
        temps=np.asarray(temps, dtype=float),
        start_temp=float(args.start_temp),
        target_temp=float(args.target_temp),
        valid=valid,
        args=args,
    )
    metrics.update({
        "kp": float(pid_vec[0]),
        "ki": float(pid_vec[1]),
        "kd": float(pid_vec[2]),
        "history_score": float(candidate.get("history_score", np.nan)),
        "source": candidate.get("source", "candidate_table"),
        "lookup_band": candidate.get("lookup_band", ""),
    })
    return metrics, temps


def scalar_rank_score(row: Dict[str, float | bool], args: argparse.Namespace) -> float:
    hist = float(row.get("history_score", 0.0))
    if not np.isfinite(hist):
        hist = 0.0
    return (
        float(row["horizon_mae"])
        + float(args.rank_motion_weight) * float(row["horizon_weighted_motion"])
        + float(args.rank_std_weight) * float(row["horizon_near_std"])
        + float(args.rank_history_weight) * hist
    )


def lexicographic_key(row: Dict[str, float | bool], args: argparse.Namespace) -> Tuple[float, float, float, float, float]:
    # Unsafe candidates are ranked primarily by overshoot. Safe candidates compete on MAE/motion/std.
    overshoot = float(row["horizon_overshoot_max"])
    unsafe = 1.0 if overshoot > float(args.overshoot_tolerance) or not bool(row.get("valid", True)) else 0.0
    return (
        unsafe,
        overshoot if unsafe else 0.0,
        float(row["horizon_mae"]),
        float(row["horizon_weighted_motion"]),
        float(row["horizon_near_std"]),
    )


def select_pid_for_state(
    *,
    state: SimState,
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    candidate_table: CandidateTable,
    decision_idx: int,
) -> Dict[str, float | bool | str | int]:
    horizon_steps = max(1, int(math.ceil(float(args.horizon_s) / float(args.dt_s))))
    candidates = select_candidate_pids(state=state, candidate_table=candidate_table, args=args)
    if not candidates:
        current = {"kp": state.kp, "ki": state.ki, "kd": state.kd, "history_score": 0.0, "source": "fallback_current"}
        candidates = [current]

    evaluated: List[Dict[str, float | bool | str]] = []
    for cand in candidates:
        metrics, _ = rollout_candidate_pid(
            initial_state=state,
            candidate=cand,
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            device=device,
            args=args,
            horizon_steps=horizon_steps,
        )
        metrics["rank_score"] = scalar_rank_score(metrics, args)
        evaluated.append(metrics)

    evaluated.sort(key=lambda r: lexicographic_key(r, args))
    selected = dict(evaluated[0])

    # Optional safety gate against changing when not clearly better than current.
    current_rows = [r for r in evaluated if str(r.get("source", "")) == "current_pid"]
    if current_rows:
        current_row = current_rows[0]
        selected_key = lexicographic_key(selected, args)
        current_key = lexicographic_key(current_row, args)
        selected_score = scalar_rank_score(selected, args)
        current_score = scalar_rank_score(current_row, args)
        margin_blocks_change = (
            selected_key == current_key
            and selected_score >= current_score - float(args.apply_margin)
        )
        if selected_key > current_key or margin_blocks_change:
            selected = dict(current_row)

    selected["changed"] = bool(
        abs(float(selected["kp"]) - float(state.kp)) > 1e-9
        or abs(float(selected["ki"]) - float(state.ki)) > 1e-9
        or abs(float(selected["kd"]) - float(state.kd)) > 1e-9
    )
    selected["n_evaluated"] = int(len(evaluated))
    selected["horizon_steps"] = int(horizon_steps)
    selected["decision_idx"] = int(decision_idx)
    selected["current_band"] = current_band(state.temp, args.target_temp, args)
    return selected


# -----------------------------------------------------------------------------
# Full simulation
# -----------------------------------------------------------------------------


def compute_final_metrics(
    *,
    trajectory: pd.DataFrame,
    args: argparse.Namespace,
    valid: bool,
    invalid_reason: str,
) -> Dict[str, float | bool | str]:
    if trajectory.empty:
        return {"valid": False, "invalid_reason": "empty trajectory"}

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
            (trajectory[["kp", "ki", "kd"]].diff().abs().sum(axis=1).fillna(0.0).to_numpy() > 1e-9)
        )),
    }


def run_simulation(
    *,
    model: GRUModel,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    window_steps: int,
    device: torch.device,
    args: argparse.Namespace,
    candidate_table: CandidateTable,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    start_temp = float(args.start_temp)
    precondition_ref = start_temp if args.precondition_ref is None else float(args.precondition_ref)
    dt_s = float(args.dt_s)
    total_steps = max(1, int(math.ceil(float(args.duration_s) / dt_s)))
    hold_steps = max(1, int(math.ceil(float(args.hold_s) / dt_s)))

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

    trajectory_rows: List[Dict[str, float | int | bool | str]] = []
    decision_rows: List[Dict[str, float | int | bool | str]] = []
    valid = True
    invalid_reason = ""
    step = 0
    decision_idx = 0

    while step < total_steps:
        decision_idx += 1
        t0 = time.perf_counter()
        decision = select_pid_for_state(
            state=state,
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            device=device,
            args=args,
            candidate_table=candidate_table,
            decision_idx=decision_idx,
        )
        optimize_ms = 1000.0 * (time.perf_counter() - t0)

        old_pid = np.asarray([state.kp, state.ki, state.kd], dtype=float)
        new_pid = clip_pid(np.asarray([decision["kp"], decision["ki"], decision["kd"]], dtype=float), args)
        state.kp, state.ki, state.kd = float(new_pid[0]), float(new_pid[1]), float(new_pid[2])

        decision_row: Dict[str, float | int | bool | str] = {
            "decision_idx": decision_idx,
            "elapsed_s": float(state.elapsed_s),
            "temp": float(state.temp),
            "error": float(args.target_temp - state.temp),
            "band": str(decision.get("current_band", "")),
            "old_kp": float(old_pid[0]),
            "old_ki": float(old_pid[1]),
            "old_kd": float(old_pid[2]),
            "kp": float(state.kp),
            "ki": float(state.ki),
            "kd": float(state.kd),
            "changed": bool(decision.get("changed", False)),
            "rank_score": float(decision.get("rank_score", np.nan)),
            "history_score": float(decision.get("history_score", np.nan)),
            "source": str(decision.get("source", "")),
            "horizon_overshoot_max": float(decision.get("horizon_overshoot_max", np.nan)),
            "horizon_mae": float(decision.get("horizon_mae", np.nan)),
            "horizon_final_abs_error": float(decision.get("horizon_final_abs_error", np.nan)),
            "horizon_weighted_motion": float(decision.get("horizon_weighted_motion", np.nan)),
            "horizon_mean_abs_velocity": float(decision.get("horizon_mean_abs_velocity", np.nan)),
            "horizon_near_std": float(decision.get("horizon_near_std", np.nan)),
            "n_evaluated": int(decision.get("n_evaluated", 0)),
            "horizon_steps": int(decision.get("horizon_steps", 0)),
            "optimize_ms": float(optimize_ms),
        }
        decision_rows.append(decision_row)

        if bool(args.print_every_decision):
            print(
                f"[decision {decision_idx:03d}] "
                f"t={state.elapsed_s:7.1f}s temp={state.temp:8.4f} "
                f"band={decision_row['band']:>4} "
                f"pid=({state.kp:7.0f}, {state.ki:8.0f}, {state.kd:7.0f}) "
                f"os={decision_row['horizon_overshoot_max']:7.4f} "
                f"mae={decision_row['horizon_mae']:7.4f} "
                f"motion={decision_row['horizon_weighted_motion']:9.6f} "
                f"src={decision_row['source']} "
                f"eval={decision_row['n_evaluated']} time={optimize_ms:7.1f}ms"
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
    metrics = compute_final_metrics(trajectory=trajectory, args=args, valid=valid, invalid_reason=invalid_reason)
    return trajectory, decisions, metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"gru_ranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    feature_names = list(checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))
    candidate_table = load_candidate_table(args)

    print("=== GRU-ranked PID scheduler ===")
    print(f"checkpoint:       {checkpoint_path}")
    print(f"device:           {device}")
    print(f"features:         {feature_names}")
    print(f"window_steps:     {window_steps}")
    print(f"scenario:         {args.start_temp} °C -> {args.target_temp} °C")
    print(f"duration/dt:      {args.duration_s}s / {args.dt_s}s")
    print(f"horizon/hold:     {args.horizon_s}s / {args.hold_s}s")
    print(f"candidate table:  {candidate_table.path} ({len(candidate_table.rows)} rows)")
    print(f"overshoot tol:    {args.overshoot_tolerance}")
    print(f"output dir:       {output_dir}")

    trajectory, decisions, metrics = run_simulation(
        model=model,
        checkpoint=checkpoint,
        feature_names=feature_names,
        window_steps=window_steps,
        device=device,
        args=args,
        candidate_table=candidate_table,
    )

    trajectory_csv = output_dir / "gru_ranked_trajectory.csv"
    decisions_csv = output_dir / "gru_ranked_decisions.csv"
    summary_json = output_dir / "gru_ranked_summary.json"

    if bool(args.save_trajectory):
        trajectory.to_csv(trajectory_csv, index=False)
    decisions.to_csv(decisions_csv, index=False)

    summary = {
        "run_id": run_id,
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
        "selector": {
            "horizon_s": float(args.horizon_s),
            "hold_s": float(args.hold_s),
            "candidates_per_decision": int(args.candidates_per_decision),
            "neighbor_pool": int(args.neighbor_pool),
            "overshoot_tolerance": float(args.overshoot_tolerance),
            "candidate_table": str(candidate_table.path),
            "candidate_table_rows": int(len(candidate_table.rows)),
        },
        "bounds": {
            "kp": [float(args.kp_min), float(args.kp_max)],
            "ki": [float(args.ki_min), float(args.ki_max)],
            "kd": [float(args.kd_min), float(args.kd_max)],
        },
        "metrics": metrics,
        "trajectory_csv": str(trajectory_csv) if bool(args.save_trajectory) else None,
        "decisions_csv": str(decisions_csv),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final metrics ===")
    for key in [
        "overshoot_max", "overshoot_rmse", "tail_mae", "tail_std",
        "final_abs_error", "end_temp", "time_to_near_s", "time_to_settle_s", "pid_changes",
    ]:
        print(f"{key:18s}: {metrics.get(key)}")

    print("\n=== Saved ===")
    if bool(args.save_trajectory):
        print(f"trajectory csv: {trajectory_csv}")
    print(f"decisions csv:  {decisions_csv}")
    print(f"summary json:   {summary_json}")


if __name__ == "__main__":
    main()
