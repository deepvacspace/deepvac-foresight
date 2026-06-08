#!/usr/bin/env python3
"""GRU-ranked GP PID controller for the chamber.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(getattr(sys, "_MEIPASS", SCRIPT_DIR))
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tcp.tcp_common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    apply_pid_update,
    publish_temp_ref_job,
    request_temperature_states,
)
from utils.bo_common import append_rows_csv, make_run_id


DEFAULT_CHECKPOINT = RUNTIME_DIR / "model" / "gru_t1.pt"
DEFAULT_GP_MODELS = RUNTIME_DIR / "model" / "band_gp_models.pkl"
DEFAULT_CANDIDATE_TABLE = RUNTIME_DIR / "data" / "gru_pid_candidate_table.csv"
DEFAULT_RANKED_DECISIONS = RUNTIME_DIR / "data" / "gru_ranked_decisions.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "live_runs"
DEFAULT_FEATURE_NAMES = [
    "temp",
    "temp_ref",
    "error",
    "temp_u",
    "temp_u_p",
    "temp_u_i",
    "temp_u_d",
    "kp",
    "ki",
    "kd",
]


def load_runtime_modules() -> None:
    global ChamberPID
    global CodesysDiff
    global SimState
    global clip_pid
    global current_band
    global initialize_feature_window
    global lexicographic_key
    global load_candidate_table
    global load_model
    global make_feature_row
    global rollout_candidate_pid
    global scalar_rank_score
    global select_candidate_pids
    global torch

    import torch as torch_module
    from gp_gru import (  # noqa: WPS433
        ChamberPID as ChamberPID_,
        CodesysDiff as CodesysDiff_,
        SimState as SimState_,
        clip_pid as clip_pid_,
        current_band as current_band_,
        initialize_feature_window as initialize_feature_window_,
        lexicographic_key as lexicographic_key_,
        load_candidate_table as load_candidate_table_,
        make_feature_row as make_feature_row_,
        rollout_candidate_pid as rollout_candidate_pid_,
        scalar_rank_score as scalar_rank_score_,
        select_candidate_pids as select_candidate_pids_,
    )
    from gru_common import load_model as load_model_  # noqa: WPS433

    torch = torch_module
    ChamberPID = ChamberPID_
    CodesysDiff = CodesysDiff_
    SimState = SimState_
    clip_pid = clip_pid_
    current_band = current_band_
    initialize_feature_window = initialize_feature_window_
    lexicographic_key = lexicographic_key_
    load_candidate_table = load_candidate_table_
    load_model = load_model_
    make_feature_row = make_feature_row_
    rollout_candidate_pid = rollout_candidate_pid_
    scalar_rank_score = scalar_rank_score_
    select_candidate_pids = select_candidate_pids_


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run live GRU-ranked GP PID control over TCP.")

    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--gp-models", default=str(DEFAULT_GP_MODELS),
                    help="Pickled band GP models. Empty string disables GP-proposed candidates.")
    ap.add_argument("--candidate-table", default=str(DEFAULT_CANDIDATE_TABLE))
    ap.add_argument("--ranked-decisions-csv", default=str(DEFAULT_RANKED_DECISIONS))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--target-temp", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=1200.0)
    ap.add_argument("--dt-s", type=float, default=2.0,
                    help="Sampling period and GRU feature step in seconds.")
    ap.add_argument("--hold-s", type=float, default=30.0,
                    help="Seconds between PID replans.")
    ap.add_argument("--horizon-s", type=float, default=60.0)
    ap.add_argument("--window-steps", type=int, default=60)

    ap.add_argument("--kp-init", type=float, default=None,
                    help="Initial fallback kp. Default uses TCP state kp.")
    ap.add_argument("--ki-init", type=float, default=None,
                    help="Initial fallback ki. Default uses TCP state ki.")
    ap.add_argument("--kd-init", type=float, default=None,
                    help="Initial fallback kd. Default uses TCP state kd.")
    ap.add_argument("--kp-min", type=float, default=1.0)
    ap.add_argument("--kp-max", type=float, default=50.0)
    ap.add_argument("--ki-min", type=float, default=1.0)
    ap.add_argument("--ki-max", type=float, default=1000.0)
    ap.add_argument("--kd-min", type=float, default=1.0)
    ap.add_argument("--kd-max", type=float, default=80.0)
    ap.add_argument("--initial-i", type=float, default=0.0)
    ap.add_argument("--initial-d", type=float, default=0.0)
    ap.add_argument("--initial-p", type=float, default=0.0)

    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--pid-period-s", type=float, default=0.1)

    ap.add_argument("--far-threshold", type=float, default=10.0)
    ap.add_argument("--near-threshold", type=float, default=3.0)
    ap.add_argument("--candidates-per-decision", type=int, default=32)
    ap.add_argument("--neighbor-pool", type=int, default=300)
    ap.add_argument("--include-current-candidate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-anchor-candidates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ranked-decision-candidates", type=int, default=24,
                    help="Unique PIDs from ranked decisions to evaluate at every replan.")
    ap.add_argument("--gp-candidates", type=int, default=16,
                    help="GP-proposed current-band candidates to evaluate at every replan.")
    ap.add_argument("--gp-random-candidates", type=int, default=4096,
                    help="Random PID samples scored by the current-band GP per replan.")
    ap.add_argument("--gp-acquisition", choices=["lcb", "ei"], default="lcb")
    ap.add_argument("--gp-lcb-kappa", type=float, default=0.10)
    ap.add_argument("--gp-xi", type=float, default=0.01)
    ap.add_argument("--max-pid-delta-frac", type=float, default=1.0)
    ap.add_argument("--history-score-weight", type=float, default=0.10)

    ap.add_argument("--history-temp-scale", type=float, default=10.0)
    ap.add_argument("--history-error-scale", type=float, default=10.0)
    ap.add_argument("--history-time-scale", type=float, default=600.0)
    ap.add_argument("--history-velocity-scale", type=float, default=0.2)

    ap.add_argument("--overshoot-tolerance", type=float, default=0.05)
    ap.add_argument("--motion-error-scale", type=float, default=6.0)
    ap.add_argument("--near-band", type=float, default=2.0)
    ap.add_argument("--settle-band", type=float, default=0.5)
    ap.add_argument("--tail-window-s", type=float, default=300.0)
    ap.add_argument("--max-abs-temp", type=float, default=100.0)
    ap.add_argument("--rank-motion-weight", type=float, default=1.0)
    ap.add_argument("--rank-std-weight", type=float, default=1.0)
    ap.add_argument("--rank-history-weight", type=float, default=0.05)
    ap.add_argument("--apply-margin", type=float, default=0.0)

    ap.add_argument("--pid-row", type=int, default=1)
    ap.add_argument("--tcp-host", default=DEFAULT_HOST)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--tcp-timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--state-wait-s", type=float, default=1.0)
    ap.add_argument("--state-max-wait-s", type=float, default=0.0,
                    help="Max seconds to wait for valid temp_u. 0 means forever.")
    ap.add_argument("--read-retries", type=int, default=2)
    ap.add_argument("--read-retry-delay-s", type=float, default=0.25)
    ap.add_argument("--max-consecutive-failures", type=int, default=0,
                    help="Max consecutive TCP read failures before aborting. 0 means keep trying forever.")
    ap.add_argument("--tcp-reconnect-delay-s", type=float, default=2.0,
                    help="Delay before retrying after TCP connect/read/write failures.")
    ap.add_argument("--tcp-write-retries", type=int, default=0,
                    help="PID/job write retries before aborting. 0 means keep trying forever.")
    ap.add_argument("--publish-job", action=argparse.BooleanOptionalAction, default=True,
                    help="Publish target temp_ref job for duration-s before control starts.")
    ap.add_argument("--apply-unchanged", action=argparse.BooleanOptionalAction, default=False,
                    help="Send PID even when selected PID equals current PID.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Read TCP and plan, but do not publish job or write PID.")

    return ap


def get_states(args: argparse.Namespace) -> Dict[str, float]:
    return request_temperature_states(
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )


def finite_value(row: Dict[str, float], key: str) -> bool:
    value = row.get(key)
    return value is not None and math.isfinite(float(value))


def read_states_with_retries(args: argparse.Namespace) -> Dict[str, float]:
    attempts = max(1, int(args.read_retries) + 1)
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return get_states(args)
        except Exception as exc:
            last_exc = exc
            if args.read_retry_delay_s > 0:
                time.sleep(float(args.read_retry_delay_s))
    raise RuntimeError("TCP get_states failed") from last_exc


def retry_forever(label: str, fn, args: argparse.Namespace):
    attempts = 0
    max_attempts = int(args.tcp_write_retries)
    while True:
        attempts += 1
        try:
            return fn()
        except Exception as exc:
            if max_attempts > 0 and attempts >= max_attempts:
                raise
            print(f"{label} failed (attempt {attempts}); retrying: {exc}")
            time.sleep(max(0.0, float(args.tcp_reconnect_delay_s)))


def publish_job_with_retries(args: argparse.Namespace) -> None:
    retry_forever(
        "publish temp_ref job",
        lambda: publish_temp_ref_job(
            temp_ref=float(args.target_temp),
            duration_s=float(args.duration_s),
            host=args.tcp_host,
            port=args.tcp_port,
            timeout=args.tcp_timeout,
        ),
        args,
    )


def apply_pid_update_with_retries(
    *,
    label: str,
    run_id: str,
    row: int,
    pid: Tuple[int, int, int],
    args: argparse.Namespace,
    events: List[Dict[str, object]],
) -> Tuple[int, int, int]:
    return retry_forever(
        f"PID update {label}",
        lambda: apply_pid_update(
            label=label,
            run_id=run_id,
            row=row,
            pid=pid,
            args=args,
            events=events,
        ),
        args,
    )


def wait_for_valid_start(args: argparse.Namespace) -> Dict[str, float]:
    start = time.time()
    attempts = 0
    while True:
        attempts += 1
        try:
            snap = get_states(args)
            if finite_value(snap, "temp_u") and finite_value(snap, "temp"):
                print(
                    f"ready after {attempts} reads: "
                    f"temp={float(snap['temp']):.4f}, temp_u={float(snap['temp_u']):.4f}"
                )
                return snap
            print(f"waiting for finite temp_u/temp, got temp_u={snap.get('temp_u')}")
        except Exception as exc:
            print(f"waiting for get_states: {exc}")

        if args.state_max_wait_s > 0 and (time.time() - start) >= float(args.state_max_wait_s):
            raise TimeoutError("Timed out waiting for finite temp_u from get_states")
        time.sleep(max(0.0, float(args.state_wait_s)))


def load_ranked_pid_candidates(path: Path, limit: int) -> List[Dict[str, float]]:
    if limit <= 0 or not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"kp", "ki", "kd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ranked decisions missing columns: {sorted(missing)}")

    out: List[Dict[str, float]] = []
    seen: set[Tuple[int, int, int]] = set()
    for row in df.dropna(subset=["kp", "ki", "kd"]).itertuples(index=False):
        pid = (int(round(float(row.kp))), int(round(float(row.ki))), int(round(float(row.kd))))
        if pid in seen:
            continue
        seen.add(pid)
        out.append({
            "kp": float(pid[0]),
            "ki": float(pid[1]),
            "kd": float(pid[2]),
            "history_score": 0.0,
            "source": "ranked_decision",
        })
        if len(out) >= limit:
            break
    return out


def load_gp_models(path_text: str) -> Dict[str, object]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        print(f"GP model pickle not found, disabling GP candidates: {path}")
        return {}
    with path.open("rb") as fh:
        models = pickle.load(fh)
    if not isinstance(models, dict):
        raise ValueError(f"GP model pickle must contain a dict, got {type(models)}")
    return models


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = float(y_best) - mu - float(xi)
    z = improvement / sigma_safe
    ei = improvement * normal_cdf(z) + sigma_safe * normal_pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def gp_pid_candidates_for_state(
    *,
    models: Dict[str, object],
    state: "SimState",
    args: argparse.Namespace,
    decision_idx: int,
) -> List[Dict[str, float]]:
    if not models or int(args.gp_candidates) <= 0 or int(args.gp_random_candidates) <= 0:
        return []

    band = current_band(state.temp, args.target_temp, args)
    model = models.get(band)
    if model is None:
        return []

    scaler = model["scaler"]
    gp = model["gp"]
    rng = np.random.default_rng(int(args.seed) + 1009 * int(decision_idx))

    lo = np.asarray([args.kp_min, args.ki_min, args.kd_min], dtype=float)
    hi = np.asarray([args.kp_max, args.ki_max, args.kd_max], dtype=float)
    pid_samples = rng.uniform(lo, hi, size=(int(args.gp_random_candidates), 3))
    X = np.column_stack([
        pid_samples,
        np.full(len(pid_samples), float(state.temp), dtype=float),
        np.full(len(pid_samples), float(args.target_temp), dtype=float),
    ])

    mu, sigma = gp.predict(scaler.transform(X), return_std=True)
    if args.gp_acquisition == "ei":
        score = expected_improvement(
            mu=mu,
            sigma=sigma,
            y_best=float(model.get("best_cost", np.min(mu))),
            xi=float(args.gp_xi),
        )
        ranked_idx = np.argsort(-score)[: int(args.gp_candidates)]
        score_key = "expected_improvement"
    else:
        score = mu - float(args.gp_lcb_kappa) * sigma
        ranked_idx = np.argsort(score)[: int(args.gp_candidates)]
        score_key = "lcb_score"

    out: List[Dict[str, float]] = []
    for idx_raw in ranked_idx:
        idx = int(idx_raw)
        out.append({
            "kp": float(pid_samples[idx, 0]),
            "ki": float(pid_samples[idx, 1]),
            "kd": float(pid_samples[idx, 2]),
            "history_score": 0.0,
            "source": "band_gp_model",
            "gp_pred_cost": float(mu[idx]),
            "gp_pred_std": float(sigma[idx]),
            score_key: float(score[idx]),
        })
    return out


def dedupe_candidates(candidates: Sequence[Dict[str, float]], args: argparse.Namespace) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    seen: set[Tuple[float, float, float]] = set()
    for candidate in candidates:
        pid = clip_pid(
            np.asarray([candidate["kp"], candidate["ki"], candidate["kd"]], dtype=float),
            args,
        )
        key = (float(pid[0]), float(pid[1]), float(pid[2]))
        if key in seen:
            continue
        seen.add(key)
        item = dict(candidate)
        item["kp"], item["ki"], item["kd"] = key
        out.append(item)
    return out


def select_live_pid(
    *,
    state: SimState,
    model: torch.nn.Module,
    checkpoint: Dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    candidate_table: CandidateTable,
    ranked_candidates: Sequence[Dict[str, float]],
    gp_models: Dict[str, object],
    decision_idx: int,
) -> Dict[str, float | bool | str | int]:
    horizon_steps = max(1, int(math.ceil(float(args.horizon_s) / float(args.dt_s))))
    candidates = select_candidate_pids(state=state, candidate_table=candidate_table, args=args)
    candidates.extend(copy.deepcopy(list(ranked_candidates)))
    candidates.extend(gp_pid_candidates_for_state(
        models=gp_models,
        state=state,
        args=args,
        decision_idx=decision_idx,
    ))
    if not candidates:
        candidates = [{
            "kp": state.kp,
            "ki": state.ki,
            "kd": state.kd,
            "history_score": 0.0,
            "source": "fallback_current",
        }]
    candidates = dedupe_candidates(candidates, args)

    evaluated: List[Dict[str, float | bool | str]] = []
    for candidate in candidates:
        metrics, _ = rollout_candidate_pid(
            initial_state=state,
            candidate=candidate,
            model=model,
            checkpoint=checkpoint,
            feature_names=feature_names,
            device=device,
            args=args,
            horizon_steps=horizon_steps,
        )
        metrics["rank_score"] = scalar_rank_score(metrics, args)
        evaluated.append(metrics)

    evaluated.sort(key=lambda row: lexicographic_key(row, args))
    selected = dict(evaluated[0])

    current_rows = [row for row in evaluated if str(row.get("source", "")) == "current_pid"]
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


def sync_pid_state_from_tcp(pid: ChamberPID, snap: Dict[str, float], feature_scale: float) -> None:
    scale = max(abs(float(feature_scale)), 1e-9)
    if finite_value(snap, "temp_u_p"):
        pid.p_part = float(snap["temp_u_p"]) / scale
    if finite_value(snap, "temp_u_i"):
        pid.i_part = float(snap["temp_u_i"]) / scale
    if finite_value(snap, "temp_u_d"):
        pid.d_part = float(snap["temp_u_d"]) / scale


def state_feature_row(
    *,
    feature_names: Sequence[str],
    snap: Dict[str, float],
    target_temp: float,
    previous_temp: float,
    dt_s: float,
    kp: float,
    ki: float,
    kd: float,
) -> np.ndarray:
    return make_feature_row(
        feature_names,
        temp=float(snap["temp"]),
        temp_ref=float(target_temp),
        previous_temp=float(previous_temp),
        dt_s=float(dt_s),
        u=float(snap["temp_u"]),
        u_p=float(snap["temp_u_p"]),
        u_i=float(snap["temp_u_i"]),
        u_d=float(snap["temp_u_d"]),
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    if args.dt_s <= 0:
        raise ValueError("--dt-s must be > 0")
    if args.hold_s <= 0:
        raise ValueError("--hold-s must be > 0")
    if args.horizon_s <= 0:
        raise ValueError("--horizon-s must be > 0")
    if not (0 <= int(args.pid_row) <= 4):
        raise ValueError("--pid-row must be in range [0, 4]")
    if args.max_consecutive_failures < 0:
        raise ValueError("--max-consecutive-failures must be >= 0")
    if args.tcp_reconnect_delay_s < 0:
        raise ValueError("--tcp-reconnect-delay-s must be >= 0")
    if args.tcp_write_retries < 0:
        raise ValueError("--tcp-write-retries must be >= 0")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    load_runtime_modules()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    run_id = make_run_id(prefix="live_gp")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))
    candidate_table = load_candidate_table(args)
    gp_models = load_gp_models(str(args.gp_models))
    ranked_candidates = load_ranked_pid_candidates(
        Path(args.ranked_decisions_csv),
        int(args.ranked_decision_candidates),
    )

    print("=== Live GRU-ranked GP controller ===")
    print(f"run_id:           {run_id}")
    print(f"checkpoint:       {Path(args.checkpoint)}")
    print(f"gp models:        {Path(args.gp_models) if args.gp_models else 'disabled'} ({len(gp_models)} bands)")
    print(f"candidate table:  {Path(args.candidate_table)} ({len(candidate_table.rows)} rows)")
    print(f"ranked decisions: {Path(args.ranked_decisions_csv)} ({len(ranked_candidates)} unique PIDs)")
    print(f"target/duration:  {args.target_temp} C / {args.duration_s}s")
    print(f"dt/hold/horizon:  {args.dt_s}s / {args.hold_s}s / {args.horizon_s}s")
    print(f"tcp:              {args.tcp_host}:{args.tcp_port} row={args.pid_row}")
    print(f"output dir:       {output_dir}")

    start_snap = wait_for_valid_start(args)
    start_temp = float(start_snap["temp"])
    start_ref = float(start_snap.get("temp_ref", start_temp))
    kp0 = float(start_snap.get("kp") if args.kp_init is None else args.kp_init)
    ki0 = float(start_snap.get("ki") if args.ki_init is None else args.ki_init)
    kd0 = float(start_snap.get("kd") if args.kd_init is None else args.kd_init)
    initial_pid = clip_pid(np.asarray([kp0, ki0, kd0], dtype=float), args)
    kp, ki, kd = float(initial_pid[0]), float(initial_pid[1]), float(initial_pid[2])

    args.start_temp = start_temp
    feature_window = initialize_feature_window(
        feature_names=feature_names,
        window_steps=window_steps,
        start_temp=start_temp,
        precondition_ref=start_ref,
        dt_s=float(args.dt_s),
        kp=kp,
        ki=ki,
        kd=kd,
    )

    pid = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    pid.p_part = float(args.initial_p)
    pid.i_part = float(args.initial_i)
    pid.d_part = float(args.initial_d)
    sync_pid_state_from_tcp(pid, start_snap, float(args.control_feature_scale))

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
        kp=kp,
        ki=ki,
        kd=kd,
    )

    if bool(args.publish_job) and not bool(args.dry_run):
        publish_job_with_retries(args)
        print(f"[run {run_id}] published temp_ref job for {args.duration_s:.1f}s")
    elif bool(args.dry_run):
        print(f"[run {run_id}] dry-run: not publishing job or writing PID")

    sample_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    pid_events: List[Dict[str, object]] = []
    next_decision_elapsed = 0.0
    decision_idx = 0
    consecutive_failures = 0
    previous_temp = start_temp

    started_at_wall = time.time()
    started_at_label = datetime.now().isoformat(timespec="seconds")
    while True:
        loop_start = time.time()
        elapsed_s = loop_start - started_at_wall
        if elapsed_s >= float(args.duration_s):
            break

        try:
            snap = read_states_with_retries(args)
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            max_failures = int(args.max_consecutive_failures)
            limit_text = "forever" if max_failures == 0 else str(max_failures)
            print(f"[run {run_id}] get_states failed ({consecutive_failures}/{limit_text}): {exc}")
            if max_failures > 0 and consecutive_failures >= max_failures:
                raise
            time.sleep(max(float(args.dt_s), float(args.tcp_reconnect_delay_s)))
            continue

        if not finite_value(snap, "temp_u"):
            print(f"[run {run_id}] skipping sample with invalid temp_u={snap.get('temp_u')}")
            time.sleep(float(args.dt_s))
            continue

        live_temp = float(snap["temp"])
        sync_pid_state_from_tcp(pid, snap, float(args.control_feature_scale))
        feature_row = state_feature_row(
            feature_names=feature_names,
            snap=snap,
            target_temp=float(args.target_temp),
            previous_temp=previous_temp,
            dt_s=float(args.dt_s),
            kp=kp,
            ki=ki,
            kd=kd,
        )
        feature_window = np.roll(feature_window, shift=-1, axis=0)
        feature_window[-1, :] = feature_row

        state.elapsed_s = float(elapsed_s)
        state.previous_temp = float(previous_temp)
        state.temp = live_temp
        state.feature_window = feature_window
        state.kp = float(kp)
        state.ki = float(ki)
        state.kd = float(kd)
        state.pid = pid
        state.diff.prev_value = previous_temp
        previous_temp = live_temp

        if elapsed_s >= next_decision_elapsed:
            decision_idx += 1
            t0 = time.perf_counter()
            decision = select_live_pid(
                state=state,
                model=model,
                checkpoint=checkpoint,
                feature_names=feature_names,
                device=device,
                args=args,
                candidate_table=candidate_table,
                ranked_candidates=ranked_candidates,
                gp_models=gp_models,
                decision_idx=decision_idx,
            )
            optimize_ms = 1000.0 * (time.perf_counter() - t0)
            selected_pid = clip_pid(
                np.asarray([decision["kp"], decision["ki"], decision["kd"]], dtype=float),
                args,
            )
            selected_tuple = (int(selected_pid[0]), int(selected_pid[1]), int(selected_pid[2]))
            changed = selected_tuple != (int(kp), int(ki), int(kd))

            if changed or bool(args.apply_unchanged):
                if not bool(args.dry_run):
                    kp_i, ki_i, kd_i = apply_pid_update_with_retries(
                        label=f"live_gp_decision_{decision_idx}@{elapsed_s:.1f}s",
                        run_id=run_id,
                        row=int(args.pid_row),
                        pid=selected_tuple,
                        args=args,
                        events=pid_events,
                    )
                    kp, ki, kd = float(kp_i), float(ki_i), float(kd_i)
                else:
                    kp, ki, kd = map(float, selected_tuple)
                    print(
                        f"[run {run_id}] dry-run PID "
                        f"decision {decision_idx}: kp={selected_tuple[0]} ki={selected_tuple[1]} kd={selected_tuple[2]}"
                    )

            decision_row = {
                "run_id": run_id,
                "decision_idx": decision_idx,
                "timestamp": loop_start,
                "elapsed_s": elapsed_s,
                "temp": live_temp,
                "temp_ref": float(args.target_temp),
                "old_kp": float(state.kp),
                "old_ki": float(state.ki),
                "old_kd": float(state.kd),
                "kp": float(kp),
                "ki": float(ki),
                "kd": float(kd),
                "changed": bool(changed),
                "band": str(decision.get("current_band", "")),
                "source": str(decision.get("source", "")),
                "rank_score": float(decision.get("rank_score", np.nan)),
                "horizon_mae": float(decision.get("horizon_mae", np.nan)),
                "horizon_final_abs_error": float(decision.get("horizon_final_abs_error", np.nan)),
                "horizon_overshoot_max": float(decision.get("horizon_overshoot_max", np.nan)),
                "horizon_weighted_motion": float(decision.get("horizon_weighted_motion", np.nan)),
                "horizon_near_std": float(decision.get("horizon_near_std", np.nan)),
                "n_evaluated": int(decision.get("n_evaluated", 0)),
                "optimize_ms": optimize_ms,
            }
            decision_rows.append(decision_row)
            print(
                f"[decision {decision_idx:03d}] t={elapsed_s:7.1f}s temp={live_temp:8.4f} "
                f"pid=({kp:4.0f},{ki:5.0f},{kd:4.0f}) src={decision_row['source']} "
                f"mae={decision_row['horizon_mae']:.4f} os={decision_row['horizon_overshoot_max']:.4f} "
                f"eval={decision_row['n_evaluated']} time={optimize_ms:.1f}ms"
            )
            next_decision_elapsed += float(args.hold_s)

        sample_rows.append({
            "run_id": run_id,
            "timestamp": loop_start,
            "elapsed_s": elapsed_s,
            "temp": live_temp,
            "temp_ref": float(args.target_temp),
            "abs_error": abs(float(args.target_temp) - live_temp),
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            "temp_u": float(snap["temp_u"]),
            "temp_u_p": float(snap["temp_u_p"]),
            "temp_u_i": float(snap["temp_u_i"]),
            "temp_u_d": float(snap["temp_u_d"]),
            "tcp_temp_ref": float(snap.get("temp_ref", np.nan)),
            "tcp_kp": float(snap.get("kp", np.nan)),
            "tcp_ki": float(snap.get("ki", np.nan)),
            "tcp_kd": float(snap.get("kd", np.nan)),
        })

        sleep_s = max(0.0, float(args.dt_s) - (time.time() - loop_start))
        time.sleep(sleep_s)

    samples_csv = output_dir / "live_samples.csv"
    decisions_csv = output_dir / "live_decisions.csv"
    events_csv = output_dir / "pid_events.csv"
    summary_json = output_dir / "live_summary.json"

    append_rows_csv(str(samples_csv), sample_rows)
    append_rows_csv(str(decisions_csv), decision_rows)
    append_rows_csv(str(events_csv), pid_events)

    summary = {
        "run_id": run_id,
        "started_at": started_at_label,
        "duration_s": float(time.time() - started_at_wall),
        "target_temp": float(args.target_temp),
        "start_temp": start_temp,
        "final_temp": None if not sample_rows else float(sample_rows[-1]["temp"]),
        "num_samples": len(sample_rows),
        "num_decisions": len(decision_rows),
        "pid_events": len(pid_events),
        "checkpoint": str(Path(args.checkpoint)),
        "gp_models": str(Path(args.gp_models)) if args.gp_models else None,
        "candidate_table": str(Path(args.candidate_table)),
        "ranked_decisions_csv": str(Path(args.ranked_decisions_csv)),
        "samples_csv": str(samples_csv),
        "decisions_csv": str(decisions_csv),
        "events_csv": str(events_csv),
    }
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Saved ===")
    print(f"samples:   {samples_csv}")
    print(f"decisions: {decisions_csv}")
    print(f"events:    {events_csv}")
    print(f"summary:   {summary_json}")


if __name__ == "__main__":
    main()
