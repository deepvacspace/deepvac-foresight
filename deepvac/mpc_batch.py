"""Vectorized GRU+MPC rollouts: score a whole candidate population in one batch.

A vectorization of digitaltwin.common.ChamberPID, digitaltwin.common.CodesysDiff and
deepvac.mpc.step_state. Only i_part, diff.prev_value and diff.filter_out carry
across steps; every other PID term is recomputed.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from deepvac.mpc import (
    CandidateTable,
    SimState,
    inject_seed_candidates,
    sample_candidates_random,
    select_history_candidates,
)
from deepvac.pid import clip_pid, pid_bounds

CostFn = Callable[..., dict[str, float]]

# digitaltwin.common.CodesysDiff / ChamberPID internal limits.
_DIFF_CLIP = 5.0
_DIFF_GAIN = 10.0
_D_PART_CLIP = 0.4


def _vec_diff_update(
    value: np.ndarray,
    prev_value: np.ndarray,
    filter_out: np.ndarray,
    dc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized digitaltwin.common.CodesysDiff.update."""
    diff_value = value - prev_value
    filter_in = np.clip(diff_value, -_DIFF_CLIP, _DIFF_CLIP)
    filter_out = dc * filter_out + (1.0 - dc) * filter_in
    out = _DIFF_GAIN * np.clip(filter_out, -_DIFF_CLIP, _DIFF_CLIP)
    return out, value.copy(), filter_out


def _vec_pid_step(
    *,
    x_target: float,
    x_measured: np.ndarray,
    kp: np.ndarray,
    ki: np.ndarray,
    kd: np.ndarray,
    diff_out: np.ndarray,
    i_part: np.ndarray,
    u_min: float,
    u_max: float,
    i_reverse_mul: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized digitaltwin.common.ChamberPID.step (enable=True).

    Mirrors the scalar ordering exactly: u sums the *unclipped* p_part with the
    clipped i_part/d_part, and p_part is only clipped afterwards for reporting.
    """
    delta = float(x_target) - x_measured

    # The scalar version returns 0 and leaves state untouched when p_coef == 0.
    live = kp != 0.0
    safe_kp = np.where(live, kp, 1.0)
    inv_kp = 1.0 / safe_kp

    p_part = inv_kp * delta

    effective_i = np.where(delta * i_part < 0.0, ki * i_reverse_mul, ki)
    delta_edge = 1.2 * kp
    can_integrate = live & (effective_i != 0.0) & (np.abs(delta) < delta_edge)
    safe_i = np.where(effective_i != 0.0, effective_i, 1.0)
    i_part = np.where(can_integrate, i_part + inv_kp * (delta * 0.1 / safe_i), i_part)

    d_part = inv_kp * (kd * -diff_out)

    i_part = np.clip(i_part, u_min, u_max)
    d_part = np.clip(d_part, -_D_PART_CLIP, _D_PART_CLIP)

    u = np.clip(p_part + i_part + d_part, u_min, u_max)
    p_part = np.clip(p_part, u_min, u_max)

    zero = np.zeros_like(u)
    return (
        np.where(live, u, zero),
        np.where(live, p_part, zero),
        i_part,
        np.where(live, d_part, zero),
    )


def _vec_run_pid_substeps(
    *,
    temp: np.ndarray,
    temp_ref: float,
    kp: np.ndarray,
    ki: np.ndarray,
    kd: np.ndarray,
    i_part: np.ndarray,
    diff_prev: np.ndarray,
    diff_filter: np.ndarray,
    dt_s: float,
    period_s: float,
    feature_scale: float,
    diff_dc: float,
    u_min: float,
    u_max: float,
    i_reverse_mul: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized deepvac.mpc.run_pid_substeps with temp_mode="hold"."""
    period = max(float(period_s), 1e-6)
    dt = max(float(dt_s), period)
    n_substeps = max(1, int(round(dt / period)))

    u = u_p = u_i = u_d = np.zeros_like(temp)
    for _ in range(n_substeps):
        # temp_mode="hold": the measured temperature is constant across substeps.
        diff_out, diff_prev, diff_filter = _vec_diff_update(temp, diff_prev, diff_filter, diff_dc)
        u, u_p, i_part, u_d = _vec_pid_step(
            x_target=temp_ref,
            x_measured=temp,
            kp=kp,
            ki=ki,
            kd=kd,
            diff_out=diff_out,
            i_part=i_part,
            u_min=u_min,
            u_max=u_max,
            i_reverse_mul=i_reverse_mul,
        )
        u_i = i_part

    terms = {
        "u": u * feature_scale,
        "u_p": u_p * feature_scale,
        "u_i": u_i * feature_scale,
        "u_d": u_d * feature_scale,
    }
    return terms, i_part, diff_prev, diff_filter


def feature_rows(
    feature_names: Sequence[str],
    *,
    temp: np.ndarray,
    temp_ref: float,
    previous_temp: np.ndarray,
    dt_s: float,
    u: np.ndarray,
    u_p: np.ndarray,
    u_i: np.ndarray,
    u_d: np.ndarray,
    kp: np.ndarray,
    ki: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Vectorized deepvac.mpc.make_feature_row. Returns (N, n_features)."""
    error = float(temp_ref) - temp
    dt_safe = max(float(dt_s), 1e-9)
    velocity = (temp - previous_temp) / dt_safe
    ones = np.ones_like(temp)

    values = {
        "temp": temp,
        "temp_ref": float(temp_ref) * ones,
        "error": error,
        "abs_error": np.abs(error),
        "dt_s": float(dt_s) * ones,
        "temp_velocity": velocity,
        "error_velocity": -velocity,
        "temp_u": u,
        "temp_u_p": u_p,
        "temp_u_i": u_i,
        "temp_u_d": u_d,
        "kp": kp,
        "ki": ki,
        "kd": kd,
    }
    zeros = np.zeros_like(temp)
    return np.stack([values.get(name, zeros) for name in feature_names], axis=1).astype(np.float32)


def batched_predict_delta(
    model: Any,
    checkpoint: dict[str, object],
    windows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Batched digitaltwin.common.predict_delta_t1 over (N, window_steps, n_features)."""
    x_scaler = checkpoint["x_scaler"]
    y_scaler = checkpoint["y_scaler"]

    n, window_steps, n_features = windows.shape
    scaled = x_scaler.transform(windows.reshape(-1, n_features)).reshape(n, window_steps, n_features)
    xb = torch.as_tensor(scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_scaled = model(xb).cpu().numpy()

    return np.asarray(y_scaler.inverse_transform(pred_scaled)[:, 0], dtype=float)


def rollout_population(
    *,
    initial_state: SimState,
    candidate_pids: np.ndarray,
    model: Any,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    horizon_steps: int,
    cost_fn: CostFn,
    diff_dc: float = 0.995,
) -> list[dict[str, float]]:
    """Roll every candidate forward over the horizon in one batched pass.

    Equivalent to calling deepvac.mpc.rollout_constant_pid once per row of
    candidate_pids, including the early-stop-on-invalid behaviour: a candidate
    whose predicted temperature goes non-finite or leaves --max-abs-temp keeps
    that step and then stops contributing further ones.
    """
    pids = clip_pid(np.atleast_2d(np.asarray(candidate_pids, dtype=float)), args)
    n = len(pids)
    kp, ki, kd = pids[:, 0].copy(), pids[:, 1].copy(), pids[:, 2].copy()

    target_temp = float(args.target_temp)
    dt_s = float(args.dt_s)
    max_abs_temp = float(args.max_abs_temp)
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)

    temp = np.full(n, float(initial_state.temp), dtype=float)
    previous_temp = np.full(n, float(initial_state.previous_temp), dtype=float)
    windows = np.repeat(
        np.asarray(initial_state.feature_window, dtype=np.float32)[None, :, :], n, axis=0
    )

    i_part = np.full(n, float(initial_state.pid.i_part), dtype=float)
    diff_prev = np.full(n, float(initial_state.diff.prev_value), dtype=float)
    diff_filter = np.full(n, float(initial_state.diff.filter_out), dtype=float)

    alive = np.ones(n, dtype=bool)
    lengths = np.zeros(n, dtype=int)
    temps = np.zeros((n, horizon_steps), dtype=float)

    for step in range(horizon_steps):
        terms, i_part, diff_prev, diff_filter = _vec_run_pid_substeps(
            temp=temp,
            temp_ref=target_temp,
            kp=kp,
            ki=ki,
            kd=kd,
            i_part=i_part,
            diff_prev=diff_prev,
            diff_filter=diff_filter,
            dt_s=dt_s,
            period_s=float(args.pid_period_s),
            feature_scale=feature_scale,
            diff_dc=diff_dc,
            u_min=float(args.u_min),
            u_max=float(args.u_max),
            i_reverse_mul=float(args.pid_i_reverse_mul),
        )

        # predict_next(): the last window row is rewritten from the *current* temp.
        windows[:, -1, :] = feature_rows(
            feature_names,
            temp=temp, temp_ref=target_temp, previous_temp=previous_temp, dt_s=dt_s,
            u=terms["u"], u_p=terms["u_p"], u_i=terms["u_i"], u_d=terms["u_d"],
            kp=kp, ki=ki, kd=kd,
        )
        next_temp = temp + batched_predict_delta(model, checkpoint, windows, device)

        bad = ~np.isfinite(next_temp) | (np.abs(next_temp) > max_abs_temp)
        # Non-finite values only; finite outliers are left alone.
        next_temp = np.nan_to_num(next_temp, nan=max_abs_temp, posinf=max_abs_temp, neginf=-max_abs_temp)

        temps[:, step] = next_temp
        lengths[alive] = step + 1

        # step_state(): roll the window and append a row built from the new temp.
        next_rows = feature_rows(
            feature_names,
            temp=next_temp, temp_ref=target_temp, previous_temp=temp, dt_s=dt_s,
            u=terms["u"], u_p=terms["u_p"], u_i=terms["u_i"], u_d=terms["u_d"],
            kp=kp, ki=ki, kd=kd,
        )
        windows = np.roll(windows, shift=-1, axis=1)
        windows[:, -1, :] = next_rows

        alive = alive & ~bad
        if not alive.any():
            break

        # Freeze dead candidates so their garbage cannot poison later batches.
        previous_temp = np.where(alive, temp, previous_temp)
        temp = np.where(alive, next_temp, temp)

    valid = lengths >= horizon_steps
    previous_pid = np.asarray([initial_state.kp, initial_state.ki, initial_state.kd], dtype=float)
    start_temp = float(args.start_temp)

    results: list[dict[str, float]] = []
    for idx in range(n):
        metrics = cost_fn(
            temps=temps[idx, : lengths[idx]],
            candidate_pid=pids[idx],
            previous_pid=previous_pid,
            start_temp=start_temp,
            target_temp=target_temp,
            valid=bool(valid[idx]),
            args=args,
        )
        metrics["kp"] = float(pids[idx, 0])
        metrics["ki"] = float(pids[idx, 1])
        metrics["kd"] = float(pids[idx, 2])
        results.append(metrics)

    return results


def optimize_pid_batched(
    *,
    state: SimState,
    model: Any,
    checkpoint: dict[str, object],
    feature_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    rng: np.random.Generator,
    cost_fn: CostFn,
    candidate_table: CandidateTable | None = None,
    decision_idx: int | None = None,
    time_budget_s: float | None = None,
) -> dict[str, float]:
    """Batched equivalent of deepvac.mpc.optimize_pid_for_state.

    Adds time_budget_s: CEM stops after the current iteration once the budget is
    spent, so a live decision degrades to fewer refinement passes instead of
    overrunning its hold interval.
    """
    lo, hi = pid_bounds(args)
    current_pid = np.asarray([state.kp, state.ki, state.kd], dtype=float)
    horizon_steps = max(1, int(math.ceil(float(args.mpc_horizon_s) / float(args.dt_s))))
    population = max(4, int(args.cem_population))
    label = f"{int(decision_idx):03d}" if decision_idx is not None else "---"
    verbose = bool(getattr(args, "print_optimizer_progress", True))

    history_candidates = select_history_candidates(
        state=state, args=args, candidate_table=candidate_table
    )

    t0 = time.perf_counter()
    current_metrics = rollout_population(
        initial_state=state, candidate_pids=current_pid[None, :], model=model,
        checkpoint=checkpoint, feature_names=feature_names, device=device, args=args,
        horizon_steps=horizon_steps, cost_fn=cost_fn,
    )[0]
    current_cost = float(current_metrics["cost"])

    def evaluate(samples: np.ndarray) -> list[dict[str, float]]:
        return rollout_population(
            initial_state=state, candidate_pids=samples, model=model,
            checkpoint=checkpoint, feature_names=feature_names, device=device, args=args,
            horizon_steps=horizon_steps, cost_fn=cost_fn,
        )

    evaluated: list[dict[str, float]] = []
    iterations_run = 0

    if args.optimizer == "random":
        samples = sample_candidates_random(rng, population, args, center=current_pid)
        seed_start = 0
        if bool(args.include_current_candidate):
            samples[0, :] = current_pid
            seed_start = 1
        inject_seed_candidates(samples, history_candidates, start_idx=seed_start)
        evaluated = evaluate(samples)
        iterations_run = 1
    else:
        mean = np.clip(current_pid, lo, hi)
        std = 0.40 * (hi - lo)
        min_std = np.maximum(float(args.cem_min_std_frac) * (hi - lo), 1e-9)
        elite_n = max(2, int(round(population * float(args.cem_elite_frac))))
        total_iters = max(1, int(args.cem_iterations))

        for cem_iter in range(total_iters):
            samples = np.clip(rng.normal(loc=mean, scale=std, size=(population, 3)), lo, hi)
            samples = np.clip(np.floor(samples + 0.5), lo, hi)
            seed_start = 0
            if bool(args.include_current_candidate):
                samples[0, :] = current_pid
                seed_start = 1
            inject_seed_candidates(samples, history_candidates, start_idx=seed_start)

            rows = evaluate(samples)
            for row in rows:
                row["cem_iter"] = cem_iter + 1
            rows.sort(key=lambda r: float(r["cost"]))
            evaluated.extend(rows)
            iterations_run = cem_iter + 1

            elites = np.asarray([[r["kp"], r["ki"], r["kd"]] for r in rows[:elite_n]], dtype=float)
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), min_std)

            if verbose:
                print(
                    f"[decision {label}] cem {cem_iter + 1}/{total_iters} "
                    f"best={float(rows[0]['cost']):.4f} "
                    f"pid=({float(rows[0]['kp']):.0f}, {float(rows[0]['ki']):.0f}, {float(rows[0]['kd']):.0f}) "
                    f"elapsed={1000.0 * (time.perf_counter() - t0):.0f}ms",
                    flush=True,
                )

            if time_budget_s is not None and (time.perf_counter() - t0) >= float(time_budget_s):
                if verbose and cem_iter + 1 < total_iters:
                    print(
                        f"[decision {label}] time budget {float(time_budget_s):.1f}s reached, "
                        f"stopping after CEM iteration {cem_iter + 1}/{total_iters}",
                        flush=True,
                    )
                break

    evaluated.sort(key=lambda r: float(r["cost"]))
    best = dict(evaluated[0])
    best_pid = np.asarray([best["kp"], best["ki"], best["kd"]], dtype=float)

    # Safety gate: keep the current PID unless the improvement is meaningful.
    if float(best["cost"]) >= current_cost - float(args.apply_margin):
        best = dict(current_metrics)
        best["kp"], best["ki"], best["kd"] = (float(v) for v in current_pid)
        best["changed"] = False
    else:
        best["changed"] = bool(np.linalg.norm(best_pid - current_pid) > 1e-9)

    best["current_cost"] = current_cost
    best["n_evaluated"] = int(len(evaluated))
    best["horizon_steps"] = int(horizon_steps)
    best["n_history_candidates"] = int(len(history_candidates))
    best["cem_iterations_run"] = int(iterations_run)
    best["decision_seconds"] = float(time.perf_counter() - t0)
    return best
