#!/usr/bin/env python3
"""Label randomized digital-twin states with CEM's chosen PID triplet, producing a
(observation, action) dataset for advisor.train_policy_bc.

Each episode starts from a random (temp, kp, ki, kd). At every decision point
the production CEM optimizer (deepvac.mpc_batch.optimize_pid_batched) is
queried for its chosen PID and the (observation, PID) pair is recorded. With
probability --explore-frac a random PID is executed instead of CEM's own
choice before advancing to the next decision.

Example:

    python -m advisor.generate_bc_dataset --checkpoint digitaltwin/gru/validation_rollout/gru_rollout.pt --n-states 20000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.mpc import SimState, add_common_mpc_args, initialize_feature_window, step_state  
from deepvac.mpc_batch import optimize_pid_batched  
from deepvac.pid import clip_pid, pid_bounds  

from digitaltwin.common import ChamberPID, CodesysDiff, load_model, predict_delta_t1

from advisor.advise_pid import add_cost_args, horizon_cost  
from advisor.train_policy_ppo import DIFF_CLIP  

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bc_dataset" / "cem_labels.npz"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Label randomized twin states with CEM's chosen PID for behavior cloning.")

    ap.add_argument("--checkpoint", required=True, help="A rollout-trained digital-twin checkpoint (GRU or LSTM).")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--n-states", type=int, default=20_000)
    ap.add_argument("--mpc-hold-s", type=float, default=20.0, help="One decision's duration.")
    ap.add_argument("--mpc-horizon-s", type=float, default=60.0, help="CEM's own lookahead per decision.")
    ap.add_argument("--temp-min", type=float, default=-5.0)
    ap.add_argument("--temp-max", type=float, default=26.0)
    ap.add_argument("--explore-frac", type=float, default=0.2,
                    help="Probability of executing a random PID instead of CEM's choice after labeling.")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="If --output already exists, load it and keep adding states on top "
                         "instead of overwriting it from scratch.")

    add_common_mpc_args(ap)
    add_cost_args(ap)
    ap.set_defaults(target_temp=0.0, print_optimizer_progress=False, print_every_decision=False)

    return ap


def compute_obs(state: SimState, step_idx: int, episode_steps: int, args: argparse.Namespace,
                 lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Builds the same 8-dim observation as advisor.train_policy_ppo.VecTwinEnv._obs(), for one state."""
    scale = max(abs(args.temp_min), abs(args.temp_max), 1e-6)
    velocity = (state.temp - state.previous_temp) / max(float(args.dt_s), 1e-9)
    i_scale = max(abs(float(args.u_min)), abs(float(args.u_max)), 1e-6)
    return np.array([
        (state.temp - args.target_temp) / scale,
        velocity / scale,
        2.0 * (state.kp - lo[0]) / (hi[0] - lo[0]) - 1.0,
        2.0 * (state.ki - lo[1]) / (hi[1] - lo[1]) - 1.0,
        2.0 * (state.kd - lo[2]) / (hi[2] - lo[2]) - 1.0,
        1.0 - step_idx / episode_steps,
        state.pid.i_part / i_scale,
        state.diff.filter_out / DIFF_CLIP,
    ], dtype=np.float32)


def sample_initial_state(args: argparse.Namespace, feature_names: list, window_steps: int,
                          lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> SimState:
    start_temp = float(rng.uniform(args.temp_min, args.temp_max))
    kp0, ki0, kd0 = (float(v) for v in rng.uniform(lo, hi))
    precondition_ref = start_temp if args.precondition_ref is None else float(args.precondition_ref)

    feature_window = initialize_feature_window(
        feature_names=feature_names, window_steps=window_steps,
        start_temp=start_temp, precondition_ref=precondition_ref,
        dt_s=float(args.dt_s), kp=kp0, ki=ki0, kd=kd0,
    )
    pid_ctrl = ChamberPID(args.u_min, args.u_max, args.pid_i_reverse_mul)
    diff_ctrl = CodesysDiff()
    diff_ctrl.prev_value = start_temp

    return SimState(
        elapsed_s=0.0, temp=start_temp, previous_temp=start_temp,
        feature_window=feature_window, pid=pid_ctrl, diff=diff_ctrl,
        kp=kp0, ki=ki0, kd=kd0,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[device] using {device}")

    twin, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", []))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))
    lo, hi = pid_bounds(args)
    dt_s = float(args.dt_s)
    hold_steps = max(1, round(float(args.mpc_hold_s) / dt_s))
    episode_steps = max(1, round(float(args.duration_s) / float(args.mpc_hold_s)))
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    pid_list: list[np.ndarray] = []
    cost_list: list[float] = []

    if args.resume and output_path.exists():
        existing = np.load(output_path)
        obs_list = list(existing["obs"])
        action_list = list(existing["action"])
        pid_list = list(existing["raw_pid"])
        cost_list = list(existing["cost"])
        print(f"[resume] loaded {len(obs_list)} existing states from {output_path}")

    seed_offset = len(obs_list)
    rng = np.random.default_rng(args.seed + seed_offset)
    rng_cem = np.random.default_rng(args.seed + seed_offset + 1)

    def save() -> None:
        np.savez(
            output_path,
            obs=np.stack(obs_list).astype(np.float32),
            action=np.stack(action_list).astype(np.float32),
            raw_pid=np.stack(pid_list).astype(np.float32),
            cost=np.asarray(cost_list, dtype=np.float32),
        )
        print(f"[save] {len(obs_list)} states -> {output_path}")

    t_start = time.time()
    episode_idx = 0

    while len(obs_list) < args.n_states:
        episode_idx += 1
        state = sample_initial_state(args, feature_names, window_steps, lo, hi, rng)

        for step_idx in range(episode_steps):
            if len(obs_list) >= args.n_states:
                break

            obs = compute_obs(state, step_idx, episode_steps, args, lo, hi)
            decision = optimize_pid_batched(
                state=state, model=twin, checkpoint=checkpoint, feature_names=feature_names,
                device=device, args=args, rng=rng_cem, cost_fn=horizon_cost, decision_idx=step_idx,
            )
            cem_pid = np.array([decision["kp"], decision["ki"], decision["kd"]], dtype=float)
            target_action = 2.0 * (cem_pid - lo) / (hi - lo) - 1.0

            obs_list.append(obs)
            action_list.append(target_action.astype(np.float32))
            pid_list.append(cem_pid.astype(np.float32))
            cost_list.append(float(decision["cost"]))

            if rng.random() < args.explore_frac:
                exec_pid = clip_pid(rng.uniform(lo, hi), args)
            else:
                exec_pid = clip_pid(cem_pid, args)
            state.kp, state.ki, state.kd = float(exec_pid[0]), float(exec_pid[1]), float(exec_pid[2])

            valid_segment = True
            for _ in range(hold_steps):
                state, _info, step_valid, _reason = step_state(
                    state=state, model=twin, checkpoint=checkpoint, feature_names=feature_names,
                    device=device, target_temp=float(args.target_temp), dt_s=dt_s,
                    pid_period_s=float(args.pid_period_s), feature_scale=feature_scale,
                    max_abs_temp=float(args.max_abs_temp), predict_fn=predict_delta_t1,
                )
                if not step_valid:
                    valid_segment = False
                    break
            if not valid_segment:
                break

            if len(obs_list) % args.save_every == 0:
                save()
                elapsed = time.time() - t_start
                rate = len(obs_list) / max(elapsed, 1e-9)
                remaining = (args.n_states - len(obs_list)) / max(rate, 1e-9)
                print(f"[progress] {len(obs_list)}/{args.n_states} states, episode {episode_idx}, "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining ({rate:.2f} states/s)")

    save()
    print(f"[done] {len(obs_list)} states across {episode_idx} episodes -> {output_path}")


if __name__ == "__main__":
    main()
