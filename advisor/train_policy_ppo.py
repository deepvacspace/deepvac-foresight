#!/usr/bin/env python3
"""Train a PPO policy network for real-time PID control against the GRU twin.

Reward is -horizon_cost() (advisor/advise_pid.py) over each --mpc-hold-s segment,
accumulated with GAE across the episode.

Example:

    python -m advisor.train_policy_ppo --checkpoint gru/validation_rollout/gru_rollout.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.datasets import set_seed
from deepvac.mpc import add_common_mpc_args, overshoot_array
from deepvac.mpc_batch import _vec_run_pid_substeps, batched_predict_delta, feature_rows
from deepvac.pid import clip_pid, pid_bounds

from gru.gru_common import load_model

from advisor.advise_pid import add_cost_args, horizon_cost, normalized_pid_distance

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "policy_ppo"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a PPO policy against the GRU twin, replacing CEM at decision time.")

    ap.add_argument("--checkpoint", required=True, help="A rollout-trained GRU checkpoint.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--mpc-hold-s", type=float, default=20.0, help="One RL step's duration.")
    ap.add_argument("--temp-min", type=float, default=-5.0)
    ap.add_argument("--temp-max", type=float, default=26.0)

    add_common_mpc_args(ap)
    add_cost_args(ap)
    ap.set_defaults(target_temp=0.0)

    ap.add_argument("--reward-scale", type=float, default=0.01,
                    help="Multiplier applied to the terminal horizon_cost() before it becomes reward.")
    ap.add_argument("--shaping-coef", type=float, default=0.02,
                    help="Weight of the dense per-segment tracking-error and control-change reward.")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--rollout-len", type=int, default=60, help="Env steps collected per env per PPO update.")
    ap.add_argument("--ppo-epochs", type=int, default=8)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus weight.")
    ap.add_argument("--min-std", type=float, default=0.05,
                    help="Floor on the policy's action std, in normalized [-1,1] action space.")
    ap.add_argument("--overshoot-shaping-coef", type=float, default=1.0,
                    help="Weight of a dense per-segment penalty on crossing past target_temp in "
                         "the overshoot direction (deepvac.mpc.overshoot_array).")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=1, help="Print/eval every this many PPO updates.")
    ap.add_argument("--eval-envs", type=int, default=64)

    ap.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                    help="Log scalars to TensorBoard. Requires the tensorboard package.")
    ap.add_argument("--tensorboard-dir", default=None,
                    help="Default: --output-dir/tensorboard/<timestamp>.")

    return ap


def setup_tensorboard(args: argparse.Namespace, output_dir: Path):
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but the tensorboard package is not installed. "
            "Install it with: pip install tensorboard, or pass --no-tensorboard."
        ) from exc

    log_dir = Path(args.tensorboard_dir or (output_dir / "tensorboard" / datetime.now().strftime("%Y%m%dT%H%M%S")))
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("args", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    print(f"[tensorboard] logging to {log_dir}")
    print(f"[tensorboard] view with: tensorboard --logdir {log_dir.parent}")
    return writer


OBS_DIM = 8
DIFF_CLIP = 5.0  # matches gru_common.CodesysDiff / deepvac.mpc_batch._DIFF_CLIP


def mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, output_activation: nn.Module | None) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(num_layers):
        layers += [nn.Linear(dim, hidden_dim), nn.Tanh()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian actor-critic over the 3 PID gains. actor_mean returns the raw,
    unbounded pre-squash mean; callers apply tanh() once to get the [-1, 1]
    action."""

    def __init__(self, hidden_dim: int, num_layers: int, min_std: float = 0.05) -> None:
        super().__init__()
        self.actor_mean = mlp(OBS_DIM, hidden_dim, num_layers, 3, None)
        self.log_std = nn.Parameter(torch.zeros(3) - 0.5)
        self.min_log_std = float(np.log(max(min_std, 1e-6)))
        self.critic = mlp(OBS_DIM, hidden_dim, num_layers, 1, None)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.log_std.clamp(min=self.min_log_std).exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self(obs)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=obs.device), value
        dist = Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        return raw_action, log_prob, value

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class VecTwinEnv:
    """N parallel episodes stepped through the GRU twin at once, one RL step per
    --mpc-hold-s segment. Auto-resets envs individually when an episode ends or
    a candidate's rollout goes invalid.
    """

    def __init__(self, args: argparse.Namespace, model, checkpoint: dict[str, object], device: torch.device, n_envs: int, seed: int) -> None:
        self.args = args
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.n = n_envs
        self.rng = np.random.default_rng(seed)

        self.feature_names = list(checkpoint.get("feature_names", []))
        self.window_steps = int(checkpoint.get("window_steps", args.window_steps))
        self.lo, self.hi = pid_bounds(args)
        self.target_temp = float(args.target_temp)
        self.dt_s = float(args.dt_s)
        self.hold_steps = max(1, int(round(float(args.mpc_hold_s) / self.dt_s)))
        self.episode_steps = max(1, int(round(float(args.duration_s) / float(args.mpc_hold_s))))
        self.feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)

        self.temp = np.zeros(self.n)
        self.previous_temp = np.zeros(self.n)
        self.windows = np.zeros((self.n, self.window_steps, len(self.feature_names)), dtype=np.float32)
        self.i_part = np.zeros(self.n)
        self.diff_prev = np.zeros(self.n)
        self.diff_filter = np.zeros(self.n)
        self.kp = np.zeros(self.n)
        self.ki = np.zeros(self.n)
        self.kd = np.zeros(self.n)
        self.start_temp = np.zeros(self.n)
        self.step_idx = np.zeros(self.n, dtype=int)
        self.episode_temps = np.zeros((self.n, self.episode_steps * self.hold_steps))
        self.episode_start_pid = np.zeros((self.n, 3))

        self._reset_all(np.arange(self.n))

    def _reset_all(self, idx: np.ndarray) -> None:
        n = len(idx)
        if n == 0:
            return
        start_temp = self.rng.uniform(self.args.temp_min, self.args.temp_max, size=n)
        kp = self.rng.uniform(self.lo[0], self.hi[0], size=n)
        ki = self.rng.uniform(self.lo[1], self.hi[1], size=n)
        kd = self.rng.uniform(self.lo[2], self.hi[2], size=n)

        zeros = np.zeros(n)
        row = np.stack([
            {
                "temp": start_temp, "temp_ref": start_temp, "error": np.zeros(n),
                "abs_error": np.zeros(n), "dt_s": np.full(n, self.dt_s),
                "temp_velocity": zeros, "error_velocity": zeros,
                "temp_u": zeros, "temp_u_p": zeros, "temp_u_i": zeros, "temp_u_d": zeros,
                "kp": kp, "ki": ki, "kd": kd,
            }.get(name, zeros)
            for name in self.feature_names
        ], axis=1).astype(np.float32)
        window = np.repeat(row[:, None, :], self.window_steps, axis=1)

        self.temp[idx] = start_temp
        self.previous_temp[idx] = start_temp
        self.windows[idx] = window
        self.i_part[idx] = 0.0
        self.diff_prev[idx] = start_temp
        self.diff_filter[idx] = 0.0
        self.kp[idx] = kp
        self.ki[idx] = ki
        self.kd[idx] = kd
        self.start_temp[idx] = start_temp
        self.step_idx[idx] = 0
        self.episode_temps[idx] = 0.0
        self.episode_start_pid[idx] = np.stack([kp, ki, kd], axis=1)

    def _obs(self) -> np.ndarray:
        """Builds the policy's 8-dim observation vector for every env."""
        scale = max(abs(self.args.temp_min), abs(self.args.temp_max), 1e-6)
        velocity = (self.temp - self.previous_temp) / max(self.dt_s, 1e-9)
        i_scale = max(abs(float(self.args.u_min)), abs(float(self.args.u_max)), 1e-6)
        obs = np.stack([
            (self.temp - self.target_temp) / scale,
            velocity / scale,
            2.0 * (self.kp - self.lo[0]) / (self.hi[0] - self.lo[0]) - 1.0,
            2.0 * (self.ki - self.lo[1]) / (self.hi[1] - self.lo[1]) - 1.0,
            2.0 * (self.kd - self.lo[2]) / (self.hi[2] - self.lo[2]) - 1.0,
            1.0 - self.step_idx / self.episode_steps,
            self.i_part / i_scale,
            self.diff_filter / DIFF_CLIP,
        ], axis=1).astype(np.float32)
        return obs

    def obs(self) -> np.ndarray:
        return self._obs()

    def step(self, action_normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Advances --mpc-hold-s worth of twin steps holding the rescaled PID
        (action_normalized in [-1, 1]^3) fixed. Reward is a dense per-segment
        tracking term plus horizon_cost() over the whole episode on the
        terminal segment."""
        prev_step_idx = self.step_idx.copy()
        pid = self.lo + (np.clip(action_normalized, -1.0, 1.0) + 1.0) / 2.0 * (self.hi - self.lo)
        pid = clip_pid(pid, self.args)
        kp, ki, kd = pid[:, 0], pid[:, 1], pid[:, 2]
        previous_pid = np.stack([self.kp, self.ki, self.kd], axis=1)

        temp = self.temp.copy()
        previous_temp = self.previous_temp.copy()
        windows = self.windows.copy()
        i_part, diff_prev, diff_filter = self.i_part.copy(), self.diff_prev.copy(), self.diff_filter.copy()
        alive = np.ones(self.n, dtype=bool)
        segment = np.zeros((self.n, self.hold_steps))

        for t in range(self.hold_steps):
            terms, i_part, diff_prev, diff_filter = _vec_run_pid_substeps(
                temp=temp, temp_ref=self.target_temp, kp=kp, ki=ki, kd=kd,
                i_part=i_part, diff_prev=diff_prev, diff_filter=diff_filter,
                dt_s=self.dt_s, period_s=float(self.args.pid_period_s),
                feature_scale=self.feature_scale, diff_dc=0.995,
                u_min=float(self.args.u_min), u_max=float(self.args.u_max),
                i_reverse_mul=float(self.args.pid_i_reverse_mul),
            )
            windows[:, -1, :] = feature_rows(
                self.feature_names, temp=temp, temp_ref=self.target_temp, previous_temp=previous_temp,
                dt_s=self.dt_s, u=terms["u"], u_p=terms["u_p"], u_i=terms["u_i"], u_d=terms["u_d"],
                kp=kp, ki=ki, kd=kd,
            )
            next_temp = temp + batched_predict_delta(self.model, self.checkpoint, windows, self.device)
            bad = ~np.isfinite(next_temp) | (np.abs(next_temp) > float(self.args.max_abs_temp))
            next_temp = np.nan_to_num(next_temp, nan=float(self.args.max_abs_temp),
                                       posinf=float(self.args.max_abs_temp), neginf=-float(self.args.max_abs_temp))
            segment[:, t] = next_temp

            next_rows = feature_rows(
                self.feature_names, temp=next_temp, temp_ref=self.target_temp, previous_temp=temp,
                dt_s=self.dt_s, u=terms["u"], u_p=terms["u_p"], u_i=terms["u_i"], u_d=terms["u_d"],
                kp=kp, ki=ki, kd=kd,
            )
            windows = np.roll(windows, shift=-1, axis=1)
            windows[:, -1, :] = next_rows
            alive = alive & ~bad
            previous_temp = np.where(alive, temp, previous_temp)
            temp = np.where(alive, next_temp, temp)

        self.step_idx += 1
        done = (self.step_idx >= self.episode_steps) | (~alive)

        reward = -self.args.shaping_coef * np.mean((self.target_temp - segment) ** 2, axis=1)
        episode_cost = np.full(self.n, np.nan)
        for i in range(self.n):
            lo_t, hi_t = prev_step_idx[i] * self.hold_steps, (prev_step_idx[i] + 1) * self.hold_steps
            self.episode_temps[i, lo_t:hi_t] = segment[i]
            reward[i] -= self.args.shaping_coef * normalized_pid_distance(pid[i], previous_pid[i], self.args)
            seg_overshoot = overshoot_array(segment[i], start_temp=float(self.start_temp[i]), target_temp=self.target_temp)
            reward[i] -= self.args.overshoot_shaping_coef * float(seg_overshoot.max())
            if done[i]:
                metrics = horizon_cost(
                    temps=self.episode_temps[i, :hi_t], candidate_pid=pid[i],
                    previous_pid=self.episode_start_pid[i], start_temp=float(self.start_temp[i]),
                    target_temp=self.target_temp, valid=bool(alive[i]), args=self.args,
                )
                episode_cost[i] = float(metrics["cost"])
                reward[i] += -episode_cost[i] * self.args.reward_scale

        self.temp, self.previous_temp, self.windows = temp, previous_temp, windows
        self.i_part, self.diff_prev, self.diff_filter = i_part, diff_prev, diff_filter
        self.kp, self.ki, self.kd = kp, ki, kd

        info = {"final_abs_error": np.abs(self.target_temp - self.temp), "temp": self.temp.copy(),
                "episode_cost": episode_cost}

        reset_idx = np.nonzero(done)[0]
        self._reset_all(reset_idx)
        obs = self._obs()
        return obs, reward, done, info


def collect_rollout(env: VecTwinEnv, model: ActorCritic, device: torch.device, rollout_len: int, obs: np.ndarray):
    obs_buf = np.zeros((rollout_len, env.n, OBS_DIM), dtype=np.float32)
    action_buf = np.zeros((rollout_len, env.n, 3), dtype=np.float32)
    logprob_buf = np.zeros((rollout_len, env.n), dtype=np.float32)
    value_buf = np.zeros((rollout_len, env.n), dtype=np.float32)
    reward_buf = np.zeros((rollout_len, env.n), dtype=np.float32)
    done_buf = np.zeros((rollout_len, env.n), dtype=np.float32)

    ep_rewards: list[float] = []
    ep_final_errors: list[float] = []
    ep_costs: list[float] = []
    running_reward = np.zeros(env.n)

    for t in range(rollout_len):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw_action, log_prob, value = model.act(obs_t)
        action_np = raw_action.cpu().numpy()
        next_obs, reward, done, info = env.step(np.tanh(action_np))

        obs_buf[t] = obs
        action_buf[t] = action_np
        logprob_buf[t] = log_prob.cpu().numpy()
        value_buf[t] = value.cpu().numpy()
        reward_buf[t] = reward
        done_buf[t] = done.astype(np.float32)

        running_reward += reward
        finished = np.nonzero(done)[0]
        for i in finished:
            ep_rewards.append(float(running_reward[i]))
            ep_final_errors.append(float(info["final_abs_error"][i]))
            ep_costs.append(float(info["episode_cost"][i]))
            running_reward[i] = 0.0

        obs = next_obs

    with torch.no_grad():
        _, _, last_value = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
    last_value = last_value.cpu().numpy()

    return (obs_buf, action_buf, logprob_buf, value_buf, reward_buf, done_buf, last_value, obs,
            ep_rewards, ep_final_errors, ep_costs)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray,
                 gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    rollout_len, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(n_envs)
    for t in reversed(range(rollout_len)):
        next_value = last_value if t == rollout_len - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer, device: torch.device,
    obs_buf, action_buf, logprob_buf, advantages, returns,
    ppo_epochs: int, minibatches: int, clip_eps: float, entropy_coef: float,
    value_coef: float, max_grad_norm: float,
) -> dict[str, float]:
    n = obs_buf.shape[0]
    obs_t = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
    action_t = torch.as_tensor(action_buf, dtype=torch.float32, device=device)
    logprob_t = torch.as_tensor(logprob_buf, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    batch_size = max(1, n // minibatches)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n_updates = 0

    for _ in range(ppo_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            new_logprob, entropy, value = model.evaluate(obs_t[idx], action_t[idx])
            ratio = (new_logprob - logprob_t[idx]).exp()
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value - return_t[idx]) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(-entropy_loss.item())
            stats["clip_frac"] += float(((ratio - 1.0).abs() > clip_eps).float().mean().item())
            n_updates += 1

    return {k: v / max(1, n_updates) for k, v in stats.items()}


def evaluate_policy(env: VecTwinEnv, model: ActorCritic, device: torch.device) -> dict[str, float]:
    """Runs env.episode_steps deterministic steps and returns mean episode cost."""
    obs = env.obs()
    for _ in range(env.episode_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _, _ = model(obs_t)
        obs, reward, done, info = env.step(np.tanh(mean.cpu().numpy()))
    return {
        "mean_episode_cost": float(np.nanmean(info["episode_cost"])),
        "mean_final_abs_error": float(np.mean(info["final_abs_error"])),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        print(f"[device] using cuda ({torch.cuda.get_device_name(device)})")
    else:
        print("[device] using cpu")

    twin, checkpoint = load_model(Path(args.checkpoint), device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = setup_tensorboard(args, output_dir)

    train_env = VecTwinEnv(args, twin, checkpoint, device, args.n_envs, args.seed)
    eval_env = VecTwinEnv(args, twin, checkpoint, device, args.eval_envs, args.seed + 1)

    model = ActorCritic(args.hidden_dim, args.num_layers, min_std=args.min_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_update = args.n_envs * args.rollout_len
    n_updates = max(1, args.total_timesteps // steps_per_update)
    print(f"[ppo] {n_updates} updates x {steps_per_update} steps/update "
          f"({args.n_envs} envs x {args.rollout_len} rollout_len)")

    obs = train_env.obs()
    best_eval_cost = float("inf")
    history: list[dict[str, float]] = []

    for update in range(1, n_updates + 1):
        (obs_buf, action_buf, logprob_buf, value_buf, reward_buf, done_buf,
         last_value, obs, ep_rewards, ep_final_errors, ep_costs) = collect_rollout(
            train_env, model, device, args.rollout_len, obs
        )
        advantages, returns = compute_gae(reward_buf, value_buf, done_buf, last_value, args.gamma, args.gae_lambda)

        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        stats = ppo_update(
            model, optimizer, device,
            flat(obs_buf), flat(action_buf), flat(logprob_buf), flat(advantages), flat(returns),
            args.ppo_epochs, args.minibatches, args.clip_eps, args.entropy_coef,
            args.value_coef, args.max_grad_norm,
        )

        if update % args.log_every == 0 or update == n_updates:
            eval_metrics = evaluate_policy(eval_env, model, device)
            row = {
                "update": update,
                "timesteps": update * steps_per_update,
                "mean_train_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
                "mean_train_final_abs_error": float(np.mean(ep_final_errors)) if ep_final_errors else float("nan"),
                "mean_train_episode_cost": float(np.mean(ep_costs)) if ep_costs else float("nan"),
                **eval_metrics,
                **stats,
            }
            history.append(row)
            if writer is not None:
                for key, value in row.items():
                    if key not in ("update", "timesteps") and np.isfinite(value):
                        writer.add_scalar(key, value, global_step=row["timesteps"])
            print(
                f"[update {update:4d}/{n_updates}] timesteps={row['timesteps']} "
                f"eval_cost={eval_metrics['mean_episode_cost']:.4f} "
                f"eval_final_err={eval_metrics['mean_final_abs_error']:.4f} "
                f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f}"
            )

            if eval_metrics["mean_episode_cost"] < best_eval_cost:
                best_eval_cost = eval_metrics["mean_episode_cost"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "min_std": args.min_std,
                    "obs_dim": OBS_DIM,
                    "temp_min": args.temp_min, "temp_max": args.temp_max,
                    "target_temp": args.target_temp,
                    "kp_min": args.kp_min, "kp_max": args.kp_max,
                    "ki_min": args.ki_min, "ki_max": args.ki_max,
                    "kd_min": args.kd_min, "kd_max": args.kd_max,
                    "mpc_hold_s": args.mpc_hold_s, "dt_s": args.dt_s,
                    "gru_checkpoint": str(Path(args.checkpoint).resolve()),
                    "best_eval_cost": best_eval_cost,
                }, output_dir / "policy.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if writer is not None:
        writer.close()
    print(f"\nbest eval cost : {best_eval_cost:.4f}")
    print(f"checkpoint     : {output_dir / 'policy.pt'}")
    print(f"history        : {output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
