#!/usr/bin/env python3
"""Behavior-clone advisor.train_policy_ppo's ActorCritic from CEM-labeled
states (advisor.generate_bc_dataset's output), by supervised regression onto
CEM's own decisions. Output checkpoint format matches train_policy_ppo's, so
it can be evaluated with the same harness or used as a PPO warm start.

Example:

    python -m advisor.train_policy_bc --checkpoint digitaltwin/gru/validation_rollout/gru_rollout.pt \
        --dataset advisor/bc_dataset/cem_labels.npz --hidden-dim 128 --num-layers 3
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.datasets import set_seed  

from digitaltwin.common import load_model

from advisor.train_policy_ppo import OBS_DIM, ActorCritic, VecTwinEnv, evaluate_policy  
from advisor.train_policy_ppo import build_arg_parser as build_ppo_arg_parser  

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "policy_bc"


def build_arg_parser():
    ap = build_ppo_arg_parser()
    ap.description = "Behavior-clone a PID policy from CEM-labeled (state, action) pairs."
    ap.set_defaults(output_dir=str(DEFAULT_OUTPUT_DIR))

    ap.add_argument("--dataset", required=True, help="npz produced by advisor.generate_bc_dataset.")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--bc-patience", type=int, default=20)
    ap.add_argument("--bc-batch-size", type=int, default=256)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--bc-weight-decay", type=float, default=1e-5)

    return ap


def predict_action(model: ActorCritic, obs: torch.Tensor) -> torch.Tensor:
    """The action the policy executes for a given observation: tanh(actor_mean(obs))."""
    mean, _, _ = model(obs)
    return torch.tanh(mean)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[device] using {device}")

    data = np.load(args.dataset)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    action = torch.as_tensor(data["action"], dtype=torch.float32, device=device)
    n = obs.shape[0]
    print(f"[data] {n} labeled states from {args.dataset}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long, device=device)
    train_idx = torch.as_tensor(perm[n_val:], dtype=torch.long, device=device)
    print(f"[data] train={len(train_idx)} val={len(val_idx)}")

    model = ActorCritic(args.hidden_dim, args.num_layers, min_std=args.min_std).to(device)
    optimizer = torch.optim.Adam(model.actor_mean.parameters(), lr=args.bc_lr, weight_decay=args.bc_weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.bc_epochs + 1):
        model.train()
        epoch_perm = train_idx[torch.randperm(len(train_idx), device=device)]
        total_loss = 0.0
        for start in range(0, len(epoch_perm), args.bc_batch_size):
            batch_idx = epoch_perm[start:start + args.bc_batch_size]
            pred = predict_action(model, obs[batch_idx])
            loss = loss_fn(pred, action[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor_mean.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)
        train_loss = total_loss / len(train_idx)

        model.eval()
        with torch.no_grad():
            val_pred = predict_action(model, obs[val_idx])
            val_loss = float(loss_fn(val_pred, action[val_idx]).item())
            val_mae_pid_norm = float((val_pred - action[val_idx]).abs().mean().item())

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mae_pid_norm": val_mae_pid_norm,
        })
        print(f"[epoch {epoch:4d}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"val_mae_norm={val_mae_pid_norm:.4f}{' *' if improved else ''}")

        if epochs_without_improve >= args.bc_patience:
            print(f"[stop] no val improvement for {args.bc_patience} epochs")
            break

    model.load_state_dict(best_state)

    print("[eval] running the same fixed-episode rollout evaluate_policy() uses for PPO...")
    twin, twin_checkpoint = load_model(Path(args.checkpoint), device)
    eval_env = VecTwinEnv(args, twin, twin_checkpoint, device, args.eval_envs, args.seed + 1)
    eval_metrics = evaluate_policy(eval_env, model, device)
    print(f"[eval] mean_episode_cost={eval_metrics['mean_episode_cost']:.4f} "
          f"mean_final_abs_error={eval_metrics['mean_final_abs_error']:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "best_eval_cost": eval_metrics["mean_episode_cost"],
        "train_method": "behavior_cloning",
        "dataset": str(Path(args.dataset).resolve()),
    }, output_dir / "policy.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"checkpoint : {output_dir / 'policy.pt'}")
    print(f"history    : {output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
