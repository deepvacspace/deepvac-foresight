#!/usr/bin/env python3
"""Train an MLP cost surrogate for CEM candidate screening.

Predicts a candidate PID's rollout cost directly from a state
(temp, current_pid, candidate_pid), instead of simulating the GRU twin.

1. Generate a labeled dataset: sample (temp, current_pid) states, score
   candidate PIDs per state via deepvac.mpc_batch.rollout_population and
   gru.advise_pid.horizon_cost.
2. Train a small MLP (gru/cost_surrogate.py) to regress cost, then report
   rank correlation and top-K% screening hit rate against true cost on
   held-out states.

--target-normalization: normalize cost within each state before training.
--candidate-sampling: how candidate PIDs are drawn per state -- uniform,
cem (replays deepvac.mpc.optimize_pid_for_state's mean/std elite-update
trajectory via batched rollouts), or both mixed.
--optuna-trials: sweep hidden_dim/num_layers/dropout/batch_size/lr/
weight_decay/huber_beta via Optuna, reusing one generated/loaded dataset.

Scoped to one --target-temp and one horizon (--mpc-horizon-s/--dt-s), the
same horizon gru/advise_pid.py must be run with.

Example:

    python -m gru.train_cost_surrogate --checkpoint gru/validation_rollout/gru_rollout.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.datasets import set_seed
from deepvac.mpc import add_common_mpc_args, sample_candidates_random
from deepvac.mpc_batch import rollout_population
from deepvac.pid import pid_bounds

from gru.advise_pid import add_cost_args, build_initial_state, horizon_cost
from gru.cost_surrogate import FEATURE_NAMES, CostSurrogateMLP
from gru.gru_common import load_model
from gru.twin_acceptance import rank_correlation

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "cost_surrogate"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a fast cost surrogate for CEM candidate screening.")

    ap.add_argument("--checkpoint", required=True, help="A rollout-trained GRU checkpoint.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--dataset-csv", default=None,
                    help="Reuse a previously generated dataset instead of sampling again.")
    ap.add_argument("--save-dataset-csv", default=None,
                    help="Where to save the generated dataset. Default: --output-dir/dataset.csv.")
    ap.add_argument("--mpc-horizon-s", type=float, default=60.0,
                    help="Horizon each sampled candidate is rolled out and scored over.")

    add_common_mpc_args(ap)
    add_cost_args(ap)
    ap.set_defaults(start_temp=24.0, target_temp=0.0)

    ap.add_argument("--temp-min", type=float, default=-5.0)
    ap.add_argument("--temp-max", type=float, default=26.0)
    ap.add_argument("--n-states", type=int, default=300)
    ap.add_argument("--n-candidates-per-state", type=int, default=300,
                    help="Candidates drawn uniformly per state. Used as-is for "
                         "--candidate-sampling uniform/mixed; ignored for cem (which uses "
                         "--cem-population x --cem-iterations instead).")
    ap.add_argument(
        "--candidate-sampling", choices=["uniform", "cem", "mixed"], default="mixed",
        help="How each state's candidate PIDs are drawn: uniform across the kp/ki/kd bounds, "
             "cem (replays the CEM elite-update trajectory, --cem-population x --cem-iterations "
             "rollouts per state), or both.",
    )

    ap.add_argument(
        "--target-normalization", choices=["none", "center", "zscore"], default="zscore",
        help="Normalize cost within each state before training: 'center' subtracts the "
             "state's median cost, 'zscore' also divides by the state's cost std, 'none' "
             "trains on raw cost.",
    )
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--huber-beta", type=float, default=1.0)

    ap.add_argument("--screening-fractions", nargs="*", type=float, default=[0.1, 0.25, 0.5])

    ap.add_argument(
        "--optuna-trials", type=int, default=0,
        help="Sweep hidden_dim/num_layers/dropout/batch_size/lr/weight_decay/huber_beta with "
             "this many Optuna trials instead of using the CLI values directly. 0 disables. "
             "The dataset is generated/loaded once and reused across all trials.",
    )
    ap.add_argument("--optuna-epochs", type=int, default=30,
                    help="Max epochs per Optuna trial. The final pass still trains for --epochs.")
    ap.add_argument("--optuna-timeout-s", type=float, default=None,
                    help="Stop starting new trials after this many seconds. Default: no timeout.")
    ap.add_argument(
        "--optuna-pruner", choices=["median", "none"], default="median",
        help="Prune weak trials early based on val_mae.",
    )
    ap.add_argument("--optuna-storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///output_dir/optuna.db, letting a "
                         "study resume across interrupted runs. Default: in-memory, one-shot.")

    return ap


def sample_cem_like_results(
    rng: np.random.Generator,
    state,
    model,
    checkpoint: dict[str, object],
    feature_names: list,
    device: torch.device,
    state_args: argparse.Namespace,
    horizon_steps: int,
) -> list[dict[str, float]]:
    """Score one state's candidates using deepvac.mpc.optimize_pid_for_state's
    CEM mean/std elite-update trajectory, scored via one batched
    rollout_population call per iteration instead of one candidate at a time.
    """
    lo, hi = pid_bounds(state_args)
    current_pid = np.asarray([state.kp, state.ki, state.kd], dtype=float)
    mean = np.clip(current_pid, lo, hi)
    std = 0.40 * (hi - lo)
    min_std = np.maximum(float(state_args.cem_min_std_frac) * (hi - lo), 1e-9)
    population = max(4, int(state_args.cem_population))
    elite_n = max(2, int(round(population * float(state_args.cem_elite_frac))))

    all_results: list[dict[str, float]] = []
    for _ in range(max(1, int(state_args.cem_iterations))):
        samples = rng.normal(loc=mean, scale=std, size=(population, 3))
        samples = np.clip(samples, lo, hi)
        samples = np.floor(samples + 0.5)
        samples = np.clip(samples, lo, hi)

        results = rollout_population(
            initial_state=state, candidate_pids=samples, model=model, checkpoint=checkpoint,
            feature_names=feature_names, device=device, args=state_args,
            horizon_steps=horizon_steps, cost_fn=horizon_cost,
        )
        all_results.extend(results)

        order = sorted(range(len(results)), key=lambda i: float(results[i]["cost"]))
        elites = samples[order[:elite_n]]
        mean = np.mean(elites, axis=0)
        std = np.maximum(np.std(elites, axis=0), min_std)

    return all_results


def generate_dataset(
    args: argparse.Namespace,
    model,
    checkpoint: dict[str, object],
    feature_names: list,
    window_steps: int,
    device: torch.device,
) -> pd.DataFrame:
    horizon_steps = max(1, int(round(float(args.mpc_horizon_s) / float(args.dt_s))))
    rng = np.random.default_rng(int(args.seed))

    rows: list[dict[str, float]] = []
    progress_every = max(1, int(args.n_states) // 20)

    for state_idx in range(int(args.n_states)):
        temp = float(rng.uniform(args.temp_min, args.temp_max))
        current_pid = sample_candidates_random(rng, 1, args)[0]

        state_args = copy.deepcopy(args)
        state_args.start_temp = temp
        state_args.context_csv = None
        state_args.kp_init, state_args.ki_init, state_args.kd_init = (float(v) for v in current_pid)

        state = build_initial_state(
            checkpoint=checkpoint, feature_names=feature_names, window_steps=window_steps,
            args=state_args, verbose=False,
        )

        results: list[dict[str, float]] = []
        sources: list[str] = []
        if args.candidate_sampling in ("uniform", "mixed"):
            uniform_candidates = sample_candidates_random(rng, int(args.n_candidates_per_state), args)
            uniform_results = rollout_population(
                initial_state=state, candidate_pids=uniform_candidates, model=model, checkpoint=checkpoint,
                feature_names=feature_names, device=device, args=state_args,
                horizon_steps=horizon_steps, cost_fn=horizon_cost,
            )
            results.extend(uniform_results)
            sources.extend(["uniform"] * len(uniform_results))
        if args.candidate_sampling in ("cem", "mixed"):
            cem_results = sample_cem_like_results(
                rng, state, model, checkpoint, feature_names, device, state_args, horizon_steps,
            )
            results.extend(cem_results)
            sources.extend(["cem"] * len(cem_results))

        for r, source in zip(results, sources, strict=True):
            rows.append({
                "state_idx": state_idx,
                "temp": temp,
                "current_kp": float(current_pid[0]),
                "current_ki": float(current_pid[1]),
                "current_kd": float(current_pid[2]),
                "candidate_kp": r["kp"], "candidate_ki": r["ki"], "candidate_kd": r["kd"],
                "cost": r["cost"], "valid": r["valid"], "source": source,
            })

        if (state_idx + 1) % progress_every == 0:
            print(f"[data] {state_idx + 1}/{args.n_states} states, {len(rows)} rows so far")

    return pd.DataFrame(rows)


def split_by_state(df: pd.DataFrame, val_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = df["state_idx"].unique()
    rng.shuffle(states)
    n_val = max(1, int(round(len(states) * val_fraction)))
    val_states = set(states[:n_val].tolist())
    return df["state_idx"].isin(val_states).to_numpy()


def add_state_normalized_target(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Add a `cost_target` column: cost normalized within each state.
    A no-op copy when method == "none"."""
    df = df.copy()
    if method == "none":
        df["cost_target"] = df["cost"]
        return df

    grouped = df.groupby("state_idx")["cost"]
    if method == "center":
        df["cost_target"] = df["cost"] - grouped.transform("median")
    elif method == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").clip(lower=1e-6)
        df["cost_target"] = (df["cost"] - mean) / std
    else:
        raise ValueError(f"Unknown --target-normalization {method!r}")
    return df


def train_surrogate(
    df: pd.DataFrame, args: argparse.Namespace, device: torch.device, trial=None,
) -> tuple[CostSurrogateMLP, object, object, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    set_seed(args.seed)
    df = add_state_normalized_target(df, args.target_normalization)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["cost_target"].to_numpy(dtype=np.float32).reshape(-1, 1)
    val_mask = split_by_state(df, args.val_fraction, args.seed)
    train_mask = ~val_mask

    x_scaler = StandardScaler().fit(X[train_mask])
    y_scaler = StandardScaler().fit(y[train_mask])

    X_train = torch.as_tensor(x_scaler.transform(X[train_mask]), dtype=torch.float32)
    y_train = torch.as_tensor(y_scaler.transform(y[train_mask]), dtype=torch.float32)
    X_val = torch.as_tensor(x_scaler.transform(X[val_mask]), dtype=torch.float32, device=device)
    y_val_raw = y[val_mask]

    model = CostSurrogateMLP(
        input_dim=len(FEATURE_NAMES), hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True,
    )

    tag = f"[trial {trial.number}]" if trial is not None else "[epoch]"
    best_val_mae = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(X_val).cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_pred_scaled)
        val_mae = float(np.mean(np.abs(val_pred - y_val_raw)))

        print(f"{tag} epoch={epoch:3d}/{args.epochs} train_loss={np.mean(losses):.4f} "
              f"val_mae={val_mae:.4f} (target={args.target_normalization})")

        if trial is not None:
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                import optuna

                print(f"[trial {trial.number}] pruned at epoch {epoch}")
                raise optuna.TrialPruned()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"{tag} stopping: no improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, x_scaler, y_scaler, val_mask


def evaluate_screening(
    df: pd.DataFrame,
    val_mask: np.ndarray,
    model: CostSurrogateMLP,
    x_scaler,
    y_scaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    val_df = df[val_mask].copy()
    x_scaled = x_scaler.transform(val_df[FEATURE_NAMES].to_numpy(dtype=np.float32))
    with torch.no_grad():
        pred_scaled = model(torch.as_tensor(x_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    val_df["predicted_score"] = y_scaler.inverse_transform(pred_scaled)[:, 0]

    correlations: list[float] = []
    hit_rates: dict[float, list[bool]] = {frac: [] for frac in args.screening_fractions}

    for _, group in val_df.groupby("state_idx"):
        if len(group) < 5:
            continue
        corr = rank_correlation(group["predicted_score"].tolist(), group["cost"].tolist())
        if corr is not None:
            correlations.append(corr)

        true_best_idx = group["cost"].idxmin()
        ordered = group.sort_values("predicted_score")
        for frac in args.screening_fractions:
            k = max(1, int(round(len(group) * frac)))
            hit_rates[frac].append(true_best_idx in set(ordered.index[:k]))

    return {
        "mean_rank_correlation": float(np.mean(correlations)) if correlations else None,
        "n_states_evaluated": len(correlations),
        "screening_hit_rate": {
            f"top_{frac:g}": float(np.mean(hits)) if hits else None
            for frac, hits in hit_rates.items()
        },
    }


def clone_args(args: argparse.Namespace, updates: dict[str, object] | None = None) -> argparse.Namespace:
    next_args = copy.deepcopy(args)
    for key, value in (updates or {}).items():
        setattr(next_args, key, value)
    return next_args


def suggest_optuna_args(trial, base_args: argparse.Namespace) -> argparse.Namespace:
    updates: dict[str, object] = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "huber_beta": trial.suggest_float("huber_beta", 0.1, 2.0),
        "epochs": base_args.optuna_epochs,
        "patience": min(base_args.patience, max(5, base_args.optuna_epochs // 2)),
    }
    return clone_args(base_args, updates)


def run_optuna(
    args: argparse.Namespace, df: pd.DataFrame, device: torch.device,
) -> argparse.Namespace:
    """Search hyperparameters against the already-generated/loaded dataset, and
    return an args namespace carrying the best trial's values."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna tuning was requested but optuna is not installed. "
            "Install it with: pip install optuna, or pass --optuna-trials 0."
        ) from exc

    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=5)
        if args.optuna_pruner == "median" else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        study_name="cost_surrogate",
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
        pruner=pruner,
    )

    def objective(trial):
        trial_args = suggest_optuna_args(trial, args)
        trial_args.seed = args.seed + trial.number
        model, x_scaler, y_scaler, val_mask = train_surrogate(df, trial_args, device, trial=trial)
        screening = evaluate_screening(df, val_mask, model, x_scaler, y_scaler, device, trial_args)
        corr = screening["mean_rank_correlation"]
        return float(corr) if corr is not None else -1.0

    print("\n=== Optuna tuning ===")
    study.optimize(
        objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout_s,
        catch=(torch.OutOfMemoryError,), gc_after_trial=True,
    )
    n_failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    if n_failed:
        print(f"[optuna] {n_failed}/{len(study.trials)} trial(s) failed and were skipped")
    if study.trials and not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f"All {len(study.trials)} Optuna trials failed -- nothing to select from.")

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mean_rank_correlation: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    (output_dir / "optuna_best_params.json").write_text(
        json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2),
        encoding="utf-8",
    )

    final_updates = dict(study.best_params)
    final_updates["epochs"] = args.epochs
    final_updates["patience"] = args.patience
    return clone_args(args, final_updates)


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    feature_names = list(checkpoint.get("feature_names", []))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_csv:
        df = pd.read_csv(args.dataset_csv)
        print(f"[data] loaded {len(df)} rows from {args.dataset_csv}")
    else:
        print(f"[data] sampling {args.n_states} states x {args.n_candidates_per_state} candidates "
              f"(--candidate-sampling {args.candidate_sampling})")
        df = generate_dataset(args, model, checkpoint, feature_names, window_steps, device)
        dataset_path = Path(args.save_dataset_csv or (output_dir / "dataset.csv"))
        df.to_csv(dataset_path, index=False)
        print(f"[data] {len(df)} rows -> {dataset_path}")

    df = df[df["valid"]].reset_index(drop=True)
    print(f"[data] {len(df)} valid rows across {df['state_idx'].nunique()} states")

    final_args = args
    if args.optuna_trials > 0:
        final_args = run_optuna(args, df, device)
        print(
            "\n=== Final training with best Optuna parameters ===\n"
            f"Using hidden_dim={final_args.hidden_dim}, num_layers={final_args.num_layers}, "
            f"dropout={final_args.dropout:.3f}, batch_size={final_args.batch_size}, "
            f"lr={final_args.lr:.6g}, weight_decay={final_args.weight_decay:.6g}, "
            f"huber_beta={final_args.huber_beta:.3f}"
        )

    print("\n=== Training ===")
    surrogate, x_scaler, y_scaler, val_mask = train_surrogate(df, final_args, device)
    screening = evaluate_screening(df, val_mask, surrogate, x_scaler, y_scaler, device, final_args)

    checkpoint_path = output_dir / "cost_surrogate.pt"
    torch.save({
        "model_state_dict": surrogate.state_dict(),
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": final_args.hidden_dim,
        "num_layers": final_args.num_layers,
        "dropout": final_args.dropout,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": FEATURE_NAMES,
        "target_normalization": final_args.target_normalization,
        "target_temp": final_args.target_temp,
        "temp_range": [final_args.temp_min, final_args.temp_max],
        "mpc_horizon_s": final_args.mpc_horizon_s,
        "dt_s": final_args.dt_s,
        "cost_weights": {
            "w_overshoot_max": final_args.w_overshoot_max, "w_overshoot_rmse": final_args.w_overshoot_rmse,
            "w_abs_error": final_args.w_abs_error, "w_final_abs_error": final_args.w_final_abs_error,
            "w_near_std": final_args.w_near_std, "w_control_change": final_args.w_control_change,
            "near_band": final_args.near_band,
        },
    }, checkpoint_path)

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "gru_checkpoint": str(Path(args.checkpoint).resolve()),
        "target_normalization": final_args.target_normalization,
        "candidate_sampling": args.candidate_sampling,
        "optuna_trials": int(args.optuna_trials),
        "n_rows": len(df),
        "n_states": int(df["state_idx"].nunique()),
        **screening,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Screening quality (held-out states) ===")
    print(f"states evaluated       : {screening['n_states_evaluated']}")
    corr = screening["mean_rank_correlation"]
    print(f"mean rank correlation  : {'n/a' if corr is None else f'{corr:.3f}'}")
    for key, value in screening["screening_hit_rate"].items():
        print(f"true best kept, {key:>8s} : {'n/a' if value is None else f'{100.0 * value:.0f}%'}")
    print(f"\ncheckpoint : {checkpoint_path}")
    print(f"report     : {report_path}")


if __name__ == "__main__":
    main()
