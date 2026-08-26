#!/usr/bin/env python3
"""Train, validate, tune, and test a one-step digital-twin plant model (GRU or
LSTM, per --model-family) from history."""

from __future__ import annotations
from deepvac.datasets import (
    FEATURE_NAMES,
    build_prediction_details,
    build_sequences,
    compute_real_metrics,
    find_run_csvs,
    plot_best_test_prediction,
    plot_best_test_temperature,
    plot_training_history,
    predict_loader,
    prepare_run_dataframe,
    scale_datasets,
    set_seed,
    split_runs,
    summarize_metrics_by_run,
)
from digitaltwin.model import MODEL_CLASSES, SequenceDataset

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = ROOT / "experiments"
DEFAULT_HISTORY_ROOT = WORK_DIR / "run_history"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train, validate, tune, and test a one-step digital-twin plant model."
    )
    ap.add_argument("--model-family", choices=sorted(MODEL_CLASSES), default="gru",
                    help="Which recurrent architecture to train.")
    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument("--telemetry-names", nargs="+",
                    default=["run_samples.csv"])
    ap.add_argument("--output-dir", default=None,
                    help="Default: <script-dir>/<model-family>/validation_t1.")
    ap.add_argument("--plots-dir", default=None,
                    help="Default: <script-dir>/<model-family>/plots_t1.")
    ap.add_argument("--window-steps", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--min-duration-s", type=float, default=300.0)
    ap.add_argument("--exclude-prefixes", nargs="*", default=[])
    ap.add_argument("--train-fraction", type=float, default=0.80)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--huber-beta", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Run this many Optuna trials before final training. Set 0 to disable.",
    )
    ap.add_argument(
        "--optuna-epochs",
        type=int,
        default=100,
        help="Maximum epochs per Optuna trial.",
    )
    ap.add_argument("--optuna-timeout-s", type=float, default=None)
    ap.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name used for both MLflow and Optuna. Default: <model-family>_t1.",
    )
    ap.add_argument(
        "--optuna-pruner",
        choices=["median", "none"],
        default="median",
        help="Prune weak trials early based on validation MAE.",
    )
    ap.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking.",
    )
    ap.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=(
            "MLflow tracking URI. Defaults to a local file store under "
            "<script-dir>/<model-family>/mlruns."
        ),
    )
    ap.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name for the parent experiment run.",
    )
    return ap


def clone_args(
    args: argparse.Namespace, updates: Optional[Dict[str, object]] = None
) -> argparse.Namespace:
    next_args = copy.deepcopy(args)
    for key, value in (updates or {}).items():
        setattr(next_args, key, value)
    return next_args


def resolve_family_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in --output-dir/--plots-dir/--experiment-name defaults that depend
    on --model-family, once it's known post-parse."""
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / args.model_family / "validation_t1")
    if args.plots_dir is None:
        args.plots_dir = str(SCRIPT_DIR / args.model_family / "plots_t1")
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_family}_t1"
    return args


def setup_mlflow(args: argparse.Namespace):
    if args.no_mlflow:
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled but mlflow is not installed. "
            "Install it with: pip install mlflow, or pass --no-mlflow."
        ) from exc

    default_mlflow_dir = SCRIPT_DIR / args.model_family / "mlruns"
    tracking_uri = args.mlflow_tracking_uri or default_mlflow_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    print(f"\nMLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {args.experiment_name}")
    return mlflow


def mlflow_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def log_params_with_prefix(mlflow, params: Dict[str, Any], prefix: str = "") -> None:
    if mlflow is None:
        return
    mlflow.log_params({f"{prefix}{key}": mlflow_value(value)
                      for key, value in params.items()})


def log_dataset_to_mlflow(mlflow, data: Dict[str, object]) -> None:
    if mlflow is None:
        return
    mlflow.log_params(
        {
            "num_train_runs": len(data["train_runs"]),
            "num_val_runs": len(data["val_runs"]),
            "num_test_runs": len(data["test_runs"]),
            "num_features": len(FEATURE_NAMES),
            "feature_names": ",".join(FEATURE_NAMES),
        }
    )
    mlflow.log_metrics(
        {
            "num_train_sequences": float(len(data["X_train_raw"])),
            "num_val_sequences": float(len(data["X_val_raw"])),
            "num_test_sequences": float(len(data["X_test_raw"])),
        }
    )


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    history_root = Path(args.history_root)
    csv_paths = find_run_csvs(history_root, args.telemetry_names)
    if not csv_paths:
        raise RuntimeError(
            f"No telemetry CSV files found under {history_root}")

    print(f"\nFound {len(csv_paths)} run CSV files.")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_meta: List[pd.DataFrame] = []
    run_summaries: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for csv_path in csv_paths:
        run_id = csv_path.parent.name
        if args.exclude_prefixes and any(
            run_id.startswith(p) for p in args.exclude_prefixes
        ):
            skipped.append({"run_id": run_id, "path": str(
                csv_path), "reason": "excluded"})
            continue

        try:
            df, run_meta = prepare_run_dataframe(csv_path, args)
            X, y, meta = build_sequences(df, run_id, args)
            if len(X) == 0:
                skipped.append(
                    {"run_id": run_id, "path": str(
                        csv_path), "reason": "no sequences"}
                )
                continue

            all_X.append(X)
            all_y.append(y)
            all_meta.append(meta)
            run_meta["num_sequences"] = int(len(X))
            run_summaries.append(run_meta)
        except Exception as exc:
            skipped.append(
                {"run_id": run_id, "path": str(csv_path), "reason": str(exc)})
            print(f"[SKIP] {run_id}: {exc}")

    if not all_X:
        raise RuntimeError("No valid sequences were created.")

    X_raw = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    meta_all = pd.concat(all_meta, ignore_index=True)

    train_runs, val_runs, test_runs = split_runs(
        run_ids=meta_all["run_id"].tolist(),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_mask = meta_all["run_id"].isin(train_runs).to_numpy()
    val_mask = meta_all["run_id"].isin(val_runs).to_numpy()
    test_mask = meta_all["run_id"].isin(test_runs).to_numpy()

    X_train_raw = X_raw[train_mask]
    y_train_raw = y_raw[train_mask]
    X_val_raw = X_raw[val_mask]
    y_val_raw = y_raw[val_mask]
    X_test_raw = X_raw[test_mask]
    y_test_raw = y_raw[test_mask]

    if len(X_train_raw) == 0 or len(X_val_raw) == 0 or len(X_test_raw) == 0:
        raise RuntimeError("Train, validation, or test split is empty.")

    X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler = scale_datasets(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw
    )

    data: Dict[str, object] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "meta_val": meta_all.loc[val_mask].reset_index(drop=True),
        "meta_test": meta_all.loc[test_mask].reset_index(drop=True),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_runs": train_runs,
        "val_runs": val_runs,
        "test_runs": test_runs,
        "run_summaries": run_summaries,
        "skipped": skipped,
    }

    print("\n=== Dataset ===")
    print(f"Dataset: {history_root}")
    print(f"Feature list: {FEATURE_NAMES}")
    print(f"Total sequences: {len(X_raw)}")
    print(f"X shape:         {X_raw.shape}")
    print(f"y shape:         {y_raw.shape}")
    print(f"Valid runs:      {meta_all['run_id'].nunique()}")
    print("\n=== Split ===")
    print(f"Train runs: {len(train_runs)}")
    print(f"Val runs:   {len(val_runs)}")
    print(f"Test runs:  {len(test_runs)}")
    print(f"Train seq:  {len(X_train_raw)}")
    print(f"Val seq:    {len(X_val_raw)}")
    print(f"Test seq:   {len(X_test_raw)}")

    return data


def make_loaders(data: Dict[str, object], args: argparse.Namespace):
    train_loader = DataLoader(
        SequenceDataset(data["X_train"], data["y_train"]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(data["X_val"], data["y_val"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(data["X_test"], data["y_test"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def train_model(
    args: argparse.Namespace,
    data: Dict[str, object],
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
    trial=None,
    mlflow=None,
    metric_prefix: str = "",
) -> Dict[str, object]:
    train_loader, val_loader, _ = make_loaders(data, args)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    model = MODEL_CLASSES[args.model_family](
        input_dim=len(FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_observed_val_mae = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_count = 0
    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        pred_train_scaled, target_train_scaled, train_eval_loss = predict_loader(
            model, train_loader, device, loss_fn
        )
        pred_val_scaled, target_val_scaled, val_loss = predict_loader(
            model, val_loader, device, loss_fn
        )

        train_metrics = compute_real_metrics(
            pred_train_scaled, target_train_scaled, data["y_scaler"]
        )
        val_metrics = compute_real_metrics(
            pred_val_scaled, target_val_scaled, data["y_scaler"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval_loss,
            "val_loss": val_loss,
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history_rows.append(row)

        if mlflow is not None:
            mlflow.log_metrics(
                {
                    f"{metric_prefix}train_loss": train_loss,
                    f"{metric_prefix}train_eval_loss": train_eval_loss,
                    f"{metric_prefix}val_loss": val_loss,
                    **{
                        f"{metric_prefix}train_{key}": value
                        for key, value in train_metrics.items()
                    },
                    **{
                        f"{metric_prefix}val_{key}": value
                        for key, value in val_metrics.items()
                    },
                },
                step=epoch,
            )

        if trial is None:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"train_mae={train_metrics['mae_delta_t1']:.6f} "
                f"val_mae={val_metrics['mae_delta_t1']:.6f} "
            )

        val_mae = val_metrics["mae_delta_t1"]
        best_observed_val_mae = min(best_observed_val_mae, val_mae)
        if trial is not None:
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_count = 0
            if trial is None:
                print("new best")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            if trial is None:
                print(
                    f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())

    if checkpoint_path is not None:
        checkpoint = {
            "model_state_dict": best_state_dict,
            "model_family": args.model_family,
            "feature_names": FEATURE_NAMES,
            "input_dim": len(FEATURE_NAMES),
            "output_dim": 1,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "window_steps": args.window_steps,
            "target": "temp(t+1 row) - temp(t)",
            "x_scaler": data["x_scaler"],
            "y_scaler": data["y_scaler"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_mae_delta_t1": best_val_mae,
            "best_metric": "val_loss",
            "best_metric_value": best_val_loss,
            "trained_from_optuna_best_params": args.optuna_trials > 0,
            "train_runs": data["train_runs"],
            "val_runs": data["val_runs"],
            "test_runs": data["test_runs"],
            "args": vars(args),
            "error_sign": "error = temp_ref - temp",
        }
        torch.save(checkpoint, checkpoint_path)

    if mlflow is not None:
        mlflow.log_metrics(
            {
                f"{metric_prefix}best_epoch": float(best_epoch),
                f"{metric_prefix}best_val_loss": float(best_val_loss),
                f"{metric_prefix}best_val_mae_delta_t1": float(best_val_mae),
                f"{metric_prefix}best_observed_val_mae_delta_t1": float(
                    best_observed_val_mae
                ),
            }
        )

    return {
        "model_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae_delta_t1": best_val_mae,
        "best_observed_val_mae_delta_t1": best_observed_val_mae,
        "history_rows": history_rows,
    }


def suggest_optuna_args(trial, base_args: argparse.Namespace) -> argparse.Namespace:
    updates = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.40),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "huber_beta": trial.suggest_float("huber_beta", 0.1, 1.0),
        "epochs": base_args.optuna_epochs,
        "patience": min(base_args.patience, max(5, base_args.optuna_epochs // 2)),
    }
    return clone_args(base_args, updates)


def run_optuna(
    args: argparse.Namespace, data: Dict[str, object], device: torch.device, mlflow=None
) -> argparse.Namespace:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna tuning was requested but optuna is not installed. "
            "Install it with: pip install optuna"
        ) from exc

    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=5)
        if args.optuna_pruner == "median"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=args.experiment_name,
        load_if_exists=True,
        pruner=pruner,
    )

    def objective(trial):
        trial_args = suggest_optuna_args(trial, args)
        set_seed(args.seed + trial.number)
        if mlflow is None:
            result = train_model(trial_args, data, device,
                                 checkpoint_path=None, trial=trial)
        else:
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
                mlflow.set_tag("phase", "optuna_trial")
                mlflow.log_param("trial_number", trial.number)
                log_params_with_prefix(mlflow, trial.params, prefix="trial_")
                result = train_model(
                    trial_args,
                    data,
                    device,
                    checkpoint_path=None,
                    trial=trial,
                    mlflow=mlflow,
                    metric_prefix="trial_",
                )
                mlflow.log_metric(
                    "trial_objective",
                    float(result["best_observed_val_mae_delta_t1"]),
                )
        return float(result["best_observed_val_mae_delta_t1"])

    print("\n=== Optuna tuning ===")
    study.optimize(objective, n_trials=args.optuna_trials,
                   timeout=args.optuna_timeout_s)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation MAE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    if mlflow is not None:
        mlflow.log_metric("optuna_best_val_mae_delta_t1",
                          float(study.best_value))
        mlflow.log_param("optuna_best_trial", study.best_trial.number)
        log_params_with_prefix(mlflow, study.best_params,
                               prefix="optuna_best_")

    output_dir = Path(args.output_dir)
    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    with open(output_dir / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_value": study.best_value, "best_params": study.best_params},
            f,
            indent=2,
        )

    final_updates = dict(study.best_params)
    final_updates["epochs"] = args.epochs
    final_updates["patience"] = args.patience
    return clone_args(args, final_updates)


def evaluate_best_model(
    args: argparse.Namespace,
    data: Dict[str, object],
    device: torch.device,
    checkpoint_path: Path,
) -> Dict[str, object]:
    _, val_loader, test_loader = make_loaders(data, args)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    best_model = MODEL_CLASSES[checkpoint.get("model_family", args.model_family)](
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    pred_val_scaled, target_val_scaled, _ = predict_loader(
        best_model, val_loader, device, loss_fn
    )
    pred_test_scaled, target_test_scaled, _ = predict_loader(
        best_model, test_loader, device, loss_fn
    )

    final_metrics = compute_real_metrics(
        pred_val_scaled, target_val_scaled, data["y_scaler"]
    )
    test_metrics = compute_real_metrics(
        pred_test_scaled, target_test_scaled, data["y_scaler"]
    )

    pred_val_real = data["y_scaler"].inverse_transform(pred_val_scaled)
    target_val_real = data["y_scaler"].inverse_transform(target_val_scaled)
    pred_test_real = data["y_scaler"].inverse_transform(pred_test_scaled)
    target_test_real = data["y_scaler"].inverse_transform(target_test_scaled)

    val_pred_df = build_prediction_details(
        data["meta_val"], pred_val_real, target_val_real)
    test_pred_df = build_prediction_details(
        data["meta_test"], pred_test_real, target_test_real)

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    val_per_run = summarize_metrics_by_run(val_pred_df)
    test_per_run = summarize_metrics_by_run(test_pred_df)
    val_per_run.to_csv(
        output_dir / "validation_metrics_by_run.csv", index=False)
    test_per_run.to_csv(output_dir / "test_metrics_by_run.csv", index=False)

    plot_best_test_prediction(
        test_pred_df, test_per_run, plots_dir / "test_best_delta_t1.png")
    plot_best_test_temperature(
        test_pred_df, test_per_run, plots_dir / "test_best_temperature_t1.png")
    plot_best_test_temperature(
        test_pred_df,
        test_per_run,
        plots_dir / "test_best_temperature_t1_tail_300s.png",
        tail_window_s=300.0,
    )

    final_report = {
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "best_val_mae_delta_t1": float(checkpoint["best_val_mae_delta_t1"]),
        "best_model_path": str(checkpoint_path),
        "model_family": args.model_family,
        "target": "temp(t+1 row) - temp(t)",
        "final_metrics": final_metrics,
        "test_metrics": test_metrics,
        "num_train_runs": len(data["train_runs"]),
        "num_val_runs": len(data["val_runs"]),
        "num_test_runs": len(data["test_runs"]),
        "num_train_sequences": int(len(data["X_train_raw"])),
        "num_val_sequences": int(len(data["X_val_raw"])),
        "num_test_sequences": int(len(data["X_test_raw"])),
        "feature_names": FEATURE_NAMES,
        "error_sign": "error = temp_ref - temp",
        "optuna_enabled": args.optuna_trials > 0,
        "trained_from_optuna_best_params": args.optuna_trials > 0,
        "final_hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "huber_beta": args.huber_beta,
        },
    }
    with open(output_dir / "validation_report_t1.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n=== Validation metrics ===")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")
    print(f"MAE:  {final_metrics['mae_delta_t1']:.6f} deg C")
    print(f"RMSE: {final_metrics['rmse_delta_t1']:.6f} deg C")
    print(f"Bias: {final_metrics['bias_delta_t1']:.6f} deg C")

    print("\n=== Test metrics ===")
    print(f"MAE:  {test_metrics['mae_delta_t1']:.6f} deg C")
    print(f"RMSE: {test_metrics['rmse_delta_t1']:.6f} deg C")
    print(f"Bias: {test_metrics['bias_delta_t1']:.6f} deg C")

    return final_report


def train_and_validate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    mlflow = setup_mlflow(args)

    run_context = (
        mlflow.start_run(
            run_name=args.mlflow_run_name) if mlflow is not None else None
    )
    if run_context is not None:
        run_context.__enter__()

    try:
        if mlflow is not None:
            mlflow.set_tag("model_family", args.model_family)
            mlflow.set_tag("target", "temp(t+1 row) - temp(t)")
            mlflow.set_tag("phase", "train_validate_test")
            log_params_with_prefix(mlflow, vars(args))

        print(f"=== {args.model_family.upper()} model ===")
        data = build_dataset(args)
        log_dataset_to_mlflow(mlflow, data)

        with open(output_dir / "split_runs.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_runs": data["train_runs"],
                    "val_runs": data["val_runs"],
                    "test_runs": data["test_runs"],
                },
                f,
                indent=2,
            )
        pd.DataFrame(data["run_summaries"]).to_csv(
            output_dir / "run_summaries.csv", index=False
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        print(f"\nDevice: {device}")
        if mlflow is not None:
            mlflow.log_param("device", str(device))

        final_args = run_optuna(
            args, data, device, mlflow) if args.optuna_trials > 0 else args
        if final_args is not args:
            print("\n=== Final training with best Optuna parameters ===")
            print(
                "Using "
                f"hidden_dim={final_args.hidden_dim}, "
                f"num_layers={final_args.num_layers}, "
                f"dropout={final_args.dropout:.3f}, "
                f"batch_size={final_args.batch_size}, "
                f"lr={final_args.lr:.6g}, "
                f"weight_decay={final_args.weight_decay:.6g}, "
                f"grad_clip={final_args.grad_clip:.3f}, "
                f"huber_beta={final_args.huber_beta:.3f}"
            )
            log_params_with_prefix(mlflow, vars(final_args), prefix="final_")

        best_checkpoint_path = output_dir / f"{args.model_family}_t1.pt"
        print("\n=== Training ===")
        set_seed(final_args.seed)
        result = train_model(
            final_args,
            data,
            device,
            checkpoint_path=best_checkpoint_path,
            mlflow=mlflow,
            metric_prefix="final_",
        )
        pd.DataFrame(result["history_rows"]).to_csv(
            output_dir / "training_history.csv", index=False
        )
        plot_training_history(
            result["history_rows"],
            plots_dir / "training_mae_t1.png",
            title=f"One-step {args.model_family.upper()} Training Curve",
        )

        final_report = evaluate_best_model(
            final_args, data, device, best_checkpoint_path)
        if mlflow is not None:
            mlflow.log_metrics(
                {
                    f"validation_{key}": float(value)
                    for key, value in final_report["final_metrics"].items()
                }
            )
            mlflow.log_metrics(
                {
                    f"test_{key}": float(value)
                    for key, value in final_report["test_metrics"].items()
                }
            )
            mlflow.log_artifact(str(best_checkpoint_path),
                                artifact_path="model")
            mlflow.log_artifacts(str(output_dir), artifact_path="outputs")
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")
    finally:
        if run_context is not None:
            run_context.__exit__(*sys.exc_info())


def main() -> None:
    args = resolve_family_defaults(build_arg_parser().parse_args())
    train_and_validate(args)


if __name__ == "__main__":
    main()
