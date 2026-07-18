"""Run-history discovery, sequence building, scaling, and plotting for the
one-step GRU/LSTM plant models.

This is a straight move of what used to be gru/utils.py: despite the name,
it was already model-agnostic and lstm/train_lstm.py imported every public
name from it directly (`from gru.utils import ...`). That cross-package
import was the concrete instance of the "model vs gru.model" import-mixing
this package resolves -- there was already exactly one implementation, just
homed under the wrong package name.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepvac.schemas import DEFAULT_FEATURE_NAMES

# Kept as FEATURE_NAMES for backward-compatible imports.
FEATURE_NAMES = DEFAULT_FEATURE_NAMES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def find_run_csvs(history_root: Path, telemetry_names: Sequence[str]) -> List[Path]:
    found: List[Path] = []

    if not history_root.exists():
        raise FileNotFoundError(f"History root does not exist: {history_root}")

    for run_dir in sorted(history_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for name in telemetry_names:
            path = run_dir / name
            if path.exists() and path.stat().st_size > 0:
                found.append(path)
                break

    return found


def infer_elapsed_s(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "elapsed_s" in df.columns:
        df["elapsed_s"] = safe_numeric(df["elapsed_s"])
        return df.sort_values("elapsed_s").reset_index(drop=True)

    if "timestamp" in df.columns:
        df["timestamp"] = safe_numeric(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"] = df["timestamp"] - float(df["timestamp"].iloc[0])
        return df

    df["elapsed_s"] = np.arange(len(df), dtype=np.float32)
    return df


def prepare_run_dataframe(
    csv_path: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    run_id = csv_path.parent.name
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("empty CSV")

    df = infer_elapsed_s(df)

    if "temp" not in df.columns:
        raise ValueError("missing required column: temp")
    df["temp"] = safe_numeric(df["temp"])

    if "temp_ref" not in df.columns:
        if "target_temp" not in df.columns:
            raise ValueError("missing required column: temp_ref or target_temp")
        df["temp_ref"] = safe_numeric(df["target_temp"])
    else:
        df["temp_ref"] = safe_numeric(df["temp_ref"])

    for col in ["kp", "ki", "kd"]:
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")
        df[col] = safe_numeric(df[col]).ffill().bfill().fillna(0.0)

    for col in ["temp_u", "temp_u_p", "temp_u_i", "temp_u_d"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = safe_numeric(df[col]).ffill().bfill().fillna(0.0)

    df["error"] = df["temp_ref"] - df["temp"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_NAMES + ["temp", "elapsed_s"]).reset_index(drop=True)

    if len(df) < args.min_samples:
        raise ValueError(f"too few samples: {len(df)} < {args.min_samples}")

    duration_s = (
        float(df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0])
        if len(df) > 1
        else 0.0
    )

    if duration_s < args.min_duration_s:
        raise ValueError(
            f"duration too short: {duration_s:.3f}s < {args.min_duration_s:.3f}s"
        )

    meta = {
        "run_id": run_id,
        "n_samples": int(len(df)),
        "duration_s": duration_s,
        "start_temp": float(df["temp"].iloc[0]),
        "target_temp": float(df["temp_ref"].iloc[0]),
    }
    return df, meta


def build_sequences(
    df: pd.DataFrame,
    run_id: str,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    temp = df["temp"].to_numpy(dtype=np.float32)
    elapsed = df["elapsed_s"].to_numpy(dtype=np.float64)

    window_steps = int(args.window_steps)

    if len(df) <= window_steps:
        return (
            np.empty((0, window_steps, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            pd.DataFrame(),
        )

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    for end_idx in range(window_steps - 1, len(df) - 1):
        next_idx = end_idx + 1
        start_idx = end_idx - window_steps + 1
        x_seq = features[start_idx : end_idx + 1]

        if len(x_seq) != window_steps:
            continue

        target_delta = float(temp[next_idx] - temp[end_idx])
        y = np.asarray([target_delta], dtype=np.float32)

        if not np.isfinite(x_seq).all() or not np.isfinite(y).all():
            continue

        X_list.append(x_seq.astype(np.float32))
        y_list.append(y)
        meta_rows.append(
            {
                "run_id": run_id,
                "end_idx": int(end_idx),
                "next_idx": int(next_idx),
                "elapsed_s": float(elapsed[end_idx]),
                "next_elapsed_s": float(elapsed[next_idx]),
                "dt_to_next_s": float(elapsed[next_idx] - elapsed[end_idx]),
                "current_temp": float(temp[end_idx]),
                "next_temp": float(temp[next_idx]),
                "current_ref": float(df["temp_ref"].iloc[end_idx]),
                "error": float(df["error"].iloc[end_idx]),
            }
        )

    if not X_list:
        return (
            np.empty((0, window_steps, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            pd.DataFrame(),
        )

    return np.stack(X_list), np.stack(y_list), pd.DataFrame(meta_rows)


def split_runs(
    run_ids: Sequence[str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    unique = sorted(set(str(r) for r in run_ids))
    if len(unique) < 3:
        raise RuntimeError("Need at least 3 valid runs for run-level train/val/test split.")

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    n_train = max(1, int(round(len(shuffled) * train_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))

    if n_train + n_val >= len(shuffled):
        overflow = n_train + n_val - (len(shuffled) - 1)
        n_train = max(1, n_train - overflow)
    if n_train + n_val >= len(shuffled):
        n_val = max(1, len(shuffled) - n_train - 1)

    train_runs = sorted(shuffled[:n_train])
    val_runs = sorted(shuffled[n_train : n_train + n_val])
    test_runs = sorted(shuffled[n_train + n_val :])

    if not train_runs or not val_runs or not test_runs:
        raise RuntimeError("Train, validation, or test split is empty.")

    return train_runs, val_runs, test_runs


def scale_datasets(
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    y_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
):
    from sklearn.preprocessing import StandardScaler

    n_features = X_train_raw.shape[-1]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train_raw.reshape(-1, n_features))
    y_scaler.fit(y_train_raw)

    X_train = x_scaler.transform(X_train_raw.reshape(-1, n_features)).reshape(
        X_train_raw.shape
    )
    X_val = x_scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(
        X_val_raw.shape
    )
    X_test = x_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(
        X_test_raw.shape
    )

    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    return X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler


def compute_real_metrics(
    pred_scaled: np.ndarray,
    target_scaled: np.ndarray,
    y_scaler,
) -> Dict[str, float]:
    pred_delta = y_scaler.inverse_transform(pred_scaled)
    target_delta = y_scaler.inverse_transform(target_scaled)

    err_delta = pred_delta - target_delta
    abs_err_delta = np.abs(err_delta)

    return {
        "mae_delta_t1": float(np.mean(abs_err_delta)),
        "rmse_delta_t1": float(np.sqrt(np.mean(err_delta**2))),
        "bias_delta_t1": float(np.mean(err_delta)),
    }


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses: List[float] = []
    pred_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            losses.append(float(loss.item()))
            pred_batches.append(pred.cpu().numpy())
            target_batches.append(yb.cpu().numpy())

    return (
        np.concatenate(pred_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        float(np.mean(losses)),
    )


def build_prediction_details(
    meta: pd.DataFrame,
    pred_real: np.ndarray,
    target_real: np.ndarray,
) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)
    out["target_delta_t1"] = target_real[:, 0]
    out["pred_delta_t1"] = pred_real[:, 0]
    out["error_delta_t1"] = out["pred_delta_t1"] - out["target_delta_t1"]
    out["abs_error_delta_t1"] = out["error_delta_t1"].abs()
    out["target_temp_t1"] = out["current_temp"] + out["target_delta_t1"]
    out["pred_temp_t1"] = out["current_temp"] + out["pred_delta_t1"]
    out["error_temp_t1"] = out["pred_temp_t1"] - out["target_temp_t1"]
    out["abs_error_temp_t1"] = out["error_temp_t1"].abs()
    return out


def summarize_metrics_by_run(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in pred_df.groupby("run_id"):
        rows.append(
            {
                "run_id": run_id,
                "n_sequences": int(len(group)),
                "mae_delta_t1": float(group["abs_error_delta_t1"].mean()),
                "bias_delta_t1": float(group["error_delta_t1"].mean()),
                "p90_abs_error_delta_t1": float(
                    group["abs_error_delta_t1"].quantile(0.90)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_delta_t1").reset_index(drop=True)


def plot_training_history(history_rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not history_rows:
        return

    history = pd.DataFrame(history_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["train_mae_delta_t1"], label="train MAE delta t+1")
    ax.plot(history["epoch"], history["val_mae_delta_t1"], label="validation MAE delta t+1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE of temp delta (deg C/sample)")
    ax.set_title("One-step GRU Training Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_test_prediction(
    pred_df: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    if pred_df.empty or per_run_metrics.empty:
        return

    best_run = str(per_run_metrics.iloc[0]["run_id"])
    run_df = pred_df[pred_df["run_id"] == best_run].sort_values("elapsed_s")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(run_df["elapsed_s"], run_df["target_delta_t1"], label="actual delta t+1")
    ax.plot(run_df["elapsed_s"], run_df["pred_delta_t1"], label="predicted delta t+1")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature delta (deg C/sample)")
    ax.set_title(f"Best Test Run One-Step Delta Prediction: {best_run}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_test_temperature(
    pred_df: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    output_path: Path,
    tail_window_s: Optional[float] = None,
) -> None:
    if pred_df.empty or per_run_metrics.empty:
        return

    best_run = str(per_run_metrics.iloc[0]["run_id"])
    run_df = pred_df[pred_df["run_id"] == best_run].sort_values("elapsed_s")
    if tail_window_s is not None and not run_df.empty:
        start = float(run_df["elapsed_s"].max()) - float(tail_window_s)
        run_df = run_df[run_df["elapsed_s"] >= start]
    if run_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(run_df["next_elapsed_s"], run_df["target_temp_t1"], label="real temp t+1")
    ax.plot(run_df["next_elapsed_s"], run_df["pred_temp_t1"], label="predicted temp t+1")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature (deg C)")
    title = f"Best Test Run One-Step Temperature: {best_run}"
    if tail_window_s is not None:
        title += f" - last {tail_window_s:g}s"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
