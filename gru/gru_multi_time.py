#!/usr/bin/env python3
"""Train, validate, and test a simple GRU plant model from DeepVac history.

This is Stage 1 of the digital twin:

    past real telemetry window -> future temperature deltas

The model predicts chamber/plant behavior using logged real control signals.
It does not yet emulate the controller in closed loop.
"""

from __future__ import annotations
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUTOMATED_DIR = ROOT / "optimization"
DEFAULT_HISTORY_ROOT = AUTOMATED_DIR / "run_history"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "validation_simple"
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent / "plots_simple"


# Simple first feature list.
# Fixed convention:
#     error = temp_ref - temp
FEATURE_NAMES = [
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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train, validate, and test a simple multi-horizon GRU plant model."
    )
    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument(
        "--telemetry-names",
        nargs="+",
        default=[
            "run_samples.csv"
        ],
    )
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
    ap.add_argument("--horizons-s", default="10,30,60,120")
    ap.add_argument(
        "--window-steps",
        type=int,
        default=60,
        help="Fixed number of previous samples used by the GRU.",
    )
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
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement.",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--huber-beta", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    return ap


class GRUDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


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


def parse_horizons(text: str) -> List[float]:
    values = sorted(set(float(x.strip())
                    for x in text.split(",") if x.strip()))
    if not values:
        raise ValueError("--horizons-s must contain at least one value")
    return values


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
            raise ValueError(
                "missing required column: temp_ref or target_temp")
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

    # Fixed sign convention:
    # Positive error means target is above current temp.
    # For cooling from 25 -> 0, error is negative while temp > target.
    df["error"] = df["temp_ref"] - df["temp"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_NAMES +
                   ["temp", "elapsed_s"]).reset_index(drop=True)

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

    start_temp = float(df["temp"].iloc[0])
    target_temp = float(df["temp_ref"].iloc[0])

    meta = {
        "run_id": run_id,
        "n_samples": int(len(df)),
        "duration_s": duration_s,
        "start_temp": start_temp,
        "target_temp": target_temp,
    }

    return df, meta


def build_sequences_for_run(
    df: pd.DataFrame,
    run_id: str,
    args: argparse.Namespace,
    horizons_s: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    elapsed = df["elapsed_s"].to_numpy(dtype=np.float64)
    features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    temp = df["temp"].to_numpy(dtype=np.float32)

    window_steps = int(args.window_steps)

    if len(df) <= window_steps:
        return (
            np.empty((0, window_steps, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0, len(horizons_s)), dtype=np.float32),
            pd.DataFrame(),
        )

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    max_horizon = max(horizons_s)

    for end_idx in range(window_steps - 1, len(df)):
        current_time = float(elapsed[end_idx])

        if current_time + max_horizon > float(elapsed[-1]):
            break

        start_idx = end_idx - window_steps + 1
        x_seq = features[start_idx: end_idx + 1]

        if len(x_seq) != window_steps:
            continue

        y_values: List[float] = []
        valid = True

        for h in horizons_s:
            target_time = current_time + float(h)
            target_idx = int(np.searchsorted(
                elapsed, target_time, side="left"))

            if target_idx >= len(df):
                valid = False
                break

            # Predict future delta, not absolute temperature.
            y_values.append(float(temp[target_idx] - temp[end_idx]))

        if not valid:
            continue

        y = np.asarray(y_values, dtype=np.float32)

        if not np.isfinite(x_seq).all() or not np.isfinite(y).all():
            continue

        X_list.append(x_seq.astype(np.float32))
        y_list.append(y)

        meta_rows.append(
            {
                "run_id": run_id,
                "end_idx": int(end_idx),
                "elapsed_s": current_time,
                "current_temp": float(temp[end_idx]),
                "current_ref": float(df["temp_ref"].iloc[end_idx]),
                "error": float(df["error"].iloc[end_idx]),
            }
        )

    if not X_list:
        return (
            np.empty((0, window_steps, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0, len(horizons_s)), dtype=np.float32),
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
        raise RuntimeError(
            "Need at least 3 valid runs for run-level train/validation/test split."
        )

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
    val_runs = sorted(shuffled[n_train: n_train + n_val])
    test_runs = sorted(shuffled[n_train + n_val:])

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
    n_features = X_train_raw.shape[-1]
    n_outputs = y_train_raw.shape[-1]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train_raw.reshape(-1, n_features))
    y_scaler.fit(y_train_raw.reshape(-1, n_outputs))

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
    y_scaler: StandardScaler,
    horizons_s: Sequence[float],
) -> Dict[str, float]:
    pred = y_scaler.inverse_transform(pred_scaled)
    target = y_scaler.inverse_transform(target_scaled)

    err = pred - target
    abs_err = np.abs(err)

    metrics: Dict[str, float] = {
        "mae_all": float(np.mean(abs_err)),
        "rmse_all": float(np.sqrt(np.mean(err**2))),
    }

    for i, h in enumerate(horizons_s):
        h_int = int(h)
        metrics[f"mae_{h_int}s"] = float(np.mean(abs_err[:, i]))
        metrics[f"rmse_{h_int}s"] = float(np.sqrt(np.mean(err[:, i] ** 2)))
        metrics[f"bias_{h_int}s"] = float(np.mean(err[:, i]))

    return metrics


def save_predictions_csv(
    output_path: Path,
    meta: pd.DataFrame,
    pred_real: np.ndarray,
    target_real: np.ndarray,
    horizons_s: Sequence[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = meta.copy().reset_index(drop=True)

    for i, h in enumerate(horizons_s):
        h_int = int(h)
        out[f"target_delta_{h_int}s"] = target_real[:, i]
        out[f"pred_delta_{h_int}s"] = pred_real[:, i]
        out[f"error_delta_{h_int}s"] = pred_real[:, i] - target_real[:, i]
        out[f"abs_error_delta_{h_int}s"] = np.abs(
            pred_real[:, i] - target_real[:, i]
        )
        out[f"target_temp_{h_int}s"] = out["current_temp"] + target_real[:, i]
        out[f"pred_temp_{h_int}s"] = out["current_temp"] + pred_real[:, i]

    out.to_csv(output_path, index=False)


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
    horizons_s: Sequence[float],
) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)

    for i, h in enumerate(horizons_s):
        h_int = int(h)
        out[f"target_delta_{h_int}s"] = target_real[:, i]
        out[f"pred_delta_{h_int}s"] = pred_real[:, i]
        out[f"error_delta_{h_int}s"] = pred_real[:, i] - target_real[:, i]
        out[f"abs_error_delta_{h_int}s"] = np.abs(
            pred_real[:, i] - target_real[:, i]
        )
        out[f"target_temp_{h_int}s"] = out["current_temp"] + target_real[:, i]
        out[f"pred_temp_{h_int}s"] = out["current_temp"] + pred_real[:, i]

    return out


def summarize_metrics_by_run(
    pred_df: pd.DataFrame,
    horizons_s: Sequence[float],
) -> pd.DataFrame:
    per_run_rows = []

    for run_id, group in pred_df.groupby("run_id"):
        row: Dict[str, object] = {
            "run_id": run_id,
            "n_sequences": int(len(group)),
        }

        all_abs_cols = []

        for h in horizons_s:
            h_int = int(h)
            abs_col = f"abs_error_delta_{h_int}s"
            err_col = f"error_delta_{h_int}s"
            all_abs_cols.append(abs_col)

            row[f"mae_{h_int}s"] = float(group[abs_col].mean())
            row[f"bias_{h_int}s"] = float(group[err_col].mean())
            row[f"p90_abs_error_{h_int}s"] = float(
                group[abs_col].quantile(0.90))

        row["mae_all"] = float(group[all_abs_cols].to_numpy().mean())
        per_run_rows.append(row)

    return pd.DataFrame(per_run_rows).sort_values("mae_all").reset_index(drop=True)


def plot_training_history(
    history_rows: Sequence[Dict[str, float]],
    output_path: Path,
) -> None:
    if not history_rows:
        return

    history = pd.DataFrame(history_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["train_mae_all"], label="train MAE")
    ax.plot(history["epoch"], history["val_mae_all"], label="validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (deg C)")
    ax.set_title("GRU Training Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_test_prediction(
    pred_df: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    horizons_s: Sequence[float],
    output_path: Path,
) -> None:
    if pred_df.empty or per_run_metrics.empty:
        return

    best_run = str(per_run_metrics.iloc[0]["run_id"])
    run_df = pred_df[pred_df["run_id"] == best_run].sort_values("elapsed_s")
    horizon = int(horizons_s[0])

    target_col = f"target_delta_{horizon}s"
    pred_col = f"pred_delta_{horizon}s"

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(run_df["elapsed_s"], run_df[target_col],
            label=f"actual delta {horizon}s")
    ax.plot(run_df["elapsed_s"], run_df[pred_col],
            label=f"predicted delta {horizon}s")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature delta (deg C)")
    ax.set_title(f"Best Test Run Prediction: {best_run}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_test_reconstructed_trajectory(
    pred_df: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    horizons_s: Sequence[float],
    output_path: Path,
    tail_window_s: Optional[float] = None,
) -> None:
    if pred_df.empty or per_run_metrics.empty:
        return

    best_run = str(per_run_metrics.iloc[0]["run_id"])
    run_df = pred_df[pred_df["run_id"] == best_run].sort_values("elapsed_s")

    if run_df.empty:
        return

    points: List[pd.DataFrame] = []

    for h in horizons_s:
        h_int = int(h)
        target_col = f"target_delta_{h_int}s"
        pred_col = f"pred_delta_{h_int}s"

        if target_col not in run_df.columns or pred_col not in run_df.columns:
            continue

        horizon_points = pd.DataFrame(
            {
                "elapsed_s": run_df["elapsed_s"].to_numpy(dtype=float) + float(h),
                "actual_temp": (
                    run_df["current_temp"].to_numpy(dtype=float)
                    + run_df[target_col].to_numpy(dtype=float)
                ),
                "predicted_temp": (
                    run_df["current_temp"].to_numpy(dtype=float)
                    + run_df[pred_col].to_numpy(dtype=float)
                ),
            }
        )
        points.append(horizon_points)

    if not points:
        return

    trajectory = pd.concat(points, ignore_index=True)
    trajectory = trajectory.replace([np.inf, -np.inf], np.nan).dropna()

    if trajectory.empty:
        return

    trajectory["time_bin_s"] = trajectory["elapsed_s"].round().astype(int)

    if tail_window_s is not None:
        tail_start_s = float(trajectory["elapsed_s"].max()) - max(
            0.0, float(tail_window_s)
        )
        trajectory = trajectory[trajectory["elapsed_s"] >= tail_start_s]
        if trajectory.empty:
            return

    binned = (
        trajectory.groupby("time_bin_s", as_index=False)[
            ["actual_temp", "predicted_temp"]
        ]
        .mean()
        .sort_values("time_bin_s")
    )

    if binned.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        binned["time_bin_s"],
        binned["actual_temp"],
        linewidth=2.4,
        label="real trajectory",
    )
    ax.plot(
        binned["time_bin_s"],
        binned["predicted_temp"],
        linewidth=2.4,
        label="predicted trajectory",
    )

    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature (deg C)")

    title = f"Best Test Run Trajectory: {best_run}"
    if tail_window_s is not None:
        title += f" - last {tail_window_s:g}s"

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def train_and_validate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    history_root = Path(args.history_root)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    horizons_s = parse_horizons(args.horizons_s)

    print("=== Simple GRU digital twin plant model ===")
    print(f"History root: {history_root}")
    print(f"Horizons:     {horizons_s}")
    print(f"Features:     {len(FEATURE_NAMES)}")
    print(f"Feature list: {FEATURE_NAMES}")

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
            skipped.append(
                {
                    "run_id": run_id,
                    "path": str(csv_path),
                }
            )
            continue

        try:
            df, run_meta = prepare_run_dataframe(csv_path, args)
            X, y, meta = build_sequences_for_run(df, run_id, args, horizons_s)

            if len(X) == 0:
                skipped.append(
                    {
                        "run_id": run_id,
                        "path": str(csv_path),
                    }
                )
                continue

            all_X.append(X)
            all_y.append(y)
            all_meta.append(meta)

            run_meta["num_sequences"] = int(len(X))
            run_summaries.append(run_meta)

        except Exception as exc:
            skipped.append({"run_id": run_id, "path": str(csv_path)})
            print(f"[SKIP] {run_id}: {exc}")

    if not all_X:
        raise RuntimeError("No valid sequences were created.")

    X_raw = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    meta_all = pd.concat(all_meta, ignore_index=True)

    print("\n=== Dataset ===")
    print(f"Total sequences: {len(X_raw)}")
    print(f"X shape:         {X_raw.shape}")
    print(f"y shape:         {y_raw.shape}")
    print(f"Valid runs:      {meta_all['run_id'].nunique()}")

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

    meta_val = meta_all.loc[val_mask].reset_index(drop=True)
    meta_test = meta_all.loc[test_mask].reset_index(drop=True)

    if len(X_train_raw) == 0 or len(X_val_raw) == 0 or len(X_test_raw) == 0:
        raise RuntimeError("Train, validation, or test split is empty.")

    print("\n=== Split ===")
    print(f"Train runs: {len(train_runs)}")
    print(f"Val runs:   {len(val_runs)}")
    print(f"Test runs:  {len(test_runs)}")
    print(f"Train seq:  {len(X_train_raw)}")
    print(f"Val seq:    {len(X_val_raw)}")
    print(f"Test seq:   {len(X_test_raw)}")

    with open(output_dir / "split_runs.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_runs": train_runs,
                "val_runs": val_runs,
                "test_runs": test_runs,
            },
            f,
            indent=2,
        )

    pd.DataFrame(run_summaries).to_csv(
        output_dir / "run_summaries.csv", index=False)
    pd.DataFrame(skipped).to_csv(output_dir / "skipped_runs.csv", index=False)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        x_scaler,
        y_scaler,
    ) = scale_datasets(
        X_train_raw,
        y_train_raw,
        X_val_raw,
        y_val_raw,
        X_test_raw,
        y_test_raw,
    )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    print(f"\nDevice: {device}")

    model = GRUDynamicsModel(
        input_dim=len(FEATURE_NAMES),
        output_dim=len(horizons_s),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_checkpoint_path = output_dir / "gru_simple_multihorizon.pt"
    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0
    history_rows: List[Dict[str, float]] = []

    print("\n=== Training ===")

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

        pred_train_scaled, target_train_scaled, eval_train_loss = predict_loader(
            model,
            train_loader,
            device,
            loss_fn,
        )
        pred_val_scaled, target_val_scaled, val_loss = predict_loader(
            model,
            val_loader,
            device,
            loss_fn,
        )

        train_metrics = compute_real_metrics(
            pred_train_scaled,
            target_train_scaled,
            y_scaler,
            horizons_s,
        )

        val_metrics = compute_real_metrics(
            pred_val_scaled,
            target_val_scaled,
            y_scaler,
            horizons_s,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": eval_train_loss,
            "val_loss": val_loss,
        }

        row.update({f"train_{key}": value for key,
                   value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history_rows.append(row)

        msg = (
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_mae_all={train_metrics['mae_all']:.6f} "
            f"val_mae_all={val_metrics['mae_all']:.6f} "
            f"val_rmse_all={val_metrics['rmse_all']:.6f}"
        )

        for h in horizons_s:
            h_int = int(h)
            msg += f" val_mae_{h_int}s={val_metrics[f'mae_{h_int}s']:.4f}"

        print(msg)

        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "input_dim": len(FEATURE_NAMES),
                "output_dim": len(horizons_s),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "window_steps": args.window_steps,
                "horizons_s": horizons_s,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_metric": "val_loss",
                "best_metric_value": best_val_loss,
                "train_runs": train_runs,
                "val_runs": val_runs,
                "test_runs": test_runs,
                "args": vars(args),
                "error_sign": "error = temp_ref - temp",
            }

            torch.save(checkpoint, best_checkpoint_path)
            print(f"  saved new best checkpoint: {best_checkpoint_path}")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    pd.DataFrame(history_rows).to_csv(
        output_dir / "training_history.csv", index=False)
    plot_training_history(history_rows, plots_dir / "training_mae.png")

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    best_model = GRUDynamicsModel(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)

    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    pred_val_scaled, target_val_scaled, _ = predict_loader(
        best_model,
        val_loader,
        device,
        loss_fn,
    )

    pred_test_scaled, target_test_scaled, _ = predict_loader(
        best_model,
        test_loader,
        device,
        loss_fn,
    )

    final_metrics = compute_real_metrics(
        pred_val_scaled,
        target_val_scaled,
        y_scaler,
        horizons_s,
    )

    test_metrics = compute_real_metrics(
        pred_test_scaled,
        target_test_scaled,
        y_scaler,
        horizons_s,
    )

    pred_val_real = y_scaler.inverse_transform(pred_val_scaled)
    target_val_real = y_scaler.inverse_transform(target_val_scaled)
    pred_test_real = y_scaler.inverse_transform(pred_test_scaled)
    target_test_real = y_scaler.inverse_transform(target_test_scaled)

    save_predictions_csv(
        output_path=output_dir / "validation_predictions.csv",
        meta=meta_val,
        pred_real=pred_val_real,
        target_real=target_val_real,
        horizons_s=horizons_s,
    )

    save_predictions_csv(
        output_path=output_dir / "test_predictions.csv",
        meta=meta_test,
        pred_real=pred_test_real,
        target_real=target_test_real,
        horizons_s=horizons_s,
    )

    val_pred_df = build_prediction_details(
        meta_val,
        pred_val_real,
        target_val_real,
        horizons_s,
    )

    test_pred_df = build_prediction_details(
        meta_test,
        pred_test_real,
        target_test_real,
        horizons_s,
    )

    val_per_run = summarize_metrics_by_run(val_pred_df, horizons_s)
    test_per_run = summarize_metrics_by_run(test_pred_df, horizons_s)

    val_per_run.to_csv(
        output_dir / "validation_metrics_by_run.csv", index=False)
    test_per_run.to_csv(output_dir / "test_metrics_by_run.csv", index=False)

    plot_best_test_prediction(
        test_pred_df,
        test_per_run,
        horizons_s,
        plots_dir / "test_best_run.png",
    )

    plot_best_test_reconstructed_trajectory(
        test_pred_df,
        test_per_run,
        horizons_s,
        plots_dir / "test_best_trajectory.png",
    )

    plot_best_test_reconstructed_trajectory(
        test_pred_df,
        test_per_run,
        horizons_s,
        plots_dir / "test_best_trajectory_tail_300s.png",
        tail_window_s=300.0,
    )

    final_report = {
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "best_model_path": str(best_checkpoint_path),
        "final_metrics": final_metrics,
        "test_metrics": test_metrics,
        "num_train_runs": len(train_runs),
        "num_val_runs": len(val_runs),
        "num_test_runs": len(test_runs),
        "num_train_sequences": int(len(X_train_raw)),
        "num_val_sequences": int(len(X_val_raw)),
        "num_test_sequences": int(len(X_test_raw)),
        "horizons_s": horizons_s,
        "feature_names": FEATURE_NAMES,
        "error_sign": "error = temp_ref - temp",
    }

    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n=== Final best validation metrics ===")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")
    print(f"MAE all horizons:  {final_metrics['mae_all']:.6f} deg C")
    print(f"RMSE all horizons: {final_metrics['rmse_all']:.6f} deg C")

    for h in horizons_s:
        h_int = int(h)
        print(
            f"h={h_int:>4}s | "
            f"MAE={final_metrics[f'mae_{h_int}s']:.6f} deg C | "
            f"RMSE={final_metrics[f'rmse_{h_int}s']:.6f} deg C | "
            f"bias={final_metrics[f'bias_{h_int}s']:.6f} deg C"
        )

    print("\n=== Final best test metrics ===")
    print(f"MAE all horizons:  {test_metrics['mae_all']:.6f} deg C")
    print(f"RMSE all horizons: {test_metrics['rmse_all']:.6f} deg C")

    for h in horizons_s:
        h_int = int(h)
        print(
            f"h={h_int:>4}s | "
            f"MAE={test_metrics[f'mae_{h_int}s']:.6f} deg C | "
            f"RMSE={test_metrics[f'rmse_{h_int}s']:.6f} deg C | "
            f"bias={test_metrics[f'bias_{h_int}s']:.6f} deg C"
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    train_and_validate(args)


if __name__ == "__main__":
    main()
