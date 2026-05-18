#!/usr/bin/env python3
"""Train, validate, and test a one-step GRU plant model from history.
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

WORK_DIR = ROOT / "optimization"
DEFAULT_HISTORY_ROOT = WORK_DIR / "history"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "validation_t1"
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent / "plots_t1"

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
        description="Train, validate, and test a one-step GRU plant model.")
    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument(
        "--telemetry-names",
        nargs="+",
        default=["run_samples.csv"],
    )
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
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
    return ap


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
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
            nn.Linear(hidden_dim, 1),
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

    meta = {
        "run_id": run_id,
        "n_samples": int(len(df)),
        "duration_s": duration_s,
        "start_temp": float(df["temp"].iloc[0]),
        "target_temp": float(df["temp_ref"].iloc[0]),
    }
    return df, meta


def build_t1_sequences_for_run(
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

    # end_idx is current t. target is next row t+1.
    for end_idx in range(window_steps - 1, len(df) - 1):
        next_idx = end_idx + 1
        start_idx = end_idx - window_steps + 1
        x_seq = features[start_idx: end_idx + 1]

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
        raise RuntimeError(
            "Need at least 3 valid runs for run-level train/val/test split.")

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

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train_raw.reshape(-1, n_features))
    y_scaler.fit(y_train_raw)

    X_train = x_scaler.transform(
        X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
    X_val = x_scaler.transform(
        X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape)
    X_test = x_scaler.transform(
        X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    return X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler


def compute_real_metrics(
    pred_scaled: np.ndarray,
    target_scaled: np.ndarray,
    y_scaler: StandardScaler,
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
                "p90_abs_error_delta_t1": float(group["abs_error_delta_t1"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_delta_t1").reset_index(drop=True)


def plot_training_history(history_rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not history_rows:
        return
    history = pd.DataFrame(history_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["train_mae_delta_t1"],
            label="train MAE delta t+1")
    ax.plot(history["epoch"], history["val_mae_delta_t1"],
            label="validation MAE delta t+1")
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
    ax.plot(run_df["elapsed_s"], run_df["target_delta_t1"],
            label="actual delta t+1")
    ax.plot(run_df["elapsed_s"], run_df["pred_delta_t1"],
            label="predicted delta t+1")
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
    ax.plot(run_df["next_elapsed_s"],
            run_df["target_temp_t1"], label="real temp t+1")
    ax.plot(run_df["next_elapsed_s"], run_df["pred_temp_t1"],
            label="predicted temp t+1")
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


def train_and_validate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    history_root = Path(args.history_root)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== GRU  model ===")

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
        if args.exclude_prefixes and any(run_id.startswith(p) for p in args.exclude_prefixes):
            skipped.append({"run_id": run_id, "path": str(csv_path)})
            continue

        try:
            df, run_meta = prepare_run_dataframe(csv_path, args)
            X, y, meta = build_t1_sequences_for_run(df, run_id, args)
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
    print(f"Dataset: {history_root}")
    print(f"Feature list: {FEATURE_NAMES}")
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
        json.dump({"train_runs": train_runs, "val_runs": val_runs,
                  "test_runs": test_runs}, f, indent=2)
    pd.DataFrame(run_summaries).to_csv(
        output_dir / "run_summaries.csv", index=False)

    X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler = scale_datasets(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw
    )

    train_loader = DataLoader(SequenceDataset(
        X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(
        X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(
        X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    print(f"\nDevice: {device}")

    model = GRUModel(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_checkpoint_path = output_dir / "gru_t1.pt"
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
        pred_train_scaled, target_train_scaled, train_eval_loss = predict_loader(
            model, train_loader, device, loss_fn)
        pred_val_scaled, target_val_scaled, val_loss = predict_loader(
            model, val_loader, device, loss_fn)

        train_metrics = compute_real_metrics(
            pred_train_scaled, target_train_scaled, y_scaler)
        val_metrics = compute_real_metrics(
            pred_val_scaled, target_val_scaled, y_scaler)

        row = {"epoch": epoch, "train_loss": train_loss,
               "train_eval_loss": train_eval_loss, "val_loss": val_loss}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history_rows.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_mae={train_metrics['mae_delta_t1']:.6f} "
            f"val_mae={val_metrics['mae_delta_t1']:.6f} "
        )

        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "input_dim": len(FEATURE_NAMES),
                "output_dim": 1,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "window_steps": args.window_steps,
                "target": "temp(t+1 row) - temp(t)",
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
            print(f"new best")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    pd.DataFrame(history_rows).to_csv(
        output_dir / "training_history.csv", index=False)
    plot_training_history(history_rows, plots_dir / "training_mae_t1.png")

    checkpoint = torch.load(best_checkpoint_path,
                            map_location=device, weights_only=False)
    best_model = GRUModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    pred_val_scaled, target_val_scaled, _ = predict_loader(
        best_model, val_loader, device, loss_fn)
    pred_test_scaled, target_test_scaled, _ = predict_loader(
        best_model, test_loader, device, loss_fn)

    final_metrics = compute_real_metrics(
        pred_val_scaled, target_val_scaled, y_scaler)
    test_metrics = compute_real_metrics(
        pred_test_scaled, target_test_scaled, y_scaler)

    pred_val_real = y_scaler.inverse_transform(pred_val_scaled)
    target_val_real = y_scaler.inverse_transform(target_val_scaled)
    pred_test_real = y_scaler.inverse_transform(pred_test_scaled)
    target_test_real = y_scaler.inverse_transform(target_test_scaled)

    val_pred_df = build_prediction_details(
        meta_val, pred_val_real, target_val_real)
    test_pred_df = build_prediction_details(
        meta_test, pred_test_real, target_test_real)

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
        "best_model_path": str(best_checkpoint_path),
        "target": "temp(t+1 row) - temp(t)",
        "final_metrics": final_metrics,
        "test_metrics": test_metrics,
        "num_train_runs": len(train_runs),
        "num_val_runs": len(val_runs),
        "num_test_runs": len(test_runs),
        "num_train_sequences": int(len(X_train_raw)),
        "num_val_sequences": int(len(X_val_raw)),
        "num_test_sequences": int(len(X_test_raw)),
        "feature_names": FEATURE_NAMES,
        "error_sign": "error = temp_ref - temp",
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


def main() -> None:
    args = build_arg_parser().parse_args()
    train_and_validate(args)


if __name__ == "__main__":
    main()
