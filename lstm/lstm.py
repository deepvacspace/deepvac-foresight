import argparse
import copy
import math
import pickle
import random
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_FEATURES = [
    "kp",
    "ki",
    "kd",
    "temp",
    "temp_ref",
    "temp_raw",
    "temp_u",
    "temp_u_p",
    "temp_u_i",
    "temp_u_d",
]

FEATURES = BASE_FEATURES + [
    "abs_error",
]

RUN_FILE_SPECS = (
    ("run_*", "run_samples.csv"),
    ("val_*", "run_samples.csv"),
)

HISTORY_DIR = Path(__file__).parents[1] / "optimization" / "run_history"
PLOT_DIR = Path(__file__).parent / "plots"

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history-dir",
        type=str,
        default=str(HISTORY_DIR),
        help=(
            "Directory containing runs"
        ),
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Sequence length for LSTM input",
    )

    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=20,
        help=(
            "Samples ahead to predict after the input sequence"
        ),
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='"cuda", "cpu", or empty for auto',
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Use 0 to disable.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print training progress every N epochs.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="0 to disable.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum scaled validation MSE improvement to reset early stopping.",
    )

    parser.add_argument(
        "--plot-samples",
        type=int,
        default=300,
        help="Number of test sequence predictions to plot.",
    )

    parser.add_argument(
        "--plot-file",
        type=str,
        default=str(PLOT_DIR / "lstm_training.png"),
        help="File name for saved plot",
    )

    parser.add_argument(
        "--model-file",
        type=str,
        default="lstm_model.pkl",
        help="File name for saved trained model bundle",
    )

    parser.add_argument(
        "--test-plot-file",
        type=str,
        default=str(PLOT_DIR / "lstm_test.png"),
        help="File name for saved held-out test run plot",
    )

    parser.add_argument(
        "--test-metrics-file",
        type=str,
        default="lstm_test_metrics.csv",
        help="File name for saved held-out test metrics CSV",
    )

    parser.add_argument(
        "--test-plot-runs",
        type=int,
        default=4,
        help="Number of held-out test runs to include in the test plot",
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show matplotlib window after saving plot.",
    )

    return parser



class TempLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()

        dropout = 0.2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_run_files(history_dir: Path) -> list[Path]:
    csv_paths: list[Path] = []

    for folder_glob, file_name in RUN_FILE_SPECS:
        csv_paths.extend(sorted(history_dir.glob(f"{folder_glob}/{file_name}")))

    return sorted(set(csv_paths))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_error"] = (df["temp_ref"] - df["temp"]).abs()
    return df


def load_runs(history_dir: Path) -> list[pd.DataFrame]:
    runs = []
    csv_paths = find_run_files(history_dir)

    if not csv_paths:
        expected = ", ".join(f"{folder}/{name}" for folder, name in RUN_FILE_SPECS)
        raise RuntimeError(f"No run files found in: {history_dir} ({expected})")

    print(f"Discovered sample files: {len(csv_paths)}")

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        missing_base = [c for c in BASE_FEATURES if c not in df.columns]
        if missing_base:
            print(f"Skipping {csv_path}: missing columns {missing_base}")
            continue

        df = add_derived_features(df)

        missing_features = [c for c in FEATURES if c not in df.columns]
        if missing_features:
            print(f"Skipping {csv_path}: missing derived/features {missing_features}")
            continue

        df = df.dropna(subset=FEATURES).reset_index(drop=True)

        if len(df) == 0:
            print(f"Skipping {csv_path}: no valid rows after dropna")
            continue

        run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else csv_path.parent.name
        df.attrs["run_id"] = run_id
        df.attrs["samples_path"] = str(csv_path)
        runs.append(df)

    if not runs:
        raise RuntimeError(
            "No valid runs were loaded. Check that sample files contain all required columns."
        )

    return runs


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    pred_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised sequences.

    Input:
        x = rows [i : i + seq_len]

    Target:
        y = temp at i + seq_len + pred_horizon - 1

    """
    vals = df[FEATURES].to_numpy(dtype=np.float32)
    temp_idx = FEATURES.index("temp")

    xs = []
    ys = []

    max_start = len(vals) - seq_len - pred_horizon + 1

    for i in range(max_start):
        x_seq = vals[i : i + seq_len]
        y_temp = vals[i + seq_len + pred_horizon - 1, temp_idx]

        xs.append(x_seq)
        ys.append(y_temp)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )

def split_runs(
    runs: list[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:

    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("--train-ratio, --val-ratio, and derived test ratio must all be > 0")

    idxs = list(range(len(runs)))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    n_runs = len(idxs)
    train_n = max(1, int(n_runs * train_ratio))
    val_n = max(1, int(n_runs * val_ratio))

    if train_n + val_n >= n_runs:
        overflow = train_n + val_n - (n_runs - 1)
        reduce_val = min(overflow, max(0, val_n - 1))
        val_n -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            train_n = max(1, train_n - overflow)

    train_set = [runs[i] for i in idxs[:train_n]]
    val_set = [runs[i] for i in idxs[train_n : train_n + val_n]]
    test_set = [runs[i] for i in idxs[train_n + val_n :]]

    return train_set, val_set, test_set


def run_ids(runs: list[pd.DataFrame]) -> list[str]:
    return [str(run.attrs.get("run_id", f"run_{idx}")) for idx, run in enumerate(runs)]


def run_sample_paths(runs: list[pd.DataFrame]) -> list[str]:
    return [str(run.attrs.get("samples_path", "")) for run in runs]


def collect_xy(
    runs: list[pd.DataFrame],
    seq_len: int,
    pred_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_x = []
    all_y = []

    for run in runs:
        x, y = build_sequences(
            run,
            seq_len=seq_len,
            pred_horizon=pred_horizon,
        )

        if len(x) > 0:
            all_x.append(x)
            all_y.append(y)

    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
    )


def fit_feature_scaler(train_x: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    flattened = train_x.reshape(-1, train_x.shape[-1])
    scaler.fit(flattened)
    return scaler


def transform_features(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(x.shape).astype(np.float32)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y).unsqueeze(1)

    ds = TensorDataset(x_t, y_t)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(math.sqrt(np.mean((pred - truth) ** 2)))
    return mae, rmse


def scaled_mse_to_real_mse(scaled_mse: float, y_scaler: StandardScaler) -> float:
    y_std = float(y_scaler.scale_[0])
    return float(scaled_mse * (y_std ** 2))


def save_model_bundle(
    out_path: Path,
    model_state_dict: dict[str, torch.Tensor],
    feat_scaler: StandardScaler,
    y_scaler: StandardScaler,
    args: argparse.Namespace,
    best_val_loss_scaled: float | None = None,
    best_val_loss_real_mse: float | None = None,
    best_epoch: int | None = None,
    split_run_ids: dict[str, list[str]] | None = None,
    split_sample_paths: dict[str, list[str]] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model_state_dict,
        "features": FEATURES,
        "seq_len": int(args.seq_len),
        "pred_horizon": int(args.pred_horizon),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "feat_scaler": feat_scaler,
        "y_scaler": y_scaler,
        "best_val_loss_scaled": best_val_loss_scaled,
        "best_val_loss_real_mse": best_val_loss_real_mse,
        "best_epoch": best_epoch,
        "split_run_ids": split_run_ids or {},
        "split_sample_paths": split_sample_paths or {},
        "history_dir": str(Path(args.history_dir).resolve()),
    }

    # Keep old key for compatibility with previous prediction scripts.
    payload["best_val_loss"] = best_val_loss_scaled

    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)


def load_model_bundle(
    bundle_path: Path,
    device: torch.device,
) -> tuple[TempLSTM, StandardScaler, StandardScaler, dict[str, object]]:
    with bundle_path.open("rb") as fh:
        payload = pickle.load(fh)

    model = TempLSTM(
        input_size=len(payload["features"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
    ).to(device)

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return model, payload["feat_scaler"], payload["y_scaler"], payload


def run_test_prediction(args: argparse.Namespace, model_out_path: Path) -> None:
    predict_script = Path(__file__).with_name("predict.py")
    cmd = [
        sys.executable,
        str(predict_script),
        "--test-set",
        "--history-dir",
        str(Path(args.history_dir)),
        "--model-file",
        str(model_out_path),
        "--plot-file",
        args.test_plot_file,
        "--metrics-csv",
        args.test_metrics_file,
    ]

    if args.device:
        cmd.extend(["--device", args.device])

    print("\nRunning prediction", flush=True)
    subprocess.run(cmd, check=True)


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    if args.pred_horizon <= 0:
        raise ValueError("--pred-horizon must be > 0")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1 so a test split remains")

    history_dir = Path(args.history_dir)

    runs = load_runs(history_dir=history_dir)

    print(f"Loaded valid runs: {len(runs)}")
    print(f"Features: {FEATURES}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prediction horizon: {args.pred_horizon} samples")

    train_runs, val_runs, test_runs = split_runs(
        runs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Train runs: {len(train_runs)}")
    print(f"Val runs: {len(val_runs)}")
    print(f"Test runs: {len(test_runs)}")

    x_train_raw, y_train = collect_xy(
        train_runs,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
    )

    x_val_raw, y_val = collect_xy(
        val_runs,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
    )

    print(f"Train sequences: {len(x_train_raw)}")
    print(f"Val sequences: {len(x_val_raw)}")

    feat_scaler = fit_feature_scaler(x_train_raw)

    x_train = transform_features(x_train_raw, feat_scaler)
    x_val = transform_features(x_val_raw, feat_scaler)

    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(
        y_train.reshape(-1, 1)
    ).astype(np.float32).ravel()

    y_val_scaled = y_scaler.transform(
        y_val.reshape(-1, 1)
    ).astype(np.float32).ravel()

    train_loader = make_loader(
        x_train,
        y_train_scaled,
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        x_val,
        y_val_scaled,
        args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Device:", device)
    print(f"Target temp mean used by scaler: {float(y_scaler.mean_[0]):.6f} degC")
    print(f"Target temp std used by scaler:  {float(y_scaler.scale_[0]):.6f} degC")

    model = TempLSTM(
        input_size=len(FEATURES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()

    train_losses_scaled = []
    val_losses_scaled = []

    train_losses_real_mse = []
    val_losses_real_mse = []
    train_losses_real_mae = []
    val_losses_real_mae = []

    best_val_loss_scaled = float("inf")
    best_val_loss_real_mse = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_train = 0.0
        total_train_abs_error_scaled = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_train += float(loss.item()) * len(xb)
            total_train_abs_error_scaled += float(torch.mean(torch.abs(pred.detach() - yb)).item()) * len(xb)
            n_train += len(xb)

        model.eval()

        total_val = 0.0
        total_val_abs_error_scaled = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                total_val += float(loss.item()) * len(xb)
                total_val_abs_error_scaled += float(torch.mean(torch.abs(pred - yb)).item()) * len(xb)
                n_val += len(xb)

        train_loss_scaled = total_train / max(1, n_train)
        val_loss_scaled = total_val / max(1, n_val)
        train_loss_real_mae = (total_train_abs_error_scaled / max(1, n_train)) * float(y_scaler.scale_[0])
        val_loss_real_mae = (total_val_abs_error_scaled / max(1, n_val)) * float(y_scaler.scale_[0])

        train_loss_real_mse = scaled_mse_to_real_mse(train_loss_scaled, y_scaler)
        val_loss_real_mse = scaled_mse_to_real_mse(val_loss_scaled, y_scaler)

        train_losses_scaled.append(train_loss_scaled)
        val_losses_scaled.append(val_loss_scaled)

        train_losses_real_mse.append(train_loss_real_mse)
        val_losses_real_mse.append(val_loss_real_mse)
        train_losses_real_mae.append(float(train_loss_real_mae))
        val_losses_real_mae.append(float(val_loss_real_mae))

        improved = val_loss_scaled < (
            best_val_loss_scaled - args.early_stopping_min_delta
        )

        if improved:
            best_val_loss_scaled = val_loss_scaled
            best_val_loss_real_mse = val_loss_real_mse
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"train_mse_scaled={train_loss_scaled:.6f} | "
                f"val_mse_scaled={val_loss_scaled:.6f} | "
                f"train_mae={train_loss_real_mae:.6f} degC | "
                f"val_mae={val_loss_real_mae:.6f} degC"
            )

        if improved:
            print(
                f"  New best validation: "
                f"val_mse_scaled={val_loss_scaled:.6f}, "
                f"val_mse_real={val_loss_real_mse:.6f} degC^2"
            )

        if (
            args.early_stopping_patience > 0
            and early_stop_counter >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch} with "
                f"val_mse_scaled={best_val_loss_scaled:.6f}, "
                f"val_mse_real={best_val_loss_real_mse:.6f} degC^2"
            )
            break

    model.load_state_dict(best_model_state)
    model.eval()

    print("\nTraining summary:")
    print(f"Prediction horizon = {args.pred_horizon} samples")
    print(f"Best validation epoch      = {best_epoch}")
    print(f"Best validation MSE scaled = {best_val_loss_scaled:.6f}")
    print(f"Best validation MSE real   = {best_val_loss_real_mse:.6f} degC^2")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(train_losses_real_mae, label="train_mae")
    ax.plot(val_losses_real_mae, label="val_mae")
    ax.set_title("Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (degC)")
    ax.legend()

    plt.tight_layout()

    out_path = Path(args.plot_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    model_out_path = Path(__file__).with_name(args.model_file)

    save_model_bundle(
        model_out_path,
        model_state_dict=best_model_state,
        feat_scaler=feat_scaler,
        y_scaler=y_scaler,
        args=args,
        best_val_loss_scaled=best_val_loss_scaled,
        best_val_loss_real_mse=best_val_loss_real_mse,
        best_epoch=best_epoch,
        split_run_ids={
            "train": run_ids(train_runs),
            "val": run_ids(val_runs),
            "test": run_ids(test_runs),
        },
        split_sample_paths={
            "train": run_sample_paths(train_runs),
            "val": run_sample_paths(val_runs),
            "test": run_sample_paths(test_runs),
        },
    )

    run_test_prediction(args, model_out_path=model_out_path)

    if args.show_plot:
        plt.show()


def main() -> None:
    run_training(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
