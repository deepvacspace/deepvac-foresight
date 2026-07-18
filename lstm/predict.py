import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lstm.lstm import (
    FEATURES,
    RUN_FILE_SPECS,
    add_derived_features,
    evaluate_metrics,
    load_model_bundle,
    transform_features,
)

DEFAULT_HISTORY_DIR = Path(__file__).parents[1] / "optimization" / "run_history"
DEFAULT_PLOT_DIR = Path(__file__).parent / "plots"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history-dir",
        type=str,
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing run sample folders. Defaults to optimization/run_history.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id / folder name to predict",
    )

    parser.add_argument(
        "--test-set",
        action="store_true",
        help="Evaluate the held-out test runs saved in the model bundle.",
    )

    parser.add_argument(
        "--model-file",
        type=str,
        default=str(Path(__file__).with_name("lstm_model.pkl")),
        help="Path to the trained LSTM model bundle",
    )

    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=None,
        help=(
            "Number of samples ahead to predict, must match model "
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='"cuda", "cpu", or empty for auto',
    )

    parser.add_argument(
        "--plot-file",
        type=str,
        default=str(DEFAULT_PLOT_DIR / "lstm_prediction.png"),
        help="File name for the prediction comparison plot",
    )

    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="lstm_predictions.csv",
        help="File name for saved prediction CSV",
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="lstm_test_metrics.csv",
        help="File name for saved test metrics CSV",
    )

    return parser


def find_samples_path(history_dir: Path, run_id: str) -> Path:
    for _, sample_name in RUN_FILE_SPECS:
        candidate = history_dir / run_id / sample_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find sample file for run id: {run_id}")


def get_test_paths(bundle: dict[str, object], history_dir: Path) -> list[Path]:
    split_sample_paths = bundle.get("split_sample_paths", {})
    if isinstance(split_sample_paths, dict):
        raw_paths = split_sample_paths.get("test", [])
        paths = [Path(str(path)) for path in raw_paths if str(path)]
        existing = [path for path in paths if path.exists()]
        if existing:
            return existing

    split_run_ids = bundle.get("split_run_ids", {})
    if not isinstance(split_run_ids, dict):
        return []

    run_ids = split_run_ids.get("test", [])
    return [find_samples_path(history_dir, str(run_id)) for run_id in run_ids]


def resolve_prediction_horizon(
    requested_horizon: int | None,
    bundle_horizon: int,
) -> int:
    if requested_horizon is None:
        return bundle_horizon

    if requested_horizon <= 0:
        raise ValueError("--pred-horizon must be greater than 0")

    if requested_horizon != bundle_horizon:
        raise ValueError(
            f"Requested --pred-horizon={requested_horizon}, but the model was trained "
            f"with pred_horizon={bundle_horizon}. Train a new model with "
            f"`python lstm.py --pred-horizon {requested_horizon}`."
        )

    return requested_horizon


def predict_run_temperatures(
    df: pd.DataFrame,
    model,
    feat_scaler,
    y_scaler,
    seq_len: int,
    pred_horizon: int,
    device: torch.device,
) -> np.ndarray:
    """
    Real-history-fed prediction.

    This is NOT recursive.

    For every target index:
        - use only the real logged samples before it
        - predict the target temperature pred_horizon samples ahead
        - do not insert previous predictions into future input windows

    General indexing:
        input window = rows [start_idx : end_idx]
        target       = temp[target_idx]

    where:
        end_idx    = target_idx - pred_horizon
        start_idx  = target_idx - pred_horizon - seq_len + 1

    Example with seq_len=20, pred_horizon=10:
        To predict temp[29], input rows are temp[0] ... temp[19].
        The prediction is plotted at index 29.
    """

    work = add_derived_features(df).reset_index(drop=True).copy()
    pred_temp = np.full(len(work), np.nan, dtype=np.float32)

    missing_features = [c for c in FEATURES if c not in work.columns]
    if missing_features:
        raise RuntimeError(f"Missing required features: {missing_features}")

    first_target_idx = seq_len + pred_horizon - 1

    if first_target_idx >= len(work):
        raise RuntimeError(
            "Run is too short for prediction. "
            f"Need at least seq_len + pred_horizon rows, got {len(work)}."
        )

    model.eval()

    with torch.no_grad():
        for target_idx in range(first_target_idx, len(work)):
            start_idx = target_idx - seq_len - pred_horizon + 1
            end_idx = target_idx - pred_horizon

            # Real-history-fed prediction:
            # This window uses only real logged data.
            # No previous prediction is inserted back into the input.
            window = work.iloc[start_idx : end_idx + 1].copy()

            if len(window) != seq_len:
                raise RuntimeError(
                    f"Internal indexing error: expected window length {seq_len}, "
                    f"got {len(window)} at target_idx={target_idx}"
                )

            x_raw = window[FEATURES].to_numpy(dtype=np.float32)[None, :, :]
            x = transform_features(x_raw, feat_scaler)

            x_t = torch.from_numpy(x).to(device)
            pred_scaled = model(x_t).cpu().numpy().ravel()[0]

            pred_value = y_scaler.inverse_transform([[pred_scaled]]).ravel()[0]
            pred_temp[target_idx] = float(pred_value)

    return pred_temp


def build_predictions_frame(
    df: pd.DataFrame,
    pred_temp: np.ndarray,
    pred_horizon: int,
) -> pd.DataFrame:
    result = df.reset_index(drop=True).copy()

    result["lstm_prediction_temp"] = pred_temp
    result["prediction_horizon"] = pred_horizon
    result["mae"] = (result["lstm_prediction_temp"] - result["temp"]).abs()
    result = result[result["lstm_prediction_temp"].notna()].reset_index(drop=True)

    drop_cols = [
        "abs_prediction_error",
        "baseline_error",
        "abs_baseline_error",
        "baseline_temp",
        "prediction_error",
    ]
    result = result.drop(columns=[col for col in drop_cols if col in result.columns])

    leading_cols = [
        col
        for col in ("run_id", "timestamp", "temp", "lstm_prediction_temp", "mae")
        if col in result.columns
    ]
    remaining_cols = [col for col in result.columns if col not in leading_cols]
    result = result[leading_cols + remaining_cols]
    return result


def save_predictions_csv(
    out_path: Path,
    df: pd.DataFrame,
    pred_temp: np.ndarray,
    pred_horizon: int,
) -> None:
    result = build_predictions_frame(
        df=df,
        pred_temp=pred_temp,
        pred_horizon=pred_horizon,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)


def save_test_predictions_csv(out_path: Path, records: list[dict[str, object]], pred_horizon: int) -> None:
    frames = [
        build_predictions_frame(
            df=record["df"],
            pred_temp=record["pred_temp"],
            pred_horizon=pred_horizon,
        )
        for record in records
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)


def evaluate_run(
    samples_path: Path,
    model,
    feat_scaler,
    y_scaler,
    seq_len: int,
    pred_horizon: int,
    device: torch.device,
) -> dict[str, object]:
    df = pd.read_csv(samples_path).reset_index(drop=True)

    pred_temp = predict_run_temperatures(
        df,
        model=model,
        feat_scaler=feat_scaler,
        y_scaler=y_scaler,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        device=device,
    )

    truth_temp = df["temp"].to_numpy(dtype=np.float32)
    valid_mask = ~np.isnan(pred_temp)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError(f"No valid predictions were generated for {samples_path}.")

    pred_mae, pred_rmse = evaluate_metrics(pred_temp[valid_mask], truth_temp[valid_mask])

    run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns and len(df) else samples_path.parent.name

    return {
        "run_id": run_id,
        "samples_path": str(samples_path),
        "df": df,
        "truth_temp": truth_temp,
        "pred_temp": pred_temp,
        "valid_indices": valid_indices,
        "prediction_points": int(np.sum(valid_mask)),
        "lstm_mae": pred_mae,
        "lstm_rmse": pred_rmse,
        "lstm_mse": float(pred_rmse**2),
    }


def save_test_metrics_csv(out_path: Path, records: list[dict[str, object]]) -> None:
    metric_rows = []
    for record in records:
        metric_rows.append(
            {
                "run_id": record["run_id"],
                "prediction_points": record["prediction_points"],
                "lstm_mae": record["lstm_mae"],
                "lstm_rmse": record["lstm_rmse"],
                "lstm_mse": record["lstm_mse"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)


def plot_test_runs(
    out_path: Path,
    records: list[dict[str, object]],
    pred_horizon: int,
) -> None:
    record = min(records, key=lambda item: float(item["lstm_mae"]))
    truth_temp = record["truth_temp"]
    pred_temp = record["pred_temp"]
    valid_indices = record["valid_indices"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(truth_temp, label="real_temp", linewidth=1.4)
    ax.plot(pred_temp, label=f"lstm", linewidth=1.3, alpha=0.85)
    ax.axvline(valid_indices[0], color="gray", linestyle="--", alpha=0.5)
    ax.set_title(
        f"Best Test Run: {record['run_id']} | "
        f"MAE={record['lstm_mae']:.3f} degC"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Temp (degC)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def print_metrics_summary(records: list[dict[str, object]]) -> None:
    lstm_mae = np.asarray([float(r["lstm_mae"]) for r in records], dtype=np.float64)
    lstm_rmse = np.asarray([float(r["lstm_rmse"]) for r in records], dtype=np.float64)

    print("Test metrics:")
    for record in records:
        print(
            f"  {record['run_id']}: "
            f"LSTM MAE={record['lstm_mae']:.4f}, RMSE={record['lstm_rmse']:.4f}"
        )

    print()
    print("mean test metrics:")
    print(f"LSTM MAE mean:      {float(np.mean(lstm_mae)):.4f} degC")
    print(f"LSTM RMSE mean:     {float(np.mean(lstm_rmse)):.4f} degC")


def main() -> None:
    args = build_arg_parser().parse_args()

    history_dir = Path(args.history_dir)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, feat_scaler, y_scaler, bundle = load_model_bundle(
        Path(args.model_file),
        device=device,
    )

    seq_len = int(bundle["seq_len"])
    bundle_pred_horizon = int(bundle["pred_horizon"])

    pred_horizon = resolve_prediction_horizon(
        requested_horizon=args.pred_horizon,
        bundle_horizon=bundle_pred_horizon,
    )

    if args.test_set:
        test_paths = get_test_paths(bundle, history_dir=history_dir)
        if not test_paths:
            raise RuntimeError(
                "Retrain with lstm.py, run predict.py --test-set."
            )

        records = [
            evaluate_run(
                samples_path=path,
                model=model,
                feat_scaler=feat_scaler,
                y_scaler=y_scaler,
                seq_len=seq_len,
                pred_horizon=pred_horizon,
                device=device,
            )
            for path in test_paths
        ]

        print(f"Model file: {args.model_file}")
        print(f"Device: {device}")
        print(f"Sequence length: {seq_len}")
        print(f"Model horizon: {bundle_pred_horizon}")
        print(f"Test runs evaluated: {len(records)}")
        print()
        print_metrics_summary(records)

        metrics_csv_path = Path(__file__).with_name(args.metrics_csv)
        save_test_metrics_csv(metrics_csv_path, records)
        print(f"Saved test metrics CSV: {metrics_csv_path}")

        predictions_csv_path = Path(__file__).with_name(args.predictions_csv)
        save_test_predictions_csv(predictions_csv_path, records, pred_horizon)
        print(f"Saved predictions CSV: {predictions_csv_path}")

        out_path = Path(args.plot_file)
        plot_test_runs(
            out_path=out_path,
            records=records,
            pred_horizon=pred_horizon,
        )

        return

    if not args.run_id:
        raise ValueError("Pass --run-id for one run, or pass --test-set for held-out test evaluation.")

    samples_path = find_samples_path(history_dir, args.run_id)

    df = pd.read_csv(samples_path).reset_index(drop=True)

    pred_temp = predict_run_temperatures(
        df,
        model=model,
        feat_scaler=feat_scaler,
        y_scaler=y_scaler,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        device=device,
    )

    truth_temp = df["temp"].to_numpy(dtype=np.float32)
    valid_mask = ~np.isnan(pred_temp)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError("No valid predictions were generated.")

    pred_mae, pred_rmse = evaluate_metrics(
        pred_temp[valid_mask],
        truth_temp[valid_mask],
    )

    print(f"Run id: {args.run_id}")
    print(f"Samples file: {samples_path}")
    print(f"Model file: {args.model_file}")
    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")
    print(f"Model horizon: {bundle_pred_horizon}")
    print(f"Prediction start index: {valid_indices[0]}")
    print(f"Prediction points: {int(np.sum(valid_mask))}")
    print()
    print(f"LSTM MAE:      {pred_mae:.4f} degC")
    print(f"LSTM RMSE:     {pred_rmse:.4f} degC")

    predictions_csv_path = Path(__file__).with_name(args.predictions_csv)

    save_predictions_csv(
        out_path=predictions_csv_path,
        df=df,
        pred_temp=pred_temp,
        pred_horizon=pred_horizon,
    )

    print(f"Saved predictions CSV: {predictions_csv_path}")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        truth_temp,
        label="real temp",
        linewidth=1.5,
    )

    ax.plot(
        pred_temp,
        label=f"lstm_prediction_temp_h{pred_horizon}",
        linewidth=1.5,
        alpha=0.85,
    )

    ax.axvline(
        valid_indices[0],
        color="gray",
        linestyle="--",
        alpha=0.6,
    )

    ax.set_title(
        f"Run Prediction: {args.run_id} | "
        f"MAE={pred_mae:.3f} degC | seq_len={seq_len} horizon={pred_horizon}"
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Temp (degC)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_path = Path(args.plot_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
