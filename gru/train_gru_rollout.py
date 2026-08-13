#!/usr/bin/env python3
"""Train the GRU plant model on multi-step rollouts instead of single steps.

  1. TRAINING. The loss is computed over an N-step unrolled rollout, with
     gradients flowing through the unroll. During the rollout the temperature is
     fed back from the model's own prediction and the PID/diff controller is
     simulated in-graph, as deepvac.mpc.step_state does at inference. Only the
     exogenous signals (temp_ref, kp, ki, kd) are read from the log.

  2. MODEL SELECTION. Early stopping and checkpointing use free-running rollout
     MAE over held-out runs at the MPC horizon, not 1-step loss. Both are
     reported every epoch.

  3. SPLITTING. --split-by pid-config keeps every run sharing a PID configuration
     in the same split, so no test configuration appears in training.

The checkpoint layout matches gru/train_gru.py, so the result is a drop-in
replacement in mpc_gru.py, mpc_batch.py, and simulate_gru.py.

Example -- fine-tune an existing 1-step checkpoint with a 40-step rollout loss:

    python -m gru.train_gru_rollout \
        --init-from gru/validation_t1/gru_t1.pt \
        --rollout-steps 40 --epochs 40 --split-by pid-config
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Binds the real package before gru_common can install its sklearn stub, whose
# missing __spec__ makes torch's dynamo tracer raise on find_spec("sklearn").
import sklearn.preprocessing  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.datasets import (  # noqa: E402
    FEATURE_NAMES,
    build_sequences,
    find_run_csvs,
    prepare_run_dataframe,
    set_seed,
)
from gru.model import GRUModel  # noqa: E402

WORK_DIR = ROOT / "optimization"
DEFAULT_HISTORY_ROOT = WORK_DIR / "run_history"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "validation_rollout"

# gru_common.CodesysDiff / ChamberPID constants.
DIFF_DC = 0.995
DIFF_CLIP = 5.0
DIFF_GAIN = 10.0
D_PART_CLIP = 0.4

# Features the rollout regenerates rather than reads from the log.
ENDOGENOUS = ("temp", "error", "temp_u", "temp_u_p", "temp_u_i", "temp_u_d")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train a GRU plant model on multi-step rollouts, selected on rollout error.",
    )

    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument("--telemetry-names", nargs="+", default=["run_samples.csv"])
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--window-steps", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--min-duration-s", type=float, default=300.0)
    ap.add_argument("--exclude-prefixes", nargs="*", default=[])
    ap.add_argument("--max-runs", type=int, default=None, help="Cap runs loaded, for quick checks.")

    # --- Splitting -------------------------------------------------------------
    ap.add_argument(
        "--split-by",
        choices=["pid-config", "run"],
        default="pid-config",
        help=(
            "pid-config keeps runs sharing a PID configuration together so no test "
            "configuration appears in training. run splits by run id."
        ),
    )
    ap.add_argument(
        "--config-key",
        choices=["triplet-set", "first-triplet"],
        default="triplet-set",
        help=(
            "What counts as one PID configuration. triplet-set = every triplet used "
            "in the run. first-triplet = the far-band entry gains only, which is "
            "coarser and therefore stricter."
        ),
    )
    ap.add_argument("--train-fraction", type=float, default=0.80)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)

    # --- Rollout ---------------------------------------------------------------
    ap.add_argument("--rollout-steps", type=int, default=40,
                    help="Unroll length used for the training loss. Match the MPC horizon.")
    ap.add_argument("--eval-horizon-steps", type=int, default=None,
                    help="Unroll length for validation/test rollouts. Default: --rollout-steps.")
    ap.add_argument("--rollout-stride", type=int, default=10,
                    help="Sample every Nth window as a rollout start.")
    ap.add_argument("--eval-stride", type=int, default=25,
                    help="Rollout start stride for validation/test.")
    ap.add_argument(
        "--control-mode",
        choices=["simulate", "teacher"],
        default="simulate",
        help=(
            "simulate runs the PID/diff controller in-graph from the predicted "
            "temperature, matching inference. teacher reads temp_u* from the log."
        ),
    )
    ap.add_argument("--pid-period-s", type=float, default=0.1)
    ap.add_argument("--dt-s", type=float, default=None,
                    help="Step size for the PID substep count. Default: each run's median sample dt.")
    ap.add_argument("--control-feature-scale", type=float, default=100.0)
    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument("--max-abs-temp", type=float, default=100.0)

    ap.add_argument(
        "--curriculum-epochs",
        type=int,
        default=0,
        help="Ramp the unroll from --curriculum-start-steps to --rollout-steps over this many epochs. 0 disables.",
    )
    ap.add_argument("--curriculum-start-steps", type=int, default=1)
    ap.add_argument(
        "--tbptt-steps",
        type=int,
        default=0,
        help="Detach the rollout state every N steps to cap graph depth. 0 keeps full BPTT.",
    )
    ap.add_argument(
        "--bias-weight",
        type=float,
        default=0.0,
        help="Extra penalty on the mean signed rollout error.",
    )

    # --- Optimization ----------------------------------------------------------
    ap.add_argument("--init-from", default=None,
                    help="Warm-start weights, scalers, and architecture from an existing checkpoint.")
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.044)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--huber-beta", type=float, default=0.134)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--checkpoint-name", default="gru_rollout.pt")
    ap.add_argument("--report-name", default="rollout_report.json")

    return ap


# -----------------------------------------------------------------------------
# Differentiable controller (torch port of gru_common.ChamberPID / CodesysDiff)
# -----------------------------------------------------------------------------


def torch_pid_substeps(
    *,
    temp: torch.Tensor,
    temp_ref: torch.Tensor,
    kp: torch.Tensor,
    ki: torch.Tensor,
    kd: torch.Tensor,
    i_part: torch.Tensor,
    diff_prev: torch.Tensor,
    diff_filter: torch.Tensor,
    n_substeps: int,
    feature_scale: float,
    u_min: float,
    u_max: float,
    i_reverse_mul: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable equivalent of deepvac.mpc.run_pid_substeps (temp_mode="hold").

    Logged kp and ki are always >= 1, so the scalar version's p_coef == 0 and
    effective_i == 0 guards are omitted.
    """
    u = u_p = u_d = torch.zeros_like(temp)

    for _ in range(n_substeps):
        # temp is held constant across substeps, so the filter decays after the
        # first pass.
        filter_in = torch.clamp(temp - diff_prev, -DIFF_CLIP, DIFF_CLIP)
        diff_filter = DIFF_DC * diff_filter + (1.0 - DIFF_DC) * filter_in
        diff_prev = temp
        diff_out = DIFF_GAIN * torch.clamp(diff_filter, -DIFF_CLIP, DIFF_CLIP)

        delta = temp_ref - temp
        inv_kp = 1.0 / kp

        u_p = inv_kp * delta
        effective_i = torch.where(delta * i_part < 0.0, ki * i_reverse_mul, ki)
        can_integrate = torch.abs(delta) < (1.2 * kp)
        i_part = torch.where(
            can_integrate,
            i_part + inv_kp * (delta * 0.1 / effective_i),
            i_part,
        )
        u_d = inv_kp * (kd * -diff_out)

        i_part = torch.clamp(i_part, u_min, u_max)
        u_d = torch.clamp(u_d, -D_PART_CLIP, D_PART_CLIP)

        # u sums the unclipped p_part with the clipped i/d parts, then clips.
        u = torch.clamp(u_p + i_part + u_d, u_min, u_max)
        u_p = torch.clamp(u_p, u_min, u_max)

    terms = {
        "temp_u": u * feature_scale,
        "temp_u_p": u_p * feature_scale,
        "temp_u_i": i_part * feature_scale,
        "temp_u_d": u_d * feature_scale,
    }
    return terms, i_part, diff_prev, diff_filter


def assemble_feature_row(
    feature_index: Dict[str, int],
    *,
    temp: torch.Tensor,
    temp_ref: torch.Tensor,
    kp: torch.Tensor,
    ki: torch.Tensor,
    kd: torch.Tensor,
    terms: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build one raw feature row per batch element, ordered like FEATURE_NAMES."""
    values = {
        "temp": temp,
        "temp_ref": temp_ref,
        "error": temp_ref - temp,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        **terms,
    }
    columns = [None] * len(feature_index)
    zero = torch.zeros_like(temp)
    for name, idx in feature_index.items():
        columns[idx] = values.get(name, zero)
    return torch.stack(columns, dim=1)


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------


@dataclass
class Scalers:
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    y_scale: torch.Tensor

    def scale_x(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.x_mean) / self.x_scale

    def unscale_y(self, scaled: torch.Tensor) -> torch.Tensor:
        return scaled * self.y_scale + self.y_mean


def unroll(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    scalers: Scalers,
    args: argparse.Namespace,
    feature_index: Dict[str, int],
    horizon: int,
) -> torch.Tensor:
    """Free-running rollout. Returns predicted temperatures, shape (B, horizon).

    Follows deepvac.mpc.step_state: the last window row is rewritten from the
    current temperature and the controller terms, the model predicts a delta, then
    the window rolls and a row built from the new temperature is appended.
    """
    window = batch["window"]              # (B, W, F) raw
    temp = batch["temp0"]                 # (B,)
    i_part = batch["i_part0"]
    diff_prev = batch["diff_prev0"]
    diff_filter = batch["diff_filter0"]
    exog = batch["exog"]                  # (B, H, 4): temp_ref, kp, ki, kd
    logged_terms = batch.get("logged_terms")  # (B, H, 4), --control-mode teacher

    n_substeps = max(1, int(round(float(batch["dt_s"][0].item()) / max(float(args.pid_period_s), 1e-6))))
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)
    tbptt = int(args.tbptt_steps)

    preds: List[torch.Tensor] = []

    for step in range(horizon):
        temp_ref = exog[:, step, 0]
        kp, ki, kd = exog[:, step, 1], exog[:, step, 2], exog[:, step, 3]

        if args.control_mode == "simulate":
            terms, i_part, diff_prev, diff_filter = torch_pid_substeps(
                temp=temp, temp_ref=temp_ref, kp=kp, ki=ki, kd=kd,
                i_part=i_part, diff_prev=diff_prev, diff_filter=diff_filter,
                n_substeps=n_substeps, feature_scale=feature_scale,
                u_min=float(args.u_min), u_max=float(args.u_max),
                i_reverse_mul=float(args.pid_i_reverse_mul),
            )
        else:
            terms = {
                name: logged_terms[:, step, j]
                for j, name in enumerate(("temp_u", "temp_u_p", "temp_u_i", "temp_u_d"))
            }

        current_row = assemble_feature_row(
            feature_index, temp=temp, temp_ref=temp_ref, kp=kp, ki=ki, kd=kd, terms=terms
        )
        window = torch.cat([window[:, :-1, :], current_row.unsqueeze(1)], dim=1)

        delta = scalers.unscale_y(model(scalers.scale_x(window))).squeeze(-1)
        next_temp = torch.clamp(temp + delta, -float(args.max_abs_temp), float(args.max_abs_temp))

        next_row = assemble_feature_row(
            feature_index, temp=next_temp, temp_ref=temp_ref, kp=kp, ki=ki, kd=kd, terms=terms
        )
        window = torch.cat([window[:, 1:, :], next_row.unsqueeze(1)], dim=1)

        preds.append(next_temp)
        temp = next_temp

        if tbptt > 0 and (step + 1) % tbptt == 0:
            window = window.detach()
            temp = temp.detach()
            i_part = i_part.detach()
            diff_prev = diff_prev.detach()
            diff_filter = diff_filter.detach()

    return torch.stack(preds, dim=1)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def pid_config_signature(df: pd.DataFrame, key: str = "triplet-set") -> str:
    """Identity of the PID configuration a run used, derived from its samples.

    triplet-set   every distinct (kp, ki, kd) applied during the run.
    first-triplet only the triplet active at the start (the far-band entry gains).
    """
    coefs = df[["kp", "ki", "kd"]].round(0).astype(int)

    if key == "first-triplet":
        first = coefs.iloc[0].to_numpy().tolist()
        return json.dumps([first], separators=(",", ":"))

    triplets = (
        coefs.drop_duplicates().sort_values(["kp", "ki", "kd"]).to_numpy().tolist()
    )
    return json.dumps(triplets, separators=(",", ":"))


def split_by_group(
    run_to_group: Dict[str, str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Assign whole groups to splits, greedily hitting the target run fractions."""
    groups: Dict[str, List[str]] = {}
    for run_id, group in run_to_group.items():
        groups.setdefault(group, []).append(run_id)

    keys = sorted(groups)
    if len(keys) < 3:
        raise RuntimeError(
            f"Need at least 3 distinct groups to split, found {len(keys)}. "
            "Use --split-by run if the history has too few distinct PID configurations."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    # Largest groups first.
    keys.sort(key=lambda k: -len(groups[k]))

    total = sum(len(groups[k]) for k in keys)
    targets = {
        "train": train_fraction * total,
        "val": val_fraction * total,
        "test": max(0.0, 1.0 - train_fraction - val_fraction) * total,
    }
    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for key in keys:
        # Whichever split is furthest below its target, in relative terms.
        name = max(targets, key=lambda n: (targets[n] - counts[n]) / max(targets[n], 1e-9))
        buckets[name].extend(groups[key])
        counts[name] += len(groups[key])

    if not all(buckets.values()):
        raise RuntimeError(f"A split came out empty: {({k: len(v) for k, v in buckets.items()})}")

    return sorted(buckets["train"]), sorted(buckets["val"]), sorted(buckets["test"])


def build_rollout_samples(
    df: pd.DataFrame,
    run_id: str,
    args: argparse.Namespace,
    horizon: int,
    stride: int,
) -> List[Dict[str, np.ndarray]]:
    """Rollout start points for one run."""
    features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    temp = df["temp"].to_numpy(dtype=np.float32)
    elapsed = df["elapsed_s"].to_numpy(dtype=np.float64)
    window_steps = int(args.window_steps)

    exog_cols = df[["temp_ref", "kp", "ki", "kd"]].to_numpy(dtype=np.float32)
    term_cols = df[["temp_u", "temp_u_p", "temp_u_i", "temp_u_d"]].to_numpy(dtype=np.float32)

    dt_s = float(args.dt_s) if args.dt_s is not None else float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else 1.0
    if not np.isfinite(dt_s) or dt_s <= 0:
        dt_s = 1.0

    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)
    samples: List[Dict[str, np.ndarray]] = []

    last_start = len(df) - horizon - 1
    for end_idx in range(window_steps - 1, last_start + 1, max(1, stride)):
        window = features[end_idx - window_steps + 1 : end_idx + 1]
        if len(window) != window_steps:
            continue
        target = temp[end_idx + 1 : end_idx + 1 + horizon]
        exog = exog_cols[end_idx : end_idx + horizon]
        terms = term_cols[end_idx : end_idx + horizon]
        if len(target) != horizon or len(exog) != horizon:
            continue
        if not (np.isfinite(window).all() and np.isfinite(target).all() and np.isfinite(exog).all()):
            continue

        samples.append({
            "window": window.astype(np.float32),
            "temp0": np.float32(temp[end_idx]),
            # Controller seeded from the logged control state at the start point.
            "i_part0": np.float32(term_cols[end_idx, 2] / feature_scale),
            "diff_prev0": np.float32(temp[end_idx]),
            "diff_filter0": np.float32(0.0),
            "exog": exog.astype(np.float32),
            "logged_terms": terms.astype(np.float32),
            "target": target.astype(np.float32),
            "dt_s": np.float32(dt_s),
            "run_id": run_id,
        })

    return samples


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "window": torch.from_numpy(s["window"]),
            "temp0": torch.tensor(s["temp0"]),
            "i_part0": torch.tensor(s["i_part0"]),
            "diff_prev0": torch.tensor(s["diff_prev0"]),
            "diff_filter0": torch.tensor(s["diff_filter0"]),
            "exog": torch.from_numpy(s["exog"]),
            "logged_terms": torch.from_numpy(s["logged_terms"]),
            "target": torch.from_numpy(s["target"]),
            "dt_s": torch.tensor(s["dt_s"]),
        }


def load_runs(args: argparse.Namespace) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], List[Dict[str, object]]]:
    csvs = find_run_csvs(Path(args.history_root), args.telemetry_names)
    if args.exclude_prefixes:
        csvs = [p for p in csvs if not any(p.parent.name.startswith(x) for x in args.exclude_prefixes)]
    if args.max_runs is not None:
        csvs = csvs[: int(args.max_runs)]

    frames: Dict[str, pd.DataFrame] = {}
    groups: Dict[str, str] = {}
    skipped: List[Dict[str, object]] = []

    for csv_path in csvs:
        try:
            df, meta = prepare_run_dataframe(csv_path, args)
        except Exception as exc:
            skipped.append({"run": csv_path.parent.name, "reason": str(exc)})
            continue
        run_id = str(meta["run_id"])
        frames[run_id] = df
        groups[run_id] = pid_config_signature(df, getattr(args, "config_key", "triplet-set"))

    if len(frames) < 3:
        raise RuntimeError(f"Only {len(frames)} usable runs under {args.history_root}")

    return frames, groups, skipped


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def evaluate_rollout(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    scalers: Scalers,
    args: argparse.Namespace,
    feature_index: Dict[str, int],
    horizon: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    errors: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = unroll(model, batch, scalers, args, feature_index, horizon)
            errors.append((pred - batch["target"]).cpu().numpy())

    err = np.concatenate(errors, axis=0)  # (N, horizon)
    per_step_mae = np.mean(np.abs(err), axis=0)
    per_step_bias = np.mean(err, axis=0)

    return {
        "n_rollouts": int(err.shape[0]),
        "horizon": int(horizon),
        "mae_temp": float(np.mean(np.abs(err))),
        "rmse_temp": float(np.sqrt(np.mean(err ** 2))),
        "bias_temp": float(np.mean(err)),
        "mae_final_step": float(per_step_mae[-1]),
        "bias_final_step": float(per_step_bias[-1]),
        "p90_abs_error_temp": float(np.percentile(np.abs(err), 90)),
        "max_abs_error_temp": float(np.max(np.abs(err))),
        # Drift-vs-horizon curve.
        "per_step_mae": [float(v) for v in per_step_mae],
        "per_step_bias": [float(v) for v in per_step_bias],
    }


def evaluate_one_step(
    model: nn.Module,
    frames: Dict[str, pd.DataFrame],
    run_ids: Sequence[str],
    scalers: Scalers,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """1-step teacher-forced metrics."""
    xs, ys = [], []
    for run_id in run_ids:
        X, y, _ = build_sequences(frames[run_id], run_id, args)
        if len(X):
            xs.append(X)
            ys.append(y)
    if not xs:
        return {"mae_delta_t1": float("nan"), "rmse_delta_t1": float("nan"), "bias_delta_t1": float("nan")}

    X = torch.from_numpy(np.concatenate(xs, axis=0))
    y = np.concatenate(ys, axis=0)[:, 0]

    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = X[i : i + 512].to(device)
            out = scalers.unscale_y(model(scalers.scale_x(xb))).squeeze(-1)
            preds.append(out.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    err = pred - y

    return {
        "mae_delta_t1": float(np.mean(np.abs(err))),
        "rmse_delta_t1": float(np.sqrt(np.mean(err ** 2))),
        "bias_delta_t1": float(np.mean(err)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def make_scalers(
    frames: Dict[str, pd.DataFrame],
    train_runs: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    init_checkpoint: Optional[Dict[str, object]],
) -> Tuple[Scalers, object, object]:
    """Reuse the warm-start checkpoint's scalers if present, else fit on train runs."""
    from sklearn.preprocessing import StandardScaler

    if init_checkpoint is not None:
        x_scaler = init_checkpoint["x_scaler"]
        y_scaler = init_checkpoint["y_scaler"]
    else:
        xs, ys = [], []
        for run_id in train_runs:
            X, y, _ = build_sequences(frames[run_id], run_id, args)
            if len(X):
                xs.append(X.reshape(-1, X.shape[-1]))
                ys.append(y)
        x_scaler = StandardScaler().fit(np.concatenate(xs, axis=0))
        y_scaler = StandardScaler().fit(np.concatenate(ys, axis=0))

    scalers = Scalers(
        x_mean=torch.tensor(np.asarray(x_scaler.mean_, dtype=np.float32), device=device),
        x_scale=torch.tensor(np.asarray(x_scaler.scale_, dtype=np.float32), device=device),
        y_mean=torch.tensor(np.asarray(y_scaler.mean_, dtype=np.float32), device=device),
        y_scale=torch.tensor(np.asarray(y_scaler.scale_, dtype=np.float32), device=device),
    )
    return scalers, x_scaler, y_scaler


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be >= 1")
    if args.rollout_stride < 1 or args.eval_stride < 1:
        raise ValueError("--rollout-stride and --eval-stride must be >= 1")
    eval_horizon = int(args.eval_horizon_steps if args.eval_horizon_steps is not None else args.rollout_steps)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}

    print(f"[data] loading runs from {args.history_root}")
    frames, groups, skipped = load_runs(args)
    print(f"[data] {len(frames)} usable runs, {len(skipped)} skipped, "
          f"{len(set(groups.values()))} distinct PID configurations (--config-key {args.config_key})")

    if args.split_by == "pid-config":
        train_runs, val_runs, test_runs = split_by_group(
            groups, args.train_fraction, args.val_fraction, args.seed
        )
    else:
        from deepvac.datasets import split_runs as split_by_run
        train_runs, val_runs, test_runs = split_by_run(
            sorted(frames), args.train_fraction, args.val_fraction, args.seed
        )

    # How many held-out configurations also appear in training.
    train_cfgs = {groups[r] for r in train_runs}
    leak = {
        "test_configs": len({groups[r] for r in test_runs}),
        "test_configs_seen_in_train": len({groups[r] for r in test_runs} & train_cfgs),
        "val_configs": len({groups[r] for r in val_runs}),
        "val_configs_seen_in_train": len({groups[r] for r in val_runs} & train_cfgs),
    }
    print(f"[split] by={args.split_by} train={len(train_runs)} val={len(val_runs)} test={len(test_runs)} runs")
    print(f"[split] test PID configs: {leak['test_configs']}, "
          f"of which already in train: {leak['test_configs_seen_in_train']}")
    if args.split_by == "run" and leak["test_configs_seen_in_train"]:
        print("[split] WARNING run-level split leaks configurations into test; "
              "test metrics measure interpolation, not generalization to unseen PIDs.")

    init_checkpoint = None
    if args.init_from:
        from gru.gru_common import _ensure_sklearn_stub

        _ensure_sklearn_stub()
        init_checkpoint = torch.load(Path(args.init_from), map_location=device, weights_only=False)
        args.hidden_dim = int(init_checkpoint["hidden_dim"])
        args.num_layers = int(init_checkpoint["num_layers"])
        args.dropout = float(init_checkpoint["dropout"])
        args.window_steps = int(init_checkpoint.get("window_steps", args.window_steps))
        print(f"[init] warm-starting from {args.init_from} "
              f"(hidden={args.hidden_dim} layers={args.num_layers} window={args.window_steps})")

    scalers, x_scaler, y_scaler = make_scalers(frames, train_runs, args, device, init_checkpoint)

    print(f"[data] building rollouts (train stride={args.rollout_stride}, eval stride={args.eval_stride})")
    train_samples = [s for r in train_runs
                     for s in build_rollout_samples(frames[r], r, args, args.rollout_steps, args.rollout_stride)]
    val_samples = [s for r in val_runs
                   for s in build_rollout_samples(frames[r], r, args, eval_horizon, args.eval_stride)]
    test_samples = [s for r in test_runs
                    for s in build_rollout_samples(frames[r], r, args, eval_horizon, args.eval_stride)]
    if not train_samples or not val_samples:
        raise RuntimeError("No rollout samples built; reduce --rollout-steps or --min-samples.")
    print(f"[data] rollouts train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    train_loader = torch.utils.data.DataLoader(
        RolloutDataset(train_samples), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(RolloutDataset(val_samples), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(RolloutDataset(test_samples), batch_size=args.batch_size)

    model = GRUModel(
        input_dim=len(FEATURE_NAMES), hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    if init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    baseline = evaluate_rollout(model, val_loader, scalers, args, feature_index, eval_horizon, device)
    print(f"[epoch  0] (init) val_rollout_mae={baseline['mae_temp']:.4f} bias={baseline['bias_temp']:+.4f}")

    best_metric = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if args.curriculum_epochs > 0:
            frac = min(1.0, (epoch - 1) / max(1, args.curriculum_epochs))
            horizon = int(round(args.curriculum_start_steps
                                + frac * (args.rollout_steps - args.curriculum_start_steps)))
            horizon = max(1, min(int(args.rollout_steps), horizon))
        else:
            horizon = int(args.rollout_steps)

        model.train()
        losses: List[float] = []
        t0 = time.perf_counter()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = unroll(model, batch, scalers, args, feature_index, horizon)
            target = batch["target"][:, :horizon]

            loss = loss_fn(pred, target)
            if args.bias_weight > 0:
                loss = loss + float(args.bias_weight) * torch.abs(torch.mean(pred - target))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate_rollout(model, val_loader, scalers, args, feature_index, eval_horizon, device)
        val_one_step = evaluate_one_step(model, frames, val_runs, scalers, args, device)
        metric = float(val_metrics["mae_temp"])

        row = {
            "epoch": epoch,
            "train_horizon": horizon,
            "train_loss": float(np.mean(losses)),
            "val_rollout_mae": metric,
            "val_rollout_bias": float(val_metrics["bias_temp"]),
            "val_rollout_final_mae": float(val_metrics["mae_final_step"]),
            "val_mae_delta_t1": float(val_one_step["mae_delta_t1"]),
            "seconds": float(time.perf_counter() - t0),
        }
        history.append(row)
        print(
            f"[epoch {epoch:2d}] h={horizon:2d} train_loss={row['train_loss']:.5f} "
            f"val_rollout_mae={metric:.4f} bias={row['val_rollout_bias']:+.4f} "
            f"final_step_mae={row['val_rollout_final_mae']:.4f} "
            f"val_1step_mae={row['val_mae_delta_t1']:.5f} ({row['seconds']:.0f}s)"
        )

        if metric < best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[stop] no rollout improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    model.to(device)

    print(f"[best] epoch {best_epoch}, val_rollout_mae={best_metric:.4f}")

    test_rollout = evaluate_rollout(model, test_loader, scalers, args, feature_index, eval_horizon, device) \
        if test_samples else {}
    test_one_step = evaluate_one_step(model, frames, test_runs, scalers, args, device)
    val_rollout = evaluate_rollout(model, val_loader, scalers, args, feature_index, eval_horizon, device)

    checkpoint_path = output_dir / args.checkpoint_name
    torch.save(
        {
            # Layout matches train_gru.py's checkpoints.
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_names": list(FEATURE_NAMES),
            "window_steps": int(args.window_steps),
            "target": "temp(t+1 row) - temp(t)",
            "training": "multi_step_rollout",
            "rollout_steps": int(args.rollout_steps),
            "control_mode": args.control_mode,
            "split_by": args.split_by,
            "best_metric": "val_rollout_mae_temp",
            "best_metric_value": float(best_metric),
            "best_epoch": int(best_epoch),
        },
        checkpoint_path,
    )

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "history_root": str(Path(args.history_root).resolve()),
        "selection_metric": "val_rollout_mae_temp",
        "eval_horizon_steps": eval_horizon,
        "best_epoch": best_epoch,
        "split": {
            "by": args.split_by,
            "config_key": args.config_key,
            "n_train_runs": len(train_runs), "n_val_runs": len(val_runs), "n_test_runs": len(test_runs),
            "n_distinct_configs": len(set(groups.values())),
            **leak,
            "train_runs": train_runs, "val_runs": val_runs, "test_runs": test_runs,
        },
        "rollout": {"train_steps": int(args.rollout_steps), "control_mode": args.control_mode,
                    "stride": int(args.rollout_stride), "tbptt_steps": int(args.tbptt_steps),
                    "bias_weight": float(args.bias_weight)},
        "val_rollout": val_rollout,
        "test_rollout": test_rollout,
        "val_one_step": evaluate_one_step(model, frames, val_runs, scalers, args, device),
        "test_one_step": test_one_step,
        "val_rollout_at_init": baseline,
        "history": history,
        "skipped_runs": skipped,
    }
    report_path = output_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Final ===")
    print(f"selection metric : val_rollout_mae_temp = {best_metric:.4f} C over {eval_horizon} steps")
    if test_rollout:
        print(f"test rollout     : mae={test_rollout['mae_temp']:.4f} bias={test_rollout['bias_temp']:+.4f} "
              f"final_step_mae={test_rollout['mae_final_step']:.4f}")
    print(f"test 1-step      : mae={test_one_step['mae_delta_t1']:.5f} (reported for comparison only)")
    print(f"checkpoint       : {checkpoint_path}")
    print(f"report           : {report_path}")


if __name__ == "__main__":
    main()
