#!/usr/bin/env python3
"""Train the GRU plant model on multi-step rollouts instead of single steps.

The loss is computed over an N-step unrolled rollout with gradients flowing through
the unroll, feeding the temperature back from the model's own prediction and
simulating the PID/diff controller in-graph as deepvac.mpc.step_state does at
inference. Early stopping and checkpointing select on free-running rollout MAE over
held-out runs. --split-by pid-config assigns whole clusters of near-duplicate PID
configurations to one split.

The checkpoint layout matches gru/train_gru.py, so the result is a drop-in
replacement in mpc_gru.py, mpc_batch.py, and simulate_gru.py.

--watch turns this into a long-running companion to whatever is collecting chamber
runs (tocero_3band.py, band_bo_gp.py, ...): it polls --history-root, and once enough
runs are new it retrains -- warm-started from the last checkpoint it promoted -- and
promotes the result only if it beats that checkpoint on a validation split that is
locked the first time it is computed and reused on every later pass, so "improved"
means the same thing across the whole run. Checkpoint writes are atomic, so whatever
is reading --promoted-checkpoint (twin_acceptance.py, MPC, a control app) never sees
a half-written file.

Examples:

    # One-shot, fine-tuning an existing 1-step checkpoint with a 40-step rollout loss.
    python -m gru.train_gru_rollout \
        --init-from gru/validation_t1/gru_t1.pt \
        --rollout-steps 40 --epochs 40 --split-by pid-config

    # Left running alongside an optimizer script collecting new chamber runs.
    python -m gru.train_gru_rollout \
        --init-from gru/validation_t1/gru_t1.pt --rollout-steps 40 \
        --watch --watch-min-new-runs 5 --watch-interval-s 300
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Bound before gru_common can install its sklearn stub, which torch's tracer rejects.
import sklearn.preprocessing  # noqa: F401,E402
import torch
import torch.nn as nn

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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train a GRU plant model on multi-step rollouts, selected on rollout error.",
    )

    ap.add_argument("--history-root", nargs="+", default=[str(DEFAULT_HISTORY_ROOT)],
                    help="One or more run-history roots; runs from all of them are pooled "
                         "together. find_run_csvs() is one level deep, not recursive, so this "
                         "is how to combine e.g. run_history/ and heatup_run_history/ -- "
                         "pointing at their shared parent directory finds nothing.")
    ap.add_argument("--telemetry-names", nargs="+", default=["run_samples.csv"])
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--window-steps", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--min-duration-s", type=float, default=300.0)
    ap.add_argument("--exclude-prefixes", nargs="*", default=[])
    ap.add_argument("--max-runs", type=int, default=None,
                    help=(
                        "Cap runs loaded, for quick checks. With --watch, folders beyond the cap "
                        "still count as 'seen' once discovered, so they are never trained on "
                        "later either -- leave this unset for real watch runs."
                    ))

    # --- Splitting -------------------------------------------------------------
    ap.add_argument(
        "--split-by",
        choices=["pid-config", "run"],
        default="pid-config",
        help="pid-config splits by PID configuration group. run splits by run id.",
    )
    ap.add_argument(
        "--config-key",
        choices=["band-table", "triplet-set", "first-triplet"],
        default="band-table",
        help=(
            "What counts as one PID configuration. band-table uses the configured "
            "far/mid/near gain table; the others use the triplets seen in the samples."
        ),
    )
    ap.add_argument(
        "--config-cluster-eps",
        type=float,
        default=0.12,
        help=(
            "Single-linkage radius merging near-duplicate band tables, in a log-scaled "
            "normalized gain space of diameter 3.0. 0 merges exact matches only."
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
        help="Extra penalty on the mean absolute per-step rollout bias.",
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

    # --- Split locking -----------------------------------------------------------
    ap.add_argument(
        "--lock-split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compute the val/test split once and reuse it on every later run so "
            "'improved' is comparable across them. New runs since the lock become "
            "training data; new runs that now cluster with a locked val/test run are "
            "excluded rather than risking leakage. Default: on for --watch, off otherwise."
        ),
    )
    ap.add_argument("--split-lock-path", default=None,
                    help="Where the locked split is stored. Default: --output-dir/split_lock.json.")

    # --- Watch mode ----------------------------------------------------------------
    ap.add_argument(
        "--watch",
        action="store_true",
        help="Poll --history-root and retrain when enough new runs appear, instead of running once.",
    )
    ap.add_argument("--watch-interval-s", type=float, default=300.0,
                    help="Seconds between polls for new runs.")
    ap.add_argument("--watch-min-new-runs", type=int, default=3,
                    help="Retrain once at least this many not-yet-seen run folders appear.")
    ap.add_argument("--watch-max-passes", type=int, default=0,
                    help="Stop after this many training passes. 0 runs until interrupted.")
    ap.add_argument("--promoted-checkpoint", default=None,
                    help="Stable path consumers should load. Default: --output-dir/promoted.pt.")
    ap.add_argument("--promote-min-improvement", type=float, default=0.0,
                    help="Minimum val_rollout_mae drop (degC) required to promote a pass's checkpoint.")
    ap.add_argument("--watch-state-path", default=None,
                    help="Default: --output-dir/watch_state.json.")
    ap.add_argument("--watch-history-csv", default=None,
                    help="Default: --output-dir/watch_history.csv.")

    return ap


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    """Save a checkpoint so a concurrent reader never observes a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}")
    tmp.write_bytes(Path(src).read_bytes())
    os.replace(tmp, dst)


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
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable equivalent of deepvac.mpc.run_pid_substeps (temp_mode="hold")."""
    u = u_p = u_d = torch.zeros_like(temp)

    for _ in range(n_substeps):
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

        u = torch.clamp(u_p + i_part + u_d, u_min, u_max)
        u_p = torch.clamp(u_p, u_min, u_max)

    terms = {
        "temp_u": u * feature_scale,
        "temp_u_p": u_p * feature_scale,
        "temp_u_i": i_part * feature_scale,
        "temp_u_d": u_d * feature_scale,
    }
    return terms, i_part, diff_prev, diff_filter


def invert_diff_filter(
    u_d_norm: np.ndarray, kp: np.ndarray, kd: np.ndarray
) -> np.ndarray:
    """Recover CodesysDiff's filter state from a logged D term.

    Inverts u_d = (1/kp) * (kd * -DIFF_GAIN * clamp(diff_filter)). Approximate where
    u_d sits on its +-D_PART_CLIP rail.
    """
    kp = np.asarray(kp, dtype=np.float64)
    kd = np.asarray(kd, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_filter = -np.asarray(u_d_norm, dtype=np.float64) * kp / (DIFF_GAIN * kd)
    diff_filter = np.where(np.isfinite(diff_filter), diff_filter, 0.0)
    return np.clip(diff_filter, -DIFF_CLIP, DIFF_CLIP).astype(np.float32)


def assemble_feature_row(
    feature_index: dict[str, int],
    *,
    temp: torch.Tensor,
    temp_ref: torch.Tensor,
    kp: torch.Tensor,
    ki: torch.Tensor,
    kd: torch.Tensor,
    terms: dict[str, torch.Tensor],
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
    batch: dict[str, torch.Tensor],
    scalers: Scalers,
    args: argparse.Namespace,
    feature_index: dict[str, int],
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
    logged_terms = batch.get("logged_terms")  # (B, H, 4)

    substeps = batch["n_substeps"]
    n_substeps = int(substeps[0].item())
    if bool(torch.any(substeps != substeps[0])):
        raise ValueError(
            "A batch mixed rollouts with different PID substep counts "
            f"({sorted(set(int(v) for v in substeps.tolist()))}); "
            "SubstepBatchSampler should have kept them apart."
        )
    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)
    tbptt = int(args.tbptt_steps)

    preds: list[torch.Tensor] = []

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


BAND_COLUMNS = [
    f"{band}_{coef}"
    for band in ("far", "mid", "near")
    for coef in ("kp", "ki", "kd")
]


def read_band_table(run_dir: Path) -> list[float] | None:
    """The far/mid/near gain table a run was configured with, read from
    run_summary.csv or band_metrics.json. None if the run used no static table."""
    summary_path = run_dir / "run_summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path, nrows=1)
        except Exception:
            summary = pd.DataFrame()
        if not summary.empty and all(col in summary.columns for col in BAND_COLUMNS):
            values = pd.to_numeric(summary.iloc[0][BAND_COLUMNS], errors="coerce")
            if values.notna().all():
                return [float(v) for v in values]

    metrics_path = run_dir / "band_metrics.json"
    if metrics_path.exists():
        try:
            records = json.loads(metrics_path.read_text(encoding="utf-8")).get("records", [])
        except (json.JSONDecodeError, OSError):
            records = []
        by_band = {str(rec.get("band")): rec for rec in records}
        if {"far", "mid", "near"} <= set(by_band):
            try:
                return [
                    float(by_band[band][coef])
                    for band in ("far", "mid", "near")
                    for coef in ("kp", "ki", "kd")
                ]
            except (KeyError, TypeError, ValueError):
                pass

    return None


def cluster_config_vectors(vectors: list[list[float]], eps: float) -> list[int]:
    """Single-linkage cluster label per band table, within radius eps.

    Gains are compared in log10 space and range-normalized per dimension, giving a
    space of diameter sqrt(9) = 3.0.
    """
    arr = np.log10(np.maximum(np.asarray(vectors, dtype=np.float64), 1.0))
    low = arr.min(axis=0)
    arr = (arr - low) / np.maximum(arr.max(axis=0) - low, 1e-9)

    n = len(arr)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        distance = np.sqrt(np.sum((arr - arr[i]) ** 2, axis=1))
        for j in np.nonzero(distance <= max(float(eps), 0.0))[0]:
            root_i, root_j = find(i), find(int(j))
            if root_i != root_j:
                parent[root_i] = root_j

    return [find(i) for i in range(n)]


def build_config_groups(
    frames: dict[str, pd.DataFrame],
    run_dirs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, object]]:
    """Map every run to the identity of the PID configuration it used."""
    key = getattr(args, "config_key", "band-table")
    if key != "band-table":
        groups = {run_id: pid_config_signature(df, key) for run_id, df in frames.items()}
        return groups, {"config_key": key}

    tables = {run_id: read_band_table(run_dirs[run_id]) for run_id in frames}
    with_table = [run_id for run_id in sorted(frames) if tables[run_id] is not None]
    without_table = [run_id for run_id in sorted(frames) if tables[run_id] is None]

    groups: dict[str, str] = {}
    if with_table:
        labels = cluster_config_vectors(
            [tables[run_id] for run_id in with_table], float(args.config_cluster_eps)
        )
        for run_id, label in zip(with_table, labels, strict=True):
            groups[run_id] = f"band-cluster-{int(label)}"
    for run_id in without_table:
        groups[run_id] = f"no-band-table-{run_id}"

    return groups, {
        "config_key": key,
        "cluster_eps": float(args.config_cluster_eps),
        "n_with_band_table": len(with_table),
        "n_without_band_table": len(without_table),
    }


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
    run_to_group: dict[str, str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Assign whole groups to splits, greedily hitting the target run fractions."""
    groups: dict[str, list[str]] = {}
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
    keys.sort(key=lambda k: -len(groups[k]))

    total = sum(len(groups[k]) for k in keys)
    targets = {
        "train": train_fraction * total,
        "val": val_fraction * total,
        "test": max(0.0, 1.0 - train_fraction - val_fraction) * total,
    }
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for key in keys:
        name = max(targets, key=lambda n: (targets[n] - counts[n]) / max(targets[n], 1e-9))
        buckets[name].extend(groups[key])
        counts[name] += len(groups[key])

    if not all(buckets.values()):
        raise RuntimeError(f"A split came out empty: {({k: len(v) for k, v in buckets.items()})}")

    return sorted(buckets["train"]), sorted(buckets["val"]), sorted(buckets["test"])


def write_split_lock(path: Path, val_runs: Sequence[str], test_runs: Sequence[str], n_runs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "val_runs": list(val_runs),
        "test_runs": list(test_runs),
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "n_runs_when_locked": int(n_runs),
    }, indent=2), encoding="utf-8")


def apply_split_lock(
    frames: dict[str, pd.DataFrame],
    groups: dict[str, str],
    lock_path: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Reuse a previously locked val/test split so improvement is comparable across
    repeated runs, folding in any run collected since the lock as training data.

    A newly collected run is excluded from training (not added to val/test either)
    if its current PID-config cluster overlaps a locked val/test run: the group
    clustering used here is a single-linkage over the whole dataset, so it can
    reassign as new points arrive, and a run must not silently join training data
    just because clustering happened to move it away from a held-out group.
    """
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_val = [r for r in lock["val_runs"] if r in frames]
    locked_test = [r for r in lock["test_runs"] if r in frames]
    missing = [r for r in (*lock["val_runs"], *lock["test_runs"]) if r not in frames]
    if missing:
        print(f"[split] WARNING {len(missing)} locked val/test run(s) are no longer under "
              f"--history-root: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    locked_ids = set(locked_val) | set(locked_test)
    locked_groups = {groups[r] for r in locked_ids}

    train_runs = sorted(r for r in frames if r not in locked_ids and groups[r] not in locked_groups)
    excluded = sorted(r for r in frames if r not in locked_ids and groups[r] in locked_groups)

    return train_runs, sorted(locked_val), sorted(locked_test), excluded


def build_rollout_samples(
    df: pd.DataFrame,
    run_id: str,
    args: argparse.Namespace,
    horizon: int,
    stride: int,
) -> list[dict[str, np.ndarray]]:
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
    n_substeps = max(1, int(round(dt_s / max(float(args.pid_period_s), 1e-6))))

    # Controller state left behind by each row, which the next row's step starts from.
    i_part_state = term_cols[:, 2] / feature_scale
    diff_filter_state = invert_diff_filter(
        term_cols[:, 3] / feature_scale, exog_cols[:, 1], exog_cols[:, 3]
    )

    samples: list[dict[str, np.ndarray]] = []

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

        prev_idx = max(end_idx - 1, 0)
        samples.append({
            "window": window.astype(np.float32),
            "temp0": np.float32(temp[end_idx]),
            "i_part0": np.float32(i_part_state[prev_idx]),
            "diff_prev0": np.float32(temp[prev_idx]),
            "diff_filter0": np.float32(diff_filter_state[prev_idx]),
            "exog": exog.astype(np.float32),
            "logged_terms": terms.astype(np.float32),
            "target": target.astype(np.float32),
            "n_substeps": np.int64(n_substeps),
            "run_id": run_id,
        })

    return samples


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict[str, np.ndarray]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
            "n_substeps": torch.tensor(s["n_substeps"]),
        }


class SubstepBatchSampler(torch.utils.data.Sampler):
    """Yields batches whose rollouts all share one PID substep count."""

    def __init__(
        self,
        samples: list[dict[str, np.ndarray]],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.buckets: dict[int, list[int]] = {}
        for idx, sample in enumerate(samples):
            self.buckets.setdefault(int(sample["n_substeps"]), []).append(idx)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return sum(
            (len(idxs) + self.batch_size - 1) // self.batch_size
            for idxs in self.buckets.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for idxs in self.buckets.values():
            order = list(idxs)
            if self.shuffle:
                self.rng.shuffle(order)
            batches.extend(
                order[i : i + self.batch_size]
                for i in range(0, len(order), self.batch_size)
            )
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)


def make_rollout_loader(
    samples: list[dict[str, np.ndarray]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    """DataLoader that never mixes substep counts within a batch."""
    return torch.utils.data.DataLoader(
        RolloutDataset(samples),
        batch_sampler=SubstepBatchSampler(samples, batch_size, shuffle=shuffle, seed=seed),
    )


def find_run_csvs_multi(history_roots: Sequence[str], telemetry_names: Sequence[str]) -> list[Path]:
    """find_run_csvs over several roots, pooled. Each root is still scanned one
    level deep, not recursively -- a run's CSV must sit directly under one of
    these roots, e.g. run_history/<run_id>/run_samples.csv."""
    csvs: list[Path] = []
    for root in history_roots:
        csvs.extend(find_run_csvs(Path(root), telemetry_names))
    return csvs


def load_runs(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, object], list[dict[str, object]]]:
    history_roots = [args.history_root] if isinstance(args.history_root, str) else list(args.history_root)
    csvs = find_run_csvs_multi(history_roots, args.telemetry_names)
    if args.exclude_prefixes:
        csvs = [p for p in csvs if not any(p.parent.name.startswith(x) for x in args.exclude_prefixes)]
    if args.max_runs is not None:
        csvs = csvs[: int(args.max_runs)]

    frames: dict[str, pd.DataFrame] = {}
    run_dirs: dict[str, Path] = {}
    skipped: list[dict[str, object]] = []

    for csv_path in csvs:
        try:
            df, meta = prepare_run_dataframe(csv_path, args)
        except Exception as exc:
            skipped.append({"run": csv_path.parent.name, "reason": str(exc)})
            continue
        run_id = str(meta["run_id"])
        if run_id in frames:
            skipped.append({"run": csv_path.parent.name,
                            "reason": f"duplicate run_id {run_id!r} already loaded from another root"})
            continue
        frames[run_id] = df
        run_dirs[run_id] = csv_path.parent

    if len(frames) < 3:
        raise RuntimeError(f"Only {len(frames)} usable runs under {history_roots}")

    groups, group_info = build_config_groups(frames, run_dirs, args)

    return frames, groups, group_info, skipped


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def evaluate_rollout(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    scalers: Scalers,
    args: argparse.Namespace,
    feature_index: dict[str, int],
    horizon: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    errors: list[np.ndarray] = []

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
        "per_step_mae": [float(v) for v in per_step_mae],
        "per_step_bias": [float(v) for v in per_step_bias],
    }


def evaluate_one_step(
    model: nn.Module,
    frames: dict[str, pd.DataFrame],
    run_ids: Sequence[str],
    scalers: Scalers,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
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
    preds: list[np.ndarray] = []
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
    frames: dict[str, pd.DataFrame],
    train_runs: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    init_checkpoint: dict[str, object] | None,
) -> tuple[Scalers, object, object]:
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


def train_and_validate(args: argparse.Namespace) -> dict[str, object]:
    """Run one full data-load/split/train/evaluate pass and return its key results."""
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

    print(f"[data] loading runs from {', '.join(args.history_root)}")
    frames, groups, group_info, skipped = load_runs(args)
    group_sizes = Counter(groups.values())
    print(f"[data] {len(frames)} usable runs, {len(skipped)} skipped, "
          f"{len(group_sizes)} distinct PID configurations (--config-key {args.config_key})")
    if args.config_key == "band-table":
        largest = max(group_sizes.values())
        print(f"[data] band tables read for {group_info['n_with_band_table']} runs, "
              f"{group_info['n_without_band_table']} without (kept as their own group); "
              f"clustered at eps={args.config_cluster_eps:g} into {len(group_sizes)} groups, "
              f"{sum(1 for n in group_sizes.values() if n == 1)} singletons, "
              f"largest {largest} runs ({100.0 * largest / len(frames):.0f}%)")
        if largest > 0.5 * len(frames):
            print("[data] WARNING one cluster holds over half the runs; --config-cluster-eps "
                  "is high enough that single-linkage is chaining distinct configurations together.")

    lock_split = bool(args.watch) if args.lock_split is None else bool(args.lock_split)
    split_lock_path = Path(args.split_lock_path or (output_dir / "split_lock.json"))
    excluded_by_lock: list[str] = []

    if args.split_by == "pid-config":
        if lock_split and split_lock_path.exists():
            train_runs, val_runs, test_runs, excluded_by_lock = apply_split_lock(
                frames, groups, split_lock_path
            )
        else:
            train_runs, val_runs, test_runs = split_by_group(
                groups, args.train_fraction, args.val_fraction, args.seed
            )
            if lock_split:
                write_split_lock(split_lock_path, val_runs, test_runs, n_runs=len(frames))
                print(f"[split] locked val/test split -> {split_lock_path}")
    else:
        from deepvac.datasets import split_runs as split_by_run
        train_runs, val_runs, test_runs = split_by_run(
            sorted(frames), args.train_fraction, args.val_fraction, args.seed
        )
        if lock_split:
            print("[split] WARNING --lock-split has no effect with --split-by run")

    if excluded_by_lock:
        print(f"[split] {len(excluded_by_lock)} run(s) excluded this pass: their current PID-config "
              f"cluster overlaps a locked val/test run -> {excluded_by_lock[:5]}"
              f"{' ...' if len(excluded_by_lock) > 5 else ''}")

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

    substep_counts = Counter(int(s["n_substeps"]) for s in train_samples)
    print(f"[data] PID substeps per model step: {dict(sorted(substep_counts.items()))}")

    train_loader = make_rollout_loader(train_samples, args.batch_size, True, args.seed)
    val_loader = make_rollout_loader(val_samples, args.batch_size, False, args.seed)
    test_loader = make_rollout_loader(test_samples, args.batch_size, False, args.seed)

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
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if args.curriculum_epochs > 0:
            frac = min(1.0, (epoch - 1) / max(1, args.curriculum_epochs))
            horizon = int(round(args.curriculum_start_steps
                                + frac * (args.rollout_steps - args.curriculum_start_steps)))
            horizon = max(1, min(int(args.rollout_steps), horizon))
        else:
            horizon = int(args.rollout_steps)

        model.train()
        losses: list[float] = []
        t0 = time.perf_counter()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = unroll(model, batch, scalers, args, feature_index, horizon)
            target = batch["target"][:, :horizon]

            loss = loss_fn(pred, target)
            if args.bias_weight > 0:
                loss = loss + float(args.bias_weight) * (pred - target).mean(dim=0).abs().mean()

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
    atomic_torch_save(
        {
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
        "history_root": [str(Path(r).resolve()) for r in args.history_root],
        "selection_metric": "val_rollout_mae_temp",
        "eval_horizon_steps": eval_horizon,
        "best_epoch": best_epoch,
        "split": {
            "by": args.split_by,
            **group_info,
            "n_train_runs": len(train_runs), "n_val_runs": len(val_runs), "n_test_runs": len(test_runs),
            "n_distinct_configs": len(group_sizes),
            "n_singleton_configs": sum(1 for n in group_sizes.values() if n == 1),
            "largest_config_group": max(group_sizes.values()),
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

    return {
        "checkpoint_path": checkpoint_path,
        "report_path": report_path,
        "val_rollout_mae": float(best_metric),
        "best_epoch": int(best_epoch),
        "n_runs": len(frames),
        "train_runs": train_runs, "val_runs": val_runs, "test_runs": test_runs,
        "excluded_by_lock": excluded_by_lock,
    }


# -----------------------------------------------------------------------------
# Watch mode
# -----------------------------------------------------------------------------


def discover_run_ids(history_roots: Sequence[str]) -> set:
    ids: set = set()
    for root in history_roots:
        root_path = Path(root)
        if root_path.exists():
            ids |= {p.name for p in root_path.iterdir() if p.is_dir()}
    return ids


def load_watch_state(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"seen_run_ids": [], "promoted_metric": None, "promoted_at": None, "n_passes": 0}


def save_watch_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_watch_history_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=write_header, index=False)


def run_watch_loop(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    promoted_path = Path(args.promoted_checkpoint or (output_dir / "promoted.pt"))
    state_path = Path(args.watch_state_path or (output_dir / "watch_state.json"))
    history_csv = Path(args.watch_history_csv or (output_dir / "watch_history.csv"))
    split_lock_path = Path(args.split_lock_path or (output_dir / "split_lock.json"))
    user_init_from = args.init_from

    state = load_watch_state(state_path)
    seen_run_ids = set(state.get("seen_run_ids") or [])

    print(f"[watch] history root(s) : {', '.join(args.history_root)}")
    print(f"[watch] promoted ckpt   : {promoted_path}")
    print(f"[watch] split lock      : {split_lock_path}")
    print(f"[watch] poll interval   : {args.watch_interval_s:g}s, min new runs: {args.watch_min_new_runs}")
    print(f"[watch] already seen    : {len(seen_run_ids)} run folders, "
          f"{state.get('n_passes', 0)} pass(es) so far, "
          f"promoted metric so far: {state.get('promoted_metric')}")

    pass_idx = int(state.get("n_passes", 0))
    try:
        while True:
            current_ids = discover_run_ids(args.history_root)
            new_ids = current_ids - seen_run_ids
            if len(new_ids) < args.watch_min_new_runs:
                print(f"[watch] {len(new_ids)} new run folder(s), waiting for "
                      f"{args.watch_min_new_runs}; next check in {args.watch_interval_s:g}s")
                time.sleep(args.watch_interval_s)
                continue

            pass_idx += 1
            pass_dir = output_dir / f"pass_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
            print(f"\n[watch] === pass {pass_idx}: {len(new_ids)} new run folder(s) -> {pass_dir} ===")

            pass_args = copy.deepcopy(args)
            pass_args.output_dir = str(pass_dir)
            pass_args.lock_split = True
            pass_args.split_lock_path = str(split_lock_path)
            if not user_init_from and promoted_path.exists():
                pass_args.init_from = str(promoted_path)
                print(f"[watch] warm-starting from the currently promoted checkpoint: {promoted_path}")

            t0 = time.perf_counter()
            try:
                result = train_and_validate(pass_args)
            except Exception as exc:
                print(f"[watch] pass {pass_idx} FAILED: {exc}")
                seen_run_ids |= new_ids
                state["seen_run_ids"] = sorted(seen_run_ids)
                save_watch_state(state_path, state)
                time.sleep(args.watch_interval_s)
                continue
            seconds = time.perf_counter() - t0

            seen_run_ids |= new_ids
            metric = float(result["val_rollout_mae"])
            prior_metric = state.get("promoted_metric")
            improved = prior_metric is None or metric <= float(prior_metric) - float(args.promote_min_improvement)

            if improved:
                atomic_copy(result["checkpoint_path"], promoted_path)
                state["promoted_metric"] = metric
                state["promoted_at"] = datetime.now(timezone.utc).isoformat()
                state["promoted_from_pass"] = str(pass_dir)
                print(f"[watch] promoted: val_rollout_mae {prior_metric} -> {metric:.4f} -> {promoted_path}")
            else:
                print(f"[watch] not promoted: val_rollout_mae {metric:.4f} did not beat "
                      f"{float(prior_metric):.4f} by --promote-min-improvement={args.promote_min_improvement:g}")

            state["seen_run_ids"] = sorted(seen_run_ids)
            state["n_passes"] = pass_idx
            save_watch_state(state_path, state)

            append_watch_history_row(history_csv, {
                "pass": pass_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_new_runs": len(new_ids),
                "n_total_runs_seen": len(seen_run_ids),
                "val_rollout_mae": metric,
                "promoted": improved,
                "best_epoch": result["best_epoch"],
                "n_excluded_by_lock": len(result["excluded_by_lock"]),
                "seconds": round(seconds, 1),
                "pass_dir": str(pass_dir),
            })

            if args.watch_max_passes and pass_idx >= args.watch_max_passes:
                print(f"[watch] reached --watch-max-passes={args.watch_max_passes}, stopping")
                return

            time.sleep(args.watch_interval_s)
    except KeyboardInterrupt:
        print(f"\n[watch] interrupted after {pass_idx} pass(es); "
              f"promoted checkpoint (if any) is unaffected: {promoted_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.watch:
        run_watch_loop(args)
    else:
        train_and_validate(args)


if __name__ == "__main__":
    main()
