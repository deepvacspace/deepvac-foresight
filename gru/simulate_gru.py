#!/usr/bin/env python3
"""Closed-loop reconstruction using a trained one-step GRU plant model.
"""

from __future__ import annotations
import torch
import pandas as pd
import numpy as np

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gru.gru_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FEATURE_NAMES,
    GRUModel,
    ChamberPID,
    CodesysDiff,
    PidCoefSelector,
    limit,
    load_model,
    predict_delta_t1,
)

WORK_DIR = ROOT / "optimization"
DEFAULT_HISTORY_ROOT = WORK_DIR / "run_history"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "simulation_t1"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Use trained one-step GRU plus ChamberPID to reconstruct one historical run."
    )

    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="Path to trained gru_t1.pt checkpoint.")
    ap.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    ap.add_argument(
        "--telemetry-names",
        nargs="+",
        default=["run_samples.csv"],
    )
    ap.add_argument("--run-id", default=None,
                    help="Specific run folder name. If omitted, choose random run.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--window-steps", type=int, default=60,
                    help="Fallback only if checkpoint lacks window_steps.")
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--u-min", type=float, default=-1.0)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument(
        "--control-feature-scale",
        type=float,
        default=100.0,
        help=(
            "Scale normalized PID output before feeding temp_u/temp_u_* "
            "features to the GRU. Historical logs store these features as "
            "percent values, e.g. -100..100."
        ),
    )
    ap.add_argument("--pid-i-reverse-mul", type=float, default=0.333)
    ap.add_argument(
        "--pid-period-s",
        type=float,
        default=0.1,
        help="Internal CODESYS PID/Diff period. Your PLC PERIOD_CONSTANT is 0.1 s.",
    )
    ap.add_argument(
        "--control-mode",
        choices=["pid_substep", "replay"],
        default="pid_substep",
        help=(
            "pid_substep reconstructs PID/Diff at --pid-period-s before each GRU step. "
            "replay uses logged temp_u/temp_u_p/temp_u_i/temp_u_d as an upper-bound rollout."
        ),
    )
    ap.add_argument(
        "--pid-substep-temp-mode",
        choices=["hold", "real_interp_debug"],
        default="hold",
        help=(
            "Temperature used inside PID substeps. "
            "hold uses current predicted temp and is the correct closed-loop mode. "
            "real_interp_debug interpolates real logged temp and is only for debugging."
        ),
    )
    ap.add_argument(
        "--diff-init-mode",
        choices=["first_temp", "infer_from_d", "warmup_real"],
        default="first_temp",
        help=(
            "How to initialize CodesysDiff at the warm-up boundary. "
            "first_temp matches the corrected validation script behavior."
        ),
    )
    ap.add_argument(
        "--diff-mode",
        choices=["codesys", "dtemp", "velocity"],
        default="codesys",
        help=(
            "Approximation for m_diff^.out. "
            "'codesys' follows gru/codesys/diff.txt with a first-order "
            "ComplFilter approximation. "
            "'dtemp' uses temp[t]-temp[t-1]. "
            "'velocity' uses (temp[t]-temp[t-1]) / dt."
        ),
    )
    ap.add_argument(
        "--init-pid-from-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize PID internal p/i/d states from the last real warm-up row.",
    )
    ap.add_argument(
        "--init-diff-from-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize CODESYS Diff state from logged temp_u_d at the warm-up row.",
    )
    ap.add_argument(
        "--pid-selector-min-range",
        type=float,
        default=None,
        help="Minimum range passed to PidCoefSelector.FB_Init. Requires --pid-selector-points.",
    )
    ap.add_argument(
        "--pid-selector-max-range",
        type=float,
        default=None,
        help="Maximum range passed to PidCoefSelector.FB_Init. Requires --pid-selector-points.",
    )
    ap.add_argument(
        "--pid-selector-points",
        default=None,
        help=(
            "Optional PidCoefSelector table as 'kp,ki,kd;kp,ki,kd;...'. "
            "When set, coefficients are selected from predicted temp instead "
            "of replayed from the historical row."
        ),
    )

    return ap


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


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


def choose_run_csv(
    csv_paths: Sequence[Path],
    run_id: Optional[str],
    seed: int,
) -> Path:
    if not csv_paths:
        raise RuntimeError("No run CSV paths available.")

    if run_id:
        matches = [p for p in csv_paths if p.parent.name == run_id]
        if not matches:
            available = ", ".join(p.parent.name for p in csv_paths[:20])
            raise RuntimeError(
                f"Requested run_id not found: {run_id}. "
                f"First available run ids: {available}"
            )
        return matches[0]

    rng = random.Random(seed)
    return rng.choice(list(csv_paths))


def prepare_run_dataframe(
    csv_path: Path,
    feature_names: Sequence[str],
    min_samples: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

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
    df["abs_error"] = df["error"].abs()

    df["dt_s"] = df["elapsed_s"].diff().fillna(0.0)
    df["dt_s"] = safe_numeric(df["dt_s"]).clip(lower=0.0, upper=60.0)

    dt_for_velocity = df["dt_s"].replace(0.0, np.nan)
    df["temp_velocity"] = df["temp"].diff() / dt_for_velocity
    df["temp_velocity"] = (
        df["temp_velocity"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Make missing checkpoint features explicit and safe.
    for name in feature_names:
        if name not in df.columns:
            df[name] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=list(feature_names) +
                   ["temp", "temp_ref", "elapsed_s", "kp", "ki", "kd"])
    df = df.reset_index(drop=True)

    if len(df) < min_samples:
        raise ValueError(
            f"too few samples after cleanup: {len(df)} < {min_samples}")

    return df


def parse_pid_selector(args: argparse.Namespace) -> Optional[PidCoefSelector]:
    if args.pid_selector_points is None:
        return None

    if args.pid_selector_min_range is None or args.pid_selector_max_range is None:
        raise ValueError(
            "--pid-selector-min-range and --pid-selector-max-range are required "
            "when --pid-selector-points is set."
        )

    points: List[Tuple[float, float, float]] = []
    for item in str(args.pid_selector_points).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "Each --pid-selector-points entry must be kp,ki,kd; "
                f"got: {item!r}"
            )
        points.append((float(parts[0]), float(parts[1]), float(parts[2])))

    return PidCoefSelector(
        points=points,
        min_range=float(args.pid_selector_min_range),
        max_range=float(args.pid_selector_max_range),
    )


def compute_diff_out(
    current_temp: float,
    previous_temp: float,
    dt_s: float,
    mode: str,
) -> float:
    dt = max(float(dt_s), 1e-6)
    if mode == "dtemp":
        return float(current_temp) - float(previous_temp)
    if mode == "velocity":
        return (float(current_temp) - float(previous_temp)) / dt
    raise ValueError(f"Unknown diff mode: {mode}")


def make_feature_row(
    feature_names: Sequence[str],
    *,
    temp: float,
    temp_ref: float,
    elapsed_s: float,
    previous_elapsed_s: float,
    previous_temp: float,
    u: float,
    u_p: float,
    u_i: float,
    u_d: float,
    kp: float,
    ki: float,
    kd: float,
) -> Dict[str, float]:
    error = float(temp_ref) - float(temp)
    dt_s = max(float(elapsed_s) - float(previous_elapsed_s), 0.0)
    temp_velocity = 0.0 if dt_s <= 1e-9 else (
        float(temp) - float(previous_temp)) / dt_s

    values = {
        "temp": float(temp),
        "temp_ref": float(temp_ref),
        "error": float(error),
        "abs_error": abs(float(error)),
        "dt_s": float(dt_s),
        "temp_velocity": float(temp_velocity),
        "temp_u": float(u),
        "temp_u_p": float(u_p),
        "temp_u_i": float(u_i),
        "temp_u_d": float(u_d),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
    }

    return {name: float(values.get(name, 0.0)) for name in feature_names}


def interpolate(a: float, b: float, alpha: float) -> float:
    return float(a) + float(alpha) * (float(b) - float(a))


def initialize_codesys_diff_for_sim(
    df: pd.DataFrame,
    warmup_end_idx: int,
    feature_scale: float,
    args: argparse.Namespace,
) -> CodesysDiff:
    """Initialize Diff at the simulation boundary.

    The corrected validation showed the important part is not one update per CSV row,
    but 0.1 s internal updates. For closed-loop simulation, the safest default is
    first_temp: prev_value=current predicted temperature and zero filter state.
    """
    diff = CodesysDiff()
    current_temp = float(df["temp"].iloc[warmup_end_idx])

    mode = str(getattr(args, "diff_init_mode", "first_temp"))

    if mode == "first_temp":
        diff.prev_value = current_temp
        diff.filter_out = 0.0
        diff.out = 0.0
        return diff

    if mode == "warmup_real":
        for temp_value in df["temp"].iloc[: warmup_end_idx + 1].to_numpy(dtype=float):
            diff.update(float(temp_value))
        return diff

    if mode == "infer_from_d":
        # Old behavior: infer diff.out from the logged D term at the warm-up row.
        # Useful for comparison, but it can inject a stale derivative state.
        warmup_kp = float(df["kp"].iloc[warmup_end_idx])
        warmup_kd = float(df["kd"].iloc[warmup_end_idx])
        diff.prev_value = current_temp
        if warmup_kd != 0.0:
            d_part = float(df["temp_u_d"].iloc[warmup_end_idx]) / feature_scale
            inferred_diff_out = -(d_part * warmup_kp) / warmup_kd
            diff.out = float(inferred_diff_out)
            diff.filter_out = float(inferred_diff_out) / 10.0
        return diff

    raise ValueError(f"Unknown diff_init_mode: {mode}")


def run_controller_substeps(
    *,
    df: pd.DataFrame,
    end_idx: int,
    current_temp_pred: float,
    previous_temp_pred: float,
    elapsed: float,
    previous_elapsed: float,
    next_elapsed: float,
    pid: ChamberPID,
    codesys_diff: Optional[CodesysDiff],
    kp: float,
    ki: float,
    kd: float,
    next_kp: float,
    next_ki: float,
    next_kd: float,
    temp_ref: float,
    next_temp_ref: float,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float, float, int]:
    """Run the CODESYS-style PID/Diff at internal 0.1 s substeps.

    CODESYS order from Chamber_Control:
        ReadInput() -> diffTemp(in := temp_measured) -> pidTemp(...)

    In closed-loop mode, the future true temperature is unknown. The default
    pid_substep_temp_mode='hold' keeps x_measured at current predicted temp during
    the internal controller updates. real_interp_debug is only an offline diagnostic.
    """
    period_s = max(float(getattr(args, "pid_period_s", 0.1)), 1e-6)
    dt_to_next = max(float(next_elapsed) - float(elapsed), period_s)
    n_substeps = max(1, int(round(dt_to_next / period_s)))

    u = u_p = u_i = u_d = 0.0
    diff_out = 0.0

    last_temp_for_diff = float(previous_temp_pred)
    last_elapsed_for_diff = float(previous_elapsed)

    mode = str(getattr(args, "pid_substep_temp_mode", "hold"))

    for k in range(1, n_substeps + 1):
        alpha = k / n_substeps

        if mode == "hold":
            temp_sub = float(current_temp_pred)
        elif mode == "real_interp_debug":
            temp0 = float(df["temp"].iloc[end_idx])
            temp1 = float(df["temp"].iloc[end_idx + 1])
            temp_sub = interpolate(temp0, temp1, alpha)
        else:
            raise ValueError(f"Unknown pid_substep_temp_mode: {mode}")

        ref_sub = interpolate(temp_ref, next_temp_ref, alpha)
        kp_sub = interpolate(kp, next_kp, alpha)
        ki_sub = interpolate(ki, next_ki, alpha)
        kd_sub = interpolate(kd, next_kd, alpha)

        if codesys_diff is not None and args.diff_mode == "codesys":
            diff_out = codesys_diff.update(temp_sub)
        else:
            sub_elapsed = interpolate(elapsed, next_elapsed, alpha)
            diff_out = compute_diff_out(
                current_temp=temp_sub,
                previous_temp=last_temp_for_diff,
                dt_s=max(sub_elapsed - last_elapsed_for_diff, period_s),
                mode=args.diff_mode,
            )
            last_temp_for_diff = temp_sub
            last_elapsed_for_diff = sub_elapsed

        u, u_p, u_i, u_d = pid.step(
            enable=True,
            x_target=ref_sub,
            x_measured=temp_sub,
            p_coef=kp_sub,
            i_coef=ki_sub,
            d_coef=kd_sub,
            diff_out=diff_out,
        )

    return u, u_p, u_i, u_d, diff_out, n_substeps


def simulate_reconstruction(
    df: pd.DataFrame,
    run_id: str,
    model: GRUModel,
    checkpoint: Dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feature_names = list(checkpoint.get(
        "feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    if len(df) <= window_steps + 1:
        raise RuntimeError(
            f"Run too short for reconstruction: len={len(df)}, window_steps={window_steps}"
        )

    feature_scale = max(abs(float(args.control_feature_scale)), 1e-9)

    pid = ChamberPID(
        u_min=args.u_min,
        u_max=args.u_max,
        pid_i_reverse_mul=args.pid_i_reverse_mul,
    )
    pid_selector = parse_pid_selector(args)

    warmup_end_idx = window_steps - 1

    if args.init_pid_from_real:
        pid.p_part = limit(args.u_min, float(
            df["temp_u_p"].iloc[warmup_end_idx]) / feature_scale, args.u_max)
        pid.i_part = limit(args.u_min, float(
            df["temp_u_i"].iloc[warmup_end_idx]) / feature_scale, args.u_max)
        pid.d_part = limit(-0.4,
                           float(df["temp_u_d"].iloc[warmup_end_idx]) / feature_scale, 0.4)

    feature_window = df.iloc[:window_steps][feature_names].to_numpy(
        dtype=np.float32).copy()

    current_temp_pred = float(df["temp"].iloc[warmup_end_idx])
    previous_temp_pred = float(df["temp"].iloc[warmup_end_idx - 1])
    previous_elapsed = float(df["elapsed_s"].iloc[warmup_end_idx - 1])

    codesys_diff: Optional[CodesysDiff] = None
    if args.diff_mode == "codesys":
        codesys_diff = initialize_codesys_diff_for_sim(
            df=df,
            warmup_end_idx=warmup_end_idx,
            feature_scale=feature_scale,
            args=args,
        )

    rows: List[Dict[str, float]] = []

    for end_idx in range(warmup_end_idx, len(df) - 1):
        next_idx = end_idx + 1

        elapsed = float(df["elapsed_s"].iloc[end_idx])
        next_elapsed = float(df["elapsed_s"].iloc[next_idx])
        dt_to_next = max(next_elapsed - elapsed, 0.0)

        temp_ref = float(df["temp_ref"].iloc[end_idx])
        next_temp_ref = float(df["temp_ref"].iloc[next_idx])

        selector_idx: Optional[int] = None
        if pid_selector is not None:
            kp, ki, kd, selector_idx = pid_selector.get_coefs(
                current_temp_pred)
            next_kp, next_ki, next_kd, _ = pid_selector.get_coefs(
                current_temp_pred)
        else:
            kp = float(df["kp"].iloc[end_idx])
            ki = float(df["ki"].iloc[end_idx])
            kd = float(df["kd"].iloc[end_idx])
            next_kp = float(df["kp"].iloc[next_idx])
            next_ki = float(df["ki"].iloc[next_idx])
            next_kd = float(df["kd"].iloc[next_idx])

        control_mode = str(getattr(args, "control_mode", "pid_substep"))

        if control_mode == "replay":
            # Upper-bound diagnostic: use real logged controller outputs.
            u_feature = float(df["temp_u"].iloc[end_idx])
            u_p_feature = float(df["temp_u_p"].iloc[end_idx])
            u_i_feature = float(df["temp_u_i"].iloc[end_idx])
            u_d_feature = float(df["temp_u_d"].iloc[end_idx])
            u = u_feature / feature_scale
            u_p = u_p_feature / feature_scale
            u_i = u_i_feature / feature_scale
            u_d = u_d_feature / feature_scale
            diff_out = float("nan")
            n_pid_substeps = 0
        elif control_mode == "pid_substep":
            # Corrected controller simulation: Diff and PID run at 0.1 s substeps.
            u, u_p, u_i, u_d, diff_out, n_pid_substeps = run_controller_substeps(
                df=df,
                end_idx=end_idx,
                current_temp_pred=current_temp_pred,
                previous_temp_pred=previous_temp_pred,
                elapsed=elapsed,
                previous_elapsed=previous_elapsed,
                next_elapsed=next_elapsed,
                pid=pid,
                codesys_diff=codesys_diff,
                kp=kp,
                ki=ki,
                kd=kd,
                next_kp=next_kp,
                next_ki=next_ki,
                next_kd=next_kd,
                temp_ref=temp_ref,
                next_temp_ref=next_temp_ref,
                args=args,
            )
            u_feature = u * feature_scale
            u_p_feature = u_p * feature_scale
            u_i_feature = u_i * feature_scale
            u_d_feature = u_d * feature_scale
        else:
            raise ValueError(f"Unknown control_mode: {control_mode}")

        current_feature_row = make_feature_row(
            feature_names,
            temp=current_temp_pred,
            temp_ref=temp_ref,
            elapsed_s=elapsed,
            previous_elapsed_s=previous_elapsed,
            previous_temp=previous_temp_pred,
            u=u_feature,
            u_p=u_p_feature,
            u_i=u_i_feature,
            u_d=u_d_feature,
            kp=kp,
            ki=ki,
            kd=kd,
        )
        feature_window[-1, :] = np.asarray([current_feature_row[name]
                                           for name in feature_names], dtype=np.float32)

        pred_delta = predict_delta_t1(
            model=model,
            checkpoint=checkpoint,
            feature_window=feature_window,
            device=device,
        )

        next_temp_pred = current_temp_pred + pred_delta
        next_temp_real = float(df["temp"].iloc[next_idx])
        current_temp_real = float(df["temp"].iloc[end_idx])
        real_delta = next_temp_real - current_temp_real

        # Teacher-forced one-step baseline from real historical features.
        real_feature_window = df.iloc[
            end_idx - window_steps + 1: end_idx + 1
        ][feature_names].to_numpy(dtype=np.float32)
        prediction_delta = predict_delta_t1(
            model=model,
            checkpoint=checkpoint,
            feature_window=real_feature_window,
            device=device,
        )
        prediction_temp_t1 = current_temp_real + prediction_delta
        prediction_error_temp_t1 = prediction_temp_t1 - next_temp_real
        prediction_error_delta_t1 = prediction_delta - real_delta

        rows.append(
            {
                "run_id": run_id,
                "end_idx": int(end_idx),
                "next_idx": int(next_idx),
                "elapsed_s": elapsed,
                "next_elapsed_s": next_elapsed,
                "dt_to_next_s": dt_to_next,
                "n_pid_substeps": int(n_pid_substeps),
                "pid_period_s": float(getattr(args, "pid_period_s", 0.1)),
                "control_mode": control_mode,
                "pid_substep_temp_mode": str(getattr(args, "pid_substep_temp_mode", "hold")),
                "temp_ref": temp_ref,
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "pid_selector_idx": selector_idx if selector_idx is not None else "",
                "real_temp_t": current_temp_real,
                "pred_temp_t": current_temp_pred,
                "real_temp_t1": next_temp_real,
                "pred_temp_t1": next_temp_pred,
                "prediction_temp_t1": prediction_temp_t1,
                "real_delta_t1": real_delta,
                "pred_delta_t1": pred_delta,
                "prediction_delta_t1": prediction_delta,
                "error_temp_t1": next_temp_pred - next_temp_real,
                "abs_error_temp_t1": abs(next_temp_pred - next_temp_real),
                "prediction_error_temp_t1": prediction_error_temp_t1,
                "prediction_abs_error_temp_t1": abs(prediction_error_temp_t1),
                "error_delta_t1": pred_delta - real_delta,
                "abs_error_delta_t1": abs(pred_delta - real_delta),
                "prediction_error_delta_t1": prediction_error_delta_t1,
                "prediction_abs_error_delta_t1": abs(prediction_error_delta_t1),
                "u": u_feature,
                "u_p": u_p_feature,
                "u_i": u_i_feature,
                "u_d": u_d_feature,
                "u_norm": u,
                "u_p_norm": u_p,
                "u_i_norm": u_i,
                "u_d_norm": u_d,
                "diff_out": diff_out,
                "diff_filter_out": codesys_diff.filter_out if codesys_diff is not None else float("nan"),
                "diff_prev_value": codesys_diff.prev_value if codesys_diff is not None else float("nan"),
                "pid_i_state": pid.i_part,
                "pid_d_state": pid.d_part,
            }
        )

        next_feature_row = make_feature_row(
            feature_names,
            temp=next_temp_pred,
            temp_ref=next_temp_ref,
            elapsed_s=next_elapsed,
            previous_elapsed_s=elapsed,
            previous_temp=current_temp_pred,
            u=u_feature,
            u_p=u_p_feature,
            u_i=u_i_feature,
            u_d=u_d_feature,
            kp=next_kp,
            ki=next_ki,
            kd=next_kd,
        )

        feature_window = np.roll(feature_window, shift=-1, axis=0)
        feature_window[-1, :] = np.asarray([next_feature_row[name]
                                           for name in feature_names], dtype=np.float32)

        previous_temp_pred = current_temp_pred
        current_temp_pred = next_temp_pred
        previous_elapsed = elapsed

    sim_df = pd.DataFrame(rows)

    err = sim_df["error_temp_t1"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    prediction_err = sim_df["prediction_error_temp_t1"].to_numpy(dtype=float)
    prediction_abs_err = np.abs(prediction_err)
    real = sim_df["real_temp_t1"].to_numpy(dtype=float)
    pred = sim_df["pred_temp_t1"].to_numpy(dtype=float)
    prediction = sim_df["prediction_temp_t1"].to_numpy(dtype=float)

    metrics = {
        "run_id": run_id,
        "n_steps": int(len(sim_df)),
        "warmup_steps": int(window_steps),
        "control_mode": str(getattr(args, "control_mode", "pid_substep")),
        "pid_period_s": float(getattr(args, "pid_period_s", 0.1)),
        "pid_substep_temp_mode": str(getattr(args, "pid_substep_temp_mode", "hold")),
        "diff_init_mode": str(getattr(args, "diff_init_mode", "first_temp")),
        "mae_temp": float(np.mean(abs_err)),
        "rmse_temp": float(np.sqrt(np.mean(err**2))),
        "bias_temp": float(np.mean(err)),
        "p50_abs_error_temp": float(np.quantile(abs_err, 0.50)),
        "p90_abs_error_temp": float(np.quantile(abs_err, 0.90)),
        "p95_abs_error_temp": float(np.quantile(abs_err, 0.95)),
        "max_abs_error_temp": float(np.max(abs_err)),
        "final_real_temp": float(real[-1]),
        "final_pred_temp": float(pred[-1]),
        "final_error_temp": float(pred[-1] - real[-1]),
        "mae_delta": float(sim_df["abs_error_delta_t1"].mean()),
        "rmse_delta": float(np.sqrt(np.mean(sim_df["error_delta_t1"].to_numpy(dtype=float) ** 2))),
        "bias_delta": float(sim_df["error_delta_t1"].mean()),
        "prediction_mae_temp": float(np.mean(prediction_abs_err)),
        "prediction_rmse_temp": float(np.sqrt(np.mean(prediction_err**2))),
        "prediction_bias_temp": float(np.mean(prediction_err)),
        "prediction_p50_abs_error_temp": float(np.quantile(prediction_abs_err, 0.50)),
        "prediction_p90_abs_error_temp": float(np.quantile(prediction_abs_err, 0.90)),
        "prediction_p95_abs_error_temp": float(np.quantile(prediction_abs_err, 0.95)),
        "prediction_max_abs_error_temp": float(np.max(prediction_abs_err)),
        "prediction_final_pred_temp": float(prediction[-1]),
        "prediction_final_error_temp": float(prediction[-1] - real[-1]),
        "prediction_mae_delta": float(sim_df["prediction_abs_error_delta_t1"].mean()),
        "prediction_rmse_delta": float(np.sqrt(np.mean(sim_df["prediction_error_delta_t1"].to_numpy(dtype=float) ** 2))),
        "prediction_bias_delta": float(sim_df["prediction_error_delta_t1"].mean()),
    }

    return sim_df, metrics


def plot_trajectory(sim_df: pd.DataFrame, metrics: Dict[str, float], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sim_df["next_elapsed_s"],
            sim_df["real_temp_t1"], label="real temp")
    ax.plot(sim_df["next_elapsed_s"], sim_df["pred_temp_t1"],
            label="GRU + PID substep reconstruction")
    ax.plot(
        sim_df["next_elapsed_s"],
        sim_df["prediction_temp_t1"],
        label="GRU one-step prediction from real features",
        alpha=0.85,
    )
    ax.plot(sim_df["next_elapsed_s"], sim_df["temp_ref"],
            linestyle="--", label="target temperature")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(
        f"Run: {metrics['run_id']} | "
        f"MAE={metrics['mae_temp']:.4f} °C, RMSE={metrics['rmse_temp']:.4f} °C"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_error(sim_df: pd.DataFrame, metrics: Dict[str, float], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sim_df["next_elapsed_s"],
            sim_df["error_temp_t1"], color="#d62728", linewidth=1.7, label="reconstruction - real")
    ax.plot(
        sim_df["next_elapsed_s"],
        sim_df["prediction_error_temp_t1"],
        color="#1f77b4",
        linewidth=1.4,
        label="one-step prediction - real",
    )
    ax.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Temperature error (deg C)")
    ax.set_title(
        f"Reconstruction error: {metrics['run_id']} | "
        f"bias={metrics['bias_temp']:.4f} °C, p90={metrics['p90_abs_error_temp']:.4f} °C"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(checkpoint_path, device)
    feature_names = list(checkpoint.get(
        "feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", args.window_steps))

    print("=== GRU + ChamberPID ===")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Device:        {device}")
    print(f"Window steps:  {window_steps}")
    print(f"Features:      {feature_names}")

    csv_paths = find_run_csvs(Path(args.history_root), args.telemetry_names)
    run_csv = choose_run_csv(csv_paths, args.run_id, args.seed)
    run_id = run_csv.parent.name

    print(f"\nSelected run:  {run_id}")

    df = prepare_run_dataframe(
        csv_path=run_csv,
        feature_names=feature_names,
        min_samples=max(args.min_samples, window_steps + 2),
    )

    sim_df, metrics = simulate_reconstruction(
        df=df,
        run_id=run_id,
        model=model,
        checkpoint=checkpoint,
        args=args,
        device=device,
    )

    safe_run_id = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)

    csv_out = output_dir / f"reconstruction_{safe_run_id}.csv"
    json_out = output_dir / f"metrics_{safe_run_id}.json"
    traj_plot_out = output_dir / f"trajectory_{safe_run_id}.png"
    err_plot_out = output_dir / f"error_{safe_run_id}.png"

    sim_df.to_csv(csv_out, index=False)

    report = {
        "checkpoint": str(checkpoint_path),
        "run_csv": str(run_csv),
        "feature_names": feature_names,
        "window_steps": window_steps,
        "diff_mode": args.diff_mode,
        "control_mode": args.control_mode,
        "pid_period_s": float(args.pid_period_s),
        "pid_substep_temp_mode": args.pid_substep_temp_mode,
        "diff_init_mode": args.diff_init_mode,
        "control_feature_scale": float(args.control_feature_scale),
        "init_pid_from_real": bool(args.init_pid_from_real),
        "init_diff_from_real": bool(args.init_diff_from_real),
        "pid_selector": {
            "enabled": bool(args.pid_selector_points),
            "min_range": args.pid_selector_min_range,
            "max_range": args.pid_selector_max_range,
            "points": args.pid_selector_points,
        },
        "metrics": metrics,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_trajectory(sim_df, metrics, traj_plot_out)
    plot_error(sim_df, metrics, err_plot_out)

    print("\n=== Reconstruction metrics ===")
    print(f"steps:            {metrics['n_steps']}")
    print(f"MAE temp:         {metrics['mae_temp']:.6f} deg C")
    print(f"RMSE temp:        {metrics['rmse_temp']:.6f} deg C")
    print(f"Bias temp:        {metrics['bias_temp']:.6f} deg C")
    print(f"P90 abs error:    {metrics['p90_abs_error_temp']:.6f} deg C")
    print(f"Max abs error:    {metrics['max_abs_error_temp']:.6f} deg C")
    print(f"Final real temp:  {metrics['final_real_temp']:.6f} deg C")
    print(f"Final pred temp:  {metrics['final_pred_temp']:.6f} deg C")
    print(f"Final error:      {metrics['final_error_temp']:.6f} deg C")

    print("\n=== Real-feature one-step prediction metrics ===")
    print(f"MAE temp:         {metrics['prediction_mae_temp']:.6f} deg C")
    print(f"RMSE temp:        {metrics['prediction_rmse_temp']:.6f} deg C")
    print(f"Bias temp:        {metrics['prediction_bias_temp']:.6f} deg C")
    print(
        f"P90 abs error:    {metrics['prediction_p90_abs_error_temp']:.6f} deg C")
    print(
        f"Max abs error:    {metrics['prediction_max_abs_error_temp']:.6f} deg C")
    print(
        f"Final pred temp:  {metrics['prediction_final_pred_temp']:.6f} deg C")
    print(
        f"Final error:      {metrics['prediction_final_error_temp']:.6f} deg C")


if __name__ == "__main__":
    main()
