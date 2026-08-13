#!/usr/bin/env python3
"""
Compute band-specific metrics and costs for each run using one Gaussian Process model per band.

--band-mode 3 (default) gives far/mid/near, matching optimization/tocero_3band.py.
--band-mode 5 gives very_far/far/mid/near/very_near, matching optimization/tocero_5band.py.

Each band gets a role from its position: the outermost band is scored on how fast
it closes the error, the innermost on how well it settles, and every band between
them on how progressively it brakes. The band boundaries must match the runner's
--cross-band-N crossings so that each band's samples come from exactly one PID triplet.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from deepvac.metrics import (
    expected_improvement,
    expected_information_gain,
    lower_confidence_bound,
    parse_bounds,
)
from deepvac.artifacts import iter_run_dirs, save_json

BANDS_3 = ("far", "mid", "near")
BANDS_5 = ("very_far", "far", "mid", "near", "very_near")

BANDS = BANDS_3

BAND_SCHEMES: Dict[int, Tuple[str, ...]] = {3: BANDS_3, 5: BANDS_5}

# Band boundaries, outermost first. Mirror the --cross-band-N crossings in
# optimization/tocero_3band.py and optimization/tocero_5band.py.
DEFAULT_THRESHOLDS: Dict[int, Tuple[float, ...]] = {
    3: (10.0, 3.0),
    5: (12.0, 8.0, 5.0, 1.0),
}

# How many ranked candidates per band feed the exported PID_SCHEDULES product.
DEFAULT_COMBO_TOP_K: Dict[int, Tuple[int, ...]] = {
    3: (1, 3, 3),
    5: (1, 1, 2, 2, 2),
}

# The outermost band gets down to the setpoint fast, the innermost one settles,
# everything between them brakes.
ROLE_APPROACH = "approach"
ROLE_BRAKE = "brake"
ROLE_SETTLE = "settle"

# Column aliases read out of band_metrics.csv by gru/.
LEGACY_METRIC_ALIASES: Dict[str, Dict[str, str]] = {
    ROLE_APPROACH: {"band_mae": "far_mae", "time_to_exit_band": "time_to_reach_mid_band"},
    ROLE_BRAKE: {"band_mae": "mid_mae"},
}

OUTPUT_DIR = Path(__file__).with_name("output")
COEF_COLS = ("kp", "ki", "kd")
FEATURE_COLS = ("kp", "ki", "kd", "start_temp", "target_temp")
REQUIRED_COLS = ("timestamp", "kp", "ki", "kd", "temp", "temp_ref")


def band_roles(bands: Tuple[str, ...]) -> Dict[str, str]:
    if len(bands) < 3:
        raise ValueError(f"Need at least 3 bands, got {bands!r}")

    roles = {band: ROLE_BRAKE for band in bands}
    roles[bands[0]] = ROLE_APPROACH
    roles[bands[-1]] = ROLE_SETTLE
    return roles


ROLE_WEIGHT_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    ROLE_APPROACH: ("mae_weight", "time_weight"),
    ROLE_BRAKE: (
        "mae_weight",
        "slope_weight",
        "brake_power",
        "brake_speed_target",
        "brake_min_weight",
    ),
    ROLE_SETTLE: (
        "tail_mae_weight",
        "jitter_weight",
        "overshoot_weight",
        "max_overshoot_weight",
    ),
}


def band_flag(band: str, suffix: str) -> str:
    return f"--{band.replace('_', '-')}-{suffix}"


def band_attr(args: argparse.Namespace, band: str, suffix: str) -> float:
    return float(getattr(args, f"{band}_{suffix.replace('-', '_')}"))


def band_weight_summary(args: argparse.Namespace, band: str, role: str) -> Dict[str, float]:
    return {
        suffix: band_attr(args, band, suffix)
        for suffix in ROLE_WEIGHT_SUFFIXES[role]
    }


def resolve_band_mode(argv: Optional[List[str]] = None) -> int:
    """Read --band-mode before the full parser exists, since the per-band flags depend on it."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--band-mode", type=int, choices=sorted(BAND_SCHEMES), default=3)
    known, _ = pre.parse_known_args(argv)
    return int(known.band_mode)


def resolve_bands(args: argparse.Namespace) -> Tuple[str, ...]:
    return BAND_SCHEMES[int(getattr(args, "band_mode", 3))]


def resolve_thresholds(args: argparse.Namespace) -> Tuple[float, ...]:
    bands = resolve_bands(args)
    raw = getattr(args, "band_thresholds", None)

    if raw:
        values = tuple(float(p.strip()) for p in str(raw).split(",") if p.strip())
    elif len(bands) == 3:
        values = (float(args.far_threshold), float(args.near_threshold))
    else:
        values = DEFAULT_THRESHOLDS[len(bands)]

    if len(values) != len(bands) - 1:
        raise ValueError(
            f"--band-thresholds needs {len(bands) - 1} values for {len(bands)} bands, got {len(values)}"
        )
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError(f"--band-thresholds must be strictly decreasing, got {values}")
    if values[-1] <= 0:
        raise ValueError(f"--band-thresholds must be positive, got {values}")

    return values


def resolve_combo_top_k(args: argparse.Namespace) -> Tuple[int, ...]:
    bands = resolve_bands(args)
    raw = getattr(args, "combo_top_k", None)

    if raw:
        values = tuple(int(p.strip()) for p in str(raw).split(",") if p.strip())
    else:
        values = DEFAULT_COMBO_TOP_K[len(bands)]

    if len(values) != len(bands):
        raise ValueError(
            f"--combo-top-k needs {len(bands)} values for {len(bands)} bands, got {len(values)}"
        )
    if any(v < 1 for v in values):
        raise ValueError(f"--combo-top-k values must be >= 1, got {values}")

    return values


def add_band_arguments(ap: argparse.ArgumentParser, bands: Tuple[str, ...]) -> None:
    roles = band_roles(bands)

    for band in bands:
        ap.add_argument(band_flag(band, "kp-bounds"), default="1,20")
        ap.add_argument(band_flag(band, "ki-bounds"), default="1,1000")
        ap.add_argument(band_flag(band, "kd-bounds"), default="1,50")

        role = roles[band]
        if role == ROLE_APPROACH:
            ap.add_argument(band_flag(band, "mae-weight"), type=float, default=0.5)
            ap.add_argument(band_flag(band, "time-weight"), type=float, default=0.03)
        elif role == ROLE_BRAKE:
            ap.add_argument(band_flag(band, "mae-weight"), type=float, default=0.5)
            ap.add_argument(band_flag(band, "slope-weight"), type=float, default=10.0)
            ap.add_argument(
                band_flag(band, "brake-power"),
                type=float,
                default=2.0,
                help=(
                    "Progressive braking curve across the braking region. "
                    "1.0 = linear, 2.0 = quadratic, 3.0 = stronger braking near the settling band."
                ),
            )
            ap.add_argument(
                band_flag(band, "brake-speed-target"),
                type=float,
                default=0.003,
                help=(
                    "Allowed abs-error reduction speed in deg/s before the braking penalty starts. "
                    "Lower values make the controller brake earlier/slower."
                ),
            )
            ap.add_argument(
                band_flag(band, "brake-min-weight"),
                type=float,
                default=0.10,
                help=(
                    "Minimum braking weight at the outer edge of the braking region. "
                    "Keeps braking active from the first braking band onward."
                ),
            )
        else:
            ap.add_argument(band_flag(band, "tail-mae-weight"), type=float, default=3.0)
            ap.add_argument(band_flag(band, "jitter-weight"), type=float, default=5.0)
            ap.add_argument(band_flag(band, "overshoot-weight"), type=float, default=5.0)
            ap.add_argument(
                band_flag(band, "max-overshoot-weight"),
                type=float,
                default=1.0,
                help=(
                    "Soft settling-band max-overshoot penalty. "
                    "This improves overshoot preference without making crossing a hard constraint."
                ),
            )


def build_arg_parser(band_mode: Optional[int] = None) -> argparse.ArgumentParser:
    if band_mode is None:
        band_mode = resolve_band_mode()
    bands = BAND_SCHEMES[int(band_mode)]

    ap = argparse.ArgumentParser(
        description=(
            "Train separate GP BO models for each band's PID coefficient triplet "
            f"({'/'.join(bands)})."
        )
    )

    ap.add_argument(
        "--band-mode",
        type=int,
        choices=sorted(BAND_SCHEMES),
        default=int(band_mode),
        help="3 = far/mid/near (tocero_3band), 5 = very_far/far/mid/near/very_near (tocero_5band).",
    )

    ap.add_argument("--history-root", default="run_history",
                    help="Root folder for runs.")
    ap.add_argument(
        "--telemetry-names",
        nargs="+",
        default=["run_samples.csv"],
        help="Telemetry file names.",
    )

    ap.add_argument(
        "--acquisition",
        choices=["ei", "lcb", "eig"],
        default="lcb",
        help=(
            "Acquisition function: ei = Expected Improvement, "
            "lcb = Lower Confidence Bound, eig = Expected Information Gain."
        ),
    )

    ap.add_argument(
        "--lcb-kappa",
        type=float,
        default=0.3,
        help="LCB exploration parameter. Higher = more exploration, lower = more exploitation.",
    )

    ap.add_argument(
        "--band-thresholds",
        default=None,
        help=(
            "Comma-separated band boundaries, outermost first "
            f"({len(bands) - 1} values). Defaults to "
            f"'{','.join(format(t, 'g') for t in DEFAULT_THRESHOLDS[len(bands)])}'. "
            "These should match the runner's --cross-band-N values."
        ),
    )
    if len(bands) == 3:
        # Alternative spelling of --band-thresholds for 3-band invocations.
        ap.add_argument("--far-threshold", type=float, default=10.0)
        ap.add_argument("--near-threshold", type=float, default=3.0)

    ap.add_argument("--min-band-samples", type=int, default=8)

    add_band_arguments(ap, bands)

    ap.add_argument("--tail-seconds", type=float, default=300.0)

    ap.add_argument("--n-candidates", type=int, default=100000)
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of ranked candidates to save per band.")
    ap.add_argument(
        "--combo-top-k",
        default=None,
        help=(
            f"Comma-separated per-band candidate counts ({len(bands)} values) whose product "
            "becomes the exported PID_SCHEDULES. Defaults to "
            f"'{','.join(str(k) for k in DEFAULT_COMBO_TOP_K[len(bands)])}'."
        ),
    )
    ap.add_argument("--suggest-start-temp", type=float, default=None,
                    help="Start temperature context for the next-test suggestion. Defaults to median observed start_temp.")
    ap.add_argument("--suggest-target-temp", type=float, default=None,
                    help="Target temperature context for the next-test suggestion. Defaults to median observed target_temp/temp_ref.")
    ap.add_argument("--xi", type=float, default=0.01,
                    help="Expected-improvement exploration parameter.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-restarts-optimizer", type=int, default=5)

    ap.add_argument("--per-run-metrics-csv", default="band_metrics.csv")
    ap.add_argument("--per-run-metrics-json", default="band_metrics.json")
    ap.add_argument("--models-out", default=str(OUTPUT_DIR / "band_gp_models.pkl"))
    ap.add_argument("--next-out", default=str(OUTPUT_DIR / "band_bo_next_params.json"))
    ap.add_argument("--combinations-out", default=str(OUTPUT_DIR / "band_bo_candidate_combinations.txt"))
    ap.add_argument("--suggestions-dir", default=str(OUTPUT_DIR / "band_bo_suggestions"))

    return ap


def format_number(value: object) -> str:
    return str(int(round(float(value))))


def candidate_combination_lines(
    suggestions: Dict[str, Dict[str, object]],
    bands: Tuple[str, ...],
    top_by_band: Tuple[int, ...],
) -> List[str]:
    """Render the ranked per-band candidates as a PID_SCHEDULES block, outermost band first.

    The tuple width is 3 * len(bands), which is what deepvac.pid._schedule_pid expects
    from the matching runner (9 for tocero_3band, 15 for tocero_5band).
    """
    lines: List[str] = ["PID_SCHEDULES = ["]
    ranked_by_band = [
        suggestions[band]["top_candidates"][:count]
        for band, count in zip(bands, top_by_band)
    ]

    for combination in product(*ranked_by_band):
        values = [
            candidate[coef]
            for candidate in combination
            for coef in COEF_COLS
        ]
        lines.append("    (" + ", ".join(format_number(v) for v in values) + "),")

    lines.append("]")
    return lines


def save_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_csv(run_dir: Path, preferred_names: List[str]) -> Optional[Path]:
    for name in preferred_names:
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p

    # Fallback: find any CSV with the required telemetry columns.
    for p in sorted(run_dir.glob("*.csv")):
        try:
            head = pd.read_csv(p, nrows=3)
        except Exception:
            continue
        if all(c in head.columns for c in REQUIRED_COLS):
            return p

    return None


def load_run_summary_context(run_dir: Path) -> Dict[str, float]:
    summary_csv = run_dir / "run_summary.csv"
    if not summary_csv.exists() or summary_csv.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(summary_csv, nrows=1)
    except Exception:
        return {}

    if df.empty:
        return {}

    row = df.iloc[0]
    context: Dict[str, float] = {}
    for out_key, in_key in (("start_temp", "start_temp"), ("target_temp", "temp_ref")):
        if in_key not in row or pd.isna(row[in_key]):
            continue
        try:
            context[out_key] = float(row[in_key])
        except (TypeError, ValueError):
            continue

    return context


def finite_or_fallback(primary: float, fallback: Optional[float]) -> float:
    if np.isfinite(primary):
        return float(primary)
    if fallback is not None and np.isfinite(fallback):
        return float(fallback)
    return float("nan")


def safe_mean(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def safe_std(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if len(arr) else float("nan")


def band_coefficients(df_band: pd.DataFrame) -> Dict[str, float]:

    return {
        "kp": float(df_band["kp"].median()),
        "ki": float(df_band["ki"].median()),
        "kd": float(df_band["kd"].median()),
    }


def signed_error(df: pd.DataFrame) -> pd.Series:

    return df["temp"] - df["temp_ref"]


def start_error_sign(df: pd.DataFrame) -> float:
    e0 = float(df["error"].iloc[0])
    if abs(e0) < 1e-9:
        return 0.0
    return float(np.sign(e0))


# -------------------------------------- DATA PROCESSING ---------------------------


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLS)).copy()
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last")

    if len(df) < 3:
        raise ValueError(f"{csv_path} has too few usable rows after cleaning.")

    return df.reset_index(drop=True)


def classify_bands(
    df: pd.DataFrame,
    bands: Tuple[str, ...],
    thresholds: Tuple[float, ...],
) -> pd.DataFrame:
    """Label each sample with the band whose PID triplet was active when it was taken.

    With bands b0..bn-1 (outermost first) and boundaries t0 > t1 > ... > tn-2:
      b0        : abs_error > t0
      bi        : t_i < abs_error <= t_i-1
      b_{n-1}   : abs_error <= t_n-2
    """
    if len(thresholds) != len(bands) - 1:
        raise ValueError(
            f"Expected {len(bands) - 1} thresholds for bands {bands!r}, got {thresholds!r}"
        )

    out = df.copy()
    out["error"] = signed_error(out)
    out["abs_error"] = out["error"].abs()

    out["band"] = bands[0]
    for band, upper in zip(bands[1:], thresholds):
        out.loc[out["abs_error"] <= upper, "band"] = band

    return out


def brake_region(thresholds: Tuple[float, ...]) -> Tuple[float, float]:
    """(lower, upper) abs-error edges of the whole braking region.

    Braking strength ramps across this region as a whole rather than resetting at
    every internal band boundary, so the penalty stays monotone as the trajectory
    closes on the setpoint. With 3 bands this is exactly the mid band.
    """
    return float(thresholds[-1]), float(thresholds[0])


def bounds_for_band(args: argparse.Namespace, band: str) -> Dict[str, Tuple[float, float]]:
    return {
        "kp": parse_bounds(getattr(args, f"{band}_kp_bounds")),
        "ki": parse_bounds(getattr(args, f"{band}_ki_bounds")),
        "kd": parse_bounds(getattr(args, f"{band}_kd_bounds")),
    }


def add_time_and_slopes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").copy()
    t0 = float(out["timestamp"].iloc[0])
    out["t_rel"] = out["timestamp"].astype(float) - t0

    dt = out["timestamp"].astype(float).diff()
    dt = dt.replace(0.0, np.nan)

    out["dtemp_dt"] = out["temp"].astype(float).diff() / dt
    out["derr_dt"] = out["error"].astype(float).diff() / dt
    out["dabs_error_dt"] = out["abs_error"].astype(float).diff() / dt

    for col in ("dtemp_dt", "derr_dt", "dabs_error_dt"):
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def overshoot_amount(df: pd.DataFrame, init_sign: float) -> pd.Series:

    if init_sign > 0:
        return (-df["error"]).clip(lower=0.0)
    if init_sign < 0:
        return (df["error"]).clip(lower=0.0)

    return pd.Series(np.zeros(len(df)), index=df.index)

# -------------------------------------- COST CALCULATION ---------------------------


def compute_approach_cost(
    df_all: pd.DataFrame,
    df_band: pd.DataFrame,
    exit_threshold: float,
    min_band_samples: int,
    mae_weight: float,
    time_weight: float,
) -> Dict[str, float | int]:
    """
    Outermost-band objective:
      - change quick
    """
    n = int(len(df_band))
    enough = n >= min_band_samples

    if n == 0:
        return {
            "n_samples": 0,
            "cost": float("nan"),
            "band_mae": float("nan"),
            "time_to_exit_band": float("nan"),
        }

    band_mae = safe_mean(df_band["abs_error"])

    reached = df_all[df_all["abs_error"] <= exit_threshold]
    duration = float(df_all["t_rel"].iloc[-1] - df_all["t_rel"].iloc[0])

    if len(reached):
        time_to_exit = float(reached["t_rel"].iloc[0])
    else:
        time_to_exit = duration * 1.5

    cost = mae_weight * band_mae + time_weight * time_to_exit

    return {
        "n_samples": n,
        "cost": float(cost) if enough else float("nan"),
        "band_mae": float(band_mae),
        "time_to_exit_band": float(time_to_exit),
    }


def compute_brake_cost(
    df_band: pd.DataFrame,
    brake_lower: float,
    brake_upper: float,
    min_band_samples: int,
    mae_weight: float,
    slope_weight: float,
    brake_power: float,
    brake_speed_target: float,
    brake_min_weight: float,
) -> Dict[str, float | int]:
    """
    Braking-band objective:
      - Reduce error.
      - Start braking immediately from the outer edge of the braking region.
      - Increase braking strength progressively as the trajectory approaches the
        settling band.

    Definitions:
      error = temp - temp_ref
      abs_error = abs(error)
      dabs_error_dt < 0 means abs_error is decreasing, so the trajectory is moving
      toward the setpoint.

    The braking penalty does not punish every movement toward the target. It only
    punishes approach speed above brake_speed_target, and it weights that excess
    speed more strongly close to the settling band. brake_lower/brake_upper span
    the whole braking region, not just this band, so the weight ramps monotonically
    across every braking band instead of resetting at each boundary.
    """
    n = int(len(df_band))
    enough = n >= min_band_samples

    if n == 0:
        return {
            "n_samples": 0,
            "cost": float("nan"),
            "band_mae": float("nan"),
            "approach_speed": float("nan"),
            "weighted_approach_speed": float("nan"),
            "brake_penalty": float("nan"),
            "mean_brake_weight": float("nan"),
        }

    band_mae = safe_mean(df_band["abs_error"])

    approach_speed = (-df_band["dabs_error_dt"]).clip(lower=0.0)

    denom = max(float(brake_upper) - float(brake_lower), 1e-9)
    progress_to_settle = (float(brake_upper) - df_band["abs_error"]) / denom
    progress_to_settle = progress_to_settle.clip(lower=0.0, upper=1.0)

    brake_min_weight = float(np.clip(brake_min_weight, 0.0, 1.0))
    brake_power = max(float(brake_power), 1e-6)
    brake_weight = brake_min_weight + (1.0 - brake_min_weight) * np.power(
        progress_to_settle,
        brake_power,
    )

    excessive_speed = (approach_speed - float(brake_speed_target)).clip(lower=0.0)

    weighted_approach_speed = safe_mean(brake_weight * approach_speed)
    brake_penalty = safe_mean(brake_weight * np.square(excessive_speed))
    mean_brake_weight = safe_mean(brake_weight)

    cost = (
        mae_weight * band_mae
        + slope_weight * brake_penalty
    )

    return {
        "n_samples": n,
        "cost": float(cost) if enough else float("nan"),
        "band_mae": float(band_mae),
        "approach_speed": float(safe_mean(approach_speed)),
        "weighted_approach_speed": float(weighted_approach_speed),
        "brake_penalty": float(brake_penalty),
        "mean_brake_weight": float(mean_brake_weight),
    }


def compute_settle_cost(
    df_band: pd.DataFrame,
    min_band_samples: int,
    tail_seconds: float,
    init_sign: float,
    tail_mae_weight: float,
    jitter_weight: float,
    overshoot_weight: float,
    max_overshoot_weight: float,
) -> Dict[str, float | int]:
    """
    Innermost-band objective:
      - Reduce tail MAE.
      - Reduce jitter/std.
      - Improve overshoot only inside the settling band.

    Overshoot is still a soft penalty, not a hard constraint. The cost uses:
      - mean squared overshoot: penalizes sustained crossing.
      - max overshoot: softly discourages one deep crossing.

    This keeps overshoot relevant without letting it dominate all other settling
    objectives.
    """
    n = int(len(df_band))
    enough = n >= min_band_samples

    if n == 0:
        return {
            "n_samples": 0,
            "cost": float("nan"),
            "tail_mae": float("nan"),
            "jitter_std": float("nan"),
            "overshoot": float("nan"),
            "mean_sq_overshoot": float("nan"),
            "max_overshoot": float("nan"),
        }

    t_last = float(df_band["t_rel"].iloc[-1])
    tail = df_band[df_band["t_rel"] >= (t_last - tail_seconds)]

    if len(tail) < max(3, min_band_samples // 2):
        tail = df_band.tail(max(3, min(len(df_band), min_band_samples)))

    tail_mae = safe_mean(tail["abs_error"])
    jitter_std = safe_std(tail["error"])

    abs_overshoot = overshoot_amount(tail, init_sign)
    mean_sq_overshoot = safe_mean(np.square(abs_overshoot))
    max_overshoot = float(np.nanmax(abs_overshoot)) if len(abs_overshoot) else 0.0

    overshoot = (
        overshoot_weight * mean_sq_overshoot
        + max_overshoot_weight * max_overshoot
    )

    cost = (
        tail_mae_weight * tail_mae
        + jitter_weight * jitter_std
        + overshoot
    )

    return {
        "n_samples": n,
        "cost": float(cost) if enough else float("nan"),
        "tail_mae": float(tail_mae),
        "jitter_std": float(jitter_std),
        "overshoot": float(overshoot),
        "mean_sq_overshoot": float(mean_sq_overshoot),
        "max_overshoot": float(max_overshoot),
    }


def compute_band_cost(
    band: str,
    role: str,
    df_all: pd.DataFrame,
    df_band: pd.DataFrame,
    thresholds: Tuple[float, ...],
    init_sign: float,
    args: argparse.Namespace,
) -> Dict[str, float | int]:
    if role == ROLE_APPROACH:
        return compute_approach_cost(
            df_all=df_all,
            df_band=df_band,
            exit_threshold=float(thresholds[0]),
            min_band_samples=args.min_band_samples,
            mae_weight=band_attr(args, band, "mae_weight"),
            time_weight=band_attr(args, band, "time_weight"),
        )

    if role == ROLE_BRAKE:
        brake_lower, brake_upper = brake_region(thresholds)
        return compute_brake_cost(
            df_band=df_band,
            brake_lower=brake_lower,
            brake_upper=brake_upper,
            min_band_samples=args.min_band_samples,
            mae_weight=band_attr(args, band, "mae_weight"),
            slope_weight=band_attr(args, band, "slope_weight"),
            brake_power=band_attr(args, band, "brake_power"),
            brake_speed_target=band_attr(args, band, "brake_speed_target"),
            brake_min_weight=band_attr(args, band, "brake_min_weight"),
        )

    return compute_settle_cost(
        df_band=df_band,
        min_band_samples=args.min_band_samples,
        tail_seconds=args.tail_seconds,
        init_sign=init_sign,
        tail_mae_weight=band_attr(args, band, "tail_mae_weight"),
        jitter_weight=band_attr(args, band, "jitter_weight"),
        overshoot_weight=band_attr(args, band, "overshoot_weight"),
        max_overshoot_weight=band_attr(args, band, "max_overshoot_weight"),
    )

def compute_run_band_metrics(
    run_dir: Path,
    telemetry_csv: Path,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    bands = resolve_bands(args)
    thresholds = resolve_thresholds(args)
    roles = band_roles(bands)

    df = load_telemetry(telemetry_csv)
    df = classify_bands(df, bands=bands, thresholds=thresholds)
    df = add_time_and_slopes(df)
    summary_context = load_run_summary_context(run_dir)

    run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns and pd.notna(
        df["run_id"].iloc[0]) else run_dir.name
    init_sign = start_error_sign(df)
    start_temp = finite_or_fallback(
        float(df["temp"].iloc[0]), summary_context.get("start_temp"))
    target_temp = finite_or_fallback(
        float(df["temp_ref"].iloc[-1]), summary_context.get("target_temp"))

    records: List[Dict[str, object]] = []

    for band in bands:
        df_band = df[df["band"] == band].copy()
        metrics = compute_band_cost(
            band=band,
            role=roles[band],
            df_all=df,
            df_band=df_band,
            thresholds=thresholds,
            init_sign=init_sign,
            args=args,
        )

        if len(df_band):
            coefs = band_coefficients(df_band)
        else:
            coefs = {"kp": float("nan"), "ki": float(
                "nan"), "kd": float("nan")}

        rec: Dict[str, object] = {
            "run_id": run_id,
            "band": band,
            "role": roles[band],
            "kp": coefs["kp"],
            "ki": coefs["ki"],
            "kd": coefs["kd"],
            "cost": metrics["cost"],
            "n_samples": metrics["n_samples"],
            "start_temp": start_temp,
            "target_temp": target_temp,
            "duration_s": float(df["t_rel"].iloc[-1] - df["t_rel"].iloc[0]),
        }

        for k, v in metrics.items():
            if k not in rec:
                rec[k] = v

        # Aliased spellings read by gru/gp_build.py and gru/mpc_build.py.
        for new_key, legacy_key in LEGACY_METRIC_ALIASES.get(roles[band], {}).items():
            if new_key in metrics:
                rec[legacy_key] = metrics[new_key]

        records.append(rec)

    # Keep per-run derived metrics next to the run's samples and summary.
    metrics_csv = run_dir / args.per_run_metrics_csv
    metrics_json = run_dir / args.per_run_metrics_json

    pd.DataFrame(records).to_csv(metrics_csv, index=False)
    save_json(metrics_json, {"run_id": run_id, "records": records})

    return records


def compute_all_run_metrics(args: argparse.Namespace) -> pd.DataFrame:
    history_root = Path(args.history_root)
    all_records: List[Dict[str, object]] = []

    for run_dir in iter_run_dirs(history_root):
        telemetry_csv = find_csv(run_dir, args.telemetry_names)
        if telemetry_csv is None:
            continue

        try:
            records = compute_run_band_metrics(run_dir, telemetry_csv, args)
            all_records.extend(records)
        except Exception as exc:
            print(f"[WARN] skipped {run_dir}: {exc}")

    if not all_records:
        raise RuntimeError(
            f"No usable run telemetry found under {history_root}")

    table = pd.DataFrame(all_records)
    for col in ("kp", "ki", "kd", "cost", "n_samples", "start_temp", "target_temp"):
        table[col] = pd.to_numeric(table[col], errors="coerce")

    return table

# -------------------------------------- FIT MODELS AND SUGGEST ---------------------------


def fit_one_band_gp(
    training_table: pd.DataFrame,
    band: str,
    n_restarts_optimizer: int,
) -> Optional[Dict[str, object]]:
    df = training_table[
        (training_table["band"] == band)
        & np.isfinite(pd.to_numeric(training_table["cost"], errors="coerce"))
    ].copy()

    df = df.dropna(subset=[*FEATURE_COLS, "cost"])

    if df.empty:
        return None

    X = df[list(FEATURE_COLS)].to_numpy(dtype=float)
    y = df["cost"].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[1.0] * len(FEATURE_COLS), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=0,
    )
    gp.fit(Xs, y)

    best_idx = int(np.argmin(y))
    return {
        "band": band,
        "scaler": scaler,
        "gp": gp,
        "n_samples": int(len(df)),
        "best_cost": float(y[best_idx]),
        "best_x": {
            "kp": float(X[best_idx, 0]),
            "ki": float(X[best_idx, 1]),
            "kd": float(X[best_idx, 2]),
            "start_temp": float(X[best_idx, 3]),
            "target_temp": float(X[best_idx, 4]),
        },
        "training_rows": df,
    }


def resolve_suggestion_context(
    training_table: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    def median_feature(col: str) -> float:
        values = pd.to_numeric(training_table[col], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            raise ValueError(f"Cannot infer suggestion context: no finite {col} values.")
        return float(values.median())

    start_temp = (
        float(args.suggest_start_temp)
        if args.suggest_start_temp is not None
        else median_feature("start_temp")
    )
    target_temp = (
        float(args.suggest_target_temp)
        if args.suggest_target_temp is not None
        else median_feature("target_temp")
    )

    return {
        "start_temp": start_temp,
        "target_temp": target_temp,
        "source": {
            "start_temp": "argument" if args.suggest_start_temp is not None else "median_training_start_temp",
            "target_temp": "argument" if args.suggest_target_temp is not None else "median_training_target_temp",
        },
    }


def suggest_for_band(
    model: Optional[Dict[str, object]],
    bounds: Dict[str, Tuple[float, float]],
    start_temp: float,
    target_temp: float,
    n_candidates: int,
    top_k: int,
    acquisition: str,
    xi: float,
    lcb_kappa: float,
    seed: int,
) -> Dict[str, object]:

    if model is None:
        raise ValueError("Cannot suggest for band because model is None.")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    rng = np.random.default_rng(seed)

    candidate_coefs = np.column_stack([
        rng.uniform(bounds["kp"][0], bounds["kp"][1], size=n_candidates),
        rng.uniform(bounds["ki"][0], bounds["ki"][1], size=n_candidates),
        rng.uniform(bounds["kd"][0], bounds["kd"][1], size=n_candidates),
    ])
    candidates = np.column_stack([
        candidate_coefs,
        np.full(n_candidates, start_temp, dtype=float),
        np.full(n_candidates, target_temp, dtype=float),
    ])

    scaler: StandardScaler = model["scaler"]
    gp: GaussianProcessRegressor = model["gp"]

    Xs = scaler.transform(candidates)
    mu, std = gp.predict(Xs, return_std=True)

    if acquisition == "ei":
        score = expected_improvement(
            mu=mu,
            sigma=std,
            y_best=float(model["best_cost"]),
            xi=xi,
        )
        ranked_idx = np.argsort(-score)[:top_k]

        acquisition_value_name = "expected_improvement"
        source = "gp_expected_improvement"

    elif acquisition == "lcb":
        score = lower_confidence_bound(
            mu=mu,
            sigma=std,
            kappa=lcb_kappa,
        )
        ranked_idx = np.argsort(score)[:top_k]

        acquisition_value_name = "lcb_score"
        source = "gp_lower_confidence_bound"

    elif acquisition == "eig":
        score = expected_information_gain(sigma=std)
        ranked_idx = np.argsort(-score)[:top_k]

        acquisition_value_name = "expected_information_gain"
        source = "gp_expected_information_gain"

    else:
        raise ValueError(f"Unknown acquisition function: {acquisition}")

    top_candidates: List[Dict[str, object]] = []
    for rank, idx_raw in enumerate(ranked_idx, start=1):
        idx = int(idx_raw)
        candidate = {
            "rank": rank,
            "kp": float(candidates[idx, 0]),
            "ki": float(candidates[idx, 1]),
            "kd": float(candidates[idx, 2]),
            "start_temp": float(candidates[idx, 3]),
            "target_temp": float(candidates[idx, 4]),
            "pred_cost": float(mu[idx]),
            "pred_std": float(std[idx]),
            "source": source,
            "acquisition": acquisition,
        }
        candidate[acquisition_value_name] = float(score[idx])
        top_candidates.append(candidate)

    best = top_candidates[0]
    result = {
        "kp": best["kp"],
        "ki": best["ki"],
        "kd": best["kd"],
        "start_temp": best["start_temp"],
        "target_temp": best["target_temp"],
        "pred_cost": best["pred_cost"],
        "pred_std": best["pred_std"],
        "source": source,
        "acquisition": acquisition,
        "top_candidates": top_candidates,
    }

    result[acquisition_value_name] = best[acquisition_value_name]

    return result


def train_and_suggest(training_table: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    bands = resolve_bands(args)
    thresholds = resolve_thresholds(args)
    roles = band_roles(bands)
    combo_top_k = resolve_combo_top_k(args)

    models: Dict[str, Optional[Dict[str, object]]] = {}
    suggestions: Dict[str, Dict[str, object]] = {}
    suggestion_context = resolve_suggestion_context(training_table, args)
    run_ids = training_table.get(
        "run_id", pd.Series(dtype=str)).fillna("").astype(str)
    run_prefix_counts = run_ids.str.split(
        "_", n=1).str[0].value_counts().to_dict()

    for i, band in enumerate(bands):
        model = fit_one_band_gp(
            training_table=training_table,
            band=band,
            n_restarts_optimizer=args.n_restarts_optimizer,
        )
        models[band] = model
        if model is None:
            raise RuntimeError(
                f"No usable training rows for band '{band}'. Every run needs at least "
                f"{args.min_band_samples} samples in that band "
                f"(--min-band-samples), and the {len(bands)}-band thresholds "
                f"{thresholds} must match the runner's --cross-band-N values."
            )

        suggestion = suggest_for_band(
            model=model,
            bounds=bounds_for_band(args, band),
            start_temp=float(suggestion_context["start_temp"]),
            target_temp=float(suggestion_context["target_temp"]),
            n_candidates=args.n_candidates,
            top_k=args.top_k,
            acquisition=args.acquisition,
            xi=args.xi,
            lcb_kappa=args.lcb_kappa,
            seed=args.seed + 1000 * i,
        )
        suggestions[band] = suggestion

    models_to_pickle: Dict[str, Optional[Dict[str, object]]] = {}
    for band, model in models.items():
        if model is None:
            models_to_pickle[band] = None
            continue
        clean = dict(model)
        clean.pop("training_rows", None)
        models_to_pickle[band] = clean

    Path(args.models_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.models_out, "wb") as fh:
        pickle.dump(models_to_pickle, fh)

    payload: Dict[str, object] = {
        "suggested_controller": {
            band: {coef: suggestions[band][coef] for coef in COEF_COLS}
            for band in bands
        },
        "band_predictions": {
            band: {
                "pred_cost": suggestions[band]["pred_cost"],
                "pred_std": suggestions[band]["pred_std"],
                "start_temp": suggestions[band]["start_temp"],
                "target_temp": suggestions[band]["target_temp"],
                "expected_improvement": suggestions[band].get("expected_improvement"),
                "lcb_score": suggestions[band].get("lcb_score"),
                "expected_information_gain": suggestions[band].get("expected_information_gain"),
                "source": suggestions[band]["source"],
                "top_candidates": suggestions[band]["top_candidates"],
                "n_training_samples": None if models[band] is None else models[band]["n_samples"],
                "best_observed_cost": None if models[band] is None else models[band]["best_cost"],
                "best_observed_triplet": None if models[band] is None else models[band]["best_x"],
            }
            for band in bands
        },
        "candidate_combinations": candidate_combination_lines(
            suggestions,
            bands=bands,
            top_by_band=combo_top_k,
        ),
        "suggestion_context": suggestion_context,
        "meta": {
            "generated_at": int(time.time()),
            "history_root": str(Path(args.history_root).resolve()),
            "acquisition": args.acquisition,
            "band_mode": len(bands),
            "bands": list(bands),
            "band_roles": {band: roles[band] for band in bands},
            "band_thresholds": [float(t) for t in thresholds],
            "min_band_samples": args.min_band_samples,
            "feature_columns": list(FEATURE_COLS),
            "band_weights": {
                band: band_weight_summary(args, band, roles[band])
                for band in bands
            },
            "tail_seconds": args.tail_seconds,
            "xi": args.xi,
            "lcb_kappa": args.lcb_kappa,
            "n_candidates": args.n_candidates,
            "top_k": args.top_k,
            "combination_export_top_by_band": {
                band: int(count) for band, count in zip(bands, combo_top_k)
            },
            "training_run_prefix_counts": {
                str(prefix): int(count)
                for prefix, count in run_prefix_counts.items()
            },
            "bounds": {
                band: {k: list(v)
                       for k, v in bounds_for_band(args, band).items()}
                for band in bands
            },
            "outputs": {
                "models": str(Path(args.models_out).resolve()),
            },
        },
    }

    next_out = Path(args.next_out)
    combinations_out = Path(args.combinations_out)
    suggestions_dir = Path(args.suggestions_dir)
    suggestions_dir.mkdir(parents=True, exist_ok=True)
    snapshot = suggestions_dir / \
        f"band_bo_next_params_{payload['meta']['generated_at']}.json"
    combinations_snapshot = suggestions_dir / \
        f"band_bo_candidate_combinations_{payload['meta']['generated_at']}.txt"

    payload["meta"]["outputs"]["next_params"] = str(
        next_out.resolve())
    payload["meta"]["outputs"]["candidate_combinations"] = str(
        combinations_out.resolve())
    payload["meta"]["outputs"]["snapshot"] = str(
        snapshot.resolve())
    payload["meta"]["outputs"]["candidate_combinations_snapshot"] = str(
        combinations_snapshot.resolve())

    save_json(next_out, payload)
    save_text(combinations_out, payload["candidate_combinations"])
    save_json(snapshot, payload)
    save_text(combinations_snapshot, payload["candidate_combinations"])

    return payload


def main() -> None:
    args = build_arg_parser().parse_args()
    bands = resolve_bands(args)
    thresholds = resolve_thresholds(args)
    width = max(len(band) for band in bands)

    training_table = compute_all_run_metrics(args)
    payload = train_and_suggest(training_table, args)

    print("\n=== Band GP BO summary ===")
    n_runs = int(training_table["run_id"].nunique()
                 ) if "run_id" in training_table.columns else 0
    print(f"Training runs computed: {n_runs}")
    print(
        f"Bands: {len(bands)} "
        f"({', '.join(bands)}) at thresholds {', '.join(format(t, 'g') for t in thresholds)}"
    )

    print("\nSuggested controller:")
    for band in bands:
        s = payload["suggested_controller"][band]
        pred = payload["band_predictions"][band]
        print(
            f"  {band:>{width}}: "
            f"kp={s['kp']:.0f}, ki={s['ki']:.0f}, kd={s['kd']:.0f} "
            f"| source={pred['source']}, "
            f"pred_cost={pred['pred_cost']}, pred_std={pred['pred_std']}"
        )

    print("\nBest observed by band:")
    for band in bands:
        pred = payload["band_predictions"][band]
        print(
            f"  {band:>{width}}: "
            f"n={pred['n_training_samples']}, "
            f"best_cost={pred['best_observed_cost']}, "
            f"best_triplet={pred['best_observed_triplet']}"
        )

    print(f"\nPID_SCHEDULES written to: {args.combinations_out}")


if __name__ == "__main__":
    main()
