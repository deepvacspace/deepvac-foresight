#!/usr/bin/env python3
"""Fit a GP BO model over far/mid/near PID coefficients from run summaries."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.metrics import append_mae_column, compute_tail_cost
from deepvac.pid import parse_bounds
from deepvac.artifacts import append_row_csv, save_json

FEATURE_COLS = [
    "far_kp", "far_ki", "far_kd",
    "mid_kp", "mid_ki", "mid_kd",
    "near_kp", "near_ki", "near_kd",
]
OUTPUT_DIR = Path(__file__).with_name("output")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--history-roots", nargs="+", default=["run_history"])
    ap.add_argument("--max-legacy-runs", type=int, default=3)

    ap.add_argument("--kp-bounds", default="6,50")
    ap.add_argument("--ki-bounds", default="200,1000")
    ap.add_argument("--kd-bounds", default="5,20")

    ap.add_argument("--far-kp-bounds", default="4,8")
    ap.add_argument("--far-ki-bounds", default="400,900")
    ap.add_argument("--far-kd-bounds", default="0,25")

    ap.add_argument("--mid-kp-bounds", default="8,14")
    ap.add_argument("--mid-ki-bounds", default="200,600")
    ap.add_argument("--mid-kd-bounds", default="10,50")

    ap.add_argument("--near-kp-bounds", default="14,24")
    ap.add_argument("--near-ki-bounds", default="100,300")
    ap.add_argument("--near-kd-bounds", default="20,80")

    ap.add_argument("--n-candidates", type=int, default=12000)
    ap.add_argument("--xi", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    ap.add_argument("--merged-runs-out", default=str(OUTPUT_DIR / "bo_all_runs.csv"))
    ap.add_argument("--model-out", default=str(OUTPUT_DIR / "bo_gp_model.pkl"))
    ap.add_argument("--next-out", default=str(OUTPUT_DIR / "bo_next_params.json"))
    ap.add_argument("--history-csv", default=str(OUTPUT_DIR / "bo_training_progress.csv"))
    ap.add_argument("--params-history", default=str(OUTPUT_DIR / "bo_params_history.json"))
    ap.add_argument("--suggestions-dir", default=str(OUTPUT_DIR / "suggestions"))

    return ap


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * _normal_cdf(z) + sigma_safe * _normal_pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def _coeff_or_none(row: pd.Series, key: str) -> Optional[float]:
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def _sample_cost_from_run_samples(samples_csv: Path, args: argparse.Namespace) -> Optional[Dict[str, float]]:
    if not samples_csv.exists() or samples_csv.stat().st_size == 0:
        return None

    samples_df = pd.read_csv(samples_csv)
    samples_df = append_mae_column(samples_df)
    return compute_tail_cost(
        samples_df,
        entry_band=args.entry_band,
        overshoot_weight=args.overshoot_weight,
    )


def _cost_from_summary_metrics(row: pd.Series, args: argparse.Namespace) -> Optional[float]:
    tail_mae = _coeff_or_none(row, "tail_mae")
    overshoot = _coeff_or_none(row, "overshoot")

    if tail_mae is None or overshoot is None:
        return None

    return float(tail_mae + args.overshoot_weight * (overshoot ** 2))


def load_runs(args: argparse.Namespace) -> pd.DataFrame:
    csv_paths: List[Path] = []
    for root in args.history_roots:
        r = Path(root)
        if r.exists():
            csv_paths.extend(sorted(r.rglob("run_summary.csv")))

    if not csv_paths:
        raise FileNotFoundError(f"No run_summary.csv files found under {args.history_roots}")

    records: List[Dict[str, object]] = []
    legacy_used = 0

    for runs_csv in csv_paths:
        try:
            df = pd.read_csv(runs_csv)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable file: {runs_csv} ({exc})")
            continue

        if df.empty:
            continue

        for _, row in df.iterrows():
            run_id = row.get("run_id")
            if pd.isna(run_id):
                continue

            cost = _cost_from_summary_metrics(row, args)
            if cost is None:
                samples_csv = runs_csv.with_name("run_samples.csv")
                try:
                    cost_info = _sample_cost_from_run_samples(samples_csv, args)
                except Exception as exc:
                    print(f"[WARN] Skipping {runs_csv}: failed cost metrics from {samples_csv} ({exc})")
                    continue
                if cost_info is None:
                    print(f"[WARN] Skipping {runs_csv}: no usable summary metrics and no readable samples")
                    continue
                cost = float(cost_info["cost"])

            is_new = all(col in row.index for col in FEATURE_COLS)
            if is_new:
                coeffs = {k: _coeff_or_none(row, k) for k in FEATURE_COLS}
                if any(v is None for v in coeffs.values()):
                    continue
                coeff_format = "multi_band"
            else:
                kp = _coeff_or_none(row, "kp")
                ki = _coeff_or_none(row, "ki")
                kd = _coeff_or_none(row, "kd")
                if kp is None or ki is None or kd is None:
                    continue
                if args.max_legacy_runs >= 0 and legacy_used >= args.max_legacy_runs:
                    continue
                coeffs = {
                    "far_kp": kp,
                    "far_ki": ki,
                    "far_kd": kd,
                    "mid_kp": kp,
                    "mid_ki": ki,
                    "mid_kd": kd,
                    "near_kp": kp,
                    "near_ki": ki,
                    "near_kd": kd,
                }
                coeff_format = "legacy_triplet_expanded"
                legacy_used += 1

            rec = {
                "run_id": str(run_id),
                "cost": float(cost),
                "source_csv": str(runs_csv),
                "coeff_format": coeff_format,
            }
            rec.update(coeffs)
            records.append(rec)

    if not records:
        raise RuntimeError("No valid runs found in mixed history")

    merged = pd.DataFrame(records)

    for col in FEATURE_COLS + ["cost"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["run_id", "cost", *FEATURE_COLS]).copy()
    merged = merged.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)

    return merged


def fit_gp_model_multi(runs_df: pd.DataFrame) -> Dict[str, object]:
    if not len(runs_df):
        raise ValueError("No runs detected")

    X = runs_df[FEATURE_COLS].to_numpy(dtype=float)
    y = runs_df["cost"].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * len(FEATURE_COLS),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp.fit(Xs, y)

    return {
        "scaler": scaler,
        "gp": gp,
        "n_samples": int(len(runs_df)),
        "best_cost": float(np.min(y)),
    }


def _resolved_bounds(args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
    base = {
        "kp": parse_bounds(args.kp_bounds),
        "ki": parse_bounds(args.ki_bounds),
        "kd": parse_bounds(args.kd_bounds),
    }

    out: Dict[str, Tuple[float, float]] = {}
    for band in ("far", "mid", "near"):
        for coef in ("kp", "ki", "kd"):
            override = getattr(args, f"{band}_{coef}_bounds")
            out[f"{band}_{coef}"] = parse_bounds(override) if override else base[coef]

    return out


def suggest_next_multi(
    model: Dict[str, object],
    bounds: Dict[str, Tuple[float, float]],
    n_candidates: int,
    xi: float,
    random_state: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)

    candidates = np.column_stack(
        [rng.uniform(bounds[col][0], bounds[col][1], size=n_candidates) for col in FEATURE_COLS]
    )

    scaler: StandardScaler = model["scaler"]
    gp: GaussianProcessRegressor = model["gp"]

    Xs = scaler.transform(candidates)
    mu, std = gp.predict(Xs, return_std=True)

    y_best = float(model["best_cost"])
    ei = expected_improvement(mu=mu, sigma=std, y_best=y_best, xi=xi)
    idx = int(np.argmax(ei))

    result = {k: float(candidates[idx, i]) for i, k in enumerate(FEATURE_COLS)}
    result.update(
        {
            "pred_cost": float(mu[idx]),
            "pred_std": float(std[idx]),
            "expected_improvement": float(ei[idx]),
        }
    )
    return result


def main() -> None:
    args = build_parser().parse_args()

    bounds = _resolved_bounds(args)
    runs = load_runs(args)

    Path(args.merged_runs_out).parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.merged_runs_out, index=False)

    model = fit_gp_model_multi(runs)
    suggestion = suggest_next_multi(
        model=model,
        bounds=bounds,
        n_candidates=args.n_candidates,
        xi=args.xi,
        random_state=args.seed,
    )

    best_idx = runs["cost"].idxmin()
    best_row = runs.loc[best_idx]

    payload = {
        "suggested": {
            "far": {
                "kp": float(suggestion["far_kp"]),
                "ki": float(suggestion["far_ki"]),
                "kd": float(suggestion["far_kd"]),
            },
            "mid": {
                "kp": float(suggestion["mid_kp"]),
                "ki": float(suggestion["mid_ki"]),
                "kd": float(suggestion["mid_kd"]),
            },
            "near": {
                "kp": float(suggestion["near_kp"]),
                "ki": float(suggestion["near_ki"]),
                "kd": float(suggestion["near_kd"]),
            },
        },
        "prediction": {
            "cost_mean": float(suggestion["pred_cost"]),
            "cost_std": float(suggestion["pred_std"]),
            "expected_improvement": float(suggestion["expected_improvement"]),
            "best_observed_cost": float(best_row["cost"]),
        },
        "best_observed_run": {
            "run_id": str(best_row["run_id"]),
            "coeff_format": str(best_row["coeff_format"]),
            "cost": float(best_row["cost"]),
            "far": {
                "kp": float(best_row["far_kp"]),
                "ki": float(best_row["far_ki"]),
                "kd": float(best_row["far_kd"]),
            },
            "mid": {
                "kp": float(best_row["mid_kp"]),
                "ki": float(best_row["mid_ki"]),
                "kd": float(best_row["mid_kd"]),
            },
            "near": {
                "kp": float(best_row["near_kp"]),
                "ki": float(best_row["near_ki"]),
                "kd": float(best_row["near_kd"]),
            },
        },
        "meta": {
            "n_training_runs": int(model["n_samples"]),
            "n_loaded_runs": int(len(runs)),
            "legacy_runs_cap": int(args.max_legacy_runs),
            "acquisition": "expected_improvement",
            "xi": float(args.xi),
            "n_candidates": int(args.n_candidates),
            "seed": int(args.seed),
            "bounds": {k: [v[0], v[1]] for k, v in bounds.items()},
        },
    }

    generated_at = int(time.time())

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as fh:
        pickle.dump(model, fh)

    save_json(args.next_out, payload)

    snapshot_dir = Path(args.suggestions_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"bo_next_params_multi_{generated_at}.json"
    save_json(str(snapshot_path), {"generated_at": generated_at, **payload})

    Path(args.params_history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params_history).write_text(json.dumps([{"generated_at": generated_at, **payload}], indent=2), encoding="utf-8")

    progress_row = {
        "generated_at": generated_at,
        "n_training_runs": int(model["n_samples"]),
        "n_loaded_runs": int(len(runs)),
        "best_run_id": str(best_row["run_id"]),
        "best_observed_cost": float(best_row["cost"]),
        "pred_cost_mean": float(suggestion["pred_cost"]),
        "pred_cost_std": float(suggestion["pred_std"]),
        "expected_improvement": float(suggestion["expected_improvement"]),
        "next_out": str(Path(args.next_out).resolve()),
        "snapshot_json": str(snapshot_path.resolve()),
    }
    append_row_csv(args.history_csv, progress_row)

    print(f"Loaded mixed runs: {len(runs)}")
    print(f"Saved merged table: {args.merged_runs_out}")
    print(f"Best run: {best_row['run_id']} (format={best_row['coeff_format']}, cost={float(best_row['cost']):.6f})")
    print("Suggested next coefficients:")
    print(
        "  far=(kp={:.6f}, ki={:.6f}, kd={:.6f})".format(
            suggestion["far_kp"], suggestion["far_ki"], suggestion["far_kd"]
        )
    )
    print(
        "  mid=(kp={:.6f}, ki={:.6f}, kd={:.6f})".format(
            suggestion["mid_kp"], suggestion["mid_ki"], suggestion["mid_kd"]
        )
    )
    print(
        "  near=(kp={:.6f}, ki={:.6f}, kd={:.6f})".format(
            suggestion["near_kp"], suggestion["near_ki"], suggestion["near_kd"]
        )
    )
    print(f"Saved model: {args.model_out}")
    print(f"Saved suggestion: {args.next_out}")
    print(f"Saved suggestion snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
