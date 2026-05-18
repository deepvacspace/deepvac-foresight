#!/usr/bin/env python3
"""Run one OPC test with manual coefficients, save, append, retrain the model
and generate the next BO suggestion"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bo_common import (
    append_rows_csv,
    append_mae_column,
    compute_tail_cost,
    fit_gp_model,
    history_run_file,
    save_json,
    suggest_next_params,
    parse_bounds,
)
from opc.opc_common import OPCNodeMap, run_opc_test

ENDPOINT = "opc.tcp://192.168.88.160:12345"

HISTORY_ENTRY_KEYS = [
    "timestamp",
    "run_id",
    "kp",
    "ki",
    "kd",
    "cost",
    "next_suggested",
    "prediction",
    "meta",
]


def append_params_history(path: str, entry: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    missing = [k for k in HISTORY_ENTRY_KEYS if k not in entry]
    if missing:
        raise ValueError(f"params history entry missing keys: {missing}")

    history: List[Dict[str, object]] = []
    if p.exists() and p.stat().st_size > 0:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("params history file must contain a JSON list")
        history = [row for row in data if isinstance(row, dict)]

    history.append(entry)
    p.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_runs_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Master runs CSV does not exist: {path}")

    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError(f"Master runs CSV is empty: {path}")

    required_cols = {"run_id", "kp", "ki", "kd", "cost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Master runs CSV missing required columns: {sorted(missing)}")

    for col in ["kp", "ki", "kd", "cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["run_id", "kp", "ki", "kd", "cost"]).copy()

    df = df.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("After cleaning, no valid training runs remain in master CSV.")

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test-duration", type=float, default=900.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0)

    ap.add_argument("--kp", type=float, required=True)
    ap.add_argument("--ki", type=float, required=True)
    ap.add_argument("--kd", type=float, required=True)

    ap.add_argument("--kp-bounds", default="6,50")
    ap.add_argument("--ki-bounds", default="200,1000")
    ap.add_argument("--kd-bounds", default="5,20")

    ap.add_argument("--n-candidates", type=int, default=7000)
    ap.add_argument("--xi", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-temp_ref", default="ns=2;s=Testa chamber.temp_ref")
    ap.add_argument("--node-temp-raw", default="ns=2;s=Testa chamber.temp_raw")
    ap.add_argument("--node-temp-u", default="ns=2;s=Testa chamber.temp_u")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")

    ap.add_argument("--all-runs-csv", default="history/bo_all_runs.csv")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--model-out", default="history/bo_gp_model.pkl")
    ap.add_argument("--next-out", default="history/bo_next_params.json")
    ap.add_argument("--params-history", default="history/bo_params_history.json")
    ap.add_argument("--result-out", default="bo_last_result.json")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    params: Dict[str, float] = {
        "kp": float(args.kp),
        "ki": float(args.ki),
        "kd": float(args.kd),
    }

    bounds: Dict[str, Tuple[float, float]] = {
        "kp": parse_bounds(args.kp_bounds),
        "ki": parse_bounds(args.ki_bounds),
        "kd": parse_bounds(args.kd_bounds),
    }

    nodes = OPCNodeMap(
        temp=args.node_temp,
        temp_ref=args.node_temp_ref,
        temp_raw=args.node_temp_raw,
        temp_u=args.node_temp_u,
        temp_u_p=args.node_p,
        temp_u_i=args.node_i,
        temp_u_d=args.node_d,
    )

    df_samples, run_summary = run_opc_test(
        endpoint=ENDPOINT,
        nodes=nodes,
        kp=params["kp"],
        ki=params["ki"],
        kd=params["kd"],
        duration_s=args.test_duration,
        dt_s=args.dt,
        progress_every_s=args.progress_every,
        verbose=True,
    )

    df_samples = append_mae_column(df_samples)

    cost_info = compute_tail_cost(
        df_samples,
        entry_band=args.entry_band,
        overshoot_weight=args.overshoot_weight,
    )

    run_summary["cost"] = float(cost_info["cost"])
    run_summary["tail_mae"] = (
        None if cost_info["tail_mae"] is None else float(cost_info["tail_mae"])
    )
    run_summary["overshoot"] = None if cost_info["overshoot"] is None else float(cost_info["overshoot"])
    run_summary["tail_start_index"] = cost_info["tail_start_index"]
    run_summary["target"] = float(cost_info["target"])
    run_summary["start_temp"] = float(cost_info["start_temp"])
    run_summary["direction"] = float(cost_info["direction"])

    run_id = str(run_summary["run_id"])

    samples_out = history_run_file(run_id, args.samples_csv)
    runs_out = history_run_file(run_id, args.runs_csv)
    result_out = history_run_file(run_id, args.result_out)

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_rows_csv(args.all_runs_csv, [run_summary])

    runs = load_runs_table(args.all_runs_csv)
    runs.to_csv(args.all_runs_csv, index=False)

    if not len(runs):
        raise RuntimeError(
            f"No runs found"
        )

    runs_for_fit = runs.copy()
    runs_for_fit["mse"] = runs_for_fit["cost"].astype(float)

    model = fit_gp_model(runs_for_fit)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as fh:
        pickle.dump(model, fh)

    suggestion = suggest_next_params(
        model=model,
        bounds=bounds,
        n_candidates=args.n_candidates,
        xi=args.xi,
        random_state=args.seed,
    )

    best_idx = runs["cost"].astype(float).idxmin()
    best_row = runs.loc[best_idx]

    next_payload = {
        "suggested": {
            "kp": float(suggestion["kp"]),
            "ki": float(suggestion["ki"]),
            "kd": float(suggestion["kd"]),
        },
        "prediction": {
            "cost_mean": float(suggestion["pred_mse"]),
            "cost_std": float(suggestion["pred_std"]),
            "expected_improvement": float(suggestion["expected_improvement"]),
            "best_observed_cost": float(best_row["cost"]),
        },
        "best_observed_run": {
            "run_id": str(best_row["run_id"]),
            "kp": float(best_row["kp"]),
            "ki": float(best_row["ki"]),
            "kd": float(best_row["kd"]),
            "cost": float(best_row["cost"]),
            "tail_mae": (
                None if pd.isna(best_row.get("tail_mae")) else float(best_row["tail_mae"])
            ),
            "overshoot": None if pd.isna(best_row.get("overshoot")) else float(best_row["overshoot"]),
        },
        "meta": {
            "run_id": run_id,
            "n_training_runs": int(model["n_samples"]),
            "acquisition": "expected_improvement",
            "active_learning_rule": "maximize_expected_improvement",
            "xi": float(args.xi),
            "n_candidates": int(args.n_candidates),
            "seed": int(args.seed),
            "bounds": {
                "kp": [bounds["kp"][0], bounds["kp"][1]],
                "ki": [bounds["ki"][0], bounds["ki"][1]],
                "kd": [bounds["kd"][0], bounds["kd"][1]],
            },
            "objective": {
                "name": "tail_mae_plus_overshoot",
                "entry_band": float(args.entry_band),
                "overshoot_weight": float(args.overshoot_weight),
            },
        },
    }

    save_json(args.next_out, next_payload)

    params_entry = {
        "timestamp": int(time.time()),
        "run_id": run_id,
        "kp": float(run_summary["kp"]),
        "ki": float(run_summary["ki"]),
        "kd": float(run_summary["kd"]),
        "cost": float(run_summary["cost"]),
        "next_suggested": next_payload["suggested"],
        "prediction": next_payload["prediction"],
        "meta": {
            "run_id": run_id,
            "n_training_runs": int(model["n_samples"]),
            "acquisition": "expected_improvement",
            "active_learning_rule": "maximize_expected_improvement",
            "xi": float(args.xi),
            "n_candidates": int(args.n_candidates),
            "seed": int(args.seed),
            "test_duration": float(args.test_duration),
            "dt": float(args.dt),
            "progress_every": float(args.progress_every),
            "objective": {
                "name": "tail_mae_plus_overshoot",
                "entry_band": float(args.entry_band),
                "overshoot_weight": float(args.overshoot_weight),
            },
        },
    }
    append_params_history(args.params_history, params_entry)

    save_json(
        result_out,
        {
            "run_summary": run_summary,
            "used_params": params,
            "objective": {
                "name": "tail_mae_plus_overshoot",
                "entry_band": float(args.entry_band),
                "overshoot_weight": float(args.overshoot_weight),
            },
            "next_suggested": next_payload["suggested"],
            "files": {
                "samples_csv": samples_out,
                "runs_csv": runs_out,
                "all_runs_csv": args.all_runs_csv,
                "model_out": args.model_out,
                "next_out": args.next_out,
                "params_history": args.params_history,
            },
        },
    )

    print("Run completed and BO model updated.")
    print(f"run_id={run_summary['run_id']}")
    print(
        f"kp={run_summary['kp']:.6f} "
        f"ki={run_summary['ki']:.6f} "
        f"kd={run_summary['kd']:.6f}"
    )
    print(f"cost={run_summary['cost']:.6f}")
    print(f"tail_mae={run_summary['tail_mae']}")
    print(f"overshoot={run_summary['overshoot']}")
    print(f"Samples saved to: {samples_out}")
    print(f"Summary saved to: {runs_out}")
    print(f"All-runs table updated to: {args.all_runs_csv}")
    print(f"Retrained model saved to: {args.model_out}")
    print(f"Next params saved to: {args.next_out}")
    print(
        f"Next suggestion: kp={next_payload['suggested']['kp']:.6f} "
        f"ki={next_payload['suggested']['ki']:.6f} "
        f"kd={next_payload['suggested']['kd']:.6f}"
    )


if __name__ == "__main__":
    main()
