#!/usr/bin/env python3
"""Manual OPC tests for multi-target experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bo_common import (
    append_rows_csv,
    append_mae_column,
    compute_tail_cost,
    history_run_file,
)
from opc.opc_common import OPCNodeMap, run_opc_test

ENDPOINT = "opc.tcp://192.168.88.160:12345"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test-duration", type=float, default=1500.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--progress-every", type=float, default=60.0, help="Print progress every N seconds")

    ap.add_argument("--kp", type=float, required=True, help="Manually applied kp value")
    ap.add_argument("--ki", type=float, required=True, help="Manually applied ki value")
    ap.add_argument("--kd", type=float, required=True, help="Manually applied kd value")

    ap.add_argument("--entry-band", type=float, default=2.0)
    ap.add_argument("--overshoot-weight", type=float, default=10.0)

    ap.add_argument("--history-root", default="history_multi")
    ap.add_argument("--samples-csv", default="run_samples.csv")
    ap.add_argument("--runs-csv", default="run_summary.csv")
    ap.add_argument("--all-runs-csv", default="history_multi/bo_all_runs.csv")

    ap.add_argument("--node-temp", default="ns=2;s=Testa chamber.temp")
    ap.add_argument("--node-temp_ref", default="ns=2;s=Testa chamber.temp_ref")
    ap.add_argument("--node-temp-raw", default="ns=2;s=Testa chamber.temp_raw")
    ap.add_argument("--node-temp-u", default="ns=2;s=Testa chamber.temp_u")
    ap.add_argument("--node-p", default="ns=2;s=Testa chamber.temp_u_p")
    ap.add_argument("--node-i", default="ns=2;s=Testa chamber.temp_u_i")
    ap.add_argument("--node-d", default="ns=2;s=Testa chamber.temp_u_d")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    kp = float(args.kp)
    ki = float(args.ki)
    kd = float(args.kd)

    nodes = OPCNodeMap(
        temp=args.node_temp,
        temp_ref=args.node_temp_ref,
        temp_raw=args.node_temp_raw,
        temp_u=args.node_temp_u,
        temp_u_p=args.node_p,
        temp_u_i=args.node_i,
        temp_u_d=args.node_d,
    )

    print(f"Testing kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")

    df_samples, run_summary = run_opc_test(
        endpoint=ENDPOINT,
        nodes=nodes,
        kp=kp,
        ki=ki,
        kd=kd,
        duration_s=args.test_duration,
        dt_s=args.dt,
        progress_every_s=args.progress_every,
        verbose=True,
    )

    df_samples = append_mae_column(df_samples)

    start_temp = float(df_samples["temp"].iloc[0])
    target_temp = float(df_samples["temp_ref"].iloc[0])
    delta_temp = start_temp - target_temp

    df_samples["start_temp"] = start_temp
    df_samples["target_temp"] = target_temp
    df_samples["delta_temp"] = delta_temp

    cost_info = compute_tail_cost(
        df_samples,
        entry_band=args.entry_band,
        overshoot_weight=args.overshoot_weight,
    )

    run_summary["start_temp"] = start_temp
    run_summary["target_temp"] = target_temp
    run_summary["delta_temp"] = delta_temp

    run_summary["cost"] = float(cost_info["cost"])
    run_summary["tail_mae"] = None if cost_info["tail_mae"] is None else float(cost_info["tail_mae"])
    run_summary["overshoot"] = None if cost_info["overshoot"] is None else float(cost_info["overshoot"])
    run_summary["tail_start_index"] = cost_info["tail_start_index"]
    run_summary["target"] = float(cost_info["target"])
    run_summary["start_temp_from_cost"] = float(cost_info["start_temp"])
    run_summary["direction"] = float(cost_info["direction"])

    run_id = str(run_summary["run_id"])

    samples_out = history_run_file(run_id, str(Path(args.history_root) / args.samples_csv, args.history_root))
    runs_out = history_run_file(run_id, str(Path(args.history_root) / args.runs_csv, args.history_root))

    append_rows_csv(samples_out, df_samples.to_dict(orient="records"))
    append_rows_csv(runs_out, [run_summary])

    Path(args.all_runs_csv).parent.mkdir(parents=True, exist_ok=True)
    append_rows_csv(args.all_runs_csv, [run_summary])

    print(f"  run_id={run_summary['run_id']}")
    print(f"  samples={run_summary['num_samples']}")
    print(f"  cost={run_summary['cost']:.3f}")
    print(f"  tail_mae={run_summary['tail_mae']}")
    print(f"  overshoot={run_summary['overshoot']}")
    print(f"  start_temp={run_summary['start_temp']}")
    print(f"  target_temp={run_summary['target_temp']}")
    print(f"  delta_temp={run_summary['delta_temp']}")
    print(f"  samples_csv={samples_out}")
    print(f"  runs_csv={runs_out}")
    print(f"  all_runs_csv={args.all_runs_csv}")


if __name__ == "__main__":
    main()
