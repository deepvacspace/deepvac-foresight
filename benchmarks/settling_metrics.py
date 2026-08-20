#!/usr/bin/env python3
"""Per-episode settling metrics, reported as four separate quantities.

Runs are split into episodes by setpoint, so a run that logs its heatup as well as
its cooldown yields one heating episode and one cooling episode. Each episode is
scored on:

  transient_overshoot  max wrong-side excursion in the arrival window
  steady_jitter        std once settled
  steady_bias          mean offset once settled
  settling_time        time from episode start to staying inside the settle band

steady_mae is also reported, but it mixes jitter and bias (mae ~ |bias| + jitter)
and is never used as a primary sort key.

Nothing is filtered out: episodes that never reach the target keep their row with
null settling metrics, and configurations that overshoot are ranked among
themselves rather than dropped.

    python -m benchmarks.settling_metrics --history-root experiments\\run_history
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepvac.artifacts import iter_run_dirs, save_json  # noqa: E402

OUTPUT_DIR = Path(__file__).with_name("output")
REQUIRED_COLS = ("timestamp", "temp", "temp_ref")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Split each run into setpoint episodes and score overshoot, jitter, bias and settling separately.",
    )
    ap.add_argument("--history-root", default=str(ROOT / "experiments" / "run_history"))
    ap.add_argument("--telemetry-names", nargs="+", default=["run_samples.csv"])
    ap.add_argument("--run-glob", default=None, help="Only include run directories matching this glob.")

    ap.add_argument("--entry-band", type=float, default=2.0,
                    help="Absolute error defining arrival at the target.")
    ap.add_argument("--settle-band", type=float, default=0.5,
                    help="Absolute error the episode must stay inside to count as settled.")
    ap.add_argument("--arrival-window-s", type=float, default=120.0,
                    help="Window after first entering the entry band that defines transient overshoot.")
    ap.add_argument("--steady-start-s", type=float, default=300.0,
                    help="Seconds after entering the entry band before the steady window begins.")
    ap.add_argument(
        "--overshoot-tol",
        type=float,
        default=0.05,
        help=(
            "Overshoot at or below this counts as zero. The default is one sensor "
            "count; the logged temperature is quantised at ~0.044 C."
        ),
    )

    ap.add_argument("--min-episode-samples", type=int, default=60)
    ap.add_argument("--min-episode-duration-s", type=float, default=180.0)
    ap.add_argument("--min-episode-span-c", type=float, default=1.0,
                    help="Skip episodes whose setpoint is within this of the starting temperature (no transient).")
    ap.add_argument("--phases", nargs="*", default=None,
                    help="Only score these phases when the telemetry has a phase column (e.g. test heatup).")

    ap.add_argument("--min-replicates", type=int, default=1,
                    help="Configurations with fewer replicates than this are reported but flagged.")
    ap.add_argument("--episodes-csv", default=str(OUTPUT_DIR / "settling_episodes.csv"))
    ap.add_argument("--configs-csv", default=str(OUTPUT_DIR / "settling_configs.csv"))
    ap.add_argument("--report-json", default=str(OUTPUT_DIR / "settling_report.json"))
    ap.add_argument("--top", type=int, default=20, help="Rows to print in the ranked table.")

    return ap


# -----------------------------------------------------------------------------
# Episode splitting
# -----------------------------------------------------------------------------


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("kp", "ki", "kd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLS))
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    if len(df) < 3:
        raise ValueError("too few usable rows")

    return df.reset_index(drop=True)


def split_episodes(df: pd.DataFrame, args: argparse.Namespace) -> List[pd.DataFrame]:
    """One episode per contiguous stretch of constant setpoint and phase."""
    out = df.copy()
    if args.phases and "phase" in out.columns:
        out = out[out["phase"].astype(str).isin(args.phases)]
        if out.empty:
            return []

    key = out["temp_ref"].round(3)
    boundaries = key.ne(key.shift())
    if "phase" in out.columns:
        boundaries = boundaries | out["phase"].astype(str).ne(out["phase"].astype(str).shift())
    out = out.assign(_episode=boundaries.cumsum())

    episodes: List[pd.DataFrame] = []
    for _, ep in out.groupby("_episode", sort=True):
        ep = ep.reset_index(drop=True)
        if len(ep) < args.min_episode_samples:
            continue
        duration = float(ep["timestamp"].iloc[-1] - ep["timestamp"].iloc[0])
        if duration < args.min_episode_duration_s:
            continue
        span = abs(float(ep["temp"].iloc[0]) - float(ep["temp_ref"].iloc[0]))
        if span < args.min_episode_span_c:
            continue
        episodes.append(ep)

    return episodes


def run_config_id(run_dir: Path) -> Optional[str]:
    """The configuration label a collector recorded for this run, if any."""
    summary = run_dir / "run_summary.csv"
    if not summary.exists() or summary.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(summary, nrows=1)
    except Exception:
        return None
    if "config_id" not in df.columns or df.empty or pd.isna(df["config_id"].iloc[0]):
        return None
    return str(df["config_id"].iloc[0])


def schedule_signature(ep: pd.DataFrame) -> Tuple[str, int]:
    """The PID schedule an episode used, in order of first appearance.

    Returns the JSON-encoded triplet sequence and its length; a one-entry
    signature means a fixed triplet.
    """
    if not all(c in ep.columns for c in ("kp", "ki", "kd")):
        return "unknown", 0

    coefs = ep[["kp", "ki", "kd"]].dropna()
    if coefs.empty:
        return "unknown", 0

    triplets: List[List[int]] = []
    for row in coefs.round(0).astype(int).itertuples(index=False):
        triplet = [int(row.kp), int(row.ki), int(row.kd)]
        if not triplets or triplets[-1] != triplet:
            triplets.append(triplet)

    return json.dumps(triplets, separators=(",", ":")), len(triplets)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def episode_metrics(ep: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    t = ep["timestamp"].to_numpy(dtype=float)
    t = t - t[0]
    temp = ep["temp"].to_numpy(dtype=float)
    target = float(ep["temp_ref"].iloc[0])
    start_temp = float(temp[0])

    # +1 when the setpoint is below the start (cooling), -1 when above (heating).
    direction = 1.0 if start_temp > target else -1.0
    deviation = temp - target
    abs_error = np.abs(deviation)
    # Past the target: below it when cooling, above it when heating.
    wrong_side = np.maximum(0.0, -deviation) if direction > 0 else np.maximum(0.0, deviation)

    signature, n_triplets = schedule_signature(ep)
    rec: Dict[str, object] = {
        "target": target,
        "start_temp": start_temp,
        "direction": "cooling" if direction > 0 else "heating",
        "n_samples": int(len(ep)),
        "duration_s": float(t[-1]),
        "phase": str(ep["phase"].iloc[0]) if "phase" in ep.columns else "",
        "schedule": signature,
        "n_triplets": int(n_triplets),
        "is_fixed": bool(n_triplets == 1),
        "overshoot_max_episode": float(np.max(wrong_side)),
        "peak_deviation": float(np.max(np.abs(deviation))),
    }

    entered = np.where(abs_error <= args.entry_band)[0]
    if not len(entered):
        rec.update({
            "reached_band": False,
            "time_to_band_s": None,
            "transient_overshoot": None,
            "steady_bias": None,
            "steady_jitter": None,
            "steady_mae": None,
            "settling_time_s": None,
            "final_abs_error": float(abs_error[-1]),
        })
        return rec

    entry_idx = int(entered[0])
    entry_t = float(t[entry_idx])

    arrival = wrong_side[(t >= entry_t) & (t <= entry_t + args.arrival_window_s)]
    transient = float(np.max(arrival)) if len(arrival) else 0.0

    steady_mask = t >= entry_t + args.steady_start_s
    if steady_mask.sum() >= 5:
        steady = deviation[steady_mask]
        steady_bias = float(np.mean(steady))
        steady_jitter = float(np.std(steady))
        steady_mae = float(np.mean(np.abs(steady)))
        steady_overshoot = float(np.max(wrong_side[steady_mask]))
    else:
        steady_bias = steady_jitter = steady_mae = steady_overshoot = None

    # Settling time: the last moment the error left the settle band.
    outside = np.where(abs_error > args.settle_band)[0]
    if not len(outside):
        settling = 0.0
    elif int(outside[-1]) + 1 < len(t):
        settling = float(t[int(outside[-1]) + 1])
    else:
        settling = None  # never settled and stayed settled

    rec.update({
        "reached_band": True,
        "time_to_band_s": entry_t,
        "transient_overshoot": transient,
        "steady_overshoot": steady_overshoot,
        "steady_bias": steady_bias,
        "steady_jitter": steady_jitter,
        "steady_mae": steady_mae,
        "settling_time_s": settling,
        "final_abs_error": float(abs_error[-1]),
        "zero_overshoot": bool(transient <= args.overshoot_tol),
    })
    return rec


def collect_episodes(args: argparse.Namespace) -> pd.DataFrame:
    history_root = Path(args.history_root)
    records: List[Dict[str, object]] = []

    for run_dir in iter_run_dirs(history_root):
        if args.run_glob and not run_dir.match(args.run_glob):
            continue

        csv_path = None
        for name in args.telemetry_names:
            candidate = run_dir / name
            if candidate.exists() and candidate.stat().st_size > 0:
                csv_path = candidate
                break
        if csv_path is None:
            continue

        try:
            df = load_telemetry(csv_path)
        except Exception as exc:
            print(f"[warn] {run_dir.name}: {exc}")
            continue

        config_id = run_config_id(run_dir)
        for idx, ep in enumerate(split_episodes(df, args)):
            rec = episode_metrics(ep, args)
            rec["run_id"] = str(df["run_id"].iloc[0]) if "run_id" in df.columns else run_dir.name
            rec["run_dir"] = run_dir.name
            rec["episode_idx"] = idx
            rec["config_id"] = config_id or ""
            # The commanded configuration, or the observed PID sequence when the
            # run carries no config_id.
            rec["config_key"] = f"{config_id}#ep{idx}" if config_id else rec["schedule"]
            records.append(rec)

    if not records:
        raise RuntimeError(f"No usable episodes under {history_root}")

    cols = ["run_id", "run_dir", "config_id", "config_key", "episode_idx", "phase",
            "direction", "target", "start_temp"]
    table = pd.DataFrame(records)
    ordered = cols + [c for c in table.columns if c not in cols]
    return table[ordered]


# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------


def aggregate_configs(episodes: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """One row per (schedule, direction), summarised across replicates.

    Overshoot is summarised by its worst replicate, settling quality by the mean.
    """
    scored = episodes[episodes["reached_band"]].copy()
    if scored.empty:
        raise RuntimeError("No episodes reached the entry band.")

    grouped = scored.groupby(["config_key", "direction"], dropna=False)
    rows: List[Dict[str, object]] = []

    for (config_key, direction), g in grouped:
        transient = g["transient_overshoot"].astype(float)
        jitter = g["steady_jitter"].astype(float)
        bias = g["steady_bias"].astype(float)
        settle = g["settling_time_s"].astype(float)

        rows.append({
            "config_key": config_key,
            "config_id": str(g["config_id"].iloc[0]),
            "schedule": str(g["schedule"].iloc[0]),
            "direction": direction,
            "n_replicates": int(len(g)),
            "n_triplets": int(g["n_triplets"].iloc[0]),
            "is_fixed": bool(g["is_fixed"].iloc[0]),
            "overshoot_worst": float(transient.max()),
            "overshoot_mean": float(transient.mean()),
            "overshoot_sd": float(transient.std()) if len(g) > 1 else float("nan"),
            "n_zero_overshoot": int((transient <= args.overshoot_tol).sum()),
            "always_zero_overshoot": bool((transient <= args.overshoot_tol).all()),
            "jitter_mean": float(jitter.mean()),
            "abs_bias_mean": float(bias.abs().mean()),
            "steady_mae_mean": float(g["steady_mae"].astype(float).mean()),
            "settling_time_mean": float(settle.mean()),
            "start_temp_sd": float(g["start_temp"].astype(float).std()) if len(g) > 1 else float("nan"),
            "underpowered": bool(len(g) < args.min_replicates),
            "run_ids": ";".join(sorted(g["run_id"].astype(str))),
        })

    configs = pd.DataFrame(rows)

    # Lexicographic: zero-overshoot configurations first, then settling quality.
    configs["settle_score"] = (
        configs["jitter_mean"].fillna(np.inf)
        + configs["abs_bias_mean"].fillna(np.inf)
    )
    configs = configs.sort_values(
        ["always_zero_overshoot", "overshoot_worst", "settle_score", "settling_time_mean"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    configs.insert(0, "rank", np.arange(1, len(configs) + 1))

    return configs


def snr_summary(episodes: pd.DataFrame, min_replicates: int = 3) -> Dict[str, object]:
    """Within- vs between-configuration spread for each metric."""
    scored = episodes[episodes["reached_band"]].copy()
    counts = scored.groupby(["config_key", "direction"])["run_id"].transform("size")
    repeated = scored[counts >= min_replicates]

    out: Dict[str, object] = {
        "min_replicates": int(min_replicates),
        "n_configs_with_replicates": int(repeated.groupby(["config_key", "direction"]).ngroups) if len(repeated) else 0,
        "n_runs": int(len(repeated)),
        "metrics": {},
    }
    if out["n_configs_with_replicates"] < 2:
        return out

    for metric in ("transient_overshoot", "steady_jitter", "steady_mae", "steady_bias"):
        g = repeated.groupby(["config_key", "direction"])[metric]
        within = float(g.std().median())
        between = float(g.mean().std())
        out["metrics"][metric] = {
            "within_config_sd": within,
            "between_config_sd": between,
            "snr": float(between / within) if within and np.isfinite(within) and within > 0 else None,
            # Replicates needed to reach an SNR of 2.
            "replicates_for_snr_2": int(np.ceil((2.0 * within / between) ** 2)) if between > 0 and within > 0 else None,
        }

    return out


def main() -> None:
    args = build_arg_parser().parse_args()

    episodes = collect_episodes(args)
    configs = aggregate_configs(episodes, args)
    snr = snr_summary(episodes)

    for path_str, frame in ((args.episodes_csv, episodes), (args.configs_csv, configs)):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    reached = episodes[episodes["reached_band"]]
    report = {
        "history_root": str(Path(args.history_root).resolve()),
        "n_runs": int(episodes["run_dir"].nunique()),
        "n_episodes": int(len(episodes)),
        "n_episodes_reached_band": int(len(reached)),
        "n_episodes_by_direction": episodes["direction"].value_counts().to_dict(),
        "settings": {
            "entry_band": args.entry_band, "settle_band": args.settle_band,
            "arrival_window_s": args.arrival_window_s, "steady_start_s": args.steady_start_s,
            "overshoot_tol": args.overshoot_tol,
        },
        "zero_overshoot_episodes": int(reached["transient_overshoot"].le(args.overshoot_tol).sum()),
        "best_transient_overshoot": float(reached["transient_overshoot"].min()) if len(reached) else None,
        "median_transient_overshoot": float(reached["transient_overshoot"].median()) if len(reached) else None,
        "signal_to_noise": snr,
        "outputs": {"episodes_csv": str(Path(args.episodes_csv).resolve()),
                    "configs_csv": str(Path(args.configs_csv).resolve())},
    }
    save_json(Path(args.report_json), report)

    print(f"\n=== Settling metrics: {report['n_episodes']} episodes from {report['n_runs']} runs ===")
    print(f"reached the +-{args.entry_band:g} C band : {report['n_episodes_reached_band']}")
    print(f"episodes by direction         : {report['n_episodes_by_direction']}")
    print(f"zero-overshoot episodes       : {report['zero_overshoot_episodes']} "
          f"(<= {args.overshoot_tol:g} C)")
    print(f"best / median overshoot       : {report['best_transient_overshoot']:.3f} / "
          f"{report['median_transient_overshoot']:.3f} C")

    if snr["metrics"]:
        print(f"\n--- Signal-to-noise over {snr['n_configs_with_replicates']} configurations "
              f"with >={snr['min_replicates']} replicates ---")
        print(f"{'metric':22s} {'within SD':>10s} {'between SD':>11s} {'SNR':>6s} {'n for SNR2':>11s}")
        for name, m in snr["metrics"].items():
            snr_v = "n/a" if m["snr"] is None else f"{m['snr']:.2f}"
            need = "n/a" if m["replicates_for_snr_2"] is None else str(m["replicates_for_snr_2"])
            print(f"{name:22s} {m['within_config_sd']:10.3f} {m['between_config_sd']:11.3f} {snr_v:>6s} {need:>11s}")

    show = configs.head(args.top)
    print(f"\n--- Top {len(show)} configurations (zero overshoot first, then jitter+bias) ---")
    print(show[[
        "rank", "direction", "n_triplets", "n_replicates", "overshoot_worst",
        "jitter_mean", "abs_bias_mean", "settling_time_mean",
    ]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nepisodes csv : {args.episodes_csv}")
    print(f"configs csv  : {args.configs_csv}")
    print(f"report json  : {args.report_json}")


if __name__ == "__main__":
    main()
