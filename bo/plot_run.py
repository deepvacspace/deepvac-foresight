import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", help="Run folder name")
    ap.add_argument(
        "--history-dir",
        default="history",
        help="Base history directory",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output image path",
    )
    ap.add_argument("--show", action="store_true", help="Show interactive plot window")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    run_dir = Path(args.history_dir) / args.run_name
    csv_path = run_dir / "run_samples.csv"

    if args.out is None:
        out_path = run_dir / "telemetry_temp_vs_time.png"
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["timestamp", "temp", "temp_ref"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["temp_ref"] = pd.to_numeric(df["temp_ref"], errors="coerce")

    df = df.dropna(subset=["timestamp", "temp", "temp_ref"]).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows found in {csv_path}")

    elapsed_minutes = (df["timestamp"] - df["timestamp"].iloc[0]) / 60.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(elapsed_minutes, df["temp"], label="Temperature", linewidth=1.8)
    ax.plot(elapsed_minutes, df["temp_ref"], "--", label="Setpoint", linewidth=1.5)

    kp = df["kp"].iloc[0] if "kp" in df.columns else None
    ki = df["ki"].iloc[0] if "ki" in df.columns else None
    kd = df["kd"].iloc[0] if "kd" in df.columns else None

    title = f"Temperature vs Time - {args.run_name}"
    if kp is not None and ki is not None and kd is not None:
        title += f" | kp={kp}, ki={ki}, kd={kd}"

    ax.set_title(title)
    ax.set_xlabel("Time since start (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
