import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path("readings")
FILES = ["20dec.csv", "21dec.csv"]
OUTPUT_DIR = DATA_DIR

SP_EPS = 1e-6               # tolerance to detect setpoint changes
STEADY_STATE_DELAY = 60.0   # seconds to ignore after each setpoint change
ERROR_TOL = 0.05            # °C tolerance band for "within tolerance" metric

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

def savefig(name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")


# =============================================================================
# Load & clean data
# =============================================================================
runs = {} 

for file in FILES:
    key = Path(file).stem

    df = pd.read_csv(
        DATA_DIR / file,
        sep=";",
        decimal=",",
        encoding="utf-8"
    )

    df.columns = [
        "timestamp",
        "volume_temperature_c",
        "temperature_setpoint_c",
        "temp_control_p_pct",
        "temp_control_i_pct",
        "temp_control_d_pct",
        "discharge_pressure_404_atm",
        "discharge_pressure_23_atm",
    ]

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%y %H:%M:%S")
    df = df.sort_values("timestamp").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]
    df["t_seconds"] = (df["timestamp"] - t0).dt.total_seconds()

    df["temp_error"] = df["temperature_setpoint_c"] - df["volume_temperature_c"]
    df["abs_temp_error"] = df["temp_error"].abs()


    df["sp_change"] = df["temperature_setpoint_c"].diff().abs().fillna(0) > SP_EPS
    df["sp_segment"] = df["sp_change"].cumsum()

    df["segment_time"] = df.groupby("sp_segment")["t_seconds"].transform(lambda x: x - x.iloc[0])

    df["is_steady"] = df["segment_time"] > STEADY_STATE_DELAY


    steady = df[df["is_steady"]].copy()

    print(f"\n=== {key} ===")
    print(f"Samples                     : {len(df):,}")
    print(f"Duration                    : {df['t_seconds'].iloc[-1]:.1f} s")
    print(f"Setpoint segments           : {df['sp_segment'].nunique():,}")
    print(f"Steady-state samples        : {len(steady):,} ({len(steady)/max(len(df),1)*100:.1f}%)")

    if len(steady) > 0:
        mae = steady["abs_temp_error"].mean()
        err_std = steady["temp_error"].std()
        p95 = steady["abs_temp_error"].quantile(0.95)
        within = (steady["abs_temp_error"] <= ERROR_TOL).mean() * 100.0

        print("\nSteady-state overall (across all setpoints):")
        print(f"  MAE(|error|)              : {mae:.6f} °C")
        print(f"  Std(error) (jitter)       : {err_std:.6f} °C")
        print(f"  95th pct |error|          : {p95:.6f} °C")
        print(f"  Within ±{ERROR_TOL:.3f}°C           : {within:.2f} %")
    else:
        print("\nNo steady-state samples (increase run length or reduce STEADY_STATE_DELAY).")

    runs[key] = df


rows = []
for run_key, df in runs.items():
    steady = df[df["is_steady"]].copy()
    if len(steady) == 0:
        continue

    g = steady.groupby("sp_segment", as_index=False)
    seg_metrics = g.agg(
        setpoint_c=("temperature_setpoint_c", "first"),
        n_samples=("temp_error", "size"),
        mae_abs_error=("abs_temp_error", "mean"),
        std_error=("temp_error", "std"),
        p95_abs_error=("abs_temp_error", lambda x: x.quantile(0.95)),
        mean_error=("temp_error", "mean"),
        within_tol_frac=("abs_temp_error", lambda x: (x <= ERROR_TOL).mean()),
    )

    seg_metrics["within_tol_pct"] = seg_metrics["within_tol_frac"] * 100.0

    seg_metrics.insert(0, "run", run_key)

    rows.append(seg_metrics)

if rows:
    metrics_df = pd.concat(rows, ignore_index=True)
    metrics_df = metrics_df.sort_values(["run", "setpoint_c", "sp_segment"]).reset_index(drop=True)

    metrics_csv_path = OUTPUT_DIR / "eda_setpoint_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved per-segment metrics table to: {metrics_csv_path}")
else:
    metrics_df = pd.DataFrame()
    print("\nNo metrics table generated (no steady-state data).")


# 1) Temperature vs Setpoint 
plt.figure()
for key, df in runs.items():
    plt.plot(df["t_seconds"], df["volume_temperature_c"], label=f"{key} Temp")
    plt.plot(df["t_seconds"], df["temperature_setpoint_c"], linestyle="--", label=f"{key} SP")
plt.xlabel("Time since start (s)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature vs Setpoint")
plt.legend()
plt.tight_layout()
savefig("temperature_vs_setpoint")
plt.show()


# 2) PID contributions 
plt.figure()
for key, df in runs.items():
    plt.plot(df["t_seconds"], df["temp_control_p_pct"], label=f"{key} P")
    plt.plot(df["t_seconds"], df["temp_control_i_pct"], label=f"{key} I")
    plt.plot(df["t_seconds"], df["temp_control_d_pct"], label=f"{key} D")
plt.xlabel("Time since start (s)")
plt.ylabel("Control output (%)")
plt.title("PID Contributions")
plt.legend(ncols=2)
plt.tight_layout()
savefig("pid_contributions")
plt.show()


# 3) Error vs time aligned to setpoint changes (steady-state only)
plt.figure()
for run_key, df in runs.items():
    for seg, gseg in df.groupby("sp_segment"):
        gseg = gseg[gseg["is_steady"]]
        if len(gseg) == 0:
            continue
        plt.plot(gseg["segment_time"], gseg["temp_error"], alpha=0.35)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Time since setpoint change (s)")
plt.ylabel("Temperature error (°C) = SP - Temp")
plt.title(f"Temperature Error After Setpoint Change (steady-state, > {STEADY_STATE_DELAY:.0f}s)")
plt.tight_layout()
savefig("error_vs_time_aligned_steady")
plt.show()

# 4) Error distribution per setpoint 
steady_all = []
for run_key, df in runs.items():
    s = df[df["is_steady"]].copy()
    if len(s) == 0:
        continue
    s["run"] = run_key
    steady_all.append(s)

if steady_all:
    steady_all = pd.concat(steady_all, ignore_index=True)

    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=steady_all,
        x="temperature_setpoint_c",
        y="temp_error",
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Setpoint (°C)")
    plt.ylabel("Temperature error (°C) = SP - Temp")
    plt.title("Error Distribution per Setpoint")
    plt.tight_layout()
    savefig("error_by_setpoint_boxplot")
    plt.show()
else:
    print("\nNo steady-state rows available for setpoint-conditioned plots.")
