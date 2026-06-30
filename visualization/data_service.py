import csv
import html
import json
import math
import os
import sqlite3
import sys
from argparse import Namespace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


SAMPLES_FILE = "run_samples.csv"
SUMMARY_FILE = "run_summary.csv"
BAND_METRICS_FILE = "band_metrics.csv"
MAX_SERIES_POINTS = 1800
CACHE_VERSION = 1
CACHE_DB = Path(__file__).resolve().parent / "deepvac_runs_cache.sqlite3"


class RunNotFound(FileNotFoundError):
    pass


def workspace_root():
    configured = os.getenv("DEEPVAC_WORKSPACE_ROOT")
    if configured:
        return Path(configured).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "optimization").exists() or (parent / "gru").exists():
            return parent
    return current.parents[min(2, len(current.parents) - 1)]


def candidate_roots():
    configured = os.getenv("DEEPVAC_DATA_ROOT")
    if configured:
        yield Path(configured)

    yield workspace_root() / "optimization"
    yield Path("/data/optimization")


def gru_root():
    configured = os.getenv("DEEPVAC_GRU_ROOT")
    if configured:
        return Path(configured).resolve()
    return (workspace_root() / "gru").resolve()


def data_root():
    for root in candidate_roots():
        if root.exists():
            return root.resolve()
    return next(candidate_roots()).resolve()


def runs_root():
    root = data_root()
    run_history = root / "run_history"
    return run_history.resolve() if run_history.exists() else root


def csv_dicts(path):
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def first_float(mapping, names):
    for name in names:
        value = to_float(mapping.get(name))
        if value is not None:
            return value
    return None


def format_ts(value):
    number = to_float(value)
    if number is None:
        return value or ""
    return datetime.fromtimestamp(number, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_dir(run_id):
    root = data_root()
    root_runs = runs_root()

    direct = (root / run_id).resolve()
    if (direct / SAMPLES_FILE).exists():
        return direct

    fallback_direct = (root_runs / run_id).resolve()
    if (fallback_direct / SAMPLES_FILE).exists():
        return fallback_direct

    for samples in root_runs.rglob(SAMPLES_FILE):
        if samples.parent.name == run_id or samples.parent.relative_to(root).as_posix() == run_id:
            return samples.parent.resolve()
    raise RunNotFound(run_id)


def read_summary(path):
    rows = csv_dicts(path / SUMMARY_FILE)
    return rows[0] if rows else {}


def read_samples(path):
    return csv_dicts(path / SAMPLES_FILE)


def cache_path():
    configured = os.getenv("DEEPVAC_VISUALIZATION_DB")
    if configured:
        return Path(configured).resolve()
    return CACHE_DB


def connect_cache():
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            key TEXT PRIMARY KEY,
            id TEXT NOT NULL,
            group_name TEXT NOT NULL,
            root_path TEXT NOT NULL,
            run_path TEXT NOT NULL,
            samples_path TEXT NOT NULL,
            source_mtime REAL NOT NULL,
            samples INTEGER NOT NULL,
            duration_s REAL,
            mae REAL,
            cost REAL,
            tail_mae REAL,
            overshoot REAL,
            settle_time_s REAL,
            start_time TEXT,
            end_time TEXT,
            columns_json TEXT NOT NULL,
            numeric_columns_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            bands_json TEXT NOT NULL,
            annotations_json TEXT NOT NULL,
            samples_json TEXT NOT NULL,
            cached_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_cached_at ON runs(cached_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_id ON runs(id)")
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('cache_version', ?)",
        (str(CACHE_VERSION),),
    )
    conn.commit()
    return conn


def source_mtime(run_path):
    mtimes = []
    for file_name in [SAMPLES_FILE, SUMMARY_FILE, BAND_METRICS_FILE]:
        path = run_path / file_name
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def cache_record_for(samples_path):
    path = samples_path.parent
    samples = read_samples(path)
    summary = read_summary(path)
    bands = csv_dicts(path / BAND_METRICS_FILE)
    record = run_record(samples_path)
    columns = list(samples[0].keys()) if samples else []
    numeric = numeric_columns(samples)
    annotations = run_annotations(samples, summary, bands)
    return {
        "record": record,
        "run_path": str(path),
        "samples_path": str(samples_path),
        "source_mtime": source_mtime(path),
        "columns": columns,
        "numeric_columns": numeric,
        "summary": summary,
        "bands": bands,
        "annotations": annotations,
        "samples_rows": samples,
    }


def sync_cache(progress=None):
    root = runs_root()
    sample_files = sorted(root.rglob(SAMPLES_FILE), key=lambda path: path.stat().st_mtime, reverse=True)
    conn = connect_cache()
    active_keys = []
    total = len(sample_files)

    try:
        for index, samples_path in enumerate(sample_files, start=1):
            key = samples_path.parent.relative_to(data_root()).as_posix()
            active_keys.append(key)
            row = conn.execute(
                "SELECT source_mtime FROM runs WHERE key = ?",
                (key,),
            ).fetchone()
            current_mtime = source_mtime(samples_path.parent)
            if row and float(row["source_mtime"]) >= current_mtime:
                if progress:
                    progress(index, total, f"Using cached run {index}/{total}")
                continue

            if progress:
                progress(index, total, f"Caching run {index}/{total}: {samples_path.parent.name}")
            cached = cache_record_for(samples_path)
            record = cached["record"]
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    key, id, group_name, root_path, run_path, samples_path, source_mtime,
                    samples, duration_s, mae, cost, tail_mae, overshoot, settle_time_s,
                    start_time, end_time, columns_json, numeric_columns_json,
                    summary_json, bands_json, annotations_json, samples_json, cached_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    record["key"],
                    record["id"],
                    record["group"],
                    str(root),
                    cached["run_path"],
                    cached["samples_path"],
                    cached["source_mtime"],
                    record["samples"],
                    record["duration_s"],
                    record["mae"],
                    record["cost"],
                    record["tail_mae"],
                    record["overshoot"],
                    record["settle_time_s"],
                    record["start_time"],
                    record["end_time"],
                    json.dumps(cached["columns"]),
                    json.dumps(cached["numeric_columns"]),
                    json.dumps(cached["summary"]),
                    json.dumps(cached["bands"]),
                    json.dumps(cached["annotations"]),
                    json.dumps(cached["samples_rows"]),
                ),
            )
            if index % 25 == 0:
                conn.commit()

        if active_keys:
            placeholders = ",".join("?" for _ in active_keys)
            conn.execute(f"DELETE FROM runs WHERE key NOT IN ({placeholders})", active_keys)
        else:
            conn.execute("DELETE FROM runs")
        conn.commit()
    finally:
        conn.close()

    return {"data_root": str(root), "runs": load_cached_runs()}


def load_cached_runs():
    conn = connect_cache()
    try:
        rows = conn.execute(
            """
            SELECT key, id, group_name, samples, duration_s, mae, cost, tail_mae,
                   overshoot, settle_time_s, start_time, end_time
            FROM runs
            ORDER BY source_mtime DESC
            """
        ).fetchall()
        return [
            {
                "key": row["key"],
                "id": row["id"],
                "group": row["group_name"],
                "samples": row["samples"],
                "duration_s": row["duration_s"],
                "mae": row["mae"],
                "cost": row["cost"],
                "tail_mae": row["tail_mae"],
                "overshoot": row["overshoot"],
                "settle_time_s": row["settle_time_s"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def cached_run_payload(run_id):
    conn = connect_cache()
    try:
        row = conn.execute("SELECT * FROM runs WHERE key = ? OR id = ?", (run_id, run_id)).fetchone()
        if not row:
            return None
        record = {
            "key": row["key"],
            "id": row["id"],
            "group": row["group_name"],
            "samples": row["samples"],
            "duration_s": row["duration_s"],
            "mae": row["mae"],
            "cost": row["cost"],
            "tail_mae": row["tail_mae"],
            "overshoot": row["overshoot"],
            "settle_time_s": row["settle_time_s"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
        }
        return {
            "run": record,
            "summary": json.loads(row["summary_json"]),
            "bands": json.loads(row["bands_json"]),
            "annotations": json.loads(row["annotations_json"]),
            "columns": json.loads(row["columns_json"]),
            "numeric_columns": json.loads(row["numeric_columns_json"]),
            "samples_rows": json.loads(row["samples_json"]),
        }
    finally:
        conn.close()


def numeric_columns(rows):
    if not rows:
        return []
    excluded = {"run_id", "timestamp", "start_temp"}
    columns = []
    for column in rows[0].keys():
        if column in excluded:
            continue
        values = [to_float(row.get(column)) for row in rows[:50]]
        if any(value is not None for value in values):
            columns.append(column)
    return columns


def run_record(samples_path):
    path = samples_path.parent
    summary = read_summary(path)
    if summary.get("num_samples"):
        sample_count = summary["num_samples"]
    else:
        with samples_path.open("r", encoding="utf-8-sig") as handle:
            sample_count = str(sum(1 for _ in handle) - 1)
    start_ts = summary.get("start_ts") or summary.get("timestamp")
    end_ts = summary.get("end_ts")
    return {
        "key": path.relative_to(data_root()).as_posix(),
        "id": path.name,
        "group": str(path.parent.relative_to(data_root())),
        "samples": int(float(sample_count)) if sample_count else 0,
        "duration_s": to_float(summary.get("duration_s")),
        "mae": to_float(summary.get("mae")),
        "cost": to_float(summary.get("cost")),
        "tail_mae": to_float(summary.get("tail_mae")),
        "overshoot": first_float(summary, ["overshoot", "overshoot_max", "max_overshoot"]),
        "settle_time_s": first_float(summary, ["settle_time_s", "time_to_settle_s", "settling_time_s"]),
        "start_time": format_ts(start_ts),
        "end_time": format_ts(end_ts),
    }


def elapsed_x(rows):
    timestamps = [to_float(row.get("timestamp")) for row in rows]
    first_timestamp = next((value for value in timestamps if value is not None), None)
    values = []
    for index, timestamp in enumerate(timestamps):
        if timestamp is not None and first_timestamp is not None:
            values.append(timestamp - first_timestamp)
        else:
            values.append(float(index))
    return values


def run_annotations(samples, summary, bands):
    if not samples:
        return []

    elapsed = elapsed_x(samples)
    sample_timestamps = [to_float(row.get("timestamp")) for row in samples]
    first_timestamp = next((value for value in sample_timestamps if value is not None), None)
    temp_values = [to_float(row.get("temp")) for row in samples]
    ref_values = [to_float(row.get("temp_ref")) for row in samples]
    target = next((value for value in ref_values if value is not None), None)
    start_temp = next((value for value in temp_values if value is not None), None)
    annotations = []

    if start_temp is not None:
        annotations.append({"type": "point", "kind": "start", "x": elapsed[0], "y": start_temp, "label": "Start temp"})
    if target is not None:
        annotations.append({"type": "line-y", "kind": "target", "y": target, "label": "Target"})

    if target is not None:
        valid_pairs = [(x_value, temp) for x_value, temp in zip(elapsed, temp_values) if temp is not None]
        if valid_pairs:
            direction = 1.0 if start_temp is not None and start_temp > target else -1.0
            if direction > 0:
                overshoot_pair = min(valid_pairs, key=lambda pair: pair[1] - target)
                overshoot_value = max(0.0, target - overshoot_pair[1])
            else:
                overshoot_pair = max(valid_pairs, key=lambda pair: pair[1] - target)
                overshoot_value = max(0.0, overshoot_pair[1] - target)
            if overshoot_value > 0:
                annotations.append({
                    "type": "point",
                    "kind": "overshoot",
                    "x": overshoot_pair[0],
                    "y": overshoot_pair[1],
                    "label": f"Max overshoot {overshoot_value:g}",
                })

    settle_time = first_float(summary, ["settle_time_s", "time_to_settle_s", "settling_time_s"])
    duration = first_float(summary, ["duration_s"])
    if settle_time is not None:
        annotations.append({
            "type": "region-x",
            "kind": "settling",
            "x0": settle_time,
            "x1": duration if duration is not None and duration > settle_time else max(elapsed),
            "label": "Settling region",
        })

    for row in bands:
        change_x = first_float(row, ["timestamp", "elapsed_s", "start_s", "tail_start_s", "time_s"])
        if change_x is not None:
            if first_timestamp is not None and change_x >= first_timestamp:
                change_x -= first_timestamp
            annotations.append({"type": "line-x", "kind": "pid", "x": change_x, "label": "PID change"})

    invalid_start = None
    for index, row in enumerate(samples):
        flag = str(row.get("valid", row.get("is_valid", ""))).strip().lower()
        failed = str(row.get("failed", row.get("status", ""))).strip().lower()
        invalid = flag in {"0", "false", "no"} or failed in {"1", "true", "failed", "invalid"}
        if invalid and invalid_start is None:
            invalid_start = elapsed[index]
        elif not invalid and invalid_start is not None:
            annotations.append({"type": "region-x", "kind": "invalid", "x0": invalid_start, "x1": elapsed[index], "label": "Invalid region"})
            invalid_start = None
    if invalid_start is not None:
        annotations.append({"type": "region-x", "kind": "invalid", "x0": invalid_start, "x1": elapsed[-1], "label": "Invalid region"})

    return annotations


def downsample(rows, max_points=MAX_SERIES_POINTS):
    if len(rows) <= max_points:
        return rows
    step = math.ceil(len(rows) / max_points)
    return rows[::step]


def list_runs(progress=None):
    return sync_cache(progress=progress)


def run_detail(run_id):
    cached = cached_run_payload(run_id)
    if cached:
        return {
            "run": cached["run"],
            "summary": cached["summary"],
            "bands": cached["bands"],
            "annotations": cached["annotations"],
            "columns": cached["columns"],
            "numeric_columns": cached["numeric_columns"],
        }

    path = run_dir(run_id)
    samples = read_samples(path)
    summary = read_summary(path)
    bands = csv_dicts(path / BAND_METRICS_FILE)
    return {
        "run": run_record(path / SAMPLES_FILE),
        "summary": summary,
        "bands": bands,
        "annotations": run_annotations(samples, summary, bands),
        "columns": list(samples[0].keys()) if samples else [],
        "numeric_columns": numeric_columns(samples),
    }


def run_series(run_id, requested_columns):
    cached = cached_run_payload(run_id)
    if cached:
        samples = downsample(cached["samples_rows"])
    else:
        path = run_dir(run_id)
        samples = downsample(read_samples(path))
    numeric = numeric_columns(samples)
    y_columns = [column for column in requested_columns if column in numeric]
    x_column = "timestamp" if "timestamp" in numeric else numeric[0] if numeric else None
    points = []
    for index, row in enumerate(samples):
        timestamp = to_float(row.get("timestamp"))
        points.append({
            "i": index,
            "t": timestamp,
            "label": format_ts(timestamp) if timestamp is not None else str(index),
            "values": {column: to_float(row.get(column)) for column in y_columns},
        })
    return {"run_id": run_id, "x_column": x_column, "columns": y_columns, "points": points}


def run_source_mtime(key):
    conn = connect_cache()
    try:
        row = conn.execute("SELECT source_mtime FROM runs WHERE key = ?", (key,)).fetchone()
        return float(row["source_mtime"]) if row else None
    finally:
        conn.close()


def run_table(run_id):
    cached = cached_run_payload(run_id)
    samples = cached["samples_rows"] if cached else read_samples(run_dir(run_id))
    return {"columns": list(samples[0].keys()) if samples else [], "rows": samples}


def make_report_html(run_id):
    cached = cached_run_payload(run_id)
    if cached:
        samples = cached["samples_rows"]
        summary = cached["summary"]
        bands = cached["bands"]
        record = cached["run"]
    else:
        path = run_dir(run_id)
        samples = read_samples(path)
        summary = read_summary(path)
        bands = csv_dicts(path / BAND_METRICS_FILE)
        record = run_record(path / SAMPLES_FILE)
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    band_columns = list(bands[0].keys()) if bands else []
    band_head = "".join(f"<th>{html.escape(column)}</th>" for column in band_columns)
    band_body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in band_columns) + "</tr>"
        for row in bands
    )

    def metric(name):
        return "-" if record.get(name) is None else record.get(name)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DeepVac Run Report - {html.escape(record["id"])}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #172033; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; border-bottom: 1px solid #ccd5e1; padding-bottom: 16px; }}
    h1 {{ margin: 0 0 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f1f5f9; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ border: 1px solid #dbe2ea; border-radius: 8px; padding: 12px; }}
    .metric span {{ display: block; color: #64748b; font-size: 11px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 20px; }}
  </style>
</head>
<body>
  <header><div><h1>{html.escape(record["id"])}</h1><div>{html.escape(record["group"])}</div></div></header>
  <section class="metrics">
    <div class="metric"><span>Cost</span><strong>{html.escape(str(metric("cost")))}</strong></div>
    <div class="metric"><span>MAE</span><strong>{html.escape(str(metric("mae")))}</strong></div>
    <div class="metric"><span>Tail MAE</span><strong>{html.escape(str(metric("tail_mae")))}</strong></div>
    <div class="metric"><span>Overshoot</span><strong>{html.escape(str(metric("overshoot")))}</strong></div>
    <div class="metric"><span>Settle Time</span><strong>{html.escape(str(metric("settle_time_s")))} s</strong></div>
  </section>
  <h2>Summary</h2>
  <table><tbody>{summary_rows}</tbody></table>
  <h2>PID Schedule</h2>
  <table><thead><tr>{band_head}</tr></thead><tbody>{band_body}</tbody></table>
  <p>{len(samples)} samples available in run data CSV.</p>
</body>
</html>"""


def make_sim_args(payload):
    return Namespace(
        start_temp=float(payload.get("start_temp", 27.0)),
        target_temp=float(payload.get("target_temp", 0.0)),
        duration_s=float(payload.get("duration_s", 1200.0)),
        dt_s=float(payload.get("dt_s", 2.0)),
        precondition_ref=None,
        window_steps=int(payload.get("window_steps", 60)),
        cpu=True,
        u_min=-1.0,
        u_max=1.0,
        control_feature_scale=100.0,
        pid_i_reverse_mul=0.333,
        pid_period_s=0.1,
        pid_substep_temp_mode="hold",
        linear_self_max_correction=5.0,
        initial_i=float(payload.get("initial_i", 0.0)),
        initial_d=float(payload.get("initial_d", 0.0)),
        initial_p=float(payload.get("initial_p", 0.0)),
        tail_window_s=300.0,
        near_band=2.0,
        settle_band=0.5,
        w_tail_mae=1.0,
        w_overshoot_rmse=10.0,
        w_overshoot_max=10.0,
        w_tail_std=0.5,
        w_final_error=0.5,
        w_time_to_near=0.001,
        w_invalid=1_000_000.0,
        max_abs_temp=100.0,
    )


def bounded_float(payload, name, default, low, high):
    value = float(payload.get(name, default))
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    return value


@lru_cache(maxsize=1)
def load_gru_model():
    root = gru_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from simulate_gru import DEFAULT_CHECKPOINT, DEFAULT_FEATURE_NAMES, load_model

    device = torch.device("cpu")
    model, checkpoint = load_model(Path(DEFAULT_CHECKPOINT), device)
    feature_names = list(checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", 60))
    return model, checkpoint, feature_names, window_steps, device, str(DEFAULT_CHECKPOINT)


def simulate_gru_run(payload):
    kp = bounded_float(payload, "kp", 7.0, 1.0, 50.0)
    ki = bounded_float(payload, "ki", 700.0, 1.0, 1000.0)
    kd = bounded_float(payload, "kd", 10.0, 1.0, 20.0)
    model, checkpoint, feature_names, window_steps, device, checkpoint_path = load_gru_model()
    from simulate_runs import simulate_candidate

    args = make_sim_args(payload)
    metrics, trajectory = simulate_candidate(
        candidate_id=0,
        kp=kp,
        ki=ki,
        kd=kd,
        model=model,
        checkpoint=checkpoint,
        feature_names=feature_names,
        window_steps=window_steps,
        args=args,
        device=device,
        save_trajectory=True,
    )

    rows = trajectory.to_dict(orient="records") if trajectory is not None else []
    points = []
    for index, row in enumerate(rows):
        elapsed_s = to_float(row.get("elapsed_s"))
        points.append({
            "i": int(row.get("step", index + 1)),
            "t": elapsed_s,
            "label": f"{elapsed_s or 0:g}s",
            "values": {
                "temp": to_float(row.get("temp")),
                "temp_ref": to_float(row.get("temp_ref")),
                "error": to_float(row.get("error")),
                "u": to_float(row.get("u")),
                "u_p": to_float(row.get("u_p")),
                "u_i": to_float(row.get("u_i")),
                "u_d": to_float(row.get("u_d")),
                "pred_delta": to_float(row.get("pred_delta")),
            },
        })

    return {
        "checkpoint": checkpoint_path,
        "metrics": metrics,
        "columns": ["temp", "temp_ref", "error", "u", "u_p", "u_i", "u_d", "pred_delta"],
        "points": points,
    }
