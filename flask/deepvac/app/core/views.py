import csv
import html
import math
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request


core = Blueprint("core", __name__)

SAMPLES_FILE = "run_samples.csv"
SUMMARY_FILE = "run_summary.csv"
BAND_METRICS_FILE = "band_metrics.csv"
MAX_SERIES_POINTS = 1800


def _workspace_root():
    configured = os.getenv("DEEPVAC_WORKSPACE_ROOT")
    if configured:
        return Path(configured).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "optimization").exists() or (parent / "gru").exists():
            return parent
    return current.parents[min(4, len(current.parents) - 1)]


def _candidate_roots():
    configured = os.getenv("DEEPVAC_DATA_ROOT")
    if configured:
        yield Path(configured)

    yield _workspace_root() / "optimization"
    yield Path("/data/optimization")


def _gru_root():
    configured = os.getenv("DEEPVAC_GRU_ROOT")
    if configured:
        return Path(configured).resolve()
    return (_workspace_root() / "gru").resolve()


def _data_root():
    for root in _candidate_roots():
        if root.exists():
            return root.resolve()
    return next(_candidate_roots()).resolve()


def _runs_root():
    root = _data_root()
    run_history = root / "run_history"
    return run_history.resolve() if run_history.exists() else root


def _csv_dicts(path):
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _first_float(mapping, names):
    for name in names:
        value = _to_float(mapping.get(name))
        if value is not None:
            return value
    return None


def _format_ts(value):
    number = _to_float(value)
    if number is None:
        return value or ""
    return datetime.fromtimestamp(number, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _run_dir(run_id):
    root = _data_root()
    runs_root = _runs_root()
    direct = (root / run_id).resolve()
    if (direct / SAMPLES_FILE).exists():
        try:
            direct.relative_to(runs_root)
        except ValueError:
            abort(404)
        return direct

    fallback_direct = (runs_root / run_id).resolve()
    if (fallback_direct / SAMPLES_FILE).exists():
        try:
            fallback_direct.relative_to(runs_root)
        except ValueError:
            abort(404)
        return fallback_direct

    for samples in runs_root.rglob(SAMPLES_FILE):
        if samples.parent.name == run_id:
            resolved = samples.parent.resolve()
            try:
                resolved.relative_to(runs_root)
            except ValueError:
                abort(404)
            return resolved
    abort(404)


def _read_summary(run_dir):
    rows = _csv_dicts(run_dir / SUMMARY_FILE)
    return rows[0] if rows else {}


def _read_samples(run_dir):
    rows = _csv_dicts(run_dir / SAMPLES_FILE)
    return rows


def _numeric_columns(rows):
    if not rows:
        return []
    excluded = {"run_id", "timestamp", "start_temp"}
    columns = []
    for column in rows[0].keys():
        if column in excluded:
            continue
        values = [_to_float(row.get(column)) for row in rows[:50]]
        if any(value is not None for value in values):
            columns.append(column)
    return columns


def _run_record(samples_path):
    run_dir = samples_path.parent
    summary = _read_summary(run_dir)
    if summary.get("num_samples"):
        sample_count = summary["num_samples"]
    else:
        with samples_path.open("r", encoding="utf-8-sig") as handle:
            sample_count = str(sum(1 for _ in handle) - 1)
    start_ts = summary.get("start_ts") or summary.get("timestamp")
    end_ts = summary.get("end_ts")
    return {
        "key": run_dir.relative_to(_data_root()).as_posix(),
        "id": run_dir.name,
        "group": str(run_dir.parent.relative_to(_data_root())),
        "samples": int(float(sample_count)) if sample_count else 0,
        "duration_s": _to_float(summary.get("duration_s")),
        "mae": _to_float(summary.get("mae")),
        "cost": _to_float(summary.get("cost")),
        "tail_mae": _to_float(summary.get("tail_mae")),
        "overshoot": _first_float(summary, ["overshoot", "overshoot_max", "max_overshoot"]),
        "settle_time_s": _first_float(summary, ["settle_time_s", "time_to_settle_s", "settling_time_s"]),
        "start_time": _format_ts(start_ts),
        "end_time": _format_ts(end_ts),
    }


def _elapsed_x(rows):
    timestamps = [_to_float(row.get("timestamp")) for row in rows]
    first_timestamp = next((value for value in timestamps if value is not None), None)
    values = []
    for index, timestamp in enumerate(timestamps):
        if timestamp is not None and first_timestamp is not None:
            values.append(timestamp - first_timestamp)
        else:
            values.append(float(index))
    return values


def _run_annotations(samples, summary, bands):
    if not samples:
        return []

    elapsed = _elapsed_x(samples)
    sample_timestamps = [_to_float(row.get("timestamp")) for row in samples]
    first_timestamp = next((value for value in sample_timestamps if value is not None), None)
    temp_values = [_to_float(row.get("temp")) for row in samples]
    ref_values = [_to_float(row.get("temp_ref")) for row in samples]
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

    settle_time = _first_float(summary, ["settle_time_s", "time_to_settle_s", "settling_time_s"])
    duration = _first_float(summary, ["duration_s"])
    if settle_time is not None:
        annotations.append({
            "type": "region-x",
            "kind": "settling",
            "x0": settle_time,
            "x1": duration if duration is not None and duration > settle_time else max(elapsed),
            "label": "Settling region",
        })

    for row in bands:
        change_x = _first_float(row, ["timestamp", "elapsed_s", "start_s", "tail_start_s", "time_s"])
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


def _downsample(rows, max_points=MAX_SERIES_POINTS):
    if len(rows) <= max_points:
        return rows
    step = math.ceil(len(rows) / max_points)
    return rows[::step]


def _make_sim_args(payload):
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


def _bounded_float(payload, name, default, low, high):
    value = float(payload.get(name, default))
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    return value


@lru_cache(maxsize=1)
def _load_gru_model():
    gru_root = _gru_root()
    if str(gru_root) not in sys.path:
        sys.path.insert(0, str(gru_root))

    import torch
    from simulate_gru import DEFAULT_CHECKPOINT, DEFAULT_FEATURE_NAMES, load_model

    device = torch.device("cpu")
    model, checkpoint = load_model(Path(DEFAULT_CHECKPOINT), device)
    feature_names = list(checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", 60))
    return model, checkpoint, feature_names, window_steps, device, str(DEFAULT_CHECKPOINT)


@core.route("/")
@core.route("/index")
def dashboard():
    return render_template("dashboard.html")


@core.route("/dashboard-api/runs")
def runs():
    root = _runs_root()
    sample_files = sorted(root.rglob(SAMPLES_FILE), key=lambda path: path.stat().st_mtime, reverse=True)
    return jsonify({
        "data_root": str(root),
        "runs": [_run_record(path) for path in sample_files],
    })


@core.route("/dashboard-api/runs/<path:run_id>")
def run_detail(run_id):
    run_dir = _run_dir(run_id)
    samples = _read_samples(run_dir)
    numeric_columns = _numeric_columns(samples)
    summary = _read_summary(run_dir)
    bands = _csv_dicts(run_dir / BAND_METRICS_FILE)

    return jsonify({
        "run": _run_record(run_dir / SAMPLES_FILE),
        "summary": summary,
        "bands": bands,
        "annotations": _run_annotations(samples, summary, bands),
        "columns": list(samples[0].keys()) if samples else [],
        "numeric_columns": numeric_columns,
    })


@core.route("/dashboard-api/runs/<path:run_id>/series")
def run_series(run_id):
    run_dir = _run_dir(run_id)
    samples = _downsample(_read_samples(run_dir))
    requested = [name for name in request.args.get("columns", "").split(",") if name]
    numeric_columns = _numeric_columns(samples)
    y_columns = [column for column in requested if column in numeric_columns]

    x_column = "timestamp" if "timestamp" in numeric_columns else numeric_columns[0] if numeric_columns else None
    points = []
    for index, row in enumerate(samples):
        timestamp = _to_float(row.get("timestamp"))
        points.append({
            "i": index,
            "t": timestamp,
            "label": _format_ts(timestamp) if timestamp is not None else str(index),
            "values": {column: _to_float(row.get(column)) for column in y_columns},
        })

    return jsonify({
        "run_id": run_id,
        "x_column": x_column,
        "columns": y_columns,
        "points": points,
    })


@core.route("/dashboard-api/runs/<path:run_id>/table")
def run_table(run_id):
    run_dir = _run_dir(run_id)
    samples = _read_samples(run_dir)
    return jsonify({
        "columns": list(samples[0].keys()) if samples else [],
        "rows": samples,
    })


@core.route("/dashboard-api/runs/<path:run_id>/report")
def run_report(run_id):
    run_dir = _run_dir(run_id)
    samples = _read_samples(run_dir)
    summary = _read_summary(run_dir)
    bands = _csv_dicts(run_dir / BAND_METRICS_FILE)
    record = _run_record(run_dir / SAMPLES_FILE)
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
    metric = lambda name: "-" if record.get(name) is None else record.get(name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
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
    @media print {{ button {{ display: none; }} body {{ margin: 18mm; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{html.escape(record["id"])}</h1>
      <div>{html.escape(record["group"])}</div>
    </div>
    <button onclick="window.print()">Print / Save PDF</button>
  </header>
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


@core.route("/dashboard-api/simulate", methods=["POST"])
def simulate_gru_run():
    payload = request.get_json(silent=True) or {}
    try:
        kp = _bounded_float(payload, "kp", 7.0, 1.0, 50.0)
        ki = _bounded_float(payload, "ki", 700.0, 1.0, 1000.0)
        kd = _bounded_float(payload, "kd", 10.0, 1.0, 20.0)
        model, checkpoint, feature_names, window_steps, device, checkpoint_path = _load_gru_model()
        from simulate_runs import simulate_candidate

        args = _make_sim_args(payload)
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
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    rows = trajectory.to_dict(orient="records") if trajectory is not None else []
    points = []
    for index, row in enumerate(rows):
        elapsed_s = _to_float(row.get("elapsed_s"))
        points.append({
            "i": int(row.get("step", index + 1)),
            "t": elapsed_s,
            "label": f"{elapsed_s or 0:g}s",
            "values": {
                "temp": _to_float(row.get("temp")),
                "temp_ref": _to_float(row.get("temp_ref")),
                "error": _to_float(row.get("error")),
                "u": _to_float(row.get("u")),
                "u_p": _to_float(row.get("u_p")),
                "u_i": _to_float(row.get("u_i")),
                "u_d": _to_float(row.get("u_d")),
                "pred_delta": _to_float(row.get("pred_delta")),
            },
        })

    return jsonify({
        "checkpoint": checkpoint_path,
        "metrics": metrics,
        "columns": ["temp", "temp_ref", "error", "u", "u_p", "u_i", "u_d", "pred_delta"],
        "points": points,
    })
