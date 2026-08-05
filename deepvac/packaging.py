"""Stage a trained checkpoint into a local folder for the two downstream
desktop apps, under --output-dir (default ./packaging):

    packaging/
      insight/            -> copy into <insight app>/app/model/
        model.pt
        simulation.py
      control2-client/    -> copy into <control2-client app>/data/model/
        {model_type}_plant_model.onnx
        {model_type}_plant_model.json

insight: model.pt plus a regenerated simulation.py. The block above the
`# === END GENERATED ===` marker is regenerated from this repo's source;
--insight-root is read (never written) to carry the hand-maintained block
below the marker (simulate_candidate, compute_metrics) into the new file.

control2-client: a self-contained ONNX graph (scaling baked in, so the
C++ side feeds a raw feature window and reads back a raw temperature
delta) plus a JSON sidecar describing the input shape and feature order.
"""

from __future__ import annotations

import inspect
import json
import re
import textwrap
from pathlib import Path

import torch
import torch.nn as nn

from deepvac.model_registry import get_model_spec, registered_model_types

ModelType = str


def _read_stamped_model_family(checkpoint_path: Path) -> ModelType | None:
    """Read checkpoint["model_family"] if present and registered, else None."""
    from gru.gru_common import _ensure_sklearn_stub

    _ensure_sklearn_stub()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    family = checkpoint.get("model_family") if isinstance(checkpoint, dict) else None
    return family if family in registered_model_types() else None


def resolve_model_type(checkpoint_path: Path, explicit: str | None) -> ModelType:
    if explicit:
        get_model_spec(explicit)  # raises ValueError if not registered
        return explicit

    stamped = _read_stamped_model_family(checkpoint_path)
    if stamped is not None:
        return stamped

    parts = {p.lower() for p in checkpoint_path.parts}
    matches = [name for name in registered_model_types() if name in parts]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Checkpoint path {checkpoint_path} matches multiple registered model types "
            f"{matches} and has no 'model_family' field to disambiguate. Pass --model-type explicitly."
        )
    raise ValueError(
        f"Could not determine the model type of checkpoint {checkpoint_path}: it has no "
        f"'model_family' field, and none of the registered model types {registered_model_types()} "
        "appear as a path component. Pass --model-type explicitly."
    )


# -----------------------------------------------------------------------------
# ONNX export (control2-client)
# -----------------------------------------------------------------------------


class _ScaledPlantModel(nn.Module):
    """Wraps a plant model so the exported graph is self-contained: raw
    feature window in (unscaled), raw predicted temperature delta out
    (unscaled). Bakes the checkpoint's fitted StandardScaler transforms into
    the graph as constant buffers so the C++ side never needs to reimplement
    them."""

    def __init__(self, base_model: nn.Module, x_mean, x_scale, y_mean, y_scale) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_scale", torch.as_tensor(x_scale, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_scale, dtype=torch.float32))

    def forward(self, feature_window: torch.Tensor) -> torch.Tensor:
        x_scaled = (feature_window - self.x_mean) / self.x_scale
        y_scaled = self.base_model(x_scaled)
        return y_scaled * self.y_scale + self.y_mean


def export_onnx(checkpoint_path: Path, model_type: ModelType, output_dir: Path) -> dict[str, Path]:
    device = torch.device("cpu")
    spec = get_model_spec(model_type)
    model, checkpoint = spec.load_model(checkpoint_path, device)

    x_scaler = checkpoint["x_scaler"]
    y_scaler = checkpoint["y_scaler"]
    feature_names = list(checkpoint.get("feature_names") or [])
    window_steps = int(checkpoint.get("window_steps", 60))
    n_features = len(feature_names) or int(checkpoint["input_dim"])

    wrapped = _ScaledPlantModel(
        model,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
    ).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_type}_plant_model.onnx"
    metadata_path = output_dir / f"{model_type}_plant_model.json"

    dummy = torch.zeros((1, window_steps, n_features), dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        input_names=["feature_window"],
        output_names=["temp_delta_t1"],
        dynamic_axes={"feature_window": {0: "batch"}, "temp_delta_t1": {0: "batch"}},
        opset_version=17,
        # The default dynamo=True exporter traces via torch.export.export(),
        # which is stricter about the environment than the traditional
        # TorchScript-based exporter and trips over the fake `sklearn` module
        # _ensure_sklearn_stub() (called by load_model() above) registers in
        # sys.modules. That stub only exists so torch.load() can unpickle a
        # checkpoint's StandardScaler without scikit-learn installed; it's
        # irrelevant to this export (the fitted scaler values are already
        # baked into _ScaledPlantModel's buffers by this point), so use the
        # traditional exporter here instead of trying to make the stub more
        # convincing.
        dynamo=False,
    )

    metadata_path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "source_checkpoint": str(checkpoint_path),
                "feature_names": feature_names,
                "window_steps": window_steps,
                "input_shape": ["batch", window_steps, n_features],
                "output_shape": ["batch", 1],
                "output_meaning": "predicted temperature delta for the next step (temp[t+1] - temp[t]), degC",
                "note": (
                    "Input/output scaling is already baked into the ONNX graph -- "
                    "feed raw feature values, read back a raw degC delta."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"onnx": onnx_path, "metadata": metadata_path}


def verify_onnx(onnx_path: Path, checkpoint_path: Path, model_type: ModelType, atol: float = 1e-4) -> float:
    """Compare the exported ONNX graph against the original PyTorch pipeline
    on a random input; returns the max absolute difference. Raises if it
    exceeds `atol`."""
    import numpy as np
    import onnxruntime as ort

    device = torch.device("cpu")
    spec = get_model_spec(model_type)
    model, checkpoint = spec.load_model(checkpoint_path, device)

    window_steps = int(checkpoint.get("window_steps", 60))
    n_features = int(checkpoint["input_dim"])
    rng = np.random.default_rng(0)
    feature_window = rng.normal(size=(window_steps, n_features)).astype(np.float32)

    torch_delta = spec.predict_delta_t1(model, checkpoint, feature_window, device)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_delta,) = session.run(None, {"feature_window": feature_window[None, :, :]})
    onnx_delta = float(onnx_delta[0, 0])

    diff = abs(torch_delta - onnx_delta)
    if diff > atol:
        raise RuntimeError(
            f"ONNX export mismatch: torch={torch_delta!r} onnx={onnx_delta!r} diff={diff!r} > atol={atol!r}"
        )
    return diff


# -----------------------------------------------------------------------------
# insight (PySide6) sync
# -----------------------------------------------------------------------------

# Simulator-view logic (simulate_candidate/compute_metrics)
 
GENERATED_END_MARKER = "# === END GENERATED (deepvac package-model) ==="

_MODERNIZE_SUBS = [
    (re.compile(r"\bDict\[([^\]]+)\]"), r"dict[\1]"),
    (re.compile(r"\bTuple\[([^\]]+)\]"), r"tuple[\1]"),
    (re.compile(r"\bList\[([^\]]+)\]"), r"list[\1]"),
    (re.compile(r"\bOptional\[([^\[\]]+)\]"), r"\1 | None"),
]


def _modernize_type_hints(src: str) -> str:
    for pattern, repl in _MODERNIZE_SUBS:
        src = pattern.sub(repl, src)
    return src


def _source_of(obj) -> str:
    return _modernize_type_hints(textwrap.dedent(inspect.getsource(obj))).rstrip("\n")


def _generate_shared_block(model_type: ModelType) -> str:
    from deepvac import mpc as _mpc
    from deepvac.schemas import DEFAULT_FEATURE_NAMES

    spec = get_model_spec(model_type)
    plant_model_class = spec.plant_model_class
    plant_model_name = plant_model_class.__name__

    feature_names_literal = json.dumps(DEFAULT_FEATURE_NAMES, indent=4)[1:-1]  # strip outer [ ]

    parts = [
        f'"""{plant_model_name} plant model + CODESYS-style PID/Diff simulation for the Simulator view.\n\n'
        "Runs a synthetic closed-loop scenario (start_temp -> target_temp) entirely\n"
        "in-process: the trained model predicts the next temperature, and a Python\n"
        "reimplementation of the CODESYS ChamberPID/Diff blocks drives the control\n"
        "signal that feeds back into the model on the next step.\n\n"
        "The block above GENERATED_END_MARKER is regenerated by\n"
        "`deepvac package-model` from this repo's deepvac/mpc.py, deepvac/schemas.py,\n"
        f"and {plant_model_class.__module__} -- do not hand-edit that section, it will\n"
        "be overwritten. Everything below the marker is this app's own Simulator-view\n"
        'logic and is preserved as-is by that command.\n"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "import types",
        "from collections.abc import Sequence",
        "from pathlib import Path",
        "from typing import Any, Protocol, cast",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "class _Scaler(Protocol):",
        '    """Shape checkpoint["x_scaler"]/["y_scaler"] are expected to have --',
        "    matches sklearn.preprocessing.StandardScaler and the minimal stub",
        "    registered by _ensure_sklearn_stub() below. checkpoint is typed as",
        "    dict[str, object] (its values are heterogeneous: tensors, ints, these",
        "    scalers, ...), so this Protocol + cast() is how callers get a typed",
        '    view of the two entries they actually call methods on."""',
        "",
        "    def transform(self, X: np.ndarray) -> np.ndarray: ...",
        "    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...",
        "",
        "",
        "# compute_metrics()/simulate_candidate()'s per-candidate result: mostly",
        "# float, but candidate_id is int, valid is bool, and invalid_reason is str.",
        "Metrics = dict[str, float | int | bool | str]",
        "",
        "",
        _source_of(gru_common._ensure_sklearn_stub),
        "",
        'DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model.pt"',
        "",
        f"DEFAULT_FEATURE_NAMES = [{feature_names_literal}]",
        "",
        "",
        _source_of(plant_model_class),
        "",
        f"PlantModel = {plant_model_name}",
        "",
        _source_of(gru_common.limit),
        "",
        _source_of(gru_common.CodesysDiff),
        "",
        _source_of(gru_common.ChamberPID),
        "",
        "",
        "def load_model(",
        "    checkpoint_path: Path,",
        "    device: torch.device,",
        ") -> tuple[PlantModel, dict[str, object]]:",
        "    _ensure_sklearn_stub()",
        "    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)",
        "",
        "    model = PlantModel(",
        '        input_dim=int(checkpoint["input_dim"]),',
        '        hidden_dim=int(checkpoint["hidden_dim"]),',
        '        num_layers=int(checkpoint["num_layers"]),',
        '        dropout=float(checkpoint["dropout"]),',
        "    ).to(device)",
        "",
        '    model.load_state_dict(checkpoint["model_state_dict"])',
        "    model.eval()",
        "",
        "    return model, checkpoint",
        "",
        "",
        "def predict_delta_t1(",
        "    model: PlantModel,",
        "    checkpoint: dict[str, object],",
        "    feature_window: np.ndarray,",
        "    device: torch.device,",
        ") -> float:",
        '    x_scaler = cast(_Scaler, checkpoint["x_scaler"])',
        '    y_scaler = cast(_Scaler, checkpoint["y_scaler"])',
        "",
        "    n_features = feature_window.shape[-1]",
        "    x_scaled = x_scaler.transform(feature_window.reshape(-1, n_features)).reshape(",
        "        feature_window.shape",
        "    )",
        "",
        "    xb = torch.as_tensor(x_scaled[None, :, :], dtype=torch.float32, device=device)",
        "",
        "    with torch.no_grad():",
        "        pred_scaled = model(xb).cpu().numpy()",
        "",
        "    pred_real = y_scaler.inverse_transform(pred_scaled)",
        "    return float(pred_real[0, 0])",
        "",
        "",
        _source_of(_mpc.make_feature_row),
        "",
        _source_of(_mpc.initialize_feature_window),
        "",
        # deepvac.mpc.run_pid_substeps takes temp_end/temp_mode (it also
        # supports interpolating toward a future temperature; used by MPC
        # rollouts). The Simulator view below only ever holds temp_start and
        # has no caller for those two parameters, so this keeps its original,
        # simpler call signature and forwards into the canonical
        # implementation with temp_mode="hold" (temp_end is unused in that
        # mode) rather than changing simulate_candidate()'s call site.
        _source_of(_mpc.run_pid_substeps).replace(
            "def run_pid_substeps(", "def _run_pid_substeps_impl("
        ),
        "",
        "",
        "def run_pid_substeps(",
        "    *,",
        "    pid: Any,",
        "    diff: Any,",
        "    temp_start: float,",
        "    temp_ref: float,",
        "    kp: float,",
        "    ki: float,",
        "    kd: float,",
        "    dt_s: float,",
        "    period_s: float,",
        "    feature_scale: float,",
        ") -> dict[str, float]:",
        "    return _run_pid_substeps_impl(",
        "        pid=pid,",
        "        diff=diff,",
        "        temp_start=temp_start,",
        "        temp_end=temp_start,",
        "        temp_ref=temp_ref,",
        "        kp=kp,",
        "        ki=ki,",
        "        kd=kd,",
        "        dt_s=dt_s,",
        "        period_s=period_s,",
        "        feature_scale=feature_scale,",
        '        temp_mode="hold",',
        "    )",
        "",
        "",
        "def predict_next(",
        "    *,",
        "    model: PlantModel,",
        "    checkpoint: dict[str, object],",
        "    feature_window: np.ndarray,",
        "    feature_names: Sequence[str],",
        "    device: torch.device,",
        "    temp: float,",
        "    previous_temp: float,",
        "    temp_ref: float,",
        "    dt_s: float,",
        "    terms: dict[str, float],",
        "    kp: float,",
        "    ki: float,",
        "    kd: float,",
        ") -> tuple[float, float, np.ndarray]:",
        "    local_window = feature_window.copy()",
        "    local_window[-1, :] = make_feature_row(",
        "        feature_names,",
        "        temp=temp,",
        "        temp_ref=temp_ref,",
        "        previous_temp=previous_temp,",
        "        dt_s=dt_s,",
        '        u=terms["u"],',
        '        u_p=terms["u_p"],',
        '        u_i=terms["u_i"],',
        '        u_d=terms["u_d"],',
        "        kp=kp,",
        "        ki=ki,",
        "        kd=kd,",
        "    )",
        "    delta = predict_delta_t1(model, checkpoint, local_window, device)",
        "    return float(temp) + float(delta), float(delta), local_window",
        "",
        "",
        GENERATED_END_MARKER,
    ]
    return "\n".join(parts) + "\n"


def stage_insight(
    checkpoint_path: Path, model_type: ModelType, output_dir: Path, insight_root: Path | None
) -> dict[str, Path]:
    """Stage model.pt and a regenerated simulation.py under output_dir. insight_root is
    only read, never written -- it supplies the existing simulation.py's hand-maintained
    tail, which is carried forward into the newly generated file."""
    existing = ""
    if insight_root is not None:
        existing_sim_path = insight_root / "app" / "model" / "simulation.py"
        if existing_sim_path.exists():
            existing = existing_sim_path.read_text(encoding="utf-8")

    marker_idx = existing.find(GENERATED_END_MARKER)
    if marker_idx == -1:
        raise RuntimeError(
            f"Could not find an existing simulation.py with a {GENERATED_END_MARKER!r} "
            "marker (looked under --insight-root's app/model/). Add the marker once, right "
            "after predict_next()'s definition and before simulate_candidate(), then re-run -- "
            "everything after the marker is preserved as-is on every future run. If this is a "
            "first-time setup, pass --insight-root pointing at the checkout that already has "
            "simulate_candidate()/compute_metrics() written by hand."
        )
    preserved_tail = existing[marker_idx + len(GENERATED_END_MARKER):]

    output_dir.mkdir(parents=True, exist_ok=True)

    model_pt_path = output_dir / "model.pt"
    model_pt_path.write_bytes(checkpoint_path.read_bytes())

    sim_path = output_dir / "simulation.py"
    sim_path.write_text(_generate_shared_block(model_type) + preserved_tail, encoding="utf-8")

    return {"model_pt": model_pt_path, "simulation_py": sim_path}
