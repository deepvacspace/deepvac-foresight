"""Integration test for deepvac/packaging.py's ONNX export (the
control2-client / C++ Qt target of `deepvac package-model`, see
DEV_GUIDE.md §7). Marked slow: torch's ONNX exporter has noticeable
per-call overhead. Skipped entirely if the `package` extra (onnx,
onnxruntime) isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

pytestmark = pytest.mark.slow

from tests.conftest import build_tiny_gru_checkpoint  # noqa: E402


def test_export_onnx_matches_pytorch_numerically(tmp_path: Path):
    from deepvac.packaging import export_onnx, verify_onnx

    checkpoint_path = build_tiny_gru_checkpoint(tmp_path / "gru" / "validation_t1" / "gru_t1.pt")
    out_dir = tmp_path / "onnx_out"

    paths = export_onnx(checkpoint_path, "gru", out_dir)
    assert paths["onnx"].exists()
    assert paths["metadata"].exists()

    diff = verify_onnx(paths["onnx"], checkpoint_path, "gru")
    assert diff < 1e-3


def test_export_onnx_metadata_describes_the_expected_input(tmp_path: Path):
    import json

    from deepvac.packaging import export_onnx

    checkpoint_path = build_tiny_gru_checkpoint(tmp_path / "gru" / "validation_t1" / "gru_t1.pt", window_steps=5)
    paths = export_onnx(checkpoint_path, "gru", tmp_path / "onnx_out")

    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["model_type"] == "gru"
    assert metadata["window_steps"] == 5
    assert metadata["input_shape"] == ["batch", 5, 10]
