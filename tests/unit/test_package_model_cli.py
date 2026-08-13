"""Unit tests for deepvac/package_model.py's argument validation (the
`deepvac package-model` CLI). The packaging work itself is covered by
tests/integration/test_package_model_onnx.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from deepvac.package_model import VALID_TARGETS, build_arg_parser
from deepvac.packaging import resolve_model_type


def test_valid_targets_are_insight_and_control2_client():
    assert set(VALID_TARGETS) == {"insight", "control2-client"}


def test_build_arg_parser_requires_checkpoint_and_target():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_arg_parser_accepts_minimal_args():
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "gru/validation_t1/gru_t1.pt", "--target", "insight"])
    assert args.checkpoint == "gru/validation_t1/gru_t1.pt"
    assert args.target == "insight"
    assert args.model_type is None


class TestResolveModelType:
    def test_infers_gru_from_path(self):
        assert resolve_model_type(Path("gru/validation_t1/gru_t1.pt"), None) == "gru"

    def test_infers_lstm_from_path(self):
        assert resolve_model_type(Path("lstm/validation_t1/lstm_t1.pt"), None) == "lstm"

    def test_explicit_type_overrides_inference(self):
        assert resolve_model_type(Path("gru/validation_t1/gru_t1.pt"), "lstm") == "lstm"

    def test_rejects_invalid_explicit_type(self):
        with pytest.raises(ValueError):
            resolve_model_type(Path("gru/gru_t1.pt"), "not-a-model-type")

    def test_rejects_ambiguous_path(self):
        with pytest.raises(ValueError):
            resolve_model_type(Path("checkpoints/model.pt"), None)

    def test_rejects_path_containing_both(self):
        with pytest.raises(ValueError):
            resolve_model_type(Path("gru/lstm/model.pt"), None)
