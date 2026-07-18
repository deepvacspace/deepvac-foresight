#!/usr/bin/env python3
"""Package a trained checkpoint for the insight (PySide6) and/or
control2-client (C++ Qt) desktop apps. See deepvac/packaging.py for what
each target actually receives.

Examples:

    # Both apps, using the default sibling-directory layout on this machine:
    deepvac package-model --checkpoint gru/validation_t1/gru_t1.pt --target insight,control2-client

    # Just the ONNX export, to a custom location:
    deepvac package-model --checkpoint lstm/validation_t1/lstm_t1.pt --model-type lstm \
        --target control2-client --control2-client-root D:\\path\\to\\control2-client
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deepvac.packaging import export_onnx, resolve_model_type, sync_pyside6_app, verify_onnx

# This repo (scripts/) is expected to sit at <root>/deepvac/scripts, with the
# PySide6 app at <root>/deepvac/insight and the C++ Qt app at
# <root>/control2-client. Override with --insight-root/--control2-client-root
# if your checkout is laid out differently.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_DEEPVAC_DIR = _SCRIPTS_DIR.parent
_DEFAULT_INSIGHT_ROOT = _DEEPVAC_DIR / "insight"
_DEFAULT_CONTROL2_ROOT = _DEEPVAC_DIR.parent / "control2-client"

VALID_TARGETS = ("insight", "control2-client")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Package a trained GRU/LSTM checkpoint for the insight (PySide6) "
        "and/or control2-client (C++ Qt) desktop apps."
    )
    ap.add_argument("--checkpoint", required=True, help="Path to a trained gru_t1.pt / lstm_t1.pt checkpoint.")
    ap.add_argument("--model-type", choices=["gru", "lstm"], default=None,
                    help="Defaults to inferring 'gru' or 'lstm' from the checkpoint path.")
    ap.add_argument("--target", required=True,
                    help=f"Comma-separated list of targets: {', '.join(VALID_TARGETS)}.")
    ap.add_argument("--insight-root", default=str(_DEFAULT_INSIGHT_ROOT),
                    help="Path to the insight (PySide6) app checkout.")
    ap.add_argument("--control2-client-root", default=str(_DEFAULT_CONTROL2_ROOT),
                    help="Path to the control2-client (C++ Qt) app checkout.")
    ap.add_argument("--control2-client-data-subdir", default="data/model",
                    help="Where under --control2-client-root to write the ONNX export.")
    ap.add_argument("--no-verify-onnx", action="store_true",
                    help="Skip the numeric parity check between the PyTorch model and the exported ONNX graph.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    unknown = [t for t in targets if t not in VALID_TARGETS]
    if unknown:
        raise ValueError(f"Unknown target(s) {unknown}; valid targets are {VALID_TARGETS}")

    model_type = resolve_model_type(checkpoint_path, args.model_type)
    print(f"checkpoint: {checkpoint_path}")
    print(f"model type: {model_type}")
    print(f"targets:    {targets}")

    if "control2-client" in targets:
        control2_root = Path(args.control2_client_root)
        if not control2_root.is_dir():
            raise FileNotFoundError(f"control2-client root not found: {control2_root}")
        out_dir = control2_root / args.control2_client_data_subdir
        paths = export_onnx(checkpoint_path, model_type, out_dir)
        print(f"\n[control2-client] wrote {paths['onnx']}")
        print(f"[control2-client] wrote {paths['metadata']}")
        if not args.no_verify_onnx:
            diff = verify_onnx(paths["onnx"], checkpoint_path, model_type)
            print(f"[control2-client] verified: PyTorch vs ONNX max abs diff = {diff:.3e}")

    if "insight" in targets:
        insight_root = Path(args.insight_root)
        if not insight_root.is_dir():
            raise FileNotFoundError(f"insight root not found: {insight_root}")
        paths = sync_pyside6_app(checkpoint_path, model_type, insight_root)
        print(f"\n[insight] wrote {paths['model_pt']}")
        print(f"[insight] regenerated {paths['simulation_py']}")


if __name__ == "__main__":
    main()
