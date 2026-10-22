#!/usr/bin/env python3
"""Stage a trained checkpoint for the insight and control2-client desktop apps."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepvac.model_registry import registered_model_types
from deepvac.packaging import export_onnx, resolve_model_type, stage_insight, verify_onnx

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_DEEPVAC_DIR = _SCRIPTS_DIR.parent
_DEFAULT_INSIGHT_ROOT = _DEEPVAC_DIR / "insight"
_DEFAULT_OUTPUT_DIR = Path("packaging")

VALID_TARGETS = ("insight", "control2-client")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage a trained checkpoint for the insight (PySide6) and "
        "control2-client (C++ Qt) desktop apps into a local packaging folder."
    )
    checkpoint_group = ap.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", help="Path to a trained checkpoint (e.g. gru_t1.pt).")
    checkpoint_group.add_argument("--latest", metavar="DIR",
                    help="Use the newest *.pt file under this directory.")
    ap.add_argument("--model-type", choices=registered_model_types(), default=None,
                    help="Model family (default: read from the checkpoint).")
    ap.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
                    help="Folder to stage outputs under.")
    ap.add_argument("--insight-root", default=str(_DEFAULT_INSIGHT_ROOT),
                    help="Path to the insight app checkout (read-only).")
    ap.add_argument("--skip", action="append", choices=VALID_TARGETS, default=[],
                    help="Skip staging this target (repeatable).")
    ap.add_argument("--no-verify-onnx", action="store_true",
                    help="Skip the PyTorch vs ONNX parity check.")
    return ap


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    latest_dir = Path(args.latest)
    if not latest_dir.is_dir():
        raise FileNotFoundError(f"--latest directory not found: {latest_dir}")
    candidates = sorted(latest_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No *.pt checkpoints found under {latest_dir}")
    return candidates[0]


def main() -> None:
    args = build_arg_parser().parse_args()

    checkpoint_path = _resolve_checkpoint_path(args)

    targets = [t for t in VALID_TARGETS if t not in args.skip]
    if not targets:
        raise ValueError("--skip removed every target; nothing to stage.")

    model_type = resolve_model_type(checkpoint_path, args.model_type)
    output_dir = Path(args.output_dir)
    if args.latest:
        print(f"checkpoint: {checkpoint_path}  (newest *.pt under --latest {args.latest})")
    else:
        print(f"checkpoint: {checkpoint_path}")
    print(f"model type: {model_type}")
    print(f"output dir: {output_dir.resolve()}")
    print(f"targets:    {targets}")

    if "control2-client" in targets:
        paths = export_onnx(checkpoint_path, model_type, output_dir / "control2-client")
        print(f"\n[control2-client] wrote {paths['onnx']}")
        print(f"[control2-client] wrote {paths['metadata']}")
        if not args.no_verify_onnx:
            diff = verify_onnx(paths["onnx"], checkpoint_path, model_type)
            print(f"[control2-client] verified: PyTorch vs ONNX max abs diff = {diff:.3e}")
        print("[control2-client] move the folder above into <control2-client checkout>/data/model/")

    if "insight" in targets:
        insight_root = Path(args.insight_root)
        if not insight_root.is_dir():
            raise FileNotFoundError(
                f"--insight-root not found: {insight_root} (needed to source simulation.py's "
                "hand-maintained tail). Pass --skip insight to opt out."
            )
        paths = stage_insight(checkpoint_path, model_type, output_dir / "insight", insight_root)
        print(f"\n[insight] wrote {paths['model_pt']}")
        print(f"[insight] wrote {paths['simulation_py']}")
        print("[insight] move the folder above into <insight checkout>/app/model/")


if __name__ == "__main__":
    main()
