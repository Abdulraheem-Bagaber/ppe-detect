"""Export to ONNX and verify output parity against PyTorch.

    python scripts/export_onnx.py --weights runs/train/baseline/weights/best.pt

The export is the easy part. The point of this script is the *check*: an ONNX
graph whose preprocessing or NMS differs from the PyTorch model silently
invalidates every metric measured on the original weights, and nothing about
the export itself would tell you.

Parity is asserted on raw tensor output, before our own decoding, so a
mismatch points at the export rather than at our NMS.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ppe.config import load_config
from ppe.detection.base import letterbox, sha256_of

# YOLOv8 emits (1, 4 + num_classes, anchors): four box coordinates followed by
# one score per class. Those two halves live on completely different scales, so
# a single absolute tolerance is meaningless for one of them.
#
# Box coordinates are in pixels on a 0-640 image, so the honest unit is "how
# far did the box move". A twentieth of a pixel is far tighter than anything
# that could affect a detection, and still leaves room for fp32 rounding.
DEFAULT_BOX_TOLERANCE_PX = 0.05

# Class scores are already in 0-1, where 1e-4 is a real change in confidence.
# Measured fp32 export error is around 1e-7, so this has plenty of headroom.
DEFAULT_SCORE_TOLERANCE = 1e-4

# Number of box-coordinate channels at the start of the output tensor.
BOX_CHANNELS = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="Defaults to model.weights.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="Defaults to the weights path with .onnx.")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--box-tolerance-px",
        type=float,
        default=DEFAULT_BOX_TOLERANCE_PX,
        help="Max box-coordinate drift in pixels.",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=DEFAULT_SCORE_TOLERANCE,
        help="Max class-score drift, on the 0-1 scale.",
    )
    parser.add_argument("--samples", type=int, default=8, help="Random frames to compare.")
    parser.add_argument("--skip-parity", action="store_true", help="Not for a released model.")
    return parser.parse_args(argv)


def random_frames(count: int, imgsz: int, seed: int) -> list[np.ndarray]:
    """Deterministic pseudo-random frames for the parity comparison.

    Random noise rather than real images on purpose: it exercises the whole
    output tensor instead of the handful of anchors a real photo activates.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(imgsz, imgsz, 3), dtype=np.uint8) for _ in range(count)]


def compare(
    torch_model: object, onnx_path: Path, frames: list[np.ndarray], imgsz: int
) -> tuple[float, float]:
    """Run both models over the same frames.

    Returns the worst box-coordinate drift in pixels and the worst class-score
    drift, reported separately because they are measured in different units
    and only one of them can be judged against a 0-1 threshold.
    """
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    worst_box = 0.0
    worst_score = 0.0
    for frame in frames:
        padded, _, _, _ = letterbox(frame, imgsz)
        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch = np.ascontiguousarray(tensor[None])

        onnx_out = session.run(None, {input_name: batch})[0]
        with torch.no_grad():
            torch_out = torch_model.model(torch.from_numpy(batch))  # type: ignore[attr-defined]
        if isinstance(torch_out, list | tuple):
            torch_out = torch_out[0]
        torch_np = torch_out.cpu().numpy()

        if torch_np.shape != onnx_out.shape:
            msg = f"shape mismatch: torch {torch_np.shape} vs onnx {onnx_out.shape}"
            raise SystemExit(f"error: {msg}")

        difference = np.abs(torch_np - onnx_out)
        worst_box = max(worst_box, float(difference[:, :BOX_CHANNELS].max()))
        worst_score = max(worst_score, float(difference[:, BOX_CHANNELS:].max()))

    return worst_box, worst_score


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    weights = Path(args.weights or config.model.weights)
    if not weights.exists():
        print(f"error: weights not found at {weights}. Train a model first.", file=sys.stderr)
        return 1

    imgsz = config.model.imgsz
    from ultralytics import YOLO

    model = YOLO(str(weights))
    print(f"exporting {weights} at imgsz={imgsz}, opset={args.opset}...")
    exported = model.export(format="onnx", imgsz=imgsz, opset=args.opset, dynamic=False)
    onnx_path = Path(args.out) if args.out else Path(exported)
    if args.out and Path(exported) != onnx_path:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        Path(exported).replace(onnx_path)

    print(f"wrote {onnx_path}")

    report: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_weights": str(weights),
        "source_sha256": sha256_of(weights),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_of(onnx_path),
        "imgsz": imgsz,
        "opset": args.opset,
    }

    if args.skip_parity:
        print("WARNING: parity check skipped; do not ship this model.")
        report["parity"] = "skipped"
    else:
        print(f"checking parity over {args.samples} frames...")
        box_drift, score_drift = compare(
            model, onnx_path, random_frames(args.samples, imgsz, config.seed), imgsz
        )
        passed = box_drift <= args.box_tolerance_px and score_drift <= args.score_tolerance
        report["parity"] = {
            "max_box_drift_px": box_drift,
            "box_tolerance_px": args.box_tolerance_px,
            "max_score_drift": score_drift,
            "score_tolerance": args.score_tolerance,
            "samples": args.samples,
            "passed": passed,
        }
        print(
            f"  box coordinates: {box_drift:.3e} px "
            f"(tolerance {args.box_tolerance_px} px)\n"
            f"  class scores   : {score_drift:.3e} "
            f"(tolerance {args.score_tolerance:.0e})"
        )

        if not passed:
            print(
                "error: the ONNX model does not match the PyTorch model.\n"
                "       Every metric measured on the original weights would be a\n"
                "       claim about a different model. Not writing a pass.",
                file=sys.stderr,
            )
            _write_report(onnx_path, report)
            return 1
        print("  parity OK")

    _write_report(onnx_path, report)
    return 0


def _write_report(onnx_path: Path, report: dict[str, object]) -> None:
    path = onnx_path.with_suffix(".export.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
