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

# Max absolute difference tolerated between PyTorch and ONNX raw output.
# fp32 export is near-exact; anything above this is a real behavioural change,
# not numerical noise.
DEFAULT_TOLERANCE = 1e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="Defaults to model.weights.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="Defaults to the weights path with .onnx.")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
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
) -> tuple[float, list[float]]:
    """Run both models over the same frames; return max and per-frame deltas."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    deltas: list[float] = []
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
        deltas.append(float(np.max(np.abs(torch_np - onnx_out))))

    return max(deltas), deltas


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
        max_delta, deltas = compare(
            model, onnx_path, random_frames(args.samples, imgsz, config.seed), imgsz
        )
        report["parity"] = {
            "max_abs_diff": max_delta,
            "tolerance": args.tolerance,
            "per_frame": deltas,
            "passed": max_delta <= args.tolerance,
        }
        print(f"  max |torch - onnx| = {max_delta:.3e}  (tolerance {args.tolerance:.0e})")

        if max_delta > args.tolerance:
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
