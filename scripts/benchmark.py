"""Latency and throughput: PyTorch vs ONNX vs INT8 ONNX.

    python scripts/benchmark.py --weights runs/train/baseline/weights/best.pt

Measures what is actually installed on the machine it runs on and refuses to
report anything it could not measure. A backend whose weights are missing is
listed as unavailable, not as a blank row that a reader will fill in with
optimism.

Reports p50 and p95 rather than a mean. A mean latency hides the occasional
slow frame that decides whether a live stream keeps up.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ppe.config import load_config
from ppe.config.schema import ModelConfig
from ppe.core.errors import PPEError
from ppe.core.protocols import Detector

WARMUP_FRAMES = 5


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One backend, measured. ``latencies_ms`` empty means it did not run."""

    backend: str
    device: str
    available: bool
    reason: str = ""
    latencies_ms: tuple[float, ...] = ()

    @property
    def p50(self) -> float | None:
        return statistics.median(self.latencies_ms) if self.latencies_ms else None

    @property
    def p95(self) -> float | None:
        if len(self.latencies_ms) < 2:
            return None
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def fps(self) -> float | None:
        p50 = self.p50
        return 1000.0 / p50 if p50 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default=None, help="PyTorch weights (.pt).")
    parser.add_argument("--onnx", default=None, help="Exported ONNX model.")
    parser.add_argument("--onnx-int8", default=None, help="Quantised ONNX model.")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--out", default="reports/benchmark.json")
    parser.add_argument("--gpu", action="store_true", help="Also measure CUDA if present.")
    return parser.parse_args(argv)


def synthetic_frames(count: int, size: int, seed: int) -> list[np.ndarray]:
    """Deterministic frames, so two runs on the same machine are comparable."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(count)]


def measure(detector: Detector, frames: list[np.ndarray]) -> tuple[float, ...]:
    """Time one predict call per frame, after a warmup."""
    for frame in frames[:WARMUP_FRAMES]:
        detector.predict(frame)

    latencies: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        detector.predict(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return tuple(latencies)


def run_backend(label: str, config: ModelConfig, frames: list[np.ndarray]) -> BenchmarkResult:
    """Build one backend and time it, or record why it could not run."""
    if not Path(config.weights).exists():
        return BenchmarkResult(
            backend=label,
            device=config.device,
            available=False,
            reason=f"weights not found at {config.weights}",
        )
    try:
        from ppe.detection.factory import build_detector

        detector = build_detector(config)
        latencies = measure(detector, frames)
    except PPEError as exc:
        return BenchmarkResult(label, config.device, available=False, reason=str(exc))
    except Exception as exc:  # a backend that crashes is a result too
        return BenchmarkResult(label, config.device, available=False, reason=repr(exc))

    return BenchmarkResult(
        backend=label,
        device=detector.metadata.device,
        available=True,
        latencies_ms=latencies,
    )


def format_table(results: list[BenchmarkResult]) -> str:
    header = f"{'backend':<18}{'device':<8}{'p50 ms':>10}{'p95 ms':>10}{'FPS':>8}"
    lines = [header, "-" * len(header)]
    for result in results:
        if not result.available:
            lines.append(f"{result.backend:<18}{result.device:<8}{'not measured':>28}")
            continue
        lines.append(
            f"{result.backend:<18}{result.device:<8}"
            f"{result.p50:>10.1f}{result.p95:>10.1f}{result.fps:>8.1f}"
        )
    lines.append("")
    for result in results:
        if not result.available:
            lines.append(f"  {result.backend}: {result.reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    frames = synthetic_frames(args.frames, config.model.imgsz, config.seed)

    plans: list[tuple[str, ModelConfig]] = []

    torch_weights = args.weights or config.model.weights
    plans.append(
        (
            "pytorch-cpu",
            config.model.model_copy(
                update={"backend": "ultralytics", "weights": torch_weights, "device": "cpu"}
            ),
        )
    )
    if args.onnx:
        plans.append(
            (
                "onnx-cpu",
                config.model.model_copy(
                    update={"backend": "onnx", "weights": args.onnx, "device": "cpu"}
                ),
            )
        )
    if args.onnx_int8:
        plans.append(
            (
                "onnx-int8-cpu",
                config.model.model_copy(
                    update={"backend": "onnx", "weights": args.onnx_int8, "device": "cpu"}
                ),
            )
        )
    if args.gpu:
        plans.append(
            (
                "pytorch-cuda",
                config.model.model_copy(
                    update={"backend": "ultralytics", "weights": torch_weights, "device": "cuda"}
                ),
            )
        )

    results = []
    for label, model_config in plans:
        print(f"measuring {label}...")
        results.append(run_backend(label, model_config, frames))

    table = format_table(results)
    print()
    print(table)

    payload = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "frames": args.frames,
        "imgsz": config.model.imgsz,
        "results": [
            {
                "backend": r.backend,
                "device": r.device,
                "available": r.available,
                "reason": r.reason,
                "p50_ms": r.p50,
                "p95_ms": r.p95,
                "fps": r.fps,
            }
            for r in results
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")

    if not any(r.available for r in results):
        print("error: nothing could be measured; no numbers to report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
