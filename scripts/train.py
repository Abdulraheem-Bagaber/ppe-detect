"""Fine-tune the baseline detector with a fixed seed.

    python scripts/train.py --data data/processed/worker-safety/data.yaml

Deliberately does no tuning. The point of Phase 2 is a *documented* baseline:
one configuration, one seed, recorded alongside the weights so the number can
be reproduced and later improvements can be shown to be improvements.

Every run writes a run manifest next to the weights containing the resolved
config, the git commit, the dataset manifest digest, and the measured metrics.
Nothing in that file is ever hand-written.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ppe.config import load_config
from ppe.data.mapping import CLASS_ORDER
from ppe.detection.base import resolve_device, sha256_of

DEFAULT_DATA = "data/processed/worker-safety/data.yaml"
BASE_WEIGHTS = "yolov8n.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset data.yaml to train on.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-weights", default=BASE_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=None, help="Defaults to model.imgsz.")
    parser.add_argument("--device", default=None, help="Defaults to model.device.")
    parser.add_argument("--name", default=None, help="Run name under runs/.")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Train on this fraction of the data. For smoke runs only.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train without the held-out test evaluation (not for a reported baseline).",
    )
    return parser.parse_args(argv)


def set_seeds(seed: int) -> None:
    """Pin every source of randomness we control."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def git_commit() -> str:
    """The commit the weights were produced from, or ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def warn_if_gpu_is_being_wasted(device: str) -> None:
    """Shout if there is an NVIDIA card here but torch cannot use it.

    This exact situation cost an hour once: `uv run` re-syncs from uv.lock,
    which pins the CPU build, silently undoing `make gpu`. Training then runs
    ~30x slower with no error anywhere. A CPU run is a legitimate choice, but
    it should never be an accident.
    """
    if device != "cpu":
        return

    import shutil

    import torch

    if getattr(torch.version, "cuda", None) is not None:
        return  # CUDA build present; the machine genuinely has no usable GPU
    if shutil.which("nvidia-smi") is None:
        return  # no NVIDIA hardware; CPU is simply correct here

    print(
        "\nWARNING: an NVIDIA GPU is present but this is a CPU-only torch build,\n"
        "         so training will be roughly 30x slower. Fix with:\n"
        "             make gpu      (Windows: .\\make.ps1 gpu)\n"
        "         then re-run with --no-sync, because plain `uv run` reverts it.\n",
        file=sys.stderr,
    )


def per_class_metrics(results: Any, names: dict[int, str]) -> dict[str, dict[str, float]]:
    """Pull per-class mAP out of an ultralytics validation result.

    Returns measured numbers only. If a class is absent from the split it is
    reported as absent rather than as zero, because those mean different
    things and only one of them is the model's fault.
    """
    box = results.box
    out: dict[str, dict[str, float]] = {}

    for position, class_index in enumerate(box.ap_class_index):
        name = names.get(int(class_index), str(class_index))
        out[name] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.ap[position]),
        }

    for name in (c.value for c in CLASS_ORDER):
        out.setdefault(name, {"absent_from_split": 1.0})

    out["all"] = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    return out


def format_table(metrics: dict[str, dict[str, float]]) -> str:
    header = f"{'class':<10}{'mAP@50':>10}{'mAP@50-95':>12}{'precision':>12}{'recall':>10}"
    lines = [header, "-" * len(header)]
    for name, values in metrics.items():
        if "absent_from_split" in values:
            lines.append(f"{name:<10}{'absent from split':>44}")
            continue
        lines.append(
            f"{name:<10}{values['map50']:>10.3f}{values['map50_95']:>12.3f}"
            f"{values['precision']:>12.3f}{values['recall']:>10.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"error: dataset not found at {data_path}. Run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        return 1

    config = load_config(args.config)
    set_seeds(config.seed)

    imgsz = args.imgsz if args.imgsz is not None else config.model.imgsz
    device = resolve_device(args.device if args.device is not None else config.model.device)
    run_name = args.name or f"baseline-{datetime.now(UTC):%Y%m%d-%H%M%S}"

    warn_if_gpu_is_being_wasted(device)
    print(f"device={device}  imgsz={imgsz}  epochs={args.epochs}  seed={config.seed}")

    from ultralytics import YOLO

    model = YOLO(args.base_weights)
    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=device,
        seed=config.seed,
        deterministic=True,
        project=str(Path(args.project).resolve()),
        name=run_name,
        exist_ok=True,
        verbose=False,
        plots=True,
        fraction=args.fraction,
    )

    # Ultralytics resolves a relative `project` under its own runs_dir, so the
    # directory is read back from the trainer rather than reconstructed.
    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    names = {i: c.value for i, c in enumerate(CLASS_ORDER)}

    metrics: dict[str, dict[str, float]] = {}
    if not args.skip_test:
        print("\nEvaluating on the held-out test split...")
        test_results = model.val(
            data=str(data_path.resolve()),
            split="test",
            imgsz=imgsz,
            device=device,
            project=str(Path(args.project).resolve()),
            name=f"{run_name}-test",
            exist_ok=True,
            verbose=False,
        )
        metrics = per_class_metrics(test_results, names)
        print()
        print(format_table(metrics))

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "seed": config.seed,
        "base_weights": args.base_weights,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": imgsz,
            "fraction": args.fraction,
        },
        "dataset": {
            "data_yaml": str(data_path),
            "contents": yaml.safe_load(data_path.read_text(encoding="utf-8")),
        },
        "weights": {
            "path": str(best_weights),
            "sha256": sha256_of(best_weights),
        },
        "test_metrics": metrics or "not evaluated",
        "classes": [c.value for c in CLASS_ORDER],
    }

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"\nweights:  {best_weights}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
