"""Helpers shared by every detector backend.

Backend-agnostic: no ultralytics, no onnxruntime. Concrete backends call these
so the class-index -> vocabulary translation and confidence filtering happen
identically no matter which model produced the raw output.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ppe.core.types import BBox, Detection, PPEClass


def sha256_of(path: str | Path) -> str:
    """Hash a weights file so a reported metric can name the exact model.

    Returns ``"missing"`` rather than raising when the file is absent: a
    metadata call should not be the thing that brings down a health check.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    """Turn ``'auto'`` into ``'cuda'`` or ``'cpu'``.

    CPU is the default assumption everywhere; CUDA is used only when actually
    present, so the same code path runs on a laptop and in the CPU-only image.
    An explicit request is honoured as given -- if someone asks for ``cuda``
    on a machine without it, they should get the loud failure, not a silent
    downgrade that makes a benchmark meaningless.
    """
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_detections(
    boxes_xyxy: npt.NDArray[np.floating],
    class_ids: npt.NDArray[np.integer],
    confidences: npt.NDArray[np.floating],
    class_map: Mapping[int, PPEClass],
    conf_threshold: float,
) -> list[Detection]:
    """Translate raw model output into our vocabulary.

    Class indices absent from ``class_map`` are dropped rather than guessed:
    a model trained on more classes than we use should lose the extras here,
    visibly, rather than have them silently become helmets.
    """
    if len(boxes_xyxy) == 0:
        return []
    if not (len(boxes_xyxy) == len(class_ids) == len(confidences)):
        msg = (
            f"ragged detector output: {len(boxes_xyxy)} boxes, "
            f"{len(class_ids)} class ids, {len(confidences)} scores"
        )
        raise ValueError(msg)

    out: list[Detection] = []
    for box, class_id, confidence in zip(boxes_xyxy, class_ids, confidences, strict=True):
        label = class_map.get(int(class_id))
        if label is None or float(confidence) < conf_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        out.append(
            Detection(
                box=BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                label=label,
                confidence=min(max(float(confidence), 0.0), 1.0),
            )
        )
    return out


def filter_by_class(detections: Sequence[Detection], label: PPEClass) -> list[Detection]:
    """All detections of one class, order preserved."""
    return [d for d in detections if d.label is label]


def letterbox_to_original(box: BBox, scale: float, pad_x: float, pad_y: float) -> BBox:
    """Undo letterbox padding, mapping a box back to original image pixels.

    Needed by the ONNX path, where preprocessing is ours rather than the
    framework's.
    """
    if scale <= 0:
        msg = f"letterbox scale must be positive, got {scale}"
        raise ValueError(msg)
    return BBox(
        x1=(box.x1 - pad_x) / scale,
        y1=(box.y1 - pad_y) / scale,
        x2=(box.x2 - pad_x) / scale,
        y2=(box.y2 - pad_y) / scale,
    )


def letterbox(
    image: npt.NDArray[np.uint8], target: int
) -> tuple[npt.NDArray[np.uint8], float, float, float]:
    """Resize preserving aspect ratio and pad to a square ``target``.

    Returns the padded image plus the ``(scale, pad_x, pad_y)`` needed to map
    predictions back. Implemented with numpy rather than cv2 so the ONNX path
    can be tested without a video stack.
    """
    height, width = image.shape[:2]
    scale = min(target / width, target / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))

    ys = (np.arange(new_h) / scale).astype(int).clip(0, height - 1)
    xs = (np.arange(new_w) / scale).astype(int).clip(0, width - 1)
    resized = image[ys][:, xs]

    canvas = np.full((target, target, image.shape[2]), 114, dtype=np.uint8)
    pad_x = (target - new_w) / 2.0
    pad_y = (target - new_h) / 2.0
    top, left = int(pad_y), int(pad_x)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, pad_x, pad_y
