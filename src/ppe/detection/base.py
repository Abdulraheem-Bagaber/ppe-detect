"""Helpers shared by every detector backend.

Backend-agnostic: no ultralytics, no onnxruntime. Concrete backends call these
so the class-index -> vocabulary translation and confidence filtering happen
identically no matter which model produced the raw output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ppe.core.types import BBox, Detection, PPEClass


def sha256_of(path: str | Path) -> str:
    """Hash a weights file so a reported metric can name the exact model."""
    raise NotImplementedError


def resolve_device(requested: str) -> str:
    """Turn ``'auto'`` into ``'cuda'`` or ``'cpu'``.

    CPU is the default assumption everywhere; CUDA is used only when actually
    present, so the same code path runs on a laptop and in the CPU-only image.
    """
    raise NotImplementedError


def to_detections(
    boxes_xyxy: npt.NDArray[np.floating],
    class_ids: npt.NDArray[np.integer],
    confidences: npt.NDArray[np.floating],
    class_map: Mapping[int, PPEClass],
    conf_threshold: float,
) -> list[Detection]:
    """Translate raw model output into our vocabulary.

    Class indices absent from ``class_map`` are dropped rather than guessed.
    """
    raise NotImplementedError


def filter_by_class(detections: Sequence[Detection], label: PPEClass) -> list[Detection]:
    """All detections of one class, order preserved."""
    raise NotImplementedError


def letterbox_to_original(box: BBox, scale: float, pad_x: float, pad_y: float) -> BBox:
    """Undo letterbox padding, mapping a box back to original image pixels.

    Needed by the ONNX path in Phase 6, where preprocessing is ours rather than
    the framework's.
    """
    raise NotImplementedError
