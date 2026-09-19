"""Ultralytics YOLO backend.

**This is the only module in the project permitted to import ultralytics.**

Ultralytics is AGPL-3.0, which is viral over network use. Confining it here
gives the licence obligation a clear boundary, and swapping in RT-DETR, YOLOX
or a permissively licensed model is a new file implementing
:class:`~ppe.core.protocols.Detector` plus a one-line config change.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from ppe.config.schema import ModelConfig
from ppe.core.errors import DetectorError
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import Detection, Frame
from ppe.detection.base import resolve_device, sha256_of, to_detections


class UltralyticsDetector:
    """Wraps an Ultralytics model behind the ``Detector`` protocol."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._model = self._load()

    def _load(self) -> Any:
        from ultralytics import YOLO

        weights = Path(self._config.weights)
        if not weights.exists():
            msg = (
                f"weights not found at {weights}. Train a baseline first "
                f"(python scripts/train.py), or point model.weights elsewhere."
            )
            raise DetectorError(msg)
        try:
            return YOLO(str(weights))
        except Exception as exc:
            msg = f"could not load ultralytics weights from {weights}: {exc}"
            raise DetectorError(msg) from exc

    @cached_property
    def metadata(self) -> DetectorMetadata:
        """Identity and settings of the loaded model.

        Hashed once: the evaluation harness embeds this in every report, so a
        metric can never be quoted without the exact weights behind it.
        """
        return DetectorMetadata(
            name="ultralytics",
            weights=str(self._config.weights),
            weights_sha256=sha256_of(self._config.weights),
            imgsz=self._config.imgsz,
            conf_threshold=self._config.conf_threshold,
            iou_threshold=self._config.iou_threshold,
            device=self._device,
            class_map=dict(self._config.class_map),
        )

    def predict(self, frame: Frame) -> list[Detection]:
        """Detect objects in a single BGR frame."""
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self._config.imgsz,
                conf=self._config.conf_threshold,
                iou=self._config.iou_threshold,
                max_det=self._config.max_detections,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            msg = f"ultralytics inference failed: {exc}"
            raise DetectorError(msg) from exc

        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        return to_detections(
            boxes_xyxy=boxes.xyxy.cpu().numpy().astype(np.float64),
            class_ids=boxes.cls.cpu().numpy().astype(int),
            confidences=boxes.conf.cpu().numpy().astype(np.float64),
            class_map=self._config.class_map,
            conf_threshold=self._config.conf_threshold,
        )
