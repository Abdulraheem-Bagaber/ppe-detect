"""Ultralytics YOLO backend.

**This is the only module in the project permitted to import ultralytics.**

Ultralytics is AGPL-3.0. Confining it here gives the licence obligation a clear
boundary, and swapping in RT-DETR, YOLOX or a permissively licensed model is a
new file implementing :class:`~ppe.core.protocols.Detector` plus a one-line
config change -- nothing else in the codebase moves.

Implemented in Phase 2.
"""

from __future__ import annotations

from ppe.config.schema import ModelConfig
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import Detection, Frame


class UltralyticsDetector:
    """Wraps an Ultralytics model behind the ``Detector`` protocol."""

    def __init__(self, config: ModelConfig) -> None:
        raise NotImplementedError

    @property
    def metadata(self) -> DetectorMetadata:
        raise NotImplementedError

    def predict(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError
