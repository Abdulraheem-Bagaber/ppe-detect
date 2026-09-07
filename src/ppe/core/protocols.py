"""Structural interfaces between layers.

These are the seams. ``Detector`` in particular exists because Ultralytics is
AGPL-3.0: every ultralytics import in this codebase lives behind this protocol
in exactly one module, so swapping in RT-DETR, YOLOX or a plain ONNX graph is a
new file plus a config change, not a refactor.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ppe.core.types import Detection, Frame, PPEClass, Track


@dataclass(frozen=True, slots=True)
class DetectorMetadata:
    """Everything needed to reproduce a number this detector produced.

    The evaluation harness embeds this verbatim in its report, so a metric can
    never be quoted without the exact weights and thresholds behind it.
    """

    name: str
    weights: str
    weights_sha256: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    device: str
    class_map: Mapping[int, PPEClass]


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a frame into detections in our own vocabulary."""

    @property
    def metadata(self) -> DetectorMetadata:
        """Identity and settings of the loaded model."""
        ...

    def predict(self, frame: Frame) -> list[Detection]:
        """Detect objects in a single BGR frame."""
        ...


@runtime_checkable
class Tracker(Protocol):
    """Assigns persistent identities to person detections across frames."""

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Track]:
        """Advance the tracker by one frame and return the live tracks."""
        ...

    def reset(self) -> None:
        """Forget all state. Called between clips so runs stay independent."""
        ...


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """A frame plus the timing metadata the debouncer needs."""

    index: int
    timestamp_s: float
    image: Frame


@runtime_checkable
class FrameSource(Protocol):
    """A video file, an RTSP stream, or a directory of images."""

    @property
    def fps(self) -> float:
        """Nominal frames per second; used to convert frame counts to seconds."""
        ...

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        ...

    def __iter__(self) -> Iterator[SourceFrame]:
        """Yield frames in order until the source is exhausted."""
        ...

    def close(self) -> None:
        """Release the underlying capture handle."""
        ...
