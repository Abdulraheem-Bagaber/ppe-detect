"""The pipeline: detect -> track -> zone-filter -> associate -> debounce.

The only module that knows about all the layers. Everything it composes is
independently testable, which is the whole reason the boundaries exist.

Implemented in Phase 4.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ppe.config.schema import AppConfig
from ppe.core.protocols import Detector, FrameSource, Tracker
from ppe.core.types import Detection, Frame, PersonAssessment, Track, ViolationEvent


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything one frame produced. The unit the eval harness replays."""

    frame_index: int
    timestamp_s: float
    detections: list[Detection]
    tracks: list[Track]
    assessments: list[PersonAssessment]
    events: list[ViolationEvent] = field(default_factory=list)


class Pipeline:
    """Composes a detector, a tracker and the rule layer."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        tracker: Tracker | None = None,
    ) -> None:
        raise NotImplementedError

    def process_frame(self, frame: Frame, frame_index: int, timestamp_s: float) -> FrameResult:
        """Run one frame end to end."""
        raise NotImplementedError

    def run(self, source: FrameSource) -> Iterator[FrameResult]:
        """Stream results for an entire source, resetting state first."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear tracker and debouncer state between clips."""
        raise NotImplementedError
