"""The pipeline: detect -> track -> zone-filter -> associate -> debounce.

The only module that knows about all the layers. Everything it composes is
independently testable, which is the whole reason the boundaries exist: this
file has almost no logic of its own, and the parts that would be hard to test
here are already tested where they live.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ppe.config.schema import AppConfig
from ppe.core.protocols import Detector, FrameSource, Tracker
from ppe.core.types import (
    Detection,
    Frame,
    PersonAssessment,
    PPEClass,
    SkipReason,
    Track,
    ViolationEvent,
)
from ppe.rules.association import associate
from ppe.rules.debounce import ViolationDebouncer
from ppe.rules.zones import in_scope, zone_for


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything one frame produced. The unit the eval harness replays."""

    frame_index: int
    timestamp_s: float
    detections: list[Detection]
    tracks: list[Track]
    assessments: list[PersonAssessment]
    events: list[ViolationEvent] = field(default_factory=list)

    @property
    def violations(self) -> list[PersonAssessment]:
        return [a for a in self.assessments if a.is_violation]


class Pipeline:
    """Composes a detector, a tracker and the rule layer."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        tracker: Tracker | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker if tracker is not None else self._default_tracker()
        self.debouncer = ViolationDebouncer(config.debounce)

    def _default_tracker(self) -> Tracker:
        from ppe.tracking.bytetrack import ByteTrackTracker, NullTracker

        if not self.config.tracking.enabled:
            return NullTracker()
        return ByteTrackTracker(self.config.tracking)

    def process_frame(self, frame: Frame, frame_index: int, timestamp_s: float) -> FrameResult:
        """Run one frame end to end."""
        height, width = frame.shape[:2]
        frame_size = (width, height)

        detections = self.detector.predict(frame)
        tracks = self.tracker.update(detections, frame_index)

        # Association works on the *tracked* person boxes rather than the raw
        # ones, so a person and their identity cannot disagree about geometry.
        items = [d for d in detections if d.label is not PPEClass.PERSON]
        people = [t.detection for t in tracks]
        track_ids: list[int | None] = [t.track_id for t in tracks]

        assessments = associate([*people, *items], self.config.association, frame_size, track_ids)
        assessments = [self._apply_zone(a) for a in assessments]

        events = self.debouncer.update(frame_index, timestamp_s, assessments)
        self.debouncer.prune(frame_index)

        return FrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            detections=detections,
            tracks=tracks,
            assessments=assessments,
            events=events,
        )

    def _apply_zone(self, assessment: PersonAssessment) -> PersonAssessment:
        """Mark people outside the work area as seen but not judged."""
        zones = self.config.zones
        if in_scope(assessment.person, zones):
            zone = zone_for(assessment.person, zones)
            if zone is None:
                return assessment
            return _replace(assessment, zone_id=zone.zone_id)
        return _replace(
            assessment,
            evaluated=False,
            skip_reason=SkipReason.OUTSIDE_ZONE,
            missing=frozenset(),
        )

    def run(self, source: FrameSource) -> Iterator[FrameResult]:
        """Stream results for an entire source, resetting state first."""
        self.reset()
        for source_frame in source:
            yield self.process_frame(
                source_frame.image, source_frame.index, source_frame.timestamp_s
            )

    def process_frames(self, frames: Sequence[tuple[int, float, Frame]]) -> list[FrameResult]:
        """Run a pre-loaded sequence. Convenience for the eval harness."""
        self.reset()
        return [self.process_frame(image, index, ts) for index, ts, image in frames]

    def reset(self) -> None:
        """Clear tracker and debouncer state between clips."""
        self.tracker.reset()
        self.debouncer.reset()


def _replace(assessment: PersonAssessment, **changes: object) -> PersonAssessment:
    """Frozen-dataclass update. Kept local rather than exported.

    ``dataclasses.replace`` does not play well with slotted frozen classes
    carrying a Mapping default, so the fields are spelled out.
    """
    fields: dict[str, object] = {
        "person": assessment.person,
        "matches": assessment.matches,
        "missing": assessment.missing,
        "track_id": assessment.track_id,
        "zone_id": assessment.zone_id,
        "evaluated": assessment.evaluated,
        "skip_reason": assessment.skip_reason,
    }
    fields.update(changes)
    return PersonAssessment(**fields)  # type: ignore[arg-type]
