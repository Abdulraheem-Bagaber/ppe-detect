"""Tests for the pipeline, using a scripted detector rather than a model.

The detector is a protocol, so the pipeline can be driven by a stub that
returns whatever detections a scenario needs. That means the end-to-end
behaviour -- tracking, zone filtering, debouncing, event emission -- is tested
without weights, without a video file and in milliseconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from ppe.config.schema import AppConfig, DebounceConfig, TrackingConfig, ZoneConfig, ZonesConfig
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import BBox, Detection, Frame, PPEClass, SkipReason, Track
from ppe.pipeline.runner import Pipeline
from ppe.pipeline.sources import ArraySource

FRAME_W, FRAME_H = 640, 480


def blank_frame() -> Frame:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def person(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), PPEClass.PERSON, conf)


def helmet(cx: float, cy: float) -> Detection:
    return Detection(BBox.from_cxcywh(cx, cy, 30.0, 22.0), PPEClass.HELMET, 0.8)


def vest(cx: float, cy: float) -> Detection:
    return Detection(BBox.from_cxcywh(cx, cy, 55.0, 70.0), PPEClass.VEST, 0.8)


WORKER = (280.0, 120.0, 360.0, 360.0)  # head band y in [120, 204]


class ScriptedDetector:
    """Returns a pre-written list of detections per frame."""

    def __init__(self, script: Sequence[Sequence[Detection]]) -> None:
        self._script = list(script)
        self.calls = 0

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name="scripted",
            weights="none",
            weights_sha256="0" * 64,
            imgsz=640,
            conf_threshold=0.0,
            iou_threshold=0.0,
            device="cpu",
            class_map={},
        )

    def predict(self, frame: Frame) -> list[Detection]:
        detections = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return list(detections)


class StickyTracker:
    """Gives every person the same id, so debounce behaviour is predictable."""

    def __init__(self, track_id: int = 1) -> None:
        self._track_id = track_id
        self._seen = 0

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Track]:
        people = [d for d in detections if d.label is PPEClass.PERSON]
        out = []
        for offset, detection in enumerate(people):
            self._seen += 1
            out.append(
                Track(
                    track_id=self._track_id + offset,
                    detection=detection,
                    frames_seen=self._seen,
                )
            )
        return out

    def reset(self) -> None:
        self._seen = 0


def config(**overrides: object) -> AppConfig:
    base: dict[str, object] = {
        "debounce": DebounceConfig(confirm_frames=3, cooldown_s=10.0),
        "tracking": TrackingConfig(enabled=True),
    }
    base.update(overrides)
    return AppConfig(**base)  # type: ignore[arg-type]


class TestProcessFrame:
    def test_a_compliant_worker_produces_no_events(self) -> None:
        detector = ScriptedDetector([[person(*WORKER), helmet(320.0, 150.0), vest(320.0, 220.0)]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        result = pipeline.process_frame(blank_frame(), 0, 0.0)
        assert result.events == []
        assert result.assessments[0].is_compliant

    def test_a_bare_headed_worker_is_a_violation_before_it_is_an_event(self) -> None:
        """One frame is a violation; it takes confirm_frames to become an event."""
        detector = ScriptedDetector([[person(*WORKER), vest(320.0, 220.0)]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        result = pipeline.process_frame(blank_frame(), 0, 0.0)
        assert result.violations
        assert result.events == []

    def test_a_sustained_violation_fires_an_event(self) -> None:
        detector = ScriptedDetector([[person(*WORKER), vest(320.0, 220.0)]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        events = [
            e for i in range(3) for e in pipeline.process_frame(blank_frame(), i, i / 30).events
        ]
        assert len(events) == 1
        assert events[0].missing == frozenset({PPEClass.HELMET})
        assert events[0].track_id == 1

    def test_detections_and_tracks_are_reported_for_inspection(self) -> None:
        detector = ScriptedDetector([[person(*WORKER), helmet(320.0, 150.0)]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        result = pipeline.process_frame(blank_frame(), 0, 0.0)
        assert len(result.detections) == 2
        assert len(result.tracks) == 1

    def test_an_empty_frame_is_handled(self) -> None:
        pipeline = Pipeline(config(), ScriptedDetector([[]]), StickyTracker())
        result = pipeline.process_frame(blank_frame(), 0, 0.0)
        assert result.assessments == []
        assert result.events == []

    def test_frame_size_comes_from_the_frame_not_config(self) -> None:
        """A cropped worker is judged against the real frame the detector saw."""
        cropped = person(280.0, -80.0, 360.0, 200.0)
        pipeline = Pipeline(config(), ScriptedDetector([[cropped]]), StickyTracker())
        result = pipeline.process_frame(blank_frame(), 0, 0.0)
        assert result.assessments[0].skip_reason is SkipReason.HEAD_OUT_OF_FRAME


class TestZones:
    def _zoned(self) -> AppConfig:
        # Left half of the frame only.
        return config(
            zones=ZonesConfig(
                zones=[
                    ZoneConfig(
                        zone_id="bay",
                        polygon=[(0.0, 0.0), (320.0, 0.0), (320.0, 480.0), (0.0, 480.0)],
                    )
                ]
            )
        )

    def test_a_worker_inside_the_zone_is_judged_and_tagged(self) -> None:
        inside = person(200.0, 120.0, 280.0, 360.0)
        pipeline = Pipeline(self._zoned(), ScriptedDetector([[inside]]), StickyTracker())
        assessment = pipeline.process_frame(blank_frame(), 0, 0.0).assessments[0]
        assert assessment.is_violation
        assert assessment.zone_id == "bay"

    def test_a_worker_outside_the_zone_is_not_judged(self) -> None:
        outside = person(400.0, 120.0, 480.0, 360.0)
        pipeline = Pipeline(self._zoned(), ScriptedDetector([[outside]]), StickyTracker())
        assessment = pipeline.process_frame(blank_frame(), 0, 0.0).assessments[0]
        assert not assessment.evaluated
        assert assessment.skip_reason is SkipReason.OUTSIDE_ZONE
        assert assessment.missing == frozenset()

    def test_someone_outside_the_zone_never_raises_an_event(self) -> None:
        outside = person(400.0, 120.0, 480.0, 360.0)
        pipeline = Pipeline(self._zoned(), ScriptedDetector([[outside]]), StickyTracker())
        events = [
            e for i in range(20) for e in pipeline.process_frame(blank_frame(), i, i / 30).events
        ]
        assert events == []


class TestRun:
    def test_runs_a_whole_source(self) -> None:
        detector = ScriptedDetector([[person(*WORKER), vest(320.0, 220.0)]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        source = ArraySource([blank_frame() for _ in range(6)])
        results = list(pipeline.run(source))
        assert len(results) == 6
        assert sum(len(r.events) for r in results) == 1

    def test_timestamps_follow_the_source_frame_rate(self) -> None:
        detector = ScriptedDetector([[]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        results = list(pipeline.run(ArraySource([blank_frame() for _ in range(3)], fps=10.0)))
        assert [r.timestamp_s for r in results] == pytest.approx([0.0, 0.1, 0.2])

    def test_state_is_reset_between_runs(self) -> None:
        """Two clips must not share a debounce streak, or clip two fires early."""
        detector = ScriptedDetector([[person(*WORKER), vest(320.0, 220.0)]])
        pipeline = Pipeline(config(), detector, StickyTracker())
        first = list(pipeline.run(ArraySource([blank_frame(), blank_frame()])))
        assert sum(len(r.events) for r in first) == 0
        second = list(pipeline.run(ArraySource([blank_frame(), blank_frame()])))
        assert sum(len(r.events) for r in second) == 0

    def test_process_frames_matches_run(self) -> None:
        script = [[person(*WORKER), vest(320.0, 220.0)]]
        a = Pipeline(config(), ScriptedDetector(script), StickyTracker())
        b = Pipeline(config(), ScriptedDetector(script), StickyTracker())
        via_run = [len(r.events) for r in a.run(ArraySource([blank_frame() for _ in range(5)]))]
        via_list = [
            len(r.events)
            for r in b.process_frames([(i, i / 30.0, blank_frame()) for i in range(5)])
        ]
        assert via_run == via_list


class TestArraySource:
    def test_reports_its_frame_size(self) -> None:
        assert ArraySource([blank_frame()]).frame_size == (FRAME_W, FRAME_H)

    def test_an_empty_source_has_no_size(self) -> None:
        assert ArraySource([]).frame_size == (0, 0)

    def test_rejects_a_nonsense_frame_rate(self) -> None:
        from ppe.core.errors import VideoSourceError

        with pytest.raises(VideoSourceError, match="fps must be positive"):
            ArraySource([blank_frame()], fps=0.0)


class TestScriptedDetectorContract:
    """Guards the stub itself: a lying stub makes every test above worthless."""

    def test_it_satisfies_the_detector_protocol(self) -> None:
        from ppe.core.protocols import Detector

        assert isinstance(ScriptedDetector([[]]), Detector)

    def test_metadata_is_a_mapping_of_class_ids(self) -> None:
        assert isinstance(ScriptedDetector([[]]).metadata.class_map, Mapping)
