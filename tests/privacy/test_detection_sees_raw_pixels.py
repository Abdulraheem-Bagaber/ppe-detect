"""Detection must run on the unblurred frame. Blurring is an output concern.

This is the most load-bearing ordering in the system and the easiest to get
backwards, because "blur the frame, then process it" reads as the more
careful option. It is not: measured on real footage, running the detector on
blurred frames costs **53% of all detections** (533 -> 249 over 144 sampled
frames). Faces sit on heads, heads sit on people, and destroying that region
destroys the person detections that everything else depends on.

The correct order is: detect on raw pixels, blur on the way out. Privacy then
costs nothing in accuracy. These tests pin that order in place, because a
future refactor that "tidies up" by blurring earlier would halve the system's
recall and no existing test would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppe.api.render import annotate
from ppe.config.schema import AppConfig, PrivacyConfig
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import BBox, Detection, Frame, PersonAssessment, PPEClass
from ppe.pipeline.runner import FrameResult, Pipeline

PERSON_BOX = BBox(60.0, 40.0, 140.0, 280.0)


def textured_frame() -> Frame:
    """Noise, so blurring is detectable as a collapse in local variance."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)


class SpyDetector:
    """Records exactly which pixels it was handed."""

    def __init__(self) -> None:
        self.seen: list[Frame] = []

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name="spy",
            weights="none",
            weights_sha256="0" * 64,
            imgsz=320,
            conf_threshold=0.0,
            iou_threshold=0.0,
            device="cpu",
            class_map={},
        )

    def predict(self, frame: Frame) -> list[Detection]:
        self.seen.append(frame.copy())
        return [Detection(PERSON_BOX, PPEClass.PERSON, 0.9)]


class NoTracker:
    def update(self, detections, frame_index):
        from ppe.core.types import Track

        return [
            Track(track_id=1, detection=d, frames_seen=1)
            for d in detections
            if d.label is PPEClass.PERSON
        ]

    def reset(self) -> None:
        return None


class TestPipelineOrdering:
    def test_the_detector_is_handed_the_raw_frame(self) -> None:
        """The whole guarantee, asserted directly."""
        frame = textured_frame()
        spy = SpyDetector()
        Pipeline(AppConfig(), spy, NoTracker()).process_frame(frame, 0, 0.0)

        assert len(spy.seen) == 1
        assert np.array_equal(spy.seen[0], frame), (
            "the detector received modified pixels; if that is a blur, recall "
            "drops by roughly half and nothing else will tell you"
        )

    def test_head_detail_survives_into_the_detector(self) -> None:
        """A weaker but independent check, in case identity comparison is fooled."""
        frame = textured_frame()
        spy = SpyDetector()
        Pipeline(AppConfig(), spy, NoTracker()).process_frame(frame, 0, 0.0)

        head = PERSON_BOX.band(0.0, AppConfig().privacy.head_band_bottom)
        y1, y2, x1, x2 = int(head.y1), int(head.y2), int(head.x1), int(head.x2)
        assert spy.seen[0][y1:y2, x1:x2].std() == pytest.approx(frame[y1:y2, x1:x2].std())

    def test_blurring_stays_on_even_while_detection_sees_raw(self) -> None:
        """Both halves must hold at once: raw in, blurred out."""
        frame = textured_frame()
        spy = SpyDetector()
        config = AppConfig()
        result = Pipeline(config, spy, NoTracker()).process_frame(frame, 0, 0.0)

        assert np.array_equal(spy.seen[0], frame)

        rendered = annotate(frame, result, config)
        head = PERSON_BOX.band(0.0, config.privacy.head_band_bottom)
        y1, y2 = int(head.y1) + 2, int(head.y2) - 2
        x1, x2 = int(head.x1) + 2, int(head.x2) - 2
        assert rendered[y1:y2, x1:x2].std() < frame[y1:y2, x1:x2].std() / 2

    def test_processing_a_frame_does_not_mutate_it(self) -> None:
        """A caller reusing the buffer must not get a blurred frame back."""
        frame = textured_frame()
        snapshot = frame.copy()
        Pipeline(AppConfig(), SpyDetector(), NoTracker()).process_frame(frame, 0, 0.0)
        assert np.array_equal(frame, snapshot)

    def test_ordering_holds_even_with_blurring_disabled(self) -> None:
        config = AppConfig(privacy=PrivacyConfig(blur_faces=False))
        frame = textured_frame()
        spy = SpyDetector()
        Pipeline(config, spy, NoTracker()).process_frame(frame, 0, 0.0)
        assert np.array_equal(spy.seen[0], frame)


class TestWhyThisMatters:
    """Demonstrates the cost of getting the order wrong, on synthetic data."""

    def test_blurring_destroys_the_region_detection_depends_on(self) -> None:
        from ppe.privacy.blur import anonymise

        frame = textured_frame()
        person = Detection(PERSON_BOX, PPEClass.PERSON, 0.9)
        blurred = anonymise(frame, [person], AppConfig().privacy)

        head = PERSON_BOX.band(0.0, AppConfig().privacy.head_band_bottom)
        y1, y2 = int(head.y1) + 2, int(head.y2) - 2
        x1, x2 = int(head.x1) + 2, int(head.x2) - 2

        before = frame[y1:y2, x1:x2].std()
        after = blurred[y1:y2, x1:x2].std()
        assert after < before / 2, (
            "if this ever stops being true the blur is not working; if the "
            "detector ever sees this region, recall halves"
        )


def test_render_receives_the_raw_frame_and_blurs_it_itself() -> None:
    """annotate() owns the blurring, so no caller can forget to do it."""
    frame = textured_frame()
    result = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        detections=[Detection(PERSON_BOX, PPEClass.PERSON, 0.9)],
        tracks=[],
        assessments=[PersonAssessment(Detection(PERSON_BOX, PPEClass.PERSON, 0.9))],
    )
    rendered = annotate(frame, result, AppConfig())
    assert not np.array_equal(rendered, frame)
