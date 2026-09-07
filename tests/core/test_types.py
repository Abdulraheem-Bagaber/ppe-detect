"""Tests for the shared value objects.

These are the only fully implemented pieces of Phase 0, so they get real tests
now. Everything downstream assumes the invariants asserted here.
"""

from __future__ import annotations

import math

import pytest

from ppe.core.errors import GeometryError
from ppe.core.types import (
    AssociationMatch,
    BBox,
    Detection,
    PersonAssessment,
    PPEClass,
    SkipReason,
    Track,
    ViolationEvent,
)


class TestBBoxInvariants:
    def test_accepts_a_well_formed_box(self) -> None:
        box = BBox(10.0, 20.0, 110.0, 220.0)
        assert box.width == 100.0
        assert box.height == 200.0
        assert box.area == 20_000.0
        assert box.centre == (60.0, 120.0)

    def test_zero_area_box_is_legal(self) -> None:
        """Degenerate but not inverted: a clipped box can legitimately collapse."""
        assert BBox(5.0, 5.0, 5.0, 5.0).area == 0.0

    @pytest.mark.parametrize(
        "coords",
        [(10.0, 0.0, 5.0, 10.0), (0.0, 10.0, 10.0, 5.0)],
        ids=["x2<x1", "y2<y1"],
    )
    def test_rejects_inverted_boxes(self, coords: tuple[float, ...]) -> None:
        with pytest.raises(GeometryError, match="negative extent"):
            BBox(*coords)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_coordinates(self, bad: float) -> None:
        with pytest.raises(GeometryError, match="non-finite"):
            BBox(0.0, 0.0, bad, 10.0)

    def test_is_immutable(self) -> None:
        box = BBox(0.0, 0.0, 10.0, 10.0)
        with pytest.raises((AttributeError, TypeError)):
            box.x1 = 5.0  # type: ignore[misc]

    def test_is_hashable_and_value_equal(self) -> None:
        assert BBox(0.0, 0.0, 1.0, 1.0) == BBox(0.0, 0.0, 1.0, 1.0)
        assert len({BBox(0.0, 0.0, 1.0, 1.0), BBox(0.0, 0.0, 1.0, 1.0)}) == 1


class TestBBoxConstruction:
    def test_from_xywh(self) -> None:
        assert BBox.from_xywh(10.0, 20.0, 30.0, 40.0) == BBox(10.0, 20.0, 40.0, 60.0)

    def test_from_cxcywh(self) -> None:
        assert BBox.from_cxcywh(50.0, 50.0, 20.0, 10.0) == BBox(40.0, 45.0, 60.0, 55.0)

    def test_round_trips_through_as_tuple(self) -> None:
        box = BBox(1.5, 2.5, 3.5, 4.5)
        assert BBox(*box.as_tuple()) == box


class TestBand:
    """The band is the heart of helmet association, so it is pinned down here."""

    def test_upper_third_of_a_person(self) -> None:
        person = BBox(100.0, 100.0, 200.0, 400.0)  # 300px tall
        band = person.band(0.0, 0.35)
        assert band.y1 == 100.0
        assert band.y2 == pytest.approx(205.0)
        assert (band.x1, band.x2) == (100.0, 200.0), "band spans the full width"

    def test_is_scale_invariant(self) -> None:
        """Identical relative geometry for a distant worker and a near one."""
        small = BBox(0.0, 0.0, 20.0, 80.0).band(0.0, 0.35)
        large = BBox(0.0, 0.0, 200.0, 800.0).band(0.0, 0.35)
        assert small.height / 80.0 == pytest.approx(large.height / 800.0)

    @pytest.mark.parametrize(
        ("top", "bottom"),
        [(0.5, 0.5), (0.6, 0.4), (-0.1, 0.5), (0.5, 1.1)],
        ids=["equal", "inverted", "below-zero", "above-one"],
    )
    def test_rejects_invalid_fractions(self, top: float, bottom: float) -> None:
        with pytest.raises(GeometryError, match="band fractions"):
            BBox(0.0, 0.0, 10.0, 10.0).band(top, bottom)


class TestClipAndBorders:
    def test_clip_to_frame(self) -> None:
        clipped = BBox(-20.0, -10.0, 700.0, 500.0).clip_to(640, 480)
        assert clipped == BBox(0.0, 0.0, 640.0, 480.0)

    def test_clip_leaves_interior_boxes_alone(self) -> None:
        box = BBox(10.0, 10.0, 100.0, 100.0)
        assert box.clip_to(640, 480) == box

    def test_detects_a_person_cropped_at_the_top(self) -> None:
        """The head may be outside the frame -- this must not become a violation."""
        person = BBox(300.0, 0.0, 400.0, 400.0)
        assert "top" in person.touching_borders(640, 480, margin=4.0)

    def test_interior_box_touches_nothing(self) -> None:
        assert BBox(100.0, 100.0, 200.0, 200.0).touching_borders(640, 480, 4.0) == frozenset()

    def test_margin_is_inclusive(self) -> None:
        box = BBox(4.0, 100.0, 200.0, 200.0)
        assert box.touching_borders(640, 480, margin=4.0) == frozenset({"left"})


class TestVocabulary:
    def test_there_is_no_negative_class(self) -> None:
        """Non-compliance is derived, never predicted. Guards the core decision."""
        names = {member.value for member in PPEClass}
        assert names == {"person", "helmet", "vest"}
        assert not any(name.startswith("no_") for name in names)

    def test_ppe_items_excludes_person(self) -> None:
        assert PPEClass.ppe_items() == frozenset({PPEClass.HELMET, PPEClass.VEST})

    def test_classes_are_yaml_friendly_strings(self) -> None:
        assert PPEClass("helmet") is PPEClass.HELMET
        assert f"{PPEClass.VEST}" == "vest"


class TestDetection:
    def test_holds_a_box_and_a_label(self) -> None:
        det = Detection(box=BBox(0.0, 0.0, 10.0, 10.0), label=PPEClass.HELMET, confidence=0.9)
        assert det.label is PPEClass.HELMET

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_rejects_out_of_range_confidence(self, confidence: float) -> None:
        with pytest.raises(GeometryError, match="confidence"):
            Detection(BBox(0.0, 0.0, 1.0, 1.0), PPEClass.PERSON, confidence)


class TestTrack:
    def test_delegates_box_and_confidence_to_its_detection(self) -> None:
        det = Detection(BBox(0.0, 0.0, 10.0, 20.0), PPEClass.PERSON, 0.8)
        track = Track(track_id=7, detection=det, frames_seen=3)
        assert track.box == det.box
        assert track.confidence == pytest.approx(0.8)


class TestPersonAssessment:
    def _person(self) -> Detection:
        return Detection(BBox(0.0, 0.0, 100.0, 300.0), PPEClass.PERSON, 0.9)

    def test_compliant_when_evaluated_and_nothing_missing(self) -> None:
        helmet = Detection(BBox(30.0, 0.0, 70.0, 40.0), PPEClass.HELMET, 0.8)
        assessment = PersonAssessment(
            person=self._person(),
            matches={PPEClass.HELMET: AssociationMatch(helmet, 0.91, "helmet_band")},
        )
        assert assessment.is_compliant
        assert not assessment.is_violation

    def test_violation_when_something_is_missing(self) -> None:
        assessment = PersonAssessment(person=self._person(), missing=frozenset({PPEClass.HELMET}))
        assert assessment.is_violation
        assert not assessment.is_compliant

    def test_unevaluated_is_neither_compliant_nor_a_violation(self) -> None:
        """The distinction the whole error breakdown depends on."""
        assessment = PersonAssessment(
            person=self._person(),
            missing=frozenset({PPEClass.HELMET}),
            evaluated=False,
            skip_reason=SkipReason.HEAD_OUT_OF_FRAME,
        )
        assert not assessment.is_violation
        assert not assessment.is_compliant
        assert assessment.skip_reason is SkipReason.HEAD_OUT_OF_FRAME

    def test_defaults_to_no_matches_without_shared_state(self) -> None:
        first = PersonAssessment(person=self._person())
        second = PersonAssessment(person=self._person())
        assert first.matches == {} and second.matches == {}
        assert first.matches is not second.matches


class TestViolationEvent:
    def _event(self, **overrides: object) -> ViolationEvent:
        kwargs: dict[str, object] = {
            "track_id": 3,
            "missing": frozenset({PPEClass.HELMET}),
            "first_frame_index": 100,
            "frame_index": 115,
            "timestamp_s": 3.83,
            "person_box": BBox(0.0, 0.0, 100.0, 300.0),
            "streak_length": 15,
            "mean_person_confidence": 0.87,
        }
        kwargs.update(overrides)
        return ViolationEvent(**kwargs)  # type: ignore[arg-type]

    def test_carries_the_streak_that_produced_it(self) -> None:
        event = self._event()
        assert event.frame_index - event.first_frame_index == event.streak_length
        assert event.zone_id is None

    def test_event_ids_are_unique_per_event(self) -> None:
        assert self._event().event_id != self._event().event_id
        assert len(self._event().event_id) == 32
