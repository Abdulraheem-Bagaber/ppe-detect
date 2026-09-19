"""Tests for the evaluation harness.

The harness is the thing that decides whether the project works, so it needs
to be at least as trustworthy as what it measures. Everything here is
synthetic: hand-built events, hand-built frame results, hand-written ground
truth. No video, no model.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppe.core.errors import ConfigError
from ppe.core.types import (
    AssociationMatch,
    BBox,
    Detection,
    PersonAssessment,
    PPEClass,
    Track,
    ViolationEvent,
)
from ppe.eval.diagnosis import FalsePositiveCause, diagnose, summarise
from ppe.eval.groundtruth import (
    ClipGroundTruth,
    EvaluationSet,
    ViolationInterval,
    load_ground_truth,
)
from ppe.eval.report import format_report, to_json
from ppe.eval.scoring import aggregate, event_matches_interval, score_clip
from ppe.pipeline.runner import FrameResult

PERSON_BOX = BBox(400.0, 200.0, 500.0, 500.0)


def event(
    timestamp_s: float = 10.0,
    missing: frozenset[PPEClass] = frozenset({PPEClass.HELMET}),
    track_id: int = 1,
    first_frame: int = 300,
    frame: int = 315,
    box: BBox = PERSON_BOX,
) -> ViolationEvent:
    return ViolationEvent(
        track_id=track_id,
        missing=missing,
        first_frame_index=first_frame,
        frame_index=frame,
        timestamp_s=timestamp_s,
        person_box=box,
        streak_length=15,
        mean_person_confidence=0.9,
    )


def interval(
    start_s: float = 5.0,
    end_s: float = 20.0,
    missing: list[PPEClass] | None = None,
    box_hint: tuple[float, float, float, float] | None = None,
) -> ViolationInterval:
    return ViolationInterval(
        person_label="worker in blue",
        missing=missing or [PPEClass.HELMET],
        start_s=start_s,
        end_s=end_s,
        box_hint=box_hint,
    )


class TestGroundTruth:
    def test_an_interval_knows_its_duration(self) -> None:
        assert interval(5.0, 20.0).duration_s == 15.0

    def test_covers_respects_tolerance(self) -> None:
        i = interval(5.0, 20.0)
        assert i.covers(21.0, tolerance_s=2.0)
        assert not i.covers(21.0, tolerance_s=0.0)

    def test_a_backwards_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be after"):
            ViolationInterval(person_label="w", missing=[PPEClass.HELMET], start_s=20.0, end_s=5.0)

    def test_an_interval_needs_at_least_one_missing_item(self) -> None:
        with pytest.raises(ValueError):
            ViolationInterval(person_label="w", missing=[], start_s=1.0, end_s=2.0)

    def test_a_violation_after_the_clip_ends_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="start after the clip ends"):
            ClipGroundTruth(clip="a.mp4", duration_s=10.0, violations=[interval(60.0, 70.0)])

    def test_totals_across_a_set(self) -> None:
        clips = EvaluationSet(
            clips=[
                ClipGroundTruth(clip="a.mp4", duration_s=60.0, violations=[interval()]),
                ClipGroundTruth(clip="b.mp4", duration_s=120.0, violations=[]),
            ]
        )
        assert clips.total_duration_s == 180.0
        assert clips.total_violations == 1

    def test_a_missing_file_explains_the_gap(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="hand-annotated"):
            load_ground_truth(tmp_path / "nope.yaml")

    def test_a_real_file_round_trips(self, tmp_path) -> None:
        path = tmp_path / "gt.yaml"
        path.write_text(
            "name: site-a\n"
            "clips:\n"
            "  - clip: clips/a.mp4\n"
            "    duration_s: 120.0\n"
            "    violations:\n"
            "      - person_label: blue jacket\n"
            "        missing: [helmet]\n"
            "        start_s: 12.0\n"
            "        end_s: 41.0\n",
            encoding="utf-8",
        )
        loaded = load_ground_truth(path)
        assert loaded.clips[0].violations[0].missing == [PPEClass.HELMET]


class TestMatching:
    def test_an_event_inside_the_interval_matches(self) -> None:
        assert event_matches_interval(event(timestamp_s=10.0), interval(5.0, 20.0))

    def test_an_event_outside_the_interval_does_not(self) -> None:
        assert not event_matches_interval(event(timestamp_s=90.0), interval(5.0, 20.0))

    def test_debounce_delay_is_forgiven_by_the_tolerance(self) -> None:
        """An event fires after the streak confirms, so it lands late by design."""
        assert event_matches_interval(event(timestamp_s=21.0), interval(5.0, 20.0))

    def test_the_wrong_missing_item_does_not_match(self) -> None:
        bare_headed = event(missing=frozenset({PPEClass.HELMET}))
        no_vest_annotated = interval(missing=[PPEClass.VEST])
        assert not event_matches_interval(bare_headed, no_vest_annotated)

    def test_overlapping_missing_items_do_match(self) -> None:
        both = event(missing=frozenset({PPEClass.HELMET, PPEClass.VEST}))
        assert event_matches_interval(both, interval(missing=[PPEClass.VEST]))

    def test_a_box_hint_disambiguates_two_workers(self) -> None:
        elsewhere = interval(box_hint=(0.0, 0.0, 50.0, 100.0))
        assert not event_matches_interval(event(), elsewhere)
        here = interval(box_hint=(400.0, 200.0, 500.0, 500.0))
        assert event_matches_interval(event(), here)


class TestScoring:
    def _clip(self, *intervals: ViolationInterval) -> ClipGroundTruth:
        return ClipGroundTruth(clip="a.mp4", duration_s=600.0, violations=list(intervals))

    def test_a_detected_violation_is_a_true_positive(self) -> None:
        score = score_clip(self._clip(interval()), [event()])
        assert score.detected_intervals == 1
        assert score.false_positives == []
        assert score.missed == []

    def test_an_unannotated_event_is_a_false_alarm(self) -> None:
        score = score_clip(self._clip(), [event()])
        assert len(score.false_positives) == 1
        assert score.detected_intervals == 0

    def test_an_unreported_violation_is_a_miss(self) -> None:
        score = score_clip(self._clip(interval()), [])
        assert len(score.missed) == 1

    def test_two_events_for_one_violation_count_once_as_detected(self) -> None:
        """A cooldown re-fire is one thing found twice, not one hit plus a lie."""
        score = score_clip(self._clip(interval(5.0, 60.0)), [event(10.0), event(50.0)])
        assert score.detected_intervals == 1
        assert score.false_positives == []

    def test_metrics_are_measured_from_the_matches(self) -> None:
        score = score_clip(self._clip(interval(), interval(100.0, 130.0)), [event(), event(200.0)])
        metrics = aggregate([score])
        assert metrics.intervals_total == 2
        assert metrics.intervals_detected == 1
        assert metrics.false_positive_events == 1
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.f1 == pytest.approx(0.5)

    def test_false_alarms_per_hour_uses_the_annotated_duration(self) -> None:
        clip = ClipGroundTruth(clip="a.mp4", duration_s=1800.0)  # 30 minutes
        metrics = aggregate([score_clip(clip, [event(), event(200.0)])])
        assert metrics.false_alarms_per_hour == pytest.approx(4.0)

    def test_metrics_with_no_data_are_none_not_zero(self) -> None:
        """A precision of zero and an unmeasured precision are different claims."""
        metrics = aggregate([score_clip(ClipGroundTruth(clip="a.mp4", duration_s=60.0), [])])
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1 is None


def frame_result(
    index: int,
    detections: list[Detection],
    tracks: list[Track],
    assessments: list[PersonAssessment],
) -> FrameResult:
    return FrameResult(
        frame_index=index,
        timestamp_s=index / 30.0,
        detections=detections,
        tracks=tracks,
        assessments=assessments,
    )


def person_detection(box: BBox = PERSON_BOX) -> Detection:
    return Detection(box, PPEClass.PERSON, 0.9)


class TestDiagnosis:
    """The false-positive breakdown: the most valuable output in the project."""

    def test_an_undetected_helmet_is_a_detection_fault(self) -> None:
        frames = [
            frame_result(i, [person_detection()], [], [PersonAssessment(person_detection())])
            for i in range(300, 316)
        ]
        result = diagnose(event(), frames)
        assert result.cause is FalsePositiveCause.MISSED_PPE_DETECTION
        assert not result.is_fixable_in_config

    def test_a_detected_but_unattributed_helmet_is_an_association_fault(self) -> None:
        """Sitting right on the person's head, never matched: tune the config."""
        worn = Detection(BBox(430.0, 205.0, 470.0, 235.0), PPEClass.HELMET, 0.8)
        frames = [
            frame_result(
                i,
                [person_detection(), worn],
                [],
                [PersonAssessment(person_detection())],
            )
            for i in range(300, 316)
        ]
        result = diagnose(event(), frames)
        assert result.cause is FalsePositiveCause.ASSOCIATION_FAILURE
        assert result.is_fixable_in_config

    def test_an_associated_helmet_elsewhere_is_not_an_association_fault(self) -> None:
        """Someone else's helmet, correctly not borrowed: the annotation differs."""
        theirs = Detection(BBox(430.0, 205.0, 470.0, 235.0), PPEClass.HELMET, 0.8)
        matched = PersonAssessment(
            person=person_detection(),
            matches={PPEClass.HELMET: AssociationMatch(theirs, 0.9, "helmet")},
        )
        frames = [
            frame_result(i, [person_detection(), theirs], [], [matched]) for i in range(300, 316)
        ]
        assert diagnose(event(), frames).cause is FalsePositiveCause.ANNOTATION_DISAGREEMENT

    def test_an_identity_change_is_a_tracking_fault(self) -> None:
        frames = []
        for i in range(300, 316):
            track_id = 1 if i < 308 else 2
            frames.append(
                frame_result(
                    i,
                    [person_detection()],
                    [Track(track_id=track_id, detection=person_detection(), frames_seen=1)],
                    [PersonAssessment(person_detection())],
                )
            )
        result = diagnose(event(), frames)
        assert result.cause is FalsePositiveCause.TRACKING_ID_SWITCH
        assert result.is_fixable_in_config

    def test_a_stable_identity_is_not_blamed_on_tracking(self) -> None:
        frames = [
            frame_result(
                i,
                [person_detection()],
                [Track(track_id=1, detection=person_detection(), frames_seen=1)],
                [PersonAssessment(person_detection())],
            )
            for i in range(300, 316)
        ]
        assert diagnose(event(), frames).cause is not FalsePositiveCause.TRACKING_ID_SWITCH

    def test_no_recorded_frames_is_admitted_rather_than_guessed(self) -> None:
        assert diagnose(event(), []).cause is FalsePositiveCause.ANNOTATION_DISAGREEMENT

    def test_summarise_counts_by_cause(self) -> None:
        frames = [
            frame_result(i, [person_detection()], [], [PersonAssessment(person_detection())])
            for i in range(300, 316)
        ]
        counts = summarise([diagnose(event(), frames), diagnose(event(), frames)])
        assert counts["missed_ppe_detection"] == 2


class TestReport:
    def _bits(self):
        clip = ClipGroundTruth(clip="a.mp4", duration_s=3600.0, violations=[interval()])
        score = score_clip(clip, [event(), event(900.0)])
        frames = [
            frame_result(i, [person_detection()], [], [PersonAssessment(person_detection())])
            for i in range(300, 316)
        ]
        diagnoses = [diagnose(m.event, frames) for m in score.false_positives]
        return aggregate([score]), [score], diagnoses

    def test_the_report_leads_with_violation_metrics(self) -> None:
        text = format_report(*self._bits())
        assert "Violation-level metrics" in text
        assert "detection mAP is not" in text

    def test_the_report_explains_every_false_alarm(self) -> None:
        text = format_report(*self._bits())
        assert "Why each false alarm happened" in text
        assert "missed_ppe_detection" in text

    def test_unmeasurable_metrics_print_as_not_available(self) -> None:
        clip = ClipGroundTruth(clip="a.mp4", duration_s=60.0)
        text = format_report(aggregate([score_clip(clip, [])]), [], [])
        assert "n/a" in text

    def test_json_output_is_diffable(self) -> None:
        metrics, scores, diagnoses = self._bits()
        payload = to_json(metrics, scores, diagnoses)
        assert payload["metrics"]["intervals_total"] == 1
        assert payload["false_positive_causes"]
        assert payload["false_positives"][0]["cause"]


class TestNoFabrication:
    """The project's first rule, enforced: never invent a number."""

    def test_an_empty_evaluation_reports_nothing_rather_than_zero(self) -> None:
        metrics = aggregate([])
        assert metrics.precision is None
        assert metrics.false_alarms_per_hour is None

    def test_frames_are_real_arrays_not_implied(self) -> None:
        assert np.zeros((4, 4, 3), dtype=np.uint8).shape == (4, 4, 3)
