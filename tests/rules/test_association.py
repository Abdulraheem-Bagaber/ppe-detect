"""Tests for the association rules. Written before the implementation.

Every box here is hand-constructed. No model, no image, no video -- which is
the entire argument for keeping this layer separate. If any of these ever need
a fixture file or a trained detector, the boundary has been broken.

The scenarios that matter are the ones a naive implementation gets wrong, and
they each have a named test below:

* two workers shoulder to shoulder sharing one helmet between them
* a helmet carried in the hand at waist height
* a person cropped by the top of the frame
* a person occluded from the chest down
"""

from __future__ import annotations

import pytest

from ppe.config.schema import AssociationConfig, ItemRule
from ppe.core.types import BBox, Detection, PPEClass, SkipReason
from ppe.rules.association import (
    assign_one_to_one,
    associate,
    candidate_matches,
    evaluability,
)

FRAME = (1280, 720)


def person(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), PPEClass.PERSON, conf)


def helmet(cx: float, cy: float, size: float = 40.0, conf: float = 0.8) -> Detection:
    return Detection(BBox.from_cxcywh(cx, cy, size, size * 0.7), PPEClass.HELMET, conf)


def vest(cx: float, cy: float, w: float = 70.0, h: float = 90.0, conf: float = 0.8) -> Detection:
    return Detection(BBox.from_cxcywh(cx, cy, w, h), PPEClass.VEST, conf)


@pytest.fixture
def config() -> AssociationConfig:
    return AssociationConfig()


# A worker standing in mid-frame: 100px wide, 300px tall, well clear of edges.
# Head band is y in [200, 305]; torso band is y in [245, 425].
UPRIGHT = (400.0, 200.0, 500.0, 500.0)


class TestCandidateMatches:
    def test_a_worn_helmet_matches_its_wearer(self, config: AssociationConfig) -> None:
        p = person(*UPRIGHT)
        matches = candidate_matches(p, [helmet(450.0, 225.0)], PPEClass.HELMET, config)
        assert len(matches) == 1
        assert matches[0].rule == "helmet"
        assert matches[0].score > 0.0

    def test_a_helmet_carried_at_waist_height_is_not_worn(self, config: AssociationConfig) -> None:
        """The spec's hard case. Fully inside the person box, still not on their head."""
        p = person(*UPRIGHT)
        carried = helmet(450.0, 400.0)  # two-thirds down the body
        assert candidate_matches(p, [carried], PPEClass.HELMET, config) == []

    def test_a_helmet_on_the_ground_matches_nobody(self, config: AssociationConfig) -> None:
        p = person(*UPRIGHT)
        assert candidate_matches(p, [helmet(450.0, 600.0)], PPEClass.HELMET, config) == []

    def test_a_neighbours_helmet_is_rejected_on_centre_offset(
        self, config: AssociationConfig
    ) -> None:
        """Overlapping boxes alone must not let one worker borrow another's helmet."""
        p = person(*UPRIGHT)
        # Vertically in band and overlapping the person box, but far off-centre.
        assert candidate_matches(p, [helmet(505.0, 225.0)], PPEClass.HELMET, config) == []

    def test_a_low_confidence_item_is_ignored(self, config: AssociationConfig) -> None:
        p = person(*UPRIGHT)
        faint = helmet(450.0, 225.0, conf=0.05)
        assert candidate_matches(p, [faint], PPEClass.HELMET, config) == []

    def test_candidates_come_back_best_first(self, config: AssociationConfig) -> None:
        p = person(*UPRIGHT)
        dead_centre = helmet(450.0, 220.0)
        off_centre = helmet(470.0, 230.0)
        matches = candidate_matches(p, [off_centre, dead_centre], PPEClass.HELMET, config)
        assert len(matches) == 2
        assert matches[0].item is dead_centre
        assert matches[0].score >= matches[1].score

    def test_only_items_of_the_requested_class_are_considered(
        self, config: AssociationConfig
    ) -> None:
        p = person(*UPRIGHT)
        assert candidate_matches(p, [vest(450.0, 320.0)], PPEClass.HELMET, config) == []

    def test_a_vest_on_the_torso_matches(self, config: AssociationConfig) -> None:
        p = person(*UPRIGHT)
        matches = candidate_matches(p, [vest(450.0, 320.0)], PPEClass.VEST, config)
        assert len(matches) == 1


class TestOneToOneAssignment:
    def test_two_overlapping_workers_cannot_share_one_helmet(
        self, config: AssociationConfig
    ) -> None:
        """The single most important test in the project.

        Two workers standing shoulder to shoulder, boxes overlapping, and one
        helmet between them. A per-person argmax hands it to both and hides a
        real violation. Greedy one-to-one gives it to the better claimant and
        leaves the other bare-headed, which is the truth.
        """
        left = person(400.0, 200.0, 500.0, 500.0)
        right = person(470.0, 200.0, 570.0, 500.0)
        one_helmet = helmet(450.0, 225.0)  # clearly over the left worker

        assigned = assign_one_to_one([left, right], [one_helmet], PPEClass.HELMET, config)

        assert set(assigned) == {0}, "only the left worker should get the helmet"
        assert assigned[0].item is one_helmet

    def test_each_worker_keeps_their_own_helmet(self, config: AssociationConfig) -> None:
        left = person(300.0, 200.0, 400.0, 500.0)
        right = person(600.0, 200.0, 700.0, 500.0)
        left_helmet = helmet(350.0, 225.0)
        right_helmet = helmet(650.0, 225.0)

        assigned = assign_one_to_one(
            [left, right], [right_helmet, left_helmet], PPEClass.HELMET, config
        )

        assert assigned[0].item is left_helmet
        assert assigned[1].item is right_helmet

    def test_an_item_is_consumed_by_exactly_one_person(self, config: AssociationConfig) -> None:
        a = person(400.0, 200.0, 500.0, 500.0)
        b = person(410.0, 200.0, 510.0, 500.0)
        assigned = assign_one_to_one([a, b], [helmet(455.0, 225.0)], PPEClass.HELMET, config)
        items = [m.item for m in assigned.values()]
        assert len(items) == len({id(i) for i in items})

    def test_no_people_yields_no_assignments(self, config: AssociationConfig) -> None:
        assert assign_one_to_one([], [helmet(450.0, 225.0)], PPEClass.HELMET, config) == {}

    def test_no_items_yields_no_assignments(self, config: AssociationConfig) -> None:
        assert assign_one_to_one([person(*UPRIGHT)], [], PPEClass.HELMET, config) == {}


class TestEvaluability:
    def test_a_clear_upright_worker_is_judgeable(self, config: AssociationConfig) -> None:
        assert evaluability(person(*UPRIGHT), PPEClass.HELMET, config, FRAME) is None

    def test_a_person_cropped_at_the_top_is_not_judged_for_a_helmet(
        self, config: AssociationConfig
    ) -> None:
        """The spec's hard case. Their head may be out of shot entirely.

        Absence of evidence is not evidence of absence: this must record a skip
        reason, never a violation.
        """
        cropped = person(400.0, -120.0, 500.0, 300.0)
        assert evaluability(cropped, PPEClass.HELMET, config, FRAME) is SkipReason.HEAD_OUT_OF_FRAME

    def test_the_same_cropped_person_is_still_judged_for_a_vest(
        self, config: AssociationConfig
    ) -> None:
        """A missing head says nothing about a visible torso."""
        cropped = person(400.0, -60.0, 500.0, 300.0)
        assert evaluability(cropped, PPEClass.VEST, config, FRAME) is None

    def test_a_person_too_small_to_judge_is_skipped(self, config: AssociationConfig) -> None:
        tiny = person(400.0, 300.0, 415.0, 340.0)
        assert evaluability(tiny, PPEClass.HELMET, config, FRAME) is SkipReason.TOO_SMALL

    def test_a_low_confidence_person_is_skipped(self, config: AssociationConfig) -> None:
        unsure = person(*UPRIGHT, conf=0.1)
        assert evaluability(unsure, PPEClass.HELMET, config, FRAME) is SkipReason.LOW_CONFIDENCE

    def test_a_person_occluded_from_the_chest_down_is_still_judged(
        self, config: AssociationConfig
    ) -> None:
        """The spec's hard case.

        Only head and chest are visible, so the detector's box is short. Because
        the bands are fractions of the box rather than absolute pixels, the
        helmet band still lands on the head and the vest band still lands on the
        chest. This case needs no special handling, and that is worth pinning.
        """
        head_and_chest = person(400.0, 200.0, 500.0, 340.0)
        assert evaluability(head_and_chest, PPEClass.HELMET, config, FRAME) is None
        assert evaluability(head_and_chest, PPEClass.VEST, config, FRAME) is None

    def test_cropping_guards_can_be_turned_off(self) -> None:
        config = AssociationConfig(skip_when_cropped=False)
        cropped = person(400.0, -120.0, 500.0, 300.0)
        assert evaluability(cropped, PPEClass.HELMET, config, FRAME) is None


class TestAssociate:
    def test_a_fully_equipped_worker_is_compliant(self, config: AssociationConfig) -> None:
        detections = [person(*UPRIGHT), helmet(450.0, 225.0), vest(450.0, 320.0)]
        [assessment] = associate(detections, config, FRAME)
        assert assessment.is_compliant
        assert assessment.missing == frozenset()
        assert set(assessment.matches) == {PPEClass.HELMET, PPEClass.VEST}

    def test_a_bare_headed_worker_is_a_violation(self, config: AssociationConfig) -> None:
        detections = [person(*UPRIGHT), vest(450.0, 320.0)]
        [assessment] = associate(detections, config, FRAME)
        assert assessment.is_violation
        assert assessment.missing == frozenset({PPEClass.HELMET})

    def test_a_worker_with_nothing_is_missing_both(self, config: AssociationConfig) -> None:
        [assessment] = associate([person(*UPRIGHT)], config, FRAME)
        assert assessment.missing == frozenset({PPEClass.HELMET, PPEClass.VEST})

    def test_the_carried_helmet_scenario_end_to_end(self, config: AssociationConfig) -> None:
        """A worker holding their helmet is not wearing it."""
        detections = [person(*UPRIGHT), helmet(450.0, 400.0), vest(450.0, 320.0)]
        [assessment] = associate(detections, config, FRAME)
        assert assessment.missing == frozenset({PPEClass.HELMET})

    def test_two_workers_one_helmet_end_to_end(self, config: AssociationConfig) -> None:
        detections = [
            person(400.0, 200.0, 500.0, 500.0),
            person(470.0, 200.0, 570.0, 500.0),
            helmet(450.0, 225.0),
            vest(450.0, 320.0),
            vest(520.0, 320.0),
        ]
        left, right = associate(detections, config, FRAME)
        assert left.missing == frozenset()
        assert right.missing == frozenset({PPEClass.HELMET})

    def test_a_cropped_worker_is_reported_as_unjudged_not_guilty(
        self, config: AssociationConfig
    ) -> None:
        detections = [person(400.0, -120.0, 500.0, 300.0)]
        [assessment] = associate(detections, config, FRAME)
        assert not assessment.evaluated
        assert not assessment.is_violation
        assert assessment.skip_reason is SkipReason.HEAD_OUT_OF_FRAME

    def test_results_come_back_in_input_order(self, config: AssociationConfig) -> None:
        a = person(300.0, 200.0, 400.0, 500.0)
        b = person(600.0, 200.0, 700.0, 500.0)
        assessments = associate([a, b], config, FRAME)
        assert [x.person for x in assessments] == [a, b]

    def test_track_ids_are_carried_through(self, config: AssociationConfig) -> None:
        a = person(300.0, 200.0, 400.0, 500.0)
        b = person(600.0, 200.0, 700.0, 500.0)
        assessments = associate([a, b], config, FRAME, track_ids=[7, 9])
        assert [x.track_id for x in assessments] == [7, 9]

    def test_no_people_yields_no_assessments(self, config: AssociationConfig) -> None:
        assert associate([helmet(450.0, 225.0)], config, FRAME) == []

    def test_requiring_only_a_helmet_ignores_a_missing_vest(self) -> None:
        """Site policy changes with config, not with a retrain."""
        config = AssociationConfig(required=[PPEClass.HELMET])
        detections = [person(*UPRIGHT), helmet(450.0, 225.0)]
        [assessment] = associate(detections, config, FRAME)
        assert assessment.is_compliant

    def test_widening_the_helmet_band_accepts_a_carried_helmet(self) -> None:
        """Proof the rule is genuinely tunable without touching the model."""
        detections = [person(*UPRIGHT), helmet(450.0, 400.0)]
        strict = AssociationConfig(required=[PPEClass.HELMET])
        assert associate(detections, strict, FRAME)[0].is_violation

        lenient = AssociationConfig(
            required=[PPEClass.HELMET],
            rules={
                PPEClass.HELMET: ItemRule(band_top=0.0, band_bottom=0.95),
                PPEClass.VEST: ItemRule(band_top=0.15, band_bottom=0.75),
            },
        )
        assert associate(detections, lenient, FRAME)[0].is_compliant

    def test_a_skipped_item_does_not_count_as_missing(self, config: AssociationConfig) -> None:
        """Judged on the vest, unjudgeable on the helmet: not a helmet violation."""
        cropped = person(400.0, -120.0, 500.0, 300.0)
        detections = [cropped, vest(450.0, 100.0)]
        [assessment] = associate(detections, config, FRAME)
        assert PPEClass.HELMET not in assessment.missing
