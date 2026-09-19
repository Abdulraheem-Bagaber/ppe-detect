"""Tests for the geometric primitives the association rules are built on."""

from __future__ import annotations

import pytest

from ppe.core.types import BBox
from ppe.rules.geometry import (
    centre_offset,
    containment,
    intersection,
    intersection_area,
    iou,
    vertical_band_overlap,
    visible_fraction,
)

UNIT = BBox(0.0, 0.0, 10.0, 10.0)


class TestIntersection:
    def test_overlapping_boxes(self) -> None:
        assert intersection(UNIT, BBox(5.0, 5.0, 15.0, 15.0)) == BBox(5.0, 5.0, 10.0, 10.0)

    def test_disjoint_boxes_return_none(self) -> None:
        assert intersection(UNIT, BBox(20.0, 20.0, 30.0, 30.0)) is None

    def test_touching_edges_do_not_count_as_overlap(self) -> None:
        """Sharing a boundary is not overlapping; a zero-area box would lie."""
        assert intersection(UNIT, BBox(10.0, 0.0, 20.0, 10.0)) is None

    def test_containment_returns_the_inner_box(self) -> None:
        inner = BBox(2.0, 2.0, 4.0, 4.0)
        assert intersection(UNIT, inner) == inner

    def test_is_symmetric(self) -> None:
        other = BBox(5.0, 5.0, 15.0, 15.0)
        assert intersection(UNIT, other) == intersection(other, UNIT)

    def test_area_is_zero_when_disjoint(self) -> None:
        assert intersection_area(UNIT, BBox(20.0, 20.0, 30.0, 30.0)) == 0.0


class TestIou:
    def test_identical_boxes_score_one(self) -> None:
        assert iou(UNIT, UNIT) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self) -> None:
        assert iou(UNIT, BBox(20.0, 20.0, 30.0, 30.0)) == 0.0

    def test_quarter_overlap(self) -> None:
        # 25 shared, 175 union.
        assert iou(UNIT, BBox(5.0, 5.0, 15.0, 15.0)) == pytest.approx(25 / 175)

    def test_is_useless_for_helmet_on_person(self) -> None:
        """Why association uses containment instead: a perfect fit scores ~0.01."""
        person = BBox(0.0, 0.0, 100.0, 300.0)
        worn_helmet = BBox(40.0, 5.0, 60.0, 25.0)
        assert iou(worn_helmet, person) < 0.02
        assert containment(worn_helmet, person) == pytest.approx(1.0)

    def test_zero_area_boxes_do_not_divide_by_zero(self) -> None:
        assert iou(BBox(0.0, 0.0, 0.0, 0.0), UNIT) == 0.0


class TestContainment:
    def test_fully_inside_scores_one(self) -> None:
        assert containment(BBox(2.0, 2.0, 4.0, 4.0), UNIT) == pytest.approx(1.0)

    def test_half_inside_scores_half(self) -> None:
        assert containment(BBox(5.0, 0.0, 15.0, 10.0), UNIT) == pytest.approx(0.5)

    def test_outside_scores_zero(self) -> None:
        assert containment(BBox(20.0, 20.0, 30.0, 30.0), UNIT) == 0.0

    def test_is_asymmetric(self) -> None:
        """A helmet is inside a person; a person is not inside a helmet."""
        small = BBox(2.0, 2.0, 4.0, 4.0)
        assert containment(small, UNIT) > containment(UNIT, small)

    def test_zero_area_inner_is_uncontained_rather_than_an_error(self) -> None:
        assert containment(BBox(5.0, 5.0, 5.0, 5.0), UNIT) == 0.0


class TestCentreOffset:
    PERSON = BBox(400.0, 200.0, 500.0, 500.0)  # centre x 450, half-width 50

    def test_dead_centre_is_zero(self) -> None:
        assert centre_offset(BBox(440.0, 210.0, 460.0, 230.0), self.PERSON) == 0.0

    def test_on_the_edge_is_one(self) -> None:
        item = BBox(490.0, 210.0, 510.0, 230.0)  # centre x 500
        assert centre_offset(item, self.PERSON) == pytest.approx(1.0)

    def test_beyond_the_box_exceeds_one(self) -> None:
        item = BBox(540.0, 210.0, 560.0, 230.0)
        assert centre_offset(item, self.PERSON) > 1.0

    def test_is_symmetric_about_the_centreline(self) -> None:
        left = BBox(400.0, 210.0, 420.0, 230.0)
        right = BBox(480.0, 210.0, 500.0, 230.0)
        assert centre_offset(left, self.PERSON) == pytest.approx(centre_offset(right, self.PERSON))

    def test_zero_width_person_yields_infinity(self) -> None:
        flat = BBox(450.0, 200.0, 450.0, 500.0)
        assert centre_offset(BBox(440.0, 210.0, 460.0, 230.0), flat) == float("inf")


class TestVerticalBandOverlap:
    BAND = BBox(0.0, 100.0, 100.0, 200.0)

    def test_entirely_within_the_band(self) -> None:
        assert vertical_band_overlap(BBox(10.0, 120.0, 30.0, 160.0), self.BAND) == 1.0

    def test_entirely_outside_the_band(self) -> None:
        assert vertical_band_overlap(BBox(10.0, 300.0, 30.0, 340.0), self.BAND) == 0.0

    def test_straddling_the_lower_edge(self) -> None:
        assert vertical_band_overlap(BBox(10.0, 180.0, 30.0, 220.0), self.BAND) == pytest.approx(
            0.5
        )

    def test_ignores_horizontal_position(self) -> None:
        near = BBox(10.0, 120.0, 30.0, 160.0)
        far = BBox(900.0, 120.0, 920.0, 160.0)
        assert vertical_band_overlap(near, self.BAND) == vertical_band_overlap(far, self.BAND)

    def test_zero_height_item_is_scored_by_position(self) -> None:
        assert vertical_band_overlap(BBox(10.0, 150.0, 30.0, 150.0), self.BAND) == 1.0
        assert vertical_band_overlap(BBox(10.0, 400.0, 30.0, 400.0), self.BAND) == 0.0


class TestVisibleFraction:
    FRAME = (640, 480)

    def test_a_box_well_inside_is_fully_visible(self) -> None:
        assert visible_fraction(BBox(100.0, 100.0, 200.0, 200.0), self.FRAME) == 1.0

    def test_a_box_half_off_the_top(self) -> None:
        assert visible_fraction(BBox(100.0, -50.0, 200.0, 50.0), self.FRAME) == pytest.approx(0.5)

    def test_a_box_entirely_outside_is_invisible(self) -> None:
        assert visible_fraction(BBox(100.0, -200.0, 200.0, -100.0), self.FRAME) == 0.0

    def test_zero_area_box_is_invisible_rather_than_an_error(self) -> None:
        assert visible_fraction(BBox(10.0, 10.0, 10.0, 10.0), self.FRAME) == 0.0
