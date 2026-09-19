"""Tests for zone filtering and violation debouncing.

The debouncer is stateful, but it has no clock and no I/O: frame index and
timestamp are arguments. So ten minutes of footage simulates in a millisecond
and two runs over the same inputs always agree.
"""

from __future__ import annotations

import pytest

from ppe.config.schema import DebounceConfig, ZoneConfig, ZonesConfig
from ppe.core.errors import GeometryError
from ppe.core.types import BBox, Detection, PersonAssessment, PPEClass
from ppe.rules.debounce import ViolationDebouncer
from ppe.rules.zones import anchor_point, in_scope, point_in_polygon, zone_for

SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def person_at(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), PPEClass.PERSON, conf)


class TestPointInPolygon:
    def test_inside(self) -> None:
        assert point_in_polygon((50.0, 50.0), SQUARE)

    def test_outside(self) -> None:
        assert not point_in_polygon((150.0, 50.0), SQUARE)

    def test_on_an_edge_counts_as_inside(self) -> None:
        """Zone boundaries get drawn along walkways; people stand on them."""
        assert point_in_polygon((50.0, 0.0), SQUARE)
        assert point_in_polygon((0.0, 50.0), SQUARE)

    def test_on_a_vertex_counts_as_inside(self) -> None:
        assert point_in_polygon((0.0, 0.0), SQUARE)

    def test_concave_polygon_excludes_the_notch(self) -> None:
        """An L-shape: the missing quadrant must not count as inside."""
        l_shape = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 50.0),
            (50.0, 50.0),
            (50.0, 100.0),
            (0.0, 100.0),
        ]
        assert point_in_polygon((25.0, 75.0), l_shape)
        assert not point_in_polygon((75.0, 75.0), l_shape)

    def test_a_degenerate_polygon_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="at least 3 vertices"):
            point_in_polygon((1.0, 1.0), [(0.0, 0.0), (1.0, 1.0)])


class TestAnchorPoint:
    BOX = BBox(100.0, 200.0, 200.0, 500.0)

    def test_bottom_centre_is_where_the_feet_are(self) -> None:
        assert anchor_point(self.BOX, "bottom_centre") == (150.0, 500.0)

    def test_centre(self) -> None:
        assert anchor_point(self.BOX, "centre") == (150.0, 350.0)

    def test_unknown_anchor_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="unknown zone anchor"):
            anchor_point(self.BOX, "elbow")


class TestZoneScope:
    def _zones(self, **kwargs: object) -> ZonesConfig:
        return ZonesConfig(
            zones=[ZoneConfig(zone_id="bay", polygon=SQUARE)],
            **kwargs,  # type: ignore[arg-type]
        )

    def test_a_worker_standing_in_the_zone_is_in_scope(self) -> None:
        worker = person_at(40.0, 20.0, 60.0, 90.0)  # feet at (50, 90)
        assert in_scope(worker, self._zones())

    def test_a_worker_outside_the_zone_is_not_evaluated(self) -> None:
        worker = person_at(400.0, 20.0, 460.0, 90.0)
        assert not in_scope(worker, self._zones())

    def test_leaning_over_the_boundary_still_belongs_to_the_zone(self) -> None:
        """Feet inside, head outside: they are standing in the work area."""
        leaning = person_at(80.0, -40.0, 120.0, 95.0)  # feet at (100, 95)
        assert in_scope(leaning, self._zones())

    def test_zone_for_names_the_zone(self) -> None:
        worker = person_at(40.0, 20.0, 60.0, 90.0)
        zone = zone_for(worker, self._zones())
        assert zone is not None
        assert zone.zone_id == "bay"

    def test_a_disabled_zone_does_not_capture_anyone(self) -> None:
        config = ZonesConfig(zones=[ZoneConfig(zone_id="bay", polygon=SQUARE, enabled=False)])
        worker = person_at(40.0, 20.0, 60.0, 90.0)
        assert zone_for(worker, config) is None

    def test_no_zones_means_the_whole_frame(self) -> None:
        assert in_scope(person_at(900.0, 900.0, 950.0, 990.0), ZonesConfig())

    def test_no_zones_can_instead_mean_nothing_is_evaluated(self) -> None:
        config = ZonesConfig(evaluate_when_no_zones=False)
        assert not in_scope(person_at(40.0, 20.0, 60.0, 90.0), config)


def violating(track_id: int, conf: float = 0.9) -> PersonAssessment:
    return PersonAssessment(
        person=person_at(400.0, 200.0, 500.0, 500.0, conf),
        missing=frozenset({PPEClass.HELMET}),
        track_id=track_id,
    )


def compliant(track_id: int) -> PersonAssessment:
    return PersonAssessment(person=person_at(400.0, 200.0, 500.0, 500.0), track_id=track_id)


def unjudged(track_id: int) -> PersonAssessment:
    return PersonAssessment(
        person=person_at(400.0, 200.0, 500.0, 500.0),
        missing=frozenset({PPEClass.HELMET}),
        track_id=track_id,
        evaluated=False,
    )


class TestDebounce:
    def _debouncer(self, **kwargs: object) -> ViolationDebouncer:
        defaults: dict[str, object] = {"confirm_frames": 3, "cooldown_s": 10.0}
        defaults.update(kwargs)
        return ViolationDebouncer(DebounceConfig(**defaults))  # type: ignore[arg-type]

    def test_one_bad_frame_does_not_fire(self) -> None:
        """A single dropped helmet detection must not alert a supervisor."""
        d = self._debouncer()
        assert d.update(0, 0.0, [violating(1)]) == []

    def test_fires_once_the_streak_is_confirmed(self) -> None:
        d = self._debouncer()
        assert d.update(0, 0.0, [violating(1)]) == []
        assert d.update(1, 0.033, [violating(1)]) == []
        events = d.update(2, 0.066, [violating(1)])
        assert len(events) == 1
        assert events[0].track_id == 1
        assert events[0].missing == frozenset({PPEClass.HELMET})
        assert events[0].streak_length == 3
        assert events[0].first_frame_index == 0

    def test_a_compliant_frame_breaks_the_streak(self) -> None:
        d = self._debouncer()
        d.update(0, 0.0, [violating(1)])
        d.update(1, 0.033, [compliant(1)])
        d.update(2, 0.066, [violating(1)])
        assert d.update(3, 0.099, [violating(1)]) == []

    def test_an_unjudged_frame_neither_extends_nor_breaks_the_streak(self) -> None:
        """Stepping behind a pillar is not the same as putting a helmet on."""
        d = self._debouncer()
        d.update(0, 0.0, [violating(1)])
        d.update(1, 0.033, [unjudged(1)])
        d.update(2, 0.066, [violating(1)])
        events = d.update(3, 0.099, [violating(1)])
        assert len(events) == 1

    def test_a_long_gap_does_break_the_streak(self) -> None:
        d = self._debouncer(max_gap_frames=2)
        d.update(0, 0.0, [violating(1)])
        d.update(1, 0.033, [violating(1)])
        assert d.update(50, 1.6, [violating(1)]) == []

    def test_cooldown_suppresses_a_repeat(self) -> None:
        d = self._debouncer()
        for i in range(3):
            d.update(i, i * 0.033, [violating(1)])
        # Same track still non-compliant, well inside the cooldown window.
        fired = [e for i in range(3, 20) for e in d.update(i, i * 0.033, [violating(1)])]
        assert fired == []

    def test_it_re_fires_after_the_cooldown_expires(self) -> None:
        d = self._debouncer()
        for i in range(3):
            d.update(i, i * 0.033, [violating(1)])
        later = [e for i in range(3, 10) for e in d.update(i, 100.0 + i, [violating(1)])]
        assert len(later) == 1

    def test_tracks_are_debounced_independently(self) -> None:
        d = self._debouncer()
        d.update(0, 0.0, [violating(1)])
        d.update(1, 0.033, [violating(1), violating(2)])
        events = d.update(2, 0.066, [violating(1), violating(2)])
        assert [e.track_id for e in events] == [1]

    def test_a_change_in_what_is_missing_restarts_the_streak(self) -> None:
        """Losing a helmet and losing a vest are different events."""
        d = self._debouncer()
        d.update(0, 0.0, [violating(1)])
        d.update(1, 0.033, [violating(1)])
        both = PersonAssessment(
            person=person_at(400.0, 200.0, 500.0, 500.0),
            missing=frozenset({PPEClass.HELMET, PPEClass.VEST}),
            track_id=1,
        )
        assert d.update(2, 0.066, [both]) == []

    def test_an_assessment_without_a_track_id_is_ignored(self) -> None:
        d = self._debouncer(confirm_frames=1)
        untracked = PersonAssessment(
            person=person_at(400.0, 200.0, 500.0, 500.0),
            missing=frozenset({PPEClass.HELMET}),
        )
        assert d.update(0, 0.0, [untracked]) == []

    def test_mean_confidence_is_measured_not_invented(self) -> None:
        d = self._debouncer()
        d.update(0, 0.0, [violating(1, conf=0.6)])
        d.update(1, 0.033, [violating(1, conf=0.8)])
        events = d.update(2, 0.066, [violating(1, conf=1.0)])
        assert events[0].mean_person_confidence == pytest.approx(0.8)

    def test_is_deterministic_across_runs(self) -> None:
        """Same inputs, same events -- the whole point of passing time in."""

        def run() -> list[tuple[int, int, int]]:
            d = self._debouncer()
            out = []
            for i in range(30):
                for event in d.update(i, i * 0.033, [violating(1), violating(2)]):
                    out.append((event.track_id, event.first_frame_index, event.streak_length))
            return out

        assert run() == run()
        assert run() != []

    def test_prune_forgets_tracks_that_have_left(self) -> None:
        d = self._debouncer(forget_after_frames=10)
        d.update(0, 0.0, [violating(1)])
        assert d.tracked_ids == frozenset({1})
        d.prune(100)
        assert d.tracked_ids == frozenset()

    def test_prune_keeps_tracks_still_present(self) -> None:
        d = self._debouncer(forget_after_frames=100)
        d.update(0, 0.0, [violating(1)])
        d.prune(10)
        assert d.tracked_ids == frozenset({1})

    def test_reset_clears_everything_between_clips(self) -> None:
        d = self._debouncer()
        for i in range(3):
            d.update(i, i * 0.033, [violating(1)])
        d.reset()
        assert d.tracked_ids == frozenset()
        assert d.update(0, 0.0, [violating(1)]) == []
