"""Tests for the ByteTrack adapter.

Exercises the real supervision tracker rather than mocking it: the point of
this file is the translation between their vocabulary and ours, and a mock
would test the mock.
"""

from __future__ import annotations

import pytest

from ppe.config.schema import TrackingConfig
from ppe.core.types import BBox, Detection, PPEClass
from ppe.tracking.bytetrack import ByteTrackTracker, NullTracker


def person(x1: float, conf: float = 0.9) -> Detection:
    return Detection(BBox(x1, 100.0, x1 + 60.0, 340.0), PPEClass.PERSON, conf)


def helmet() -> Detection:
    return Detection(BBox(120.0, 105.0, 150.0, 130.0), PPEClass.HELMET, 0.8)


@pytest.fixture
def tracker() -> ByteTrackTracker:
    return ByteTrackTracker(TrackingConfig())


class TestByteTrackAdapter:
    def test_it_satisfies_the_tracker_protocol(self, tracker: ByteTrackTracker) -> None:
        from ppe.core.protocols import Tracker

        assert isinstance(tracker, Tracker)

    def test_a_steady_person_keeps_one_id(self, tracker: ByteTrackTracker) -> None:
        ids = set()
        for frame in range(8):
            for track in tracker.update([person(100.0 + frame * 2)], frame):
                ids.add(track.track_id)
        assert len(ids) == 1, f"expected a single stable identity, got {ids}"

    def test_two_separated_people_get_different_ids(self, tracker: ByteTrackTracker) -> None:
        ids = set()
        for frame in range(8):
            for track in tracker.update([person(100.0), person(400.0)], frame):
                ids.add(track.track_id)
        assert len(ids) == 2

    def test_ppe_items_are_ignored(self, tracker: ByteTrackTracker) -> None:
        """Only people are tracked; a helmet must not become a track."""
        for frame in range(5):
            tracks = tracker.update([person(100.0), helmet()], frame)
        assert all(t.detection.label is PPEClass.PERSON for t in tracks)
        assert len(tracks) <= 1

    def test_an_empty_frame_yields_no_tracks(self, tracker: ByteTrackTracker) -> None:
        assert tracker.update([], 0) == []

    def test_frames_seen_accumulates(self, tracker: ByteTrackTracker) -> None:
        last = []
        for frame in range(6):
            last = tracker.update([person(100.0)], frame)
        assert last and last[0].frames_seen > 1

    def test_reset_starts_identities_over(self, tracker: ByteTrackTracker) -> None:
        for frame in range(6):
            tracker.update([person(100.0)], frame)
        tracker.reset()
        after = []
        for frame in range(6):
            after = tracker.update([person(100.0)], frame)
        assert after and after[0].frames_seen <= 6

    def test_boxes_survive_the_round_trip(self, tracker: ByteTrackTracker) -> None:
        tracks = []
        for frame in range(5):
            tracks = tracker.update([person(100.0)], frame)
        assert tracks
        box = tracks[0].box
        assert box.x1 == pytest.approx(100.0, abs=1.0)
        assert box.y2 == pytest.approx(340.0, abs=1.0)

    def test_confidence_stays_in_range(self, tracker: ByteTrackTracker) -> None:
        tracks = []
        for frame in range(5):
            tracks = tracker.update([person(100.0, conf=0.95)], frame)
        assert tracks
        assert 0.0 <= tracks[0].confidence <= 1.0


class TestNullTracker:
    def test_it_satisfies_the_tracker_protocol(self) -> None:
        from ppe.core.protocols import Tracker

        assert isinstance(NullTracker(), Tracker)

    def test_every_frame_gets_a_fresh_identity(self) -> None:
        """Deliberate: it is the control case for the Phase 5 error breakdown."""
        t = NullTracker()
        first = t.update([person(100.0)], 0)
        second = t.update([person(100.0)], 1)
        assert first[0].track_id != second[0].track_id

    def test_it_ignores_ppe_items(self) -> None:
        assert NullTracker().update([helmet()], 0) == []

    def test_reset_restarts_numbering(self) -> None:
        t = NullTracker()
        t.update([person(100.0)], 0)
        t.reset()
        assert t.update([person(100.0)], 0)[0].track_id == 1
