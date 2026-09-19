"""ByteTrack via ``supervision``.

Only person detections are fed in. Helmets and vests are re-associated every
frame instead, so a helmet flickering out for two frames cannot manufacture a
tracking identity change -- which would otherwise show up in the Phase 5 error
breakdown as a tracking fault when it was really a detection fault.

``supervision.ByteTrack`` is deprecated upstream and disappears in 0.31, hence
the version pin. This adapter is the only place that matters: replacing
ByteTrack with another tracker means rewriting this file and nothing else.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from ppe.config.schema import TrackingConfig
from ppe.core.types import BBox, Detection, PPEClass, Track


class ByteTrackTracker:
    """Adapter from ``supervision``'s ByteTrack to the ``Tracker`` protocol."""

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._frames_seen: dict[int, int] = {}
        self._last_frame: dict[int, int] = {}
        self._tracker = self._build()

    def _build(self) -> object:
        # Import here so `import ppe.tracking.bytetrack` stays cheap and the
        # deprecation warning is raised once, where we can silence it.
        from supervision.tracker import ByteTrack

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return ByteTrack(
                track_activation_threshold=self._config.track_activation_threshold,
                lost_track_buffer=self._config.lost_track_buffer,
                minimum_matching_threshold=self._config.minimum_matching_threshold,
                frame_rate=self._config.frame_rate,
            )

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Track]:
        """Feed one frame of person detections; return the live tracks.

        Non-person detections are ignored rather than rejected, so callers can
        hand over the whole frame without filtering first.
        """
        import supervision as sv

        people = [d for d in detections if d.label is PPEClass.PERSON]
        if not people:
            # ByteTrack still needs the frame to advance its internal clock,
            # otherwise a gap in detections does not age the lost tracks.
            self._tracker.update_with_detections(sv.Detections.empty())  # type: ignore[attr-defined]
            return []

        sv_detections = sv.Detections(
            xyxy=np.array([d.box.as_tuple() for d in people], dtype=np.float64),
            confidence=np.array([d.confidence for d in people], dtype=np.float64),
            class_id=np.zeros(len(people), dtype=int),
        )
        tracked = self._tracker.update_with_detections(sv_detections)  # type: ignore[attr-defined]

        out: list[Track] = []
        for index in range(len(tracked)):
            track_id = tracked.tracker_id[index]
            if track_id is None:
                continue
            track_id = int(track_id)
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[index])
            confidence = float(tracked.confidence[index]) if tracked.confidence is not None else 0.0

            self._frames_seen[track_id] = self._frames_seen.get(track_id, 0) + 1
            gap = frame_index - self._last_frame.get(track_id, frame_index)
            self._last_frame[track_id] = frame_index

            out.append(
                Track(
                    track_id=track_id,
                    detection=Detection(
                        box=BBox(x1, y1, x2, y2),
                        label=PPEClass.PERSON,
                        confidence=min(max(confidence, 0.0), 1.0),
                    ),
                    frames_seen=self._frames_seen[track_id],
                    frames_since_update=max(gap, 0),
                )
            )
        return out

    def reset(self) -> None:
        """Forget all state. Called between clips so runs stay independent."""
        self._frames_seen.clear()
        self._last_frame.clear()
        self._tracker = self._build()


class NullTracker:
    """A tracker that assigns a fresh id to every detection, every frame.

    Used when tracking is disabled in config and by the evaluation harness to
    measure how much of the violation error is attributable to tracking: run
    the same clip with and without real identities and diff the events.
    """

    def __init__(self) -> None:
        self._next_id = 0

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Track]:  # noqa: ARG002
        out: list[Track] = []
        for detection in detections:
            if detection.label is not PPEClass.PERSON:
                continue
            self._next_id += 1
            out.append(Track(track_id=self._next_id, detection=detection, frames_seen=1))
        return out

    def reset(self) -> None:
        self._next_id = 0
