"""Temporal confirmation of violations.

The one stateful thing in ``ppe.rules``. Confirming "N consecutive frames"
cannot be a pure function, so instead of pretending otherwise this is an
explicit state machine with two properties that keep it testable:

* **No clock.** ``frame_index`` and ``timestamp_s`` are arguments. A test can
  simulate ten minutes of footage in a millisecond, and two runs over the same
  inputs always produce identical events.
* **No I/O.** It returns events; something else decides what to do with them.

Without debouncing, a single dropped helmet detection in one frame of a
thirty-frame second produces an alert. The false-alarms-per-hour figure in
Phase 5 is largely a measurement of how well this module is tuned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ppe.config.schema import DebounceConfig
from ppe.core.types import PersonAssessment, PPEClass, ViolationEvent


@dataclass(slots=True)
class TrackState:
    """Per-track debounce bookkeeping. Internal, exposed for tests."""

    track_id: int
    streak_start_frame: int | None = None
    streak_length: int = 0
    last_seen_frame: int = 0
    last_event_timestamp_s: float | None = None
    missing_items: frozenset[PPEClass] = frozenset()
    confidence_sum: float = 0.0

    @property
    def mean_confidence(self) -> float:
        """Mean person confidence over the current streak."""
        if self.streak_length == 0:
            return 0.0
        return self.confidence_sum / self.streak_length

    def reset_streak(self) -> None:
        self.streak_start_frame = None
        self.streak_length = 0
        self.missing_items = frozenset()
        self.confidence_sum = 0.0


@dataclass(slots=True)
class ViolationDebouncer:
    """Turns per-frame assessments into confirmed, rate-limited events."""

    config: DebounceConfig
    _states: dict[int, TrackState] = field(default_factory=dict, init=False)

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        assessments: Sequence[PersonAssessment],
    ) -> list[ViolationEvent]:
        """Advance one frame and return any events that fired.

        Assessments with ``evaluated=False`` or no ``track_id`` neither extend
        nor break a streak: a worker who steps behind a pillar for four frames
        has not become compliant, and pretending otherwise would reset the
        counter every time the view is briefly obstructed.
        """
        events: list[ViolationEvent] = []

        for assessment in assessments:
            track_id = assessment.track_id
            if track_id is None or not assessment.evaluated:
                continue

            state = self._states.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id)
                self._states[track_id] = state

            # A gap longer than max_gap_frames means we lost them for too long
            # to claim continuity.
            if (
                state.streak_start_frame is not None
                and frame_index - state.last_seen_frame > self.config.max_gap_frames
            ):
                state.reset_streak()

            state.last_seen_frame = frame_index

            if not assessment.missing:
                state.reset_streak()
                continue

            # A change in *what* is missing starts a fresh streak: "no helmet"
            # and "no vest" are different events, not one continuing one.
            if state.missing_items != assessment.missing:
                state.reset_streak()
                state.missing_items = assessment.missing

            if state.streak_start_frame is None:
                state.streak_start_frame = frame_index
            state.streak_length += 1
            state.confidence_sum += assessment.person.confidence

            if state.streak_length < self.config.confirm_frames:
                continue

            if not self._cooldown_elapsed(state, timestamp_s):
                continue

            events.append(
                ViolationEvent(
                    track_id=track_id,
                    missing=assessment.missing,
                    first_frame_index=state.streak_start_frame,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    person_box=assessment.person.box,
                    streak_length=state.streak_length,
                    mean_person_confidence=state.mean_confidence,
                    zone_id=assessment.zone_id,
                )
            )
            state.last_event_timestamp_s = timestamp_s
            # Restart the count so a sustained violation re-fires on the
            # cooldown schedule rather than on every subsequent frame.
            state.reset_streak()
            state.missing_items = assessment.missing

        return events

    def _cooldown_elapsed(self, state: TrackState, timestamp_s: float) -> bool:
        if state.last_event_timestamp_s is None:
            return True
        return timestamp_s - state.last_event_timestamp_s >= self.config.cooldown_s

    def prune(self, frame_index: int) -> None:
        """Drop state for tracks unseen for ``forget_after_frames``."""
        stale = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_seen_frame > self.config.forget_after_frames
        ]
        for track_id in stale:
            del self._states[track_id]

    def reset(self) -> None:
        """Forget everything. Called between clips."""
        self._states.clear()

    @property
    def tracked_ids(self) -> frozenset[int]:
        """Track ids currently held in state. For tests and diagnostics."""
        return frozenset(self._states)
