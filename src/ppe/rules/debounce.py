"""Temporal confirmation of violations. Implemented in Phase 4.

The one stateful thing in ``ppe.rules``. Confirming "N consecutive frames"
cannot be a pure function, so instead of pretending otherwise this is an
explicit state machine with two properties that keep it testable:

* **No clock.** ``frame_index`` and ``timestamp_s`` are arguments. A test can
  simulate ten minutes of footage in a millisecond, and two runs over the same
  inputs always produce identical events.
* **No I/O.** It returns events; something else decides what to do with them.

Without debouncing, a single dropped helmet detection in one frame of a
thirty-frame second produces an alert. The false-alarms-per-hour figure in
Phase 5 is mostly a measurement of how well this module is tuned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ppe.config.schema import DebounceConfig
from ppe.core.types import PersonAssessment, PPEClass, ViolationEvent


@dataclass(slots=True)
class TrackState:
    """Per-track debounce bookkeeping. Internal, exposed only for tests."""

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
        raise NotImplementedError

    def reset_streak(self) -> None:
        raise NotImplementedError


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
        nor break a streak -- a worker who steps behind a pillar for four
        frames has not become compliant.
        """
        raise NotImplementedError

    def prune(self, frame_index: int) -> None:
        """Drop state for tracks unseen for ``forget_after_frames``."""
        raise NotImplementedError

    def reset(self) -> None:
        """Forget everything. Called between clips."""
        raise NotImplementedError
