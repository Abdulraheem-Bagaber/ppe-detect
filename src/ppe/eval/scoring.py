"""Matching predicted events to ground-truth intervals, and scoring them.

Pure functions over data. The unit is the **violation event**, not the frame
and not the detection: a supervisor cares whether they were told about the
right person at roughly the right time, not whether a bounding box was
pixel-perfect.

Matching rules, stated once:

* An event matches an interval when the event's timestamp falls inside the
  interval (plus a tolerance), the event's missing items overlap the
  interval's, and -- if the interval carries a box hint -- the event's person
  box overlaps it.
* One interval can absorb several events. A worker bare-headed for thirty
  seconds, alerted twice because of the cooldown, is one thing detected twice,
  not one hit and one false alarm.
* An interval with no event is a miss. An event matching no interval is a
  false alarm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ppe.core.types import BBox, ViolationEvent
from ppe.eval.groundtruth import ClipGroundTruth, ViolationInterval
from ppe.rules.geometry import iou

# How far outside an annotated interval an event may land and still count.
# Generous on purpose: debounce delays an event by design, and a human
# annotating a clip is not accurate to the frame.
DEFAULT_TOLERANCE_S = 2.0

# How much an event's person box must overlap a box hint to be the same person.
DEFAULT_BOX_IOU = 0.1


@dataclass(frozen=True, slots=True)
class EventMatch:
    """One predicted event and the interval it was attributed to, if any."""

    event: ViolationEvent
    interval: ViolationInterval | None
    clip: str

    @property
    def is_true_positive(self) -> bool:
        return self.interval is not None

    @property
    def is_false_positive(self) -> bool:
        return self.interval is None


@dataclass(frozen=True, slots=True)
class ClipScore:
    """What happened on one clip."""

    clip: str
    duration_s: float
    matches: list[EventMatch] = field(default_factory=list)
    missed: list[ViolationInterval] = field(default_factory=list)

    @property
    def true_positive_events(self) -> list[EventMatch]:
        return [m for m in self.matches if m.is_true_positive]

    @property
    def false_positives(self) -> list[EventMatch]:
        return [m for m in self.matches if m.is_false_positive]

    @property
    def detected_intervals(self) -> int:
        """Intervals with at least one event, counted once each."""
        return len({id(m.interval) for m in self.true_positive_events})


@dataclass(frozen=True, slots=True)
class Metrics:
    """Violation-level scores. Every field is measured, none are defaults."""

    intervals_total: int
    intervals_detected: int
    false_positive_events: int
    true_positive_events: int
    duration_s: float

    @property
    def precision(self) -> float | None:
        fired = self.true_positive_events + self.false_positive_events
        return self.true_positive_events / fired if fired else None

    @property
    def recall(self) -> float | None:
        if self.intervals_total == 0:
            return None
        return self.intervals_detected / self.intervals_total

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def false_alarms_per_hour(self) -> float | None:
        if self.duration_s <= 0:
            return None
        return self.false_positive_events / (self.duration_s / 3600.0)


def event_matches_interval(
    event: ViolationEvent,
    interval: ViolationInterval,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    box_iou_threshold: float = DEFAULT_BOX_IOU,
) -> bool:
    """Whether one event can be attributed to one annotated interval."""
    if not interval.covers(event.timestamp_s, tolerance_s):
        return False
    if not (event.missing & frozenset(interval.missing)):
        return False
    if interval.box_hint is None:
        return True
    hint = BBox(*interval.box_hint)
    return iou(event.person_box, hint) >= box_iou_threshold


def score_clip(
    ground_truth: ClipGroundTruth,
    events: Sequence[ViolationEvent],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    box_iou_threshold: float = DEFAULT_BOX_IOU,
) -> ClipScore:
    """Attribute every event on one clip, and find the intervals we missed."""
    matches: list[EventMatch] = []
    hit: set[int] = set()

    for event in events:
        attributed: ViolationInterval | None = None
        for index, interval in enumerate(ground_truth.violations):
            if event_matches_interval(event, interval, tolerance_s, box_iou_threshold):
                attributed = interval
                hit.add(index)
                break
        matches.append(EventMatch(event=event, interval=attributed, clip=ground_truth.clip))

    missed = [v for i, v in enumerate(ground_truth.violations) if i not in hit]
    return ClipScore(
        clip=ground_truth.clip,
        duration_s=ground_truth.duration_s,
        matches=matches,
        missed=missed,
    )


def aggregate(scores: Sequence[ClipScore]) -> Metrics:
    """Roll clip scores into one set of numbers."""
    return Metrics(
        intervals_total=sum(s.detected_intervals + len(s.missed) for s in scores),
        intervals_detected=sum(s.detected_intervals for s in scores),
        false_positive_events=sum(len(s.false_positives) for s in scores),
        true_positive_events=sum(len(s.true_positive_events) for s in scores),
        duration_s=sum(s.duration_s for s in scores),
    )
