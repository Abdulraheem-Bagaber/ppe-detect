"""Why did each false alarm happen?

The most useful output in the project, and the reason the pipeline keeps every
intermediate result rather than only its events. A precision figure tells you
that you have a problem; this tells you which of three very different problems
it is, and therefore where a day of work would actually pay.

The three causes are genuinely distinct pieces of engineering:

* **missed_ppe_detection** -- the worker *was* wearing it, the detector never
  saw it. Fix: more or better training data for that class.
* **association_failure** -- the item was detected, sitting right there, and
  the rule layer did not attribute it to the person. Fix: tune the bands and
  thresholds in config. No retraining.
* **tracking_id_switch** -- the person's identity changed mid-violation, so
  the debouncer started a fresh streak and fired again. Fix: tracker
  parameters.

Attribution works off the recorded per-frame results, so re-running inference
is not required to regenerate the breakdown.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ppe.core.types import BBox, PPEClass, ViolationEvent
from ppe.pipeline.runner import FrameResult
from ppe.rules.geometry import containment, iou


class FalsePositiveCause(StrEnum):
    """Why a false alarm fired."""

    MISSED_PPE_DETECTION = "missed_ppe_detection"
    ASSOCIATION_FAILURE = "association_failure"
    TRACKING_ID_SWITCH = "tracking_id_switch"
    ANNOTATION_DISAGREEMENT = "annotation_disagreement"


# How much of an item must sit inside the person box before we call it "there
# but unassociated" rather than "somebody else's".
NEARBY_CONTAINMENT = 0.3

# Person-box IoU above which two tracks in adjacent frames are the same body.
SAME_BODY_IOU = 0.5


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """One false positive, explained."""

    event: ViolationEvent
    cause: FalsePositiveCause
    detail: str

    @property
    def is_fixable_in_config(self) -> bool:
        """Association and tracking faults are config work; misses are not."""
        return self.cause in {
            FalsePositiveCause.ASSOCIATION_FAILURE,
            FalsePositiveCause.TRACKING_ID_SWITCH,
        }


def diagnose(event: ViolationEvent, frames: Sequence[FrameResult]) -> Diagnosis:
    """Work out why one false-positive event fired.

    Args:
        event: The event that should not have fired.
        frames: Every frame result from the run that produced it.
    """
    window = [f for f in frames if event.first_frame_index <= f.frame_index <= event.frame_index]
    if not window:
        return Diagnosis(
            event=event,
            cause=FalsePositiveCause.ANNOTATION_DISAGREEMENT,
            detail="no recorded frames cover this event's streak",
        )

    if _identity_changed(event, window):
        return Diagnosis(
            event=event,
            cause=FalsePositiveCause.TRACKING_ID_SWITCH,
            detail=(
                f"the body at this position carried more than one track id across "
                f"frames {event.first_frame_index}-{event.frame_index}"
            ),
        )

    for item in sorted(event.missing, key=lambda c: c.value):
        nearby = _nearby_unassociated(item, event.person_box, window)
        if nearby:
            return Diagnosis(
                event=event,
                cause=FalsePositiveCause.ASSOCIATION_FAILURE,
                detail=(
                    f"a {item.value} was detected inside the person box in "
                    f"{nearby} of {len(window)} frames but never associated; "
                    f"this is a config fix, not a training one"
                ),
            )

    for item in sorted(event.missing, key=lambda c: c.value):
        if not _any_detection_of(item, window):
            return Diagnosis(
                event=event,
                cause=FalsePositiveCause.MISSED_PPE_DETECTION,
                detail=(
                    f"no {item.value} was detected anywhere in the frame across "
                    f"{len(window)} frames; the detector never saw it"
                ),
            )

    return Diagnosis(
        event=event,
        cause=FalsePositiveCause.ANNOTATION_DISAGREEMENT,
        detail=(
            "the item was detected elsewhere in frame and correctly not attributed "
            "to this person; the annotation and the system disagree about the facts"
        ),
    )


def _identity_changed(event: ViolationEvent, window: Sequence[FrameResult]) -> bool:
    """Did the body at this position carry more than one track id?"""
    ids: set[int] = set()
    for frame in window:
        for track in frame.tracks:
            if iou(track.box, event.person_box) >= SAME_BODY_IOU:
                ids.add(track.track_id)
    return len(ids) > 1


def _nearby_unassociated(item: PPEClass, person_box: BBox, window: Sequence[FrameResult]) -> int:
    """Frames where an item of this class sat inside the person but unmatched."""
    count = 0
    for frame in window:
        matched = {
            id(match.item)
            for assessment in frame.assessments
            for match in assessment.matches.values()
        }
        for detection in frame.detections:
            if detection.label is not item or id(detection) in matched:
                continue
            if containment(detection.box, person_box) >= NEARBY_CONTAINMENT:
                count += 1
                break
    return count


def _any_detection_of(item: PPEClass, window: Sequence[FrameResult]) -> bool:
    return any(d.label is item for frame in window for d in frame.detections)


def summarise(diagnoses: Sequence[Diagnosis]) -> Counter[str]:
    """Count false positives by cause."""
    return Counter(d.cause.value for d in diagnoses)
