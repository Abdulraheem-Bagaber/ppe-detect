"""Value objects shared by every layer.

Coordinate convention, stated once and never re-litigated elsewhere:

* Boxes are ``xyxy`` in **absolute pixel** coordinates, as ``float``.
* The origin is the top-left of the frame; ``y`` grows downward.
* Frames are ``H x W x 3`` ``uint8`` arrays in **BGR** order (OpenCV's layout).

Everything here is frozen and slotted. Detections flow forward through the
pipeline and are never mutated in place, which keeps the rule layer honest and
makes any stage trivially replayable in a test.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ppe.core.errors import GeometryError

if TYPE_CHECKING:
    from typing import Self

# An OpenCV-style frame: H x W x 3, uint8, BGR.
Frame = npt.NDArray[np.uint8]


class PPEClass(StrEnum):
    """The three independent object classes the detector is trained on.

    Note what is *absent*: there is no ``no_helmet``. Non-compliance is derived
    by the rule layer from the absence of an association, never predicted
    directly by the model.
    """

    PERSON = "person"
    HELMET = "helmet"
    VEST = "vest"

    @classmethod
    def ppe_items(cls) -> frozenset[PPEClass]:
        """The classes that can be *worn* -- i.e. everything except ``PERSON``."""
        return frozenset({cls.HELMET, cls.VEST})


class SkipReason(StrEnum):
    """Why a person was seen but not judged.

    Recording this rather than silently emitting a violation is what makes the
    Phase 5 false-positive breakdown possible without re-running inference.
    """

    OUTSIDE_ZONE = "outside_zone"
    HEAD_OUT_OF_FRAME = "head_out_of_frame"
    TORSO_OUT_OF_FRAME = "torso_out_of_frame"
    TOO_SMALL = "too_small"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True, slots=True)
class BBox:
    """An axis-aligned box in absolute pixel coordinates.

    Only *intrinsic* properties live here. Binary relations between two boxes
    (IoU, containment, band overlap) live in :mod:`ppe.rules.geometry`, so the
    association rules are all readable in one file.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coords = (self.x1, self.y1, self.x2, self.y2)
        if not all(np.isfinite(coords)):
            msg = f"box has non-finite coordinates: {coords}"
            raise GeometryError(msg)
        if self.x2 < self.x1 or self.y2 < self.y1:
            msg = f"box has negative extent: {coords}"
            raise GeometryError(msg)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def band(self, top: float, bottom: float) -> BBox:
        """The horizontal slice of this box between two fractions of its height.

        ``band(0.0, 0.35)`` is the upper third -- where a worn helmet must be.
        Fractions are of the box's own height, so the rule is scale-invariant:
        it behaves identically for a person 80px tall and one 800px tall.
        """
        if not 0.0 <= top < bottom <= 1.0:
            msg = f"band fractions must satisfy 0 <= top < bottom <= 1, got {top}, {bottom}"
            raise GeometryError(msg)
        return BBox(
            x1=self.x1,
            y1=self.y1 + top * self.height,
            x2=self.x2,
            y2=self.y1 + bottom * self.height,
        )

    def clip_to(self, width: int, height: int) -> BBox:
        """Clamp to a frame of the given size."""
        return BBox(
            x1=min(max(self.x1, 0.0), float(width)),
            y1=min(max(self.y1, 0.0), float(height)),
            x2=min(max(self.x2, 0.0), float(width)),
            y2=min(max(self.y2, 0.0), float(height)),
        )

    def touching_borders(self, width: int, height: int, margin: float) -> frozenset[str]:
        """Which frame edges this box sits within ``margin`` pixels of.

        Used to detect people cropped by the frame: if the top edge is touched
        the head may be outside the image, and asserting "no helmet" would be
        an unfounded claim rather than an honest detection failure.
        """
        touched: set[str] = set()
        if self.x1 <= margin:
            touched.add("left")
        if self.y1 <= margin:
            touched.add("top")
        if self.x2 >= width - margin:
            touched.add("right")
        if self.y2 >= height - margin:
            touched.add("bottom")
        return frozenset(touched)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Self:
        """Build from top-left corner plus extent."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> Self:
        """Build from centre plus extent (YOLO's native layout)."""
        return cls(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)


@dataclass(frozen=True, slots=True)
class Detection:
    """One object found in one frame, in the system's own vocabulary.

    Detector backends translate into this type; nothing downstream ever sees a
    backend-specific result object.
    """

    box: BBox
    label: PPEClass
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be in [0, 1], got {self.confidence}"
            raise GeometryError(msg)


@dataclass(frozen=True, slots=True)
class Track:
    """A person detection bound to a persistent identity across frames.

    Only people are tracked. Helmets and vests are small, jittery and
    low-confidence; tracking them would add ID-switch noise to the error
    breakdown while buying nothing, because association is recomputed per frame.
    """

    track_id: int
    detection: Detection
    frames_seen: int
    frames_since_update: int = 0

    @property
    def box(self) -> BBox:
        return self.detection.box

    @property
    def confidence(self) -> float:
        return self.detection.confidence


@dataclass(frozen=True, slots=True)
class AssociationMatch:
    """The winning link between a person and one PPE item.

    ``score`` and ``rule`` are kept so a reviewer -- or the Phase 5 error
    breakdown -- can ask *why* this helmet was attributed to this person.
    """

    item: Detection
    score: float
    rule: str


@dataclass(frozen=True, slots=True)
class PersonAssessment:
    """One person, one frame: the entire output of the rule layer.

    ``evaluated=False`` means "seen but deliberately not judged" -- outside the
    work zone, cropped by the frame edge, or too small to assess. That is
    materially different from "judged compliant", and conflating the two is how
    evaluation harnesses end up flattering themselves.
    """

    person: Detection
    matches: Mapping[PPEClass, AssociationMatch] = field(default_factory=dict)
    missing: frozenset[PPEClass] = frozenset()
    track_id: int | None = None
    zone_id: str | None = None
    evaluated: bool = True
    skip_reason: SkipReason | None = None

    @property
    def is_compliant(self) -> bool:
        """True only when the person was actually judged and lacked nothing."""
        return self.evaluated and not self.missing

    @property
    def is_violation(self) -> bool:
        return self.evaluated and bool(self.missing)


@dataclass(frozen=True, slots=True)
class ViolationEvent:
    """A confirmed, debounced non-compliance event.

    This is the unit the Phase 5 harness scores precision and recall over --
    not individual frames, and not individual detections.
    """

    track_id: int
    missing: frozenset[PPEClass]
    first_frame_index: int
    frame_index: int
    timestamp_s: float
    person_box: BBox
    streak_length: int
    mean_person_confidence: float
    zone_id: str | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
