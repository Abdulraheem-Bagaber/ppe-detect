"""Binary geometric relations between boxes. Pure functions.

These are kept separate from :class:`~ppe.core.types.BBox` on purpose: a box
should know its own width, but "how much of this helmet lies inside that
person" is a *rule*, and every rule in the system should be readable in one
place.
"""

from __future__ import annotations

from ppe.core.types import BBox


def intersection(a: BBox, b: BBox) -> BBox | None:
    """The overlapping region of two boxes, or ``None`` if they are disjoint.

    Boxes that merely touch along an edge return ``None`` rather than a
    zero-area box: sharing a boundary is not overlapping.
    """
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def intersection_area(a: BBox, b: BBox) -> float:
    """Area of overlap; ``0.0`` when disjoint."""
    overlap = intersection(a, b)
    return overlap.area if overlap is not None else 0.0


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union.

    Poorly suited to helmet/person association -- a helmet is tiny relative to
    a person, so a perfectly worn helmet still scores a near-zero IoU. Provided
    for evaluation and NMS, not for association; use :func:`containment` there.
    """
    overlap = intersection_area(a, b)
    union = a.area + b.area - overlap
    return overlap / union if union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``, in ``[0, 1]``.

    This is the right measure for "is this helmet on this person": it is
    asymmetric and scale-tolerant, unlike IoU. A zero-area ``inner`` is treated
    as uncontained rather than dividing by zero.
    """
    if inner.area <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner.area


def centre_offset(item: BBox, person: BBox) -> float:
    """Horizontal distance from the person's centreline, normalised.

    Expressed as a fraction of half the person's width, so ``0.0`` is dead
    centre and ``1.0`` sits on the box edge. Values above ``1.0`` are outside
    the person box entirely. Discriminates between two adjacent workers when
    containment alone would accept either.

    A zero-width person yields ``inf``: nothing can be centred on nothing.
    """
    half_width = person.width / 2.0
    if half_width <= 0:
        return float("inf")
    return abs(item.centre[0] - person.centre[0]) / half_width


def vertical_band_overlap(item: BBox, band: BBox) -> float:
    """Fraction of ``item``'s height falling within ``band``'s vertical extent.

    Used with :meth:`ppe.core.types.BBox.band` to express "a worn helmet sits
    in the upper third of the person" without caring about horizontal position.
    A zero-height item is scored by whether its single line falls in the band.
    """
    top = max(item.y1, band.y1)
    bottom = min(item.y2, band.y2)
    covered = max(0.0, bottom - top)
    if item.height <= 0:
        return 1.0 if band.y1 <= item.y1 <= band.y2 else 0.0
    return covered / item.height


def visible_fraction(box: BBox, frame_size: tuple[int, int]) -> float:
    """How much of ``box`` lies inside the frame, in ``[0, 1]``.

    Drives the "do not accuse someone whose head is out of shot" guard.
    """
    if box.area <= 0:
        return 0.0
    width, height = frame_size
    frame = BBox(x1=0.0, y1=0.0, x2=float(width), y2=float(height))
    return intersection_area(box, frame) / box.area
