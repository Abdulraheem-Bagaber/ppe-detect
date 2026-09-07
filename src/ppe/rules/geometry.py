"""Binary geometric relations between boxes. Pure functions, Phase 3.

These are kept separate from :class:`~ppe.core.types.BBox` on purpose: a box
should know its own width, but "how much of this helmet lies inside that
person" is a *rule*, and every rule in the system should be readable in one
place.
"""

from __future__ import annotations

from ppe.core.types import BBox


def intersection(a: BBox, b: BBox) -> BBox | None:
    """The overlapping region of two boxes, or ``None`` if they are disjoint."""
    raise NotImplementedError


def intersection_area(a: BBox, b: BBox) -> float:
    """Area of overlap; ``0.0`` when disjoint."""
    raise NotImplementedError


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union.

    Poorly suited to helmet/person association -- a helmet is tiny relative to
    a person, so a perfectly worn helmet still scores a near-zero IoU. Provided
    for evaluation and NMS, not for association; use :func:`containment` there.
    """
    raise NotImplementedError


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``, in ``[0, 1]``.

    This is the right measure for "is this helmet on this person": it is
    asymmetric and scale-tolerant, unlike IoU.
    """
    raise NotImplementedError


def centre_offset(item: BBox, person: BBox) -> float:
    """Horizontal distance from the person's centreline, normalised.

    Expressed as a fraction of half the person's width, so ``0.0`` is dead
    centre and ``1.0`` sits on the box edge. Values above ``1.0`` are outside
    the person box entirely. Discriminates between two adjacent workers when
    containment alone would accept either.
    """
    raise NotImplementedError


def vertical_band_overlap(item: BBox, band: BBox) -> float:
    """Fraction of ``item``'s height falling within ``band``'s vertical extent.

    Used with :meth:`ppe.core.types.BBox.band` to express "a worn helmet sits
    in the upper third of the person" without caring about horizontal position.
    """
    raise NotImplementedError
