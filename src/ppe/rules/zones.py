"""Work-area polygons. Pure functions.

Deliberately *not* ``supervision.PolygonZone``: supervision imports ``cv2``,
which this package must never pull in. Point-in-polygon by ray casting is
about thirty lines and is far easier to unit-test than a wrapper. Supervision
is still used for tracking and annotation, where it belongs.
"""

from __future__ import annotations

from collections.abc import Sequence

from ppe.config.schema import ZoneConfig, ZonesConfig
from ppe.core.errors import GeometryError
from ppe.core.types import BBox, Detection

Point = tuple[float, float]
Polygon = Sequence[Point]

# Anchors, in fractions of the box. bottom_centre is where the feet are, which
# is the right default: a worker leaning over a barrier belongs to the area
# they are standing in, not the one their head is above.
ANCHORS: dict[str, tuple[float, float]] = {
    "bottom_centre": (0.5, 1.0),
    "centre": (0.5, 0.5),
    "top_centre": (0.5, 0.0),
}


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting containment test. Points on an edge count as inside.

    The on-edge case is handled explicitly rather than left to floating-point
    luck, because a zone boundary drawn along a walkway will have people
    standing exactly on it.
    """
    if len(polygon) < 3:
        msg = f"a polygon needs at least 3 vertices, got {len(polygon)}"
        raise GeometryError(msg)

    x, y = point
    inside = False
    count = len(polygon)

    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]

        if _on_segment(point, (x1, y1), (x2, y2)):
            return True

        # Does a ray cast to the right from `point` cross this edge?
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing_x > x:
                inside = not inside

    return inside


def _on_segment(point: Point, a: Point, b: Point, tolerance: float = 1e-9) -> bool:
    """Whether ``point`` lies on the segment ``a``-``b``."""
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    return min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance and (
        min(ay, by) - tolerance <= py <= max(ay, by) + tolerance
    )


def anchor_point(box: BBox, anchor: str) -> Point:
    """The point on a person box tested against the zone."""
    try:
        fx, fy = ANCHORS[anchor]
    except KeyError as exc:
        msg = f"unknown zone anchor {anchor!r}; expected one of {sorted(ANCHORS)}"
        raise GeometryError(msg) from exc
    return box.x1 + fx * box.width, box.y1 + fy * box.height


def zone_for(person: Detection, config: ZonesConfig) -> ZoneConfig | None:
    """The first enabled zone containing this person, if any."""
    point = anchor_point(person.box, config.anchor)
    for zone in config.zones:
        if zone.enabled and point_in_polygon(point, zone.polygon):
            return zone
    return None


def in_scope(person: Detection, config: ZonesConfig) -> bool:
    """Whether this person should be evaluated at all.

    With no enabled zones configured and ``evaluate_when_no_zones`` set, the
    whole frame is the work area.
    """
    if not any(zone.enabled for zone in config.zones):
        return config.evaluate_when_no_zones
    return zone_for(person, config) is not None
