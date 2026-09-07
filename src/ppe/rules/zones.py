"""Work-area polygons. Pure functions, implemented in Phase 4.

Deliberately *not* ``supervision.PolygonZone``: supervision imports ``cv2``,
which this package must never pull in. Point-in-polygon by ray casting is
about twenty lines of numpy and is far easier to unit-test than a wrapper.
``supervision`` is still used for tracking and annotation, where it belongs.
"""

from __future__ import annotations

from collections.abc import Sequence

from ppe.config.schema import ZoneConfig, ZonesConfig
from ppe.core.types import BBox, Detection

Point = tuple[float, float]
Polygon = Sequence[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting containment test. Points on an edge count as inside."""
    raise NotImplementedError


def anchor_point(box: BBox, anchor: str) -> Point:
    """The point on a person box tested against the zone.

    ``bottom_centre`` -- where the feet are -- is the default: a worker leaning
    over a barrier should count as inside the area they are standing in, not
    the one their head is over.
    """
    raise NotImplementedError


def zone_for(person: Detection, config: ZonesConfig) -> ZoneConfig | None:
    """The first enabled zone containing this person, if any."""
    raise NotImplementedError


def in_scope(person: Detection, config: ZonesConfig) -> bool:
    """Whether this person should be evaluated at all.

    With no zones configured and ``evaluate_when_no_zones`` set, the whole
    frame is in scope.
    """
    raise NotImplementedError
