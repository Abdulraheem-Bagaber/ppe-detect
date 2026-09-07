"""ByteTrack via ``supervision``. Implemented in Phase 4.

Only person detections are fed in. Helmets and vests are re-associated every
frame instead, so a helmet flickering out for two frames cannot manufacture a
tracking identity change.
"""

from __future__ import annotations

from collections.abc import Sequence

from ppe.config.schema import TrackingConfig
from ppe.core.types import Detection, Track


class ByteTrackTracker:
    """Adapter from ``supervision.ByteTrack`` to the ``Tracker`` protocol."""

    def __init__(self, config: TrackingConfig) -> None:
        raise NotImplementedError

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Track]:
        """Feed one frame of person detections; return the live tracks."""
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
