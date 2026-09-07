"""Wire format for the API. Phase 6.

Separate from ``ppe.core.types`` deliberately: the internal vocabulary should
be free to change without breaking a published JSON contract.
"""

from __future__ import annotations

from pydantic import BaseModel


class BoxModel(BaseModel):
    """A box as it appears on the wire."""

    x1: float
    y1: float
    x2: float
    y2: float


class ViolationEventModel(BaseModel):
    """A violation event as it appears on the wire."""

    event_id: str
    track_id: int
    missing: list[str]
    frame_index: int
    timestamp_s: float
    person_box: BoxModel
    zone_id: str | None = None


class AnalyseResponse(BaseModel):
    """Result of analysing an uploaded clip or stream segment."""

    source: str
    frames_processed: int
    duration_s: float
    events: list[ViolationEventModel]


class HealthResponse(BaseModel):
    """Liveness plus enough identity to know what is actually running."""

    status: str
    version: str
    backend: str
    device: str
