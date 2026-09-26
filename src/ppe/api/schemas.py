"""Wire format for the API.

Separate from ``ppe.core.types`` deliberately: the internal vocabulary should
be free to change without breaking a published JSON contract, and the two
change for different reasons.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoxModel(BaseModel):
    """A box as it appears on the wire, in absolute pixels."""

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
    streak_length: int
    mean_person_confidence: float


class AnalyseRequest(BaseModel):
    """Ask the service to watch a bounded window of a stream."""

    rtsp_url: str
    max_frames: int = Field(default=900, gt=0, le=100_000)


class AnalyseResponse(BaseModel):
    """Result of analysing an uploaded clip or stream segment."""

    source: str
    frames_processed: int
    duration_s: float
    events: list[ViolationEventModel]


class ReviewResponse(BaseModel):
    """Result of a debug-viewer run: events plus a rendered video to watch."""

    source: str
    frames_processed: int
    duration_s: float
    fps: float
    blur_faces: bool
    video_url: str | None = None
    contact_sheet_url: str | None = None
    events: list[ViolationEventModel]
    skipped: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of people seen but deliberately not judged, by reason.",
    )
    detections: dict[str, int] = Field(
        default_factory=dict, description="Raw detection counts by class."
    )


class HealthResponse(BaseModel):
    """Liveness plus enough identity to know what is actually running.

    ``blur_faces`` is reported so an operator can verify the privacy posture
    of a running instance without reading its config file.
    """

    status: str
    version: str
    backend: str
    device: str
    weights_sha256: str | None = None
    blur_faces: bool = True
    detail: str | None = None
