"""HTTP routes.

Two ways in: upload a clip, or point the service at an RTSP URL. Both return
the same JSON event list, because the thing a caller wants -- who was
non-compliant, when -- does not depend on how the video arrived.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ppe.api.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    BoxModel,
    HealthResponse,
    ViolationEventModel,
)
from ppe.core.errors import PPEError, VideoSourceError
from ppe.core.types import ViolationEvent
from ppe.pipeline.runner import Pipeline
from ppe.pipeline.sources import RtspSource, VideoFileSource

router = APIRouter()


def _event_to_model(event: ViolationEvent) -> ViolationEventModel:
    return ViolationEventModel(
        event_id=event.event_id,
        track_id=event.track_id,
        missing=sorted(c.value for c in event.missing),
        frame_index=event.frame_index,
        timestamp_s=event.timestamp_s,
        person_box=BoxModel(
            x1=event.person_box.x1,
            y1=event.person_box.y1,
            x2=event.person_box.x2,
            y2=event.person_box.y2,
        ),
        zone_id=event.zone_id,
        streak_length=event.streak_length,
        mean_person_confidence=event.mean_person_confidence,
    )


def _require_detector(request: Request) -> object:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        reason = getattr(request.app.state, "detector_error", None) or "detector unavailable"
        raise HTTPException(status_code=503, detail=reason)
    return detector


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness plus enough identity to know what is actually running."""
    from ppe import __version__

    config = request.app.state.config
    detector = getattr(request.app.state, "detector", None)
    return HealthResponse(
        status="ok" if detector is not None else "degraded",
        version=__version__,
        backend=config.model.backend,
        device=detector.metadata.device if detector is not None else "none",
        weights_sha256=detector.metadata.weights_sha256 if detector is not None else None,
        blur_faces=config.privacy.blur_faces,
        detail=getattr(request.app.state, "detector_error", None),
    )


@router.get("/config")
def effective_config(request: Request) -> dict[str, object]:
    """The thresholds actually in force.

    Exposed deliberately: the argument for a separate rule layer is that the
    policy is inspectable, and a policy you cannot read from the running
    service is not inspectable.
    """
    dumped: dict[str, object] = request.app.state.config.model_dump(mode="json")
    return dumped


@router.post("/analyse/upload", response_model=AnalyseResponse)
async def analyse_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="Video clip to analyse.")],
) -> AnalyseResponse:
    """Analyse an uploaded video file."""
    detector = _require_detector(request)
    config = request.app.state.config

    max_bytes = config.api.max_upload_mb * 1024 * 1024
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        written = 0
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > max_bytes:
                handle.close()
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds api.max_upload_mb ({config.api.max_upload_mb} MB)",
                )
            handle.write(chunk)

    try:
        pipeline = Pipeline(config, detector)  # type: ignore[arg-type]
        with VideoFileSource(temp_path) as source:
            results = list(pipeline.run(source))
        events = [e for r in results for e in r.events]
        duration = results[-1].timestamp_s if results else 0.0
        return AnalyseResponse(
            source=file.filename or temp_path.name,
            frames_processed=len(results),
            duration_s=duration,
            events=[_event_to_model(e) for e in events],
        )
    except VideoSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PPEError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/analyse/stream", response_model=AnalyseResponse)
def analyse_stream(request: Request, body: AnalyseRequest) -> AnalyseResponse:
    """Analyse a bounded window of an RTSP stream.

    Bounded on purpose: an unbounded request would hold a worker forever. For
    continuous monitoring, run the pipeline as a process rather than behind an
    HTTP request.
    """
    detector = _require_detector(request)
    config = request.app.state.config

    if not body.rtsp_url.startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="rtsp_url must start with rtsp://")

    try:
        pipeline = Pipeline(config, detector)  # type: ignore[arg-type]
        results = []
        with RtspSource(body.rtsp_url) as source:
            for result in pipeline.run(source):
                results.append(result)
                if len(results) >= body.max_frames:
                    break
        events = [e for r in results for e in r.events]
        return AnalyseResponse(
            source=body.rtsp_url,
            frames_processed=len(results),
            duration_s=results[-1].timestamp_s if results else 0.0,
            events=[_event_to_model(e) for e in events],
        )
    except VideoSourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except PPEError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
