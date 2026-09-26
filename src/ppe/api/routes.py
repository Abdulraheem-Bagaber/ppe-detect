"""HTTP routes.

Two ways in: upload a clip, or point the service at an RTSP URL. Both return
the same JSON event list, because the thing a caller wants -- who was
non-compliant, when -- does not depend on how the video arrived.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from ppe.api.render import contact_sheet, write_video
from ppe.api.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    BoxModel,
    HealthResponse,
    ReviewResponse,
    ViolationEventModel,
)
from ppe.config.schema import AppConfig
from ppe.core.errors import PPEError, VideoSourceError
from ppe.core.types import Frame, ViolationEvent
from ppe.pipeline.runner import FrameResult, Pipeline
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


async def _save_upload(file: UploadFile, config: AppConfig) -> Path:
    """Stream an upload to a temp file, refusing anything over the size cap.

    Streamed in chunks rather than read whole: the cap exists to stop a large
    upload exhausting memory, and reading it all to check the size first would
    defeat that.
    """
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
    return temp_path


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

    temp_path = await _save_upload(file, config)

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


# --- local debug viewer --------------------------------------------------
# Not part of the deliverable. The product surface is the JSON event stream
# above and the CLI; everything below exists so a developer can watch what the
# rule layer decided and argue with it. Held to the same privacy rule as the
# rest of the service: faces are blurred before a frame is written anywhere.

REVIEW_ROOT = Path(tempfile.gettempdir()) / "ppe-review"
MAX_REVIEW_SECONDS = 300


@router.post("/review/upload", response_model=ReviewResponse)
async def review_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="Video clip to inspect.")],
    stride: Annotated[int, Query(ge=1, le=10)] = 1,
) -> ReviewResponse:
    """Analyse a clip and render an annotated copy to watch.

    Synchronous and bounded. A debug tool that needs a job queue is a product,
    and this is deliberately not one.
    """
    import cv2

    detector = _require_detector(request)
    config = request.app.state.config
    temp_path = await _save_upload(file, config)

    review_id = uuid.uuid4().hex
    out_dir = REVIEW_ROOT / review_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = Pipeline(config, detector)  # type: ignore[arg-type]
        detections: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        events: list[ViolationEvent] = []
        event_frames: list[tuple[Frame, FrameResult]] = []
        pairs: list[tuple[Frame, FrameResult]] = []

        with VideoFileSource(temp_path, stride=stride) as source:
            fps = source.fps
            limit = int(MAX_REVIEW_SECONDS * fps / stride)
            for source_frame in source:
                if len(pairs) >= limit:
                    break
                result = pipeline.process_frame(
                    source_frame.image, source_frame.index, source_frame.timestamp_s
                )
                pairs.append((source_frame.image, result))
                for d in result.detections:
                    detections[d.label.value] += 1
                for a in result.assessments:
                    if a.skip_reason is not None:
                        skipped[a.skip_reason.value] += 1
                if result.events:
                    events.extend(result.events)
                    event_frames.append((source_frame.image, result))

        if not pairs:
            raise HTTPException(status_code=400, detail="no frames could be read from that file")

        video_path = out_dir / "annotated.mp4"
        written = write_video(iter(pairs), video_path, fps / stride, config)

        sheet_url = None
        sheet = contact_sheet(event_frames, config)
        if sheet is not None:
            cv2.imwrite(str(out_dir / "events.png"), sheet)
            sheet_url = f"/review/{review_id}/events.png"

        return ReviewResponse(
            source=file.filename or temp_path.name,
            frames_processed=written,
            duration_s=pairs[-1][1].timestamp_s,
            fps=fps,
            blur_faces=config.privacy.blur_faces,
            video_url=f"/review/{review_id}/annotated.mp4",
            contact_sheet_url=sheet_url,
            events=[_event_to_model(e) for e in events],
            skipped=dict(skipped),
            detections=dict(detections),
        )
    except VideoSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PPEError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/review/{review_id}/{name}")
def review_asset(review_id: str, name: str) -> FileResponse:
    """Serve a rendered artefact. Ids are uuid4 hex, names are fixed."""
    if not re.fullmatch(r"[0-9a-f]{32}", review_id) or name not in {
        "annotated.mp4",
        "events.png",
    }:
        raise HTTPException(status_code=404, detail="not found")
    path = REVIEW_ROOT / review_id / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


@router.get("/", response_class=HTMLResponse)
def viewer() -> HTMLResponse:
    """The debug viewer page."""
    page = Path(__file__).parent / "static" / "viewer.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="viewer page missing")
    return HTMLResponse(page.read_text(encoding="utf-8"))
