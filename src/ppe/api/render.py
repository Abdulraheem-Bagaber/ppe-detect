"""Drawing detections onto frames, for the local debug viewer.

Not part of the served product. The deliverables are the JSON event stream and
the CLI; this exists so a developer can look at what the rule layer decided
and disagree with it.

One rule is not negotiable here: :func:`ppe.privacy.blur.anonymise` runs before
anything is drawn, so every frame this module emits has already had faces
obscured. A debug tool that leaked faces would be a worse leak than the API,
not a lesser one, because nobody audits debug tools.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from ppe.config.schema import AppConfig
from ppe.core.types import Frame, PersonAssessment, PPEClass
from ppe.pipeline.runner import FrameResult
from ppe.privacy.blur import anonymise

# BGR, because OpenCV.
COLOURS: dict[str, tuple[int, int, int]] = {
    PPEClass.PERSON.value: (255, 0, 255),
    PPEClass.HELMET.value: (0, 220, 255),
    PPEClass.VEST.value: (0, 165, 255),
}
COMPLIANT = (0, 200, 0)
VIOLATION = (0, 0, 255)
UNJUDGED = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def assessment_colour(assessment: PersonAssessment) -> tuple[int, int, int]:
    """Green compliant, red violation, grey not judged.

    The grey is the point of the whole design: "we could not tell" is a third
    outcome, and a viewer that only had two colours would hide it.
    """
    if not assessment.evaluated:
        return UNJUDGED
    return VIOLATION if assessment.missing else COMPLIANT


def annotate(frame: Frame, result: FrameResult, config: AppConfig) -> Frame:
    """Draw one frame's detections and verdicts. Faces are blurred first."""
    people = [d for d in result.detections if d.label is PPEClass.PERSON]
    canvas = anonymise(frame, people, config.privacy).copy()

    # Raw detections underneath, thin, so the verdict boxes read on top.
    for detection in result.detections:
        if detection.label is PPEClass.PERSON:
            continue
        box = detection.box
        colour = COLOURS[detection.label.value]
        cv2.rectangle(canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), colour, 1)
        cv2.putText(
            canvas,
            f"{detection.label.value} {detection.confidence:.2f}",
            (int(box.x1), max(10, int(box.y1) - 3)),
            FONT,
            0.35,
            colour,
            1,
            cv2.LINE_AA,
        )

    for assessment in result.assessments:
        _draw_person(canvas, assessment)

    _draw_banner(canvas, result, config)
    return canvas


def _draw_person(canvas: Frame, assessment: PersonAssessment) -> None:
    box = assessment.person.box
    colour = assessment_colour(assessment)
    cv2.rectangle(canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), colour, 2)

    if not assessment.evaluated:
        label = f"not judged: {assessment.skip_reason.value if assessment.skip_reason else '?'}"
    elif assessment.missing:
        label = "NO " + " + ".join(sorted(c.value for c in assessment.missing))
    else:
        label = "ok"
    if assessment.track_id is not None:
        label = f"#{assessment.track_id} {label}"

    (width, height), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    top = max(height + 4, int(box.y1))
    cv2.rectangle(
        canvas, (int(box.x1), top - height - 4), (int(box.x1) + width + 6, top), colour, -1
    )
    cv2.putText(
        canvas, label, (int(box.x1) + 3, top - 3), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )


def _draw_banner(canvas: Frame, result: FrameResult, config: AppConfig) -> None:
    """Timestamp, and a loud strip whenever an event fires on this frame."""
    cv2.putText(
        canvas,
        f"t={result.timestamp_s:6.2f}s  f={result.frame_index}",
        (8, 18),
        FONT,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if config.privacy.blur_faces:
        cv2.putText(canvas, "faces blurred", (8, 34), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    if not result.events:
        return
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, height - 28), (width, height), VIOLATION, -1)
    missing = " / ".join(
        f"#{e.track_id} {'+'.join(sorted(c.value for c in e.missing))}" for e in result.events
    )
    cv2.putText(
        canvas,
        f"VIOLATION  {missing}",
        (8, height - 9),
        FONT,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def write_video(
    frames: Iterator[tuple[Frame, FrameResult]],
    destination: Path,
    fps: float,
    config: AppConfig,
) -> int:
    """Annotate and encode. Returns the number of frames written.

    mp4v rather than H.264: OpenCV's H.264 support depends on how the wheel was
    built, and a debug tool should not fail because of a codec. Browsers play
    mp4v inconsistently, so the viewer falls back to frame-by-frame if needed.
    """
    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        for frame, result in frames:
            annotated = annotate(frame, result, config)
            if writer is None:
                height, width = annotated.shape[:2]
                # cv2.VideoWriter_fourcc is generated at runtime, so mypy
                # cannot see it on the module.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))
                if not writer.isOpened():
                    msg = f"could not open a video writer for {destination}"
                    raise RuntimeError(msg)
            writer.write(annotated)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return written


def contact_sheet(
    frames: Sequence[tuple[Frame, FrameResult]], config: AppConfig, columns: int = 3
) -> Frame | None:
    """A grid of the frames where events fired. Handy when video playback fails."""
    if not frames:
        return None
    tiles = [annotate(frame, result, config) for frame, result in frames]
    height, width = tiles[0].shape[:2]
    blank = np.zeros_like(tiles[0])

    rows = []
    for start in range(0, len(tiles), columns):
        row = list(tiles[start : start + columns])
        row += [blank] * (columns - len(row))
        rows.append(np.hstack([cv2.resize(t, (width, height)) for t in row]))
    return np.vstack(rows)
