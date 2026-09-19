"""Face blurring.

Saudi PDPL treats identifiable faces in a workplace video stream as personal
data. Blurring is therefore applied *before* any frame is written to disk or
returned by the API, and it defaults to on: a misconfiguration should fail
closed, leaking nothing, rather than fail open.

Detection still runs on the unblurred frame, so privacy costs no accuracy.
That ordering is the whole design -- blur on the way out, never on the way in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ppe.config.schema import PrivacyConfig
from ppe.core.types import BBox, Detection, Frame, PPEClass


def head_regions(persons: Sequence[Detection], config: PrivacyConfig) -> list[BBox]:
    """Head boxes derived from person boxes.

    Uses the person box's upper band rather than a face detector: it needs no
    extra model, it cannot miss a face that the face model would have missed,
    and over-blurring is the safe direction to err in. A face detector would
    be more precise and less safe.
    """
    return [
        person.box.band(0.0, config.head_band_bottom)
        for person in persons
        if person.label is PPEClass.PERSON and person.box.height > 0
    ]


def blur_regions(frame: Frame, regions: Sequence[BBox], config: PrivacyConfig) -> Frame:
    """Return a copy of ``frame`` with each region obscured.

    Never mutates the input: the unblurred frame stays available to the
    detector, which must not be handicapped by the privacy layer.
    """
    output = frame.copy()
    height, width = frame.shape[:2]

    for region in regions:
        clipped = region.clip_to(width, height)
        x1, y1 = int(clipped.x1), int(clipped.y1)
        x2, y2 = int(np.ceil(clipped.x2)), int(np.ceil(clipped.y2))
        if x2 <= x1 or y2 <= y1:
            continue

        patch = output[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        output[y1:y2, x1:x2] = (
            _pixelate(patch, config) if config.method == "pixelate" else _box_blur(patch, config)
        )

    return output


def _kernel_for(patch: Frame, config: PrivacyConfig) -> int:
    """Kernel size as a fraction of the patch, so blur scales with distance.

    A fixed pixel kernel leaves a distant worker's face perfectly legible while
    turning a near one into mush.
    """
    span = max(patch.shape[0], patch.shape[1])
    return max(3, int(span * config.kernel_fraction) | 1)


def _box_blur(patch: Frame, config: PrivacyConfig) -> Frame:
    """Separable box blur, applied repeatedly to approximate a Gaussian.

    Implemented with numpy rather than cv2 so the privacy guarantee does not
    depend on the video stack being importable -- this code must work in the
    slimmest container we can build.
    """
    kernel = _kernel_for(patch, config)
    blurred = patch.astype(np.float32)
    for _ in range(3):  # three box passes approximate a Gaussian well
        blurred = _box_pass(blurred, kernel)
    return blurred.astype(np.uint8)


def _box_pass(patch: np.ndarray, kernel: int) -> np.ndarray:
    padded = np.pad(patch, ((kernel, kernel), (kernel, kernel), (0, 0)), mode="edge")
    cumulative = padded.cumsum(axis=0).cumsum(axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0), (0, 0)), mode="constant")

    height, width = patch.shape[:2]
    half = kernel // 2
    top = kernel - half
    left = kernel - half
    y0, y1 = top, top + height
    x0, x1 = left, left + width
    size = kernel

    total = (
        cumulative[y0 + size : y1 + size, x0 + size : x1 + size]
        - cumulative[y0:y1, x0 + size : x1 + size]
        - cumulative[y0 + size : y1 + size, x0:x1]
        + cumulative[y0:y1, x0:x1]
    )
    return np.asarray(total / (size * size), dtype=np.float32)


def _pixelate(patch: Frame, config: PrivacyConfig) -> Frame:
    """Downsample then nearest-neighbour upsample. Irreversible by construction."""
    kernel = _kernel_for(patch, config)
    height, width = patch.shape[:2]
    small_h = max(1, height // kernel)
    small_w = max(1, width // kernel)

    ys = np.linspace(0, height - 1, small_h).astype(int)
    xs = np.linspace(0, width - 1, small_w).astype(int)
    small = patch[ys][:, xs]

    up_y = np.linspace(0, small_h - 1, height).astype(int)
    up_x = np.linspace(0, small_w - 1, width).astype(int)
    return np.asarray(small[up_y][:, up_x], dtype=np.uint8)


def anonymise(frame: Frame, persons: Sequence[Detection], config: PrivacyConfig) -> Frame:
    """The single entry point every output path must call before emitting.

    With ``strict`` set, a blur failure raises rather than returning the
    original frame. Emitting an unblurred face because the blur crashed is the
    one outcome this module exists to prevent.
    """
    if not config.blur_faces:
        return frame
    try:
        return blur_regions(frame, head_regions(persons, config), config)
    except Exception:
        if config.strict:
            raise
        # Fail closed even when not strict: a blank frame leaks nothing.
        return np.zeros_like(frame)
