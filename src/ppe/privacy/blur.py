"""Face blurring. Implemented in Phase 7.

Saudi PDPL treats identifiable faces in a workplace video stream as personal
data. Blurring is therefore applied *before* any frame is written to disk or
returned by the API, and it defaults to on: a misconfiguration should fail
closed, leaking nothing, rather than fail open.

With ``PrivacyConfig.strict`` set, a blur failure drops the frame instead of
emitting it unprocessed.
"""

from __future__ import annotations

from collections.abc import Sequence

from ppe.config.schema import PrivacyConfig
from ppe.core.types import BBox, Detection, Frame


def head_regions(persons: Sequence[Detection], config: PrivacyConfig) -> list[BBox]:
    """Head boxes derived from person boxes.

    Uses the person box's upper band rather than a face detector: it needs no
    extra model, and over-blurring is the safe direction to err in.
    """
    raise NotImplementedError


def blur_regions(frame: Frame, regions: Sequence[BBox], config: PrivacyConfig) -> Frame:
    """Return a copy of ``frame`` with each region obscured.

    Never mutates the input -- the unblurred frame stays available to the
    detector, which must not be handicapped by the privacy layer.
    """
    raise NotImplementedError


def anonymise(frame: Frame, persons: Sequence[Detection], config: PrivacyConfig) -> Frame:
    """The single entry point every output path must call before emitting."""
    raise NotImplementedError
