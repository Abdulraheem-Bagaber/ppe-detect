"""Frame sources. ``cv2`` lives here and in ``ppe.privacy``, nowhere else.

Implemented in Phase 6.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ppe.core.protocols import SourceFrame


class VideoFileSource:
    """Frames from a video file on disk."""

    def __init__(self, path: str | Path, stride: int = 1) -> None:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def frame_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[SourceFrame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class RtspSource:
    """Frames from a live RTSP stream.

    Differs from a file source in one important way: it drops frames under load
    rather than falling behind, so ``SourceFrame.index`` is not dense.
    Everything downstream keys off the index rather than assuming continuity.
    """

    def __init__(self, url: str, reconnect_attempts: int = 3) -> None:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def frame_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[SourceFrame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
