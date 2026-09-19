"""Frame sources. ``cv2`` lives here and in ``ppe.privacy``, nowhere else."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ppe.core.errors import VideoSourceError
from ppe.core.protocols import SourceFrame

DEFAULT_FPS = 30.0


class VideoFileSource:
    """Frames from a video file on disk."""

    def __init__(self, path: str | Path, stride: int = 1) -> None:
        if stride < 1:
            msg = f"stride must be at least 1, got {stride}"
            raise VideoSourceError(msg)
        self._path = Path(path)
        self._stride = stride
        if not self._path.exists():
            msg = f"video file not found: {self._path}"
            raise VideoSourceError(msg)
        self._capture = self._open()

    def _open(self) -> object:
        import cv2

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            msg = f"could not open video: {self._path}"
            raise VideoSourceError(msg)
        return capture

    @property
    def fps(self) -> float:
        import cv2

        reported = float(self._capture.get(cv2.CAP_PROP_FPS))  # type: ignore[attr-defined]
        # A malformed container reports 0 or nan; a wrong frame rate silently
        # corrupts every timestamp and therefore the cooldown maths.
        return reported if reported and np.isfinite(reported) and reported > 0 else DEFAULT_FPS

    @property
    def frame_size(self) -> tuple[int, int]:
        import cv2

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[attr-defined]
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[attr-defined]
        return width, height

    def __iter__(self) -> Iterator[SourceFrame]:
        fps = self.fps
        index = 0
        while True:
            ok, image = self._capture.read()  # type: ignore[attr-defined]
            if not ok:
                break
            if index % self._stride == 0:
                yield SourceFrame(index=index, timestamp_s=index / fps, image=image)
            index += 1

    def close(self) -> None:
        self._capture.release()  # type: ignore[attr-defined]

    def __enter__(self) -> VideoFileSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RtspSource:
    """Frames from a live RTSP stream.

    Differs from a file source in one important way: it drops frames under load
    rather than falling behind, so ``SourceFrame.index`` is not dense.
    Everything downstream keys off the index rather than assuming continuity,
    and the debouncer's ``max_gap_frames`` is what absorbs the difference.
    """

    def __init__(
        self, url: str, reconnect_attempts: int = 3, reconnect_delay_s: float = 2.0
    ) -> None:
        self._url = url
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay_s = reconnect_delay_s
        self._capture = self._open()

    def _open(self) -> object:
        import cv2

        capture = cv2.VideoCapture(self._url)
        if not capture.isOpened():
            msg = f"could not open RTSP stream: {self._url}"
            raise VideoSourceError(msg)
        # Keep the buffer shallow so we get the newest frame, not a stale one.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @property
    def fps(self) -> float:
        import cv2

        reported = float(self._capture.get(cv2.CAP_PROP_FPS))  # type: ignore[attr-defined]
        return reported if reported and np.isfinite(reported) and reported > 0 else DEFAULT_FPS

    @property
    def frame_size(self) -> tuple[int, int]:
        import cv2

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[attr-defined]
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[attr-defined]
        return width, height

    def __iter__(self) -> Iterator[SourceFrame]:
        index = 0
        start = time.monotonic()
        failures = 0
        while True:
            ok, image = self._capture.read()  # type: ignore[attr-defined]
            if not ok:
                failures += 1
                if failures > self._reconnect_attempts:
                    return
                time.sleep(self._reconnect_delay_s)
                self.close()
                self._capture = self._open()
                continue
            failures = 0
            # Wall-clock timestamps: a live stream has no frame count to trust.
            yield SourceFrame(index=index, timestamp_s=time.monotonic() - start, image=image)
            index += 1

    def close(self) -> None:
        self._capture.release()  # type: ignore[attr-defined]

    def __enter__(self) -> RtspSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ArraySource:
    """Frames from an in-memory list. For tests and the evaluation harness.

    Lets the whole pipeline be exercised without a video file, which keeps the
    Phase 5 harness reproducible and the tests fast.
    """

    def __init__(self, frames: list[np.ndarray], fps: float = DEFAULT_FPS) -> None:
        if fps <= 0:
            msg = f"fps must be positive, got {fps}"
            raise VideoSourceError(msg)
        self._frames = frames
        self._fps = fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        if not self._frames:
            return (0, 0)
        height, width = self._frames[0].shape[:2]
        return width, height

    def __iter__(self) -> Iterator[SourceFrame]:
        for index, image in enumerate(self._frames):
            yield SourceFrame(index=index, timestamp_s=index / self._fps, image=image)

    def close(self) -> None:
        """Nothing to release."""
