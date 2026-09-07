"""Exception hierarchy.

A single root lets callers (the CLI, the API) distinguish "our" failures from
genuine bugs without catching bare ``Exception``.
"""

from __future__ import annotations


class PPEError(Exception):
    """Root of every error this package raises deliberately."""


class ConfigError(PPEError):
    """Configuration is missing, malformed, or internally inconsistent."""


class GeometryError(PPEError):
    """A box or polygon violates an invariant (negative extent, too few vertices)."""


class DetectorError(PPEError):
    """A detector backend failed to load weights or run inference."""


class VideoSourceError(PPEError):
    """A video file or RTSP stream could not be opened or read."""
