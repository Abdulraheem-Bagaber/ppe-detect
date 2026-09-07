"""ONNX Runtime backend -- the served inference path. Phase 6.

Preprocessing and NMS are ours here rather than the framework's, which is
exactly why Phase 6 requires a parity test asserting the maximum absolute
difference against the PyTorch model stays under threshold. An export that
silently changes behaviour is worse than no export.
"""

from __future__ import annotations

from ppe.config.schema import ModelConfig
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import Detection, Frame


class OnnxDetector:
    """Wraps an ONNX graph behind the ``Detector`` protocol."""

    def __init__(self, config: ModelConfig) -> None:
        raise NotImplementedError

    @property
    def metadata(self) -> DetectorMetadata:
        raise NotImplementedError

    def predict(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError
