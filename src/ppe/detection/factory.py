"""Backend selection.

Imports the chosen backend lazily, so running the rule-layer tests never loads
torch or onnxruntime.
"""

from __future__ import annotations

from ppe.config.schema import ModelConfig
from ppe.core.errors import ConfigError
from ppe.core.protocols import Detector

BACKENDS = ("ultralytics", "onnx")


def build_detector(config: ModelConfig) -> Detector:
    """Construct the detector named by ``config.backend``.

    Raises:
        ConfigError: The backend name is unknown.
    """
    if config.backend == "ultralytics":
        from ppe.detection.ultralytics_detector import UltralyticsDetector

        return UltralyticsDetector(config)
    if config.backend == "onnx":
        from ppe.detection.onnx_detector import OnnxDetector

        return OnnxDetector(config)

    msg = f"unknown detector backend {config.backend!r}; expected one of {BACKENDS}"
    raise ConfigError(msg)
