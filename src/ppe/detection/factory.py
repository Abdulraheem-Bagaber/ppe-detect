"""Backend selection.

Imports the chosen backend lazily, so running the rule-layer tests never loads
torch or onnxruntime.
"""

from __future__ import annotations

from ppe.config.schema import ModelConfig
from ppe.core.protocols import Detector


def build_detector(config: ModelConfig) -> Detector:
    """Construct the detector named by ``config.backend``.

    Raises:
        ConfigError: The backend name is unknown.
    """
    raise NotImplementedError
