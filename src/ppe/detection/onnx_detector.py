"""ONNX Runtime backend -- the served inference path.

Preprocessing and NMS are ours here rather than the framework's, which is
exactly why the export script asserts output parity against the PyTorch model.
An export that silently changes behaviour is worse than no export: every
metric measured on the trained weights would quietly stop applying to what is
actually running in production.

YOLOv8 exports a single output of shape ``(1, 4 + num_classes, num_anchors)``:
four box coordinates in ``cxcywh`` on the letterboxed image, then one score
per class. There is no objectness channel and no NMS baked in.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ppe.config.schema import ModelConfig
from ppe.core.errors import DetectorError
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import BBox, Detection, Frame
from ppe.detection.base import letterbox, resolve_device, sha256_of, to_detections


class OnnxDetector:
    """Wraps an ONNX graph behind the ``Detector`` protocol."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._session = self._load()

    def _load(self) -> Any:
        import onnxruntime as ort

        weights = Path(self._config.weights)
        if not weights.exists():
            msg = (
                f"ONNX model not found at {weights}. Export one first: "
                f"python scripts/export_onnx.py"
            )
            raise DetectorError(msg)

        providers = ["CPUExecutionProvider"]
        if (
            self._device.startswith("cuda")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")

        try:
            return ort.InferenceSession(str(weights), providers=providers)
        except Exception as exc:
            msg = f"could not load ONNX model from {weights}: {exc}"
            raise DetectorError(msg) from exc

    @cached_property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name="onnx",
            weights=str(self._config.weights),
            weights_sha256=sha256_of(self._config.weights),
            imgsz=self._config.imgsz,
            conf_threshold=self._config.conf_threshold,
            iou_threshold=self._config.iou_threshold,
            device=self._device,
            class_map=dict(self._config.class_map),
        )

    def predict(self, frame: Frame) -> list[Detection]:
        """Detect objects in a single BGR frame."""
        padded, scale, pad_x, pad_y = letterbox(frame, self._config.imgsz)

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 in [0, 1].
        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor[None])

        try:
            outputs = self._session.run(None, {self._session.get_inputs()[0].name: tensor})
        except Exception as exc:
            msg = f"ONNX inference failed: {exc}"
            raise DetectorError(msg) from exc

        boxes, class_ids, scores = self._decode(outputs[0], scale, pad_x, pad_y)
        return to_detections(
            boxes_xyxy=boxes,
            class_ids=class_ids,
            confidences=scores,
            class_map=self._config.class_map,
            conf_threshold=self._config.conf_threshold,
        )

    def _decode(
        self, raw: npt.NDArray[np.floating], scale: float, pad_x: float, pad_y: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Turn raw YOLOv8 output into boxes in original-image pixels."""
        predictions = np.squeeze(raw, axis=0).T  # (anchors, 4 + classes)
        if predictions.shape[1] < 5:
            msg = f"unexpected ONNX output shape {raw.shape}; expected (1, 4+nc, anchors)"
            raise DetectorError(msg)

        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)

        keep = scores >= self._config.conf_threshold
        predictions, class_ids, scores = predictions[keep], class_ids[keep], scores[keep]
        if len(predictions) == 0:
            return (
                np.empty((0, 4), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float64),
            )

        cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        # Undo the letterbox before NMS, so the IoU threshold means the same
        # thing here as it does everywhere else in the system.
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

        kept = non_max_suppression(
            xyxy, scores, class_ids, self._config.iou_threshold, self._config.max_detections
        )
        return xyxy[kept].astype(np.float64), class_ids[kept], scores[kept].astype(np.float64)


def non_max_suppression(
    boxes: npt.NDArray[np.floating],
    scores: npt.NDArray[np.floating],
    class_ids: npt.NDArray[np.integer],
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Class-wise greedy NMS in numpy.

    Class-wise rather than global: a helmet sitting on a person's head overlaps
    that person heavily, and a global NMS would suppress one of them. That
    would break association in a way that looks like a rule-layer bug.
    """
    keep: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(-scores[indices])]

        while order.size > 0:
            best = int(order[0])
            keep.append(best)
            if order.size == 1:
                break
            rest = order[1:]
            overlaps = _iou_batch(boxes[best], boxes[rest])
            order = rest[overlaps < iou_threshold]

    keep.sort(key=lambda i: -scores[i])
    return keep[:max_detections]


def _iou_batch(
    box: npt.NDArray[np.floating], others: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - overlap
    return np.where(union > 0, overlap / union, 0.0)


def boxes_as_bboxes(xyxy: npt.NDArray[np.floating]) -> list[BBox]:
    """Convenience for the parity test, which compares boxes not detections."""
    return [BBox(float(a), float(b), float(c), float(d)) for a, b, c, d in xyxy]
