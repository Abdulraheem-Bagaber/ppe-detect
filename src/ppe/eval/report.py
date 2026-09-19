"""Rendering evaluation results.

Two rules here, both inherited from the project's first principle about
numbers. Nothing is rounded into looking better than it is, and a metric with
no data behind it prints ``n/a`` rather than ``0.00`` -- because a precision of
zero and a precision nobody measured are very different claims.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ppe.core.protocols import DetectorMetadata
from ppe.eval.diagnosis import Diagnosis, FalsePositiveCause, summarise
from ppe.eval.scoring import ClipScore, Metrics


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_report(
    metrics: Metrics,
    scores: Sequence[ClipScore],
    diagnoses: Sequence[Diagnosis],
    detector: DetectorMetadata | None = None,
) -> str:
    """The human-readable evaluation report."""
    lines: list[str] = ["PPE violation evaluation", "=" * 48, ""]

    if detector is not None:
        lines += [
            "Model under test",
            f"  backend   {detector.name}",
            f"  weights   {detector.weights}",
            f"  sha256    {detector.weights_sha256[:16]}...",
            f"  imgsz     {detector.imgsz}   conf {detector.conf_threshold}   "
            f"iou {detector.iou_threshold}",
            f"  device    {detector.device}",
            "",
        ]

    lines += [
        "Violation-level metrics  (the headline; detection mAP is not)",
        f"  annotated violations   {metrics.intervals_total}",
        f"  detected               {metrics.intervals_detected}",
        f"  events fired           {metrics.true_positive_events + metrics.false_positive_events}",
        f"  false alarms           {metrics.false_positive_events}",
        "",
        f"  precision              {_pct(metrics.precision)}",
        f"  recall                 {_pct(metrics.recall)}",
        f"  F1                     {_pct(metrics.f1)}",
        "  false alarms / hour    "
        + (
            "n/a"
            if metrics.false_alarms_per_hour is None
            else f"{metrics.false_alarms_per_hour:.2f}"
        ),
        f"  footage scored         {metrics.duration_s / 60:.1f} min",
        "",
    ]

    lines += _false_positive_section(diagnoses)
    lines += _missed_section(scores)
    lines += _per_clip_section(scores)
    return "\n".join(lines)


def _false_positive_section(diagnoses: Sequence[Diagnosis]) -> list[str]:
    lines = ["Why each false alarm happened", "-" * 48]
    if not diagnoses:
        lines += ["  no false alarms to explain", ""]
        return lines

    counts = summarise(diagnoses)
    total = sum(counts.values())
    for cause in FalsePositiveCause:
        count = counts.get(cause.value, 0)
        share = f"{100.0 * count / total:5.1f}%" if total else "  n/a"
        lines.append(f"  {cause.value:<26}{count:>5}  {share}")

    fixable = sum(1 for d in diagnoses if d.is_fixable_in_config)
    lines += [
        "",
        f"  {fixable} of {total} are config work, not retraining.",
        "",
        "  Examples:",
    ]
    for diagnosis in diagnoses[:5]:
        lines.append(
            f"    frame {diagnosis.event.frame_index:>6}  track {diagnosis.event.track_id:<4} "
            f"{diagnosis.cause.value}"
        )
        lines.append(f"      {diagnosis.detail}")
    lines.append("")
    return lines


def _missed_section(scores: Sequence[ClipScore]) -> list[str]:
    missed = [(s.clip, v) for s in scores for v in s.missed]
    lines = ["Violations we failed to raise", "-" * 48]
    if not missed:
        lines += ["  none", ""]
        return lines
    for clip, interval in missed[:10]:
        items = ", ".join(sorted(c.value for c in interval.missing))
        lines.append(
            f"  {clip}  {interval.person_label:<12} {items:<16} "
            f"{interval.start_s:.1f}-{interval.end_s:.1f}s"
        )
    if len(missed) > 10:
        lines.append(f"  ... and {len(missed) - 10} more")
    lines.append("")
    return lines


def _per_clip_section(scores: Sequence[ClipScore]) -> list[str]:
    header = f"  {'clip':<32}{'annotated':>10}{'found':>7}{'false':>7}{'mins':>7}"
    lines = ["Per clip", "-" * 48, header]
    for score in scores:
        annotated = score.detected_intervals + len(score.missed)
        lines.append(
            f"  {score.clip[:32]:<32}{annotated:>10}{score.detected_intervals:>7}"
            f"{len(score.false_positives):>7}{score.duration_s / 60:>7.1f}"
        )
    lines.append("")
    return lines


def to_json(
    metrics: Metrics,
    scores: Sequence[ClipScore],
    diagnoses: Sequence[Diagnosis],
    detector: DetectorMetadata | None = None,
) -> dict[str, Any]:
    """Machine-readable results, for diffing two runs against each other."""
    return {
        "detector": None
        if detector is None
        else {
            "name": detector.name,
            "weights": detector.weights,
            "weights_sha256": detector.weights_sha256,
            "imgsz": detector.imgsz,
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "device": detector.device,
        },
        "metrics": {
            "intervals_total": metrics.intervals_total,
            "intervals_detected": metrics.intervals_detected,
            "true_positive_events": metrics.true_positive_events,
            "false_positive_events": metrics.false_positive_events,
            "duration_s": metrics.duration_s,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "false_alarms_per_hour": metrics.false_alarms_per_hour,
        },
        "false_positive_causes": dict(summarise(diagnoses)),
        "false_positives": [
            {
                "event_id": d.event.event_id,
                "track_id": d.event.track_id,
                "frame_index": d.event.frame_index,
                "timestamp_s": d.event.timestamp_s,
                "missing": sorted(c.value for c in d.event.missing),
                "cause": d.cause.value,
                "detail": d.detail,
            }
            for d in diagnoses
        ],
        "missed": [
            {
                "clip": score.clip,
                "person_label": interval.person_label,
                "missing": sorted(c.value for c in interval.missing),
                "start_s": interval.start_s,
                "end_s": interval.end_s,
            }
            for score in scores
            for interval in score.missed
        ],
    }


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"
