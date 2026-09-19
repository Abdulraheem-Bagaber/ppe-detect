"""Violation-level precision, recall, F1, false alarms/hour, error breakdown.

    python scripts/evaluate.py --ground-truth configs/eval/clips.yaml

This is the script that decides whether the system is any good. Detection mAP
is a proxy; what a site cares about is whether the right person was flagged and
how much noise came with it.

Rerunnable by design: everything it reports comes from one pass over the clips,
and the JSON output can be diffed between two runs to show that a change
helped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ppe.config import load_config
from ppe.core.errors import ConfigError, DetectorError, VideoSourceError
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import ViolationEvent
from ppe.detection.factory import build_detector
from ppe.eval.diagnosis import Diagnosis, diagnose
from ppe.eval.groundtruth import ClipGroundTruth, load_ground_truth
from ppe.eval.report import dumps, format_report, to_json
from ppe.eval.scoring import ClipScore, aggregate, score_clip
from ppe.pipeline.runner import FrameResult, Pipeline
from ppe.pipeline.sources import VideoFileSource


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default="configs/eval/clips.yaml")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default=None, help="Overrides model.weights.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--out", default="reports/evaluation.txt")
    parser.add_argument("--json", default="reports/evaluation.json")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=2.0,
        help="How far outside an annotated interval an event may still count.",
    )
    return parser.parse_args(argv)


def run_clip(
    pipeline: Pipeline, clip: ClipGroundTruth, stride: int
) -> tuple[list[ViolationEvent], list[FrameResult]]:
    """Process one clip, returning its events and every frame result.

    The frame results are what make the false-positive breakdown possible
    without a second inference pass.
    """
    with VideoFileSource(clip.clip_path, stride=stride) as source:
        frames = list(pipeline.run(source))
    events = [event for frame in frames for event in frame.events]
    return events, frames


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        evaluation_set = load_ground_truth(args.ground_truth)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not evaluation_set.clips:
        print(
            "error: the evaluation set contains no clips. Violation-level metrics\n"
            "       require hand-annotated footage; there is no public dataset for\n"
            "       this. Until clips exist, the README tables stay TBD.",
            file=sys.stderr,
        )
        return 1

    config = load_config(args.config)
    if args.weights:
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"weights": args.weights})}
        )

    try:
        detector = build_detector(config.model)
    except (ConfigError, DetectorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metadata: DetectorMetadata = detector.metadata
    pipeline = Pipeline(config, detector)

    scores: list[ClipScore] = []
    diagnoses: list[Diagnosis] = []

    for clip in evaluation_set.clips:
        print(f"scoring {clip.clip} ({clip.duration_s / 60:.1f} min)...")
        try:
            events, frames = run_clip(pipeline, clip, args.stride)
        except VideoSourceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        score = score_clip(clip, events, tolerance_s=args.tolerance_s)
        scores.append(score)
        diagnoses.extend(diagnose(m.event, frames) for m in score.false_positives)

    metrics = aggregate(scores)
    report = format_report(metrics, scores, diagnoses, metadata)
    print()
    print(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8", newline="\n")

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        dumps(to_json(metrics, scores, diagnoses, metadata)), encoding="utf-8", newline="\n"
    )

    print(f"wrote {out}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
