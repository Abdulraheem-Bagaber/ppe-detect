"""Ground truth for violation-level evaluation.

Detection datasets are images. A violation is *temporal* -- "this person was
bare-headed from 0:12 to 0:41" -- and no public dataset provides that, so the
ground truth here is a small hand-written file per clip.

Kept deliberately cheap to author. Annotating twenty minutes of footage is the
longest-lead task in the project, and a format that takes an afternoon to
learn would not survive contact with that.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ppe.core.errors import ConfigError
from ppe.core.types import PPEClass


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ViolationInterval(_Base):
    """One person, non-compliant, over a stretch of wall-clock time.

    ``person_label`` is a human's name for the worker ("the one in blue"), not
    a track id. Tying ground truth to track ids would make the evaluation
    depend on the tracker it is supposed to be evaluating.
    """

    person_label: str
    missing: list[PPEClass] = Field(min_length=1)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    box_hint: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "Optional xyxy box roughly locating the person, in pixels. Used to "
            "disambiguate when two people are non-compliant at the same moment."
        ),
    )
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> ViolationInterval:
        if self.end_s <= self.start_s:
            msg = f"end_s ({self.end_s}) must be after start_s ({self.start_s})"
            raise ValueError(msg)
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def covers(self, timestamp_s: float, tolerance_s: float = 0.0) -> bool:
        """Whether a moment falls inside this interval, within tolerance."""
        return self.start_s - tolerance_s <= timestamp_s <= self.end_s + tolerance_s


class ClipGroundTruth(_Base):
    """Every violation a human found in one clip.

    ``duration_s`` is required rather than inferred from the video, because the
    false-alarms-per-hour figure divides by it and a harness that silently
    guesses its own denominator is not measuring anything.
    """

    clip: str
    duration_s: float = Field(gt=0.0)
    fps: float = Field(default=30.0, gt=0.0)
    violations: list[ViolationInterval] = Field(default_factory=list)
    annotator: str = ""
    note: str = ""

    @property
    def clip_path(self) -> Path:
        return Path(self.clip)

    @model_validator(mode="after")
    def _intervals_within_clip(self) -> ClipGroundTruth:
        overruns = [v for v in self.violations if v.start_s > self.duration_s]
        if overruns:
            labels = [v.person_label for v in overruns]
            msg = f"violations start after the clip ends: {labels}"
            raise ValueError(msg)
        return self


class EvaluationSet(_Base):
    """All annotated clips scored together."""

    name: str = "unnamed"
    clips: list[ClipGroundTruth] = Field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return sum(clip.duration_s for clip in self.clips)

    @property
    def total_violations(self) -> int:
        return sum(len(clip.violations) for clip in self.clips)


def load_ground_truth(path: str | Path) -> EvaluationSet:
    """Read an evaluation set from YAML."""
    gt_path = Path(path)
    if not gt_path.exists():
        msg = (
            f"ground truth not found: {gt_path}. Violation-level metrics need "
            "hand-annotated clips; see docs in the README."
        )
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(gt_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"could not parse {gt_path}: {exc}"
        raise ConfigError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"ground truth must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    try:
        return EvaluationSet(**raw)
    except Exception as exc:
        msg = f"invalid ground truth in {gt_path}: {exc}"
        raise ConfigError(msg) from exc
