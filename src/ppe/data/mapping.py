"""Declarative source-dataset -> our-vocabulary mapping.

One of these per source dataset. Adding a second dataset is a new YAML file,
not a new code path -- which matters, because no single public dataset has
enough person+helmet+vest images to stand alone.

Mapping is by class *name*. Index-based mapping is the classic way to corrupt
a dataset silently: a re-export reorders the classes, every helmet becomes a
vest, and nothing raises.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ppe.core.errors import ConfigError
from ppe.core.types import PPEClass

# The canonical index order for our own exports. Fixed here so every dataset
# we ever convert agrees, and a model trained on one can be evaluated on
# another without a translation table.
CLASS_ORDER: tuple[PPEClass, ...] = (PPEClass.PERSON, PPEClass.HELMET, PPEClass.VEST)
CLASS_INDEX: dict[PPEClass, int] = {label: i for i, label in enumerate(CLASS_ORDER)}


class DatasetMapping(BaseModel):
    """How one source dataset translates into our three classes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source_url: str
    licence: str
    attribution: str
    splits: dict[str, str] = Field(
        default={"train": "train", "val": "valid", "test": "test"},
        description="Our split name -> the directory name in the source export.",
    )
    class_map: dict[str, PPEClass]
    drop: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mapped_and_dropped_are_disjoint(self) -> DatasetMapping:
        overlap = set(self.class_map) & set(self.drop)
        if overlap:
            msg = f"classes both mapped and dropped: {sorted(overlap)}"
            raise ValueError(msg)
        if not self.class_map:
            msg = "class_map is empty; nothing would be converted"
            raise ValueError(msg)
        missing = set(CLASS_ORDER) - set(self.class_map.values())
        if missing:
            msg = (
                f"source provides no mapping for {sorted(m.value for m in missing)}; "
                "a dataset without all three classes cannot train this model alone"
            )
            raise ValueError(msg)
        if set(self.splits) != {"train", "val", "test"}:
            msg = f"splits must be exactly train/val/test, got {sorted(self.splits)}"
            raise ValueError(msg)
        return self

    def target_index(self, source_name: str) -> int | None:
        """Our class index for a source class name, or ``None`` to drop it.

        Raises:
            KeyError: The name is neither mapped nor explicitly dropped. That
                is deliberate -- an unrecognised class is a decision nobody
                made, and guessing is how label noise gets in.
        """
        if source_name in self.class_map:
            return CLASS_INDEX[self.class_map[source_name]]
        if source_name in self.drop:
            return None
        raise KeyError(source_name)


def load_mapping(path: str | Path) -> DatasetMapping:
    """Read a dataset mapping file."""
    mapping_path = Path(path)
    if not mapping_path.exists():
        msg = f"dataset mapping not found: {mapping_path}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"could not parse {mapping_path}: {exc}"
        raise ConfigError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"dataset mapping must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    try:
        return DatasetMapping(**raw)
    except Exception as exc:
        msg = f"invalid dataset mapping in {mapping_path}: {exc}"
        raise ConfigError(msg) from exc
