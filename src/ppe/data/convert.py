"""Convert a source dataset export into our three-class YOLO layout.

The label rewriting itself is a pure function over text, so the interesting
failure modes -- an unknown class, a malformed row, a coordinate outside
[0, 1] -- are unit-testable without a single image on disk.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ppe.core.errors import ConfigError
from ppe.data.mapping import CLASS_ORDER, DatasetMapping

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class LabelFormatError(ConfigError):
    """A label row is malformed in a way that cannot be safely guessed at."""


@dataclass(frozen=True, slots=True)
class SplitResult:
    """What conversion did to one split."""

    split: str
    images: int
    labels_written: int
    instances_kept: Counter[str] = field(default_factory=Counter)
    instances_dropped: Counter[str] = field(default_factory=Counter)
    images_without_labels: int = 0
    missing_label_files: list[str] = field(default_factory=list)


def parse_label_row(row: str) -> tuple[int, tuple[float, ...]]:
    """Split one YOLO label row into its class index and normalised geometry.

    Raises:
        LabelFormatError: Wrong field count, non-numeric values, or
            coordinates outside [0, 1]. Rounding a stray 1.0001 would be
            harmless; silently accepting a 47.0 would not.
    """
    parts = row.split()
    if len(parts) != 5:
        msg = f"expected 5 fields in a YOLO label row, got {len(parts)}: {row!r}"
        raise LabelFormatError(msg)
    try:
        class_id = int(parts[0])
        coords = tuple(float(p) for p in parts[1:])
    except ValueError as exc:
        msg = f"non-numeric value in label row {row!r}: {exc}"
        raise LabelFormatError(msg) from exc
    if any(not 0.0 <= c <= 1.0 for c in coords):
        msg = f"label coordinates must be normalised to [0, 1], got {coords} in {row!r}"
        raise LabelFormatError(msg)
    return class_id, coords


def remap_label_text(
    text: str,
    source_names: list[str],
    mapping: DatasetMapping,
) -> tuple[str, Counter[str], Counter[str]]:
    """Rewrite one label file's contents into our class indices.

    Geometry is passed through verbatim -- the source strings are reused rather
    than reformatted, so no precision is lost in the round trip.

    Returns:
        The new file contents, a count of kept instances by our class name, and
        a count of dropped instances by their source class name.
    """
    kept: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    out_rows: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        row = raw.strip()
        if not row:
            continue
        class_id, _ = parse_label_row(row)
        if not 0 <= class_id < len(source_names):
            msg = (
                f"line {lineno}: class index {class_id} is outside the source "
                f"class list of length {len(source_names)}"
            )
            raise LabelFormatError(msg)
        source_name = source_names[class_id]
        try:
            target = mapping.target_index(source_name)
        except KeyError as exc:
            msg = (
                f"line {lineno}: source class {source_name!r} is neither mapped nor "
                f"dropped in {mapping.name}. Add it to class_map or drop explicitly."
            )
            raise LabelFormatError(msg) from exc

        if target is None:
            dropped[source_name] += 1
            continue
        kept[CLASS_ORDER[target].value] += 1
        out_rows.append(" ".join([str(target), *row.split()[1:]]))

    return ("\n".join(out_rows) + "\n" if out_rows else ""), kept, dropped


def read_source_names(source_root: Path) -> list[str]:
    """Read the class list from a source export's ``data.yaml``."""
    data_yaml = source_root / "data.yaml"
    if not data_yaml.exists():
        msg = (
            f"no data.yaml in {source_root}. Expected a Roboflow-style YOLO export; "
            "unzip the download so that data.yaml sits at the top level."
        )
        raise ConfigError(msg)
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = raw.get("names") if isinstance(raw, dict) else None
    if isinstance(names, dict):  # some exports use {index: name}
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        msg = f"could not read a class name list from {data_yaml}"
        raise ConfigError(msg)
    return names


def convert_split(
    source_root: Path,
    target_root: Path,
    split: str,
    source_dir: str,
    source_names: list[str],
    mapping: DatasetMapping,
    *,
    copy_images: bool = True,
) -> SplitResult:
    """Convert one split's labels and place its images."""
    src_images = source_root / source_dir / "images"
    src_labels = source_root / source_dir / "labels"
    if not src_images.is_dir():
        msg = f"missing image directory for split {split!r}: {src_images}"
        raise ConfigError(msg)

    dst_images = target_root / split / "images"
    dst_labels = target_root / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    kept: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    missing: list[str] = []
    images = 0
    labels_written = 0
    empty = 0

    for image_path in sorted(src_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        images += 1
        if copy_images:
            shutil.copy2(image_path, dst_images / image_path.name)

        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            # A background image with no objects is legitimate; record it so
            # the stats report can show how many there are.
            missing.append(image_path.name)
            (dst_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            empty += 1
            continue

        try:
            text, file_kept, file_dropped = remap_label_text(
                label_path.read_text(encoding="utf-8"), source_names, mapping
            )
        except LabelFormatError as exc:
            msg = f"{label_path}: {exc}"
            raise LabelFormatError(msg) from exc

        (dst_labels / f"{image_path.stem}.txt").write_text(text, encoding="utf-8", newline="\n")
        labels_written += 1
        kept.update(file_kept)
        dropped.update(file_dropped)
        if not text:
            empty += 1

    return SplitResult(
        split=split,
        images=images,
        labels_written=labels_written,
        instances_kept=kept,
        instances_dropped=dropped,
        images_without_labels=empty,
        missing_label_files=missing,
    )


def write_data_yaml(target_root: Path, mapping: DatasetMapping) -> Path:
    """Emit the ``data.yaml`` the training script will consume."""
    payload = {
        "path": str(target_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_ORDER),
        "names": [c.value for c in CLASS_ORDER],
        "source": mapping.source_url,
        "licence": mapping.licence,
        "attribution": mapping.attribution,
    }
    path = target_root / "data.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path
