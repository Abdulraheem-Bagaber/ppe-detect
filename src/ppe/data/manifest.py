"""The split manifest: which image went into which split, and from what.

Committed to git. A reported metric is only reproducible if the exact split
that produced it is recoverable, and "I used the Roboflow train/valid/test
folders" stops being checkable the moment the dataset is re-exported.

Filenames are listed in full rather than summarised to a hash. At this scale
the file is a few hundred kilobytes, and being able to ``git diff`` exactly
which images moved between splits is worth more than the disk space.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ppe.data.convert import SplitResult
from ppe.data.mapping import CLASS_ORDER, DatasetMapping

MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to reconstruct or verify a split."""

    payload: dict[str, object]

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path


def file_digest(path: Path) -> str:
    """SHA-256 of a file, for spotting a silently re-exported dataset."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    mapping: DatasetMapping,
    target_root: Path,
    results: list[SplitResult],
    source_names: list[str],
) -> Manifest:
    """Assemble the manifest for a completed conversion."""
    splits: dict[str, object] = {}
    for result in results:
        images_dir = target_root / result.split / "images"
        files = sorted(p.name for p in images_dir.iterdir()) if images_dir.is_dir() else []
        splits[result.split] = {
            "images": len(files),
            "instances": dict(sorted(result.instances_kept.items())),
            "dropped_instances": dict(sorted(result.instances_dropped.items())),
            "background_images": result.images_without_labels,
            "files": files,
        }

    totals: Counter[str] = Counter()
    for result in results:
        totals.update(result.instances_kept)

    return Manifest(
        {
            "manifest_version": MANIFEST_VERSION,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": {
                "name": mapping.name,
                "source_url": mapping.source_url,
                "licence": mapping.licence,
                "attribution": mapping.attribution,
                "source_classes": source_names,
                "class_map": {k: v.value for k, v in mapping.class_map.items()},
                "dropped_classes": list(mapping.drop),
            },
            "classes": [c.value for c in CLASS_ORDER],
            "totals": dict(sorted(totals.items())),
            "splits": splits,
        }
    )
