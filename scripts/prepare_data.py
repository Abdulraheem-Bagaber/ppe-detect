"""Convert a source dataset export into our three-class YOLO layout.

    python scripts/prepare_data.py --mapping configs/datasets/worker_safety.yaml

Downloads nothing. Point it at an export you have already unzipped; see the
Data section of the README for where to get one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ppe.core.errors import ConfigError
from ppe.data.convert import convert_split, read_source_names, write_data_yaml
from ppe.data.manifest import build_manifest
from ppe.data.mapping import load_mapping


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        default="configs/datasets/worker_safety.yaml",
        help="Dataset mapping file describing the source classes.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Unzipped source export (default: data/raw/<dataset name>).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Where to write the converted dataset (default: data/processed/<name>).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Where to write the split manifest (default: configs/manifests/<name>.json).",
    )
    parser.add_argument(
        "--link-only",
        action="store_true",
        help="Write labels but do not copy images, for a quick dry run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        mapping = load_mapping(args.mapping)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    source_root = Path(args.source or f"data/raw/{mapping.name}")
    target_root = Path(args.target or f"data/processed/{mapping.name}")
    manifest_path = Path(args.manifest or f"configs/manifests/{mapping.name}.json")

    if not source_root.is_dir():
        print(
            f"error: source export not found at {source_root}\n"
            f"       Download it from {mapping.source_url} (YOLOv8 format) and unzip\n"
            f"       so that {source_root / 'data.yaml'} exists.",
            file=sys.stderr,
        )
        return 1

    try:
        source_names = read_source_names(source_root)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"source classes: {source_names}")
    unmapped = [n for n in source_names if n not in mapping.class_map and n not in mapping.drop]
    if unmapped:
        print(
            f"error: source classes {unmapped} are neither mapped nor dropped in "
            f"{args.mapping}.\n"
            f"       Add each to class_map or drop. Refusing to guess.",
            file=sys.stderr,
        )
        return 1

    results = []
    for split, source_dir in mapping.splits.items():
        try:
            result = convert_split(
                source_root,
                target_root,
                split,
                source_dir,
                source_names,
                mapping,
                copy_images=not args.link_only,
            )
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        results.append(result)
        kept = sum(result.instances_kept.values())
        dropped = sum(result.instances_dropped.values())
        print(
            f"{split:<6} {result.images:>5} images  {kept:>6} instances kept  {dropped:>6} dropped"
        )

    data_yaml = write_data_yaml(target_root, mapping)
    manifest = build_manifest(mapping, target_root, results, source_names)
    manifest.write(manifest_path)

    print(f"\nwrote {data_yaml}")
    print(f"wrote {manifest_path}")
    print(f"\nNext: python scripts/dataset_stats.py --dataset {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
