"""Report image counts, class balance and dataset sanity checks.

    python scripts/dataset_stats.py --dataset data/processed/worker-safety

Reads only a converted dataset; it never touches the source export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ppe.data.stats import collect_split_stats, format_report

SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/processed/worker-safety",
        help="Converted dataset root, containing train/ val/ test/.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Nominal image size used to turn normalised extents into aspect ratios.",
    )
    parser.add_argument("--out", default=None, help="Also write the report to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.dataset)
    if not root.is_dir():
        print(
            f"error: no converted dataset at {root}. Run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        return 1

    stats = [
        collect_split_stats(root / split, split, (args.imgsz, args.imgsz))
        for split in SPLITS
        if (root / split).is_dir()
    ]
    if not stats:
        print(f"error: {root} contains no train/ val/ test/ directories", file=sys.stderr)
        return 1

    report = format_report(stats)
    print(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8", newline="\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
