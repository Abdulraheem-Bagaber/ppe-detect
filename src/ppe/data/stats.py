"""Dataset statistics, including the checks that catch a wrong dataset early.

Class balance is the obvious report. The less obvious one is the person-box
shape check: this project's association geometry assumes ``person`` boxes are
**full bodies**, because a helmet is matched against the upper band of the
person's box. Several popular "helmet" datasets label *heads* instead, and
that failure is invisible until association mysteriously never works.

A full-body box is taller than it is wide -- typically 2-3x. A head box is
roughly square. One aspect-ratio histogram tells you which you have.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ppe.data.convert import IMAGE_SUFFIXES, parse_label_row
from ppe.data.mapping import CLASS_ORDER

# Below this median height:width ratio, `person` boxes are probably heads.
FULL_BODY_RATIO_FLOOR = 1.5

# Below this many instances, a class is too sparse to train or honestly
# evaluate. Calibrated against SH17, which has 530 safety-vest instances
# across 8,099 images and only 97 in its test split.
SPARSE_INSTANCE_FLOOR = 1000


@dataclass(frozen=True, slots=True)
class SplitStats:
    """Counts and shapes for one split."""

    split: str
    images: int
    instances: Counter[str] = field(default_factory=Counter)
    images_per_class: Counter[str] = field(default_factory=Counter)
    background_images: int = 0
    person_aspect_ratios: list[float] = field(default_factory=list)

    @property
    def total_instances(self) -> int:
        return sum(self.instances.values())

    def median_person_aspect(self) -> float | None:
        """Median height:width of person boxes, or ``None`` if there are none."""
        if not self.person_aspect_ratios:
            return None
        return statistics.median(self.person_aspect_ratios)


def collect_split_stats(split_root: Path, split: str, image_shape: tuple[int, int]) -> SplitStats:
    """Walk one converted split and tally it.

    ``image_shape`` is ``(width, height)`` and is only used to turn normalised
    box extents back into an aspect ratio; the exact value cancels out for the
    ratio as long as it is consistent, so the nominal training size is fine.
    """
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    width, height = image_shape

    instances: Counter[str] = Counter()
    images_per_class: Counter[str] = Counter()
    ratios: list[float] = []
    images = 0
    background = 0

    for image_path in sorted(images_dir.iterdir()) if images_dir.is_dir() else []:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        images += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            background += 1
            continue
        rows = [r for r in label_path.read_text(encoding="utf-8").splitlines() if r.strip()]
        if not rows:
            background += 1
            continue
        seen: set[str] = set()
        for row in rows:
            class_id, (_, _, w, h) = parse_label_row(row)
            name = CLASS_ORDER[class_id].value
            instances[name] += 1
            seen.add(name)
            if name == "person" and w > 0:
                ratios.append((h * height) / (w * width))
        images_per_class.update(seen)

    return SplitStats(
        split=split,
        images=images,
        instances=instances,
        images_per_class=images_per_class,
        background_images=background,
        person_aspect_ratios=ratios,
    )


def format_report(all_stats: list[SplitStats]) -> str:
    """Render the stats as a plain-text table plus the sanity verdicts."""
    lines: list[str] = []
    names = [c.value for c in CLASS_ORDER]

    header = f"{'split':<8}{'images':>8}{'bg':>7}" + "".join(f"{n:>11}" for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    for s in all_stats:
        row = f"{s.split:<8}{s.images:>8}{s.background_images:>7}"
        row += "".join(f"{s.instances.get(n, 0):>11}" for n in names)
        lines.append(row)

    totals: Counter[str] = Counter()
    for s in all_stats:
        totals.update(s.instances)
    total_images = sum(s.images for s in all_stats)
    row = f"{'TOTAL':<8}{total_images:>8}{sum(s.background_images for s in all_stats):>7}"
    row += "".join(f"{totals.get(n, 0):>11}" for n in names)
    lines.append("-" * len(header))
    lines.append(row)

    lines.append("")
    lines.append("Class balance (share of all instances):")
    grand = sum(totals.values()) or 1
    for n in names:
        share = 100.0 * totals.get(n, 0) / grand
        lines.append(f"  {n:<8} {totals.get(n, 0):>7}  {share:5.1f}%")

    lines.append("")
    lines.extend(_verdicts(all_stats, totals))
    return "\n".join(lines)


def _verdicts(all_stats: list[SplitStats], totals: Counter[str]) -> list[str]:
    """The checks worth failing a dataset over."""
    out = ["Sanity checks:"]

    ratios = [r for s in all_stats for r in s.person_aspect_ratios]
    if not ratios:
        out.append("  [!!] no person boxes at all -- this dataset cannot drive association")
    else:
        median = statistics.median(ratios)
        if median < FULL_BODY_RATIO_FLOOR:
            out.append(
                f"  [!!] median person box is {median:.2f}x taller than wide. Full-body "
                "boxes are usually 2-3x; this looks like HEAD boxes, which the "
                "association rules cannot use. Check the dataset before training."
            )
        else:
            out.append(f"  [ok] median person box {median:.2f}x taller than wide (full-body)")

    for name in ("helmet", "vest"):
        count = totals.get(name, 0)
        if count == 0:
            out.append(f"  [!!] no {name} instances -- required by the default rule set")
        elif count < SPARSE_INSTANCE_FLOOR:
            out.append(
                f"  [!] only {count} {name} instances (want {SPARSE_INSTANCE_FLOOR}+); "
                f"expect a weak {name} detector and a noisy per-class mAP"
            )
        else:
            out.append(f"  [ok] {count} {name} instances")

    for s in all_stats:
        if s.images == 0:
            out.append(f"  [!!] split {s.split!r} is empty")
        missing = [c.value for c in CLASS_ORDER if s.instances.get(c.value, 0) == 0]
        if missing and s.images:
            out.append(f"  [!] split {s.split!r} has no {', '.join(missing)} instances")

    return out
