"""Dataset statistics, including the checks that catch a wrong dataset early.

Class balance is the obvious report. The two that actually matter are checks
on *geometry*, because this project's rule layer makes assumptions about the
world that a dataset can silently violate:

1. **Person boxes must be full bodies.** A helmet is matched against the upper
   band of a person's box. Several popular "helmet" datasets label *heads*
   instead, which the rules cannot use. Full-body boxes run 2-3x taller than
   wide; head boxes are roughly square.

2. **Helmets must sit near the top of their person.** This is the assumption
   the entire association rule rests on, so it is worth measuring directly
   rather than inferring. It also catches the specific way an off-the-shelf
   export can be poisoned for this task: Roboflow and friends offer 90-degree
   rotation and vertical-flip augmentation, which produces upside-down training
   images where the helmet is below the boots. Nothing about class balance
   reveals that, and a model trained on it learns that a helmet above the head
   is optional.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ppe.core.types import PPEClass
from ppe.data.convert import IMAGE_SUFFIXES, parse_label_row
from ppe.data.mapping import CLASS_INDEX, CLASS_ORDER

# Below this median height:width ratio, `person` boxes are probably heads.
FULL_BODY_RATIO_FLOOR = 1.5

# Below this many instances, a class is too sparse to train or honestly
# evaluate. Calibrated against SH17, which has 530 safety-vest instances
# across 8,099 images and only 97 in its test split.
SPARSE_INSTANCE_FLOOR = 1000

# A helmet this far down its person's box (0.0 = crown, 1.0 = feet) is not
# being worn in a normally-oriented image.
HELMET_BAND_LIMIT = 0.5

# Tolerated share of helmets below that line before the dataset is suspect.
HELMET_BELOW_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class SplitStats:
    """Counts and geometry for one split."""

    split: str
    images: int
    instances: Counter[str] = field(default_factory=Counter)
    images_per_class: Counter[str] = field(default_factory=Counter)
    background_images: int = 0
    person_aspect_ratios: list[float] = field(default_factory=list)
    helmet_relative_heights: list[float] = field(default_factory=list)

    @property
    def total_instances(self) -> int:
        return sum(self.instances.values())

    def median_person_aspect(self) -> float | None:
        """Median height:width of person boxes, or ``None`` if there are none."""
        if not self.person_aspect_ratios:
            return None
        return statistics.median(self.person_aspect_ratios)

    def share_of_helmets_below_midline(self) -> float | None:
        """Fraction of helmets sitting in the lower half of their person's box."""
        if not self.helmet_relative_heights:
            return None
        below = sum(r >= HELMET_BAND_LIMIT for r in self.helmet_relative_heights)
        return below / len(self.helmet_relative_heights)


def relative_helmet_heights(
    persons: list[tuple[float, float, float, float]],
    helmets: list[tuple[float, float]],
) -> list[float]:
    """Where each helmet sits inside the person box containing it.

    Normalised coordinates throughout, so image size is irrelevant. Returns
    0.0 for a helmet at the crown and 1.0 for one at the feet. A helmet inside
    no person box is skipped rather than attributed to a guess.
    """
    out: list[float] = []
    for hx, hy in helmets:
        for px, py, pw, ph in persons:
            if ph <= 0:
                continue
            if abs(hx - px) <= pw / 2 and abs(hy - py) <= ph / 2:
                out.append((hy - (py - ph / 2)) / ph)
                break
    return out


def collect_split_stats(split_root: Path, split: str) -> SplitStats:
    """Walk one converted split and tally it."""
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"

    person_id = CLASS_INDEX[PPEClass.PERSON]
    helmet_id = CLASS_INDEX[PPEClass.HELMET]

    instances: Counter[str] = Counter()
    images_per_class: Counter[str] = Counter()
    ratios: list[float] = []
    helmet_heights: list[float] = []
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

        with Image.open(image_path) as im:
            width, height = im.size

        persons: list[tuple[float, float, float, float]] = []
        helmets: list[tuple[float, float]] = []
        seen: set[str] = set()

        for row in rows:
            class_id, (cx, cy, w, h) = parse_label_row(row)
            name = CLASS_ORDER[class_id].value
            instances[name] += 1
            seen.add(name)
            if class_id == person_id:
                persons.append((cx, cy, w, h))
                if w > 0:
                    ratios.append((h * height) / (w * width))
            elif class_id == helmet_id:
                helmets.append((cx, cy))

        helmet_heights.extend(relative_helmet_heights(persons, helmets))
        images_per_class.update(seen)

    return SplitStats(
        split=split,
        images=images,
        instances=instances,
        images_per_class=images_per_class,
        background_images=background,
        person_aspect_ratios=ratios,
        helmet_relative_heights=helmet_heights,
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
    out.extend(_helmet_orientation_verdict(all_stats))
    out.extend(_person_shape_verdict(all_stats))

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


def _helmet_orientation_verdict(all_stats: list[SplitStats]) -> list[str]:
    """The check that matters most: is a helmet actually above its wearer?"""
    heights = [r for s in all_stats for r in s.helmet_relative_heights]
    if not heights:
        return ["  [!] no helmet sits inside any person box; cannot check orientation"]

    below = sum(r >= HELMET_BAND_LIMIT for r in heights) / len(heights)
    median = statistics.median(heights)
    if below <= HELMET_BELOW_TOLERANCE:
        return [
            f"  [ok] {1 - below:.1%} of helmets sit in the upper half of their person "
            f"(median {median:.2f} down the box)"
        ]
    return [
        f"  [!!] {below:.1%} of helmets sit in the LOWER half of their person box "
        f"(median {median:.2f} down the box).",
        "       The association rule assumes a worn helmet is near the top of the",
        "       person, and this dataset violates that. The usual cause is rotation",
        "       or vertical-flip augmentation baked into the export: check the source",
        "       README for a '90-degree rotation' or 'flip' step and re-export without",
        "       it. Training on this teaches the detector that a helmet below the",
        "       boots is perfectly normal.",
    ]


def _person_shape_verdict(all_stats: list[SplitStats]) -> list[str]:
    """Full bodies, or heads?"""
    ratios = [r for s in all_stats for r in s.person_aspect_ratios]
    if not ratios:
        return ["  [!!] no person boxes at all -- this dataset cannot drive association"]

    median = statistics.median(ratios)
    if median >= FULL_BODY_RATIO_FLOOR:
        return [f"  [ok] median person box {median:.2f}x taller than wide (full-body)"]
    return [
        f"  [!!] median person box is only {median:.2f}x taller than wide; full-body "
        "boxes run 2-3x.",
        "       Either the dataset labels HEADS rather than bodies, which the",
        "       association rules cannot use at all, or the images are rotation-",
        "       augmented. The helmet-orientation check above separates the two.",
    ]
