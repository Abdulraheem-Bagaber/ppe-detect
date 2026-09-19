"""Associating PPE items with people. Pure functions.

The algorithm:

1. **Filter.** Drop items and people below their configured confidence floors.
2. **Score every (person, item) pair** of a given class. A pair is a
   *candidate* only if it clears all three gates: the item's centre lies in
   the person's configured vertical band, ``containment`` is at least
   ``min_containment``, and ``centre_offset`` is at most ``max_centre_offset``.
   The score blends containment, band overlap and proximity to the centreline.
3. **Assign greedily, one-to-one.** Sort candidate pairs by score descending
   and consume them, skipping any pair whose person or item is already taken.
   A per-person ``argmax`` would hand the same helmet to two overlapping
   workers and silently hide a violation.
4. **Guard before accusing.** A person whose required band falls outside the
   frame, or who is too small or too uncertain to judge, is returned with
   ``evaluated=False`` and a :class:`~ppe.core.types.SkipReason`. Absence of
   evidence is not recorded as evidence of absence.

Nothing here imports a model, a video library or a web framework, and a test
enforces that.
"""

from __future__ import annotations

from collections.abc import Sequence

from ppe.config.schema import AssociationConfig, ItemRule
from ppe.core.types import (
    AssociationMatch,
    BBox,
    Detection,
    PersonAssessment,
    PPEClass,
    SkipReason,
)
from ppe.rules.geometry import (
    centre_offset,
    containment,
    vertical_band_overlap,
    visible_fraction,
)


def associate(
    detections: Sequence[Detection],
    config: AssociationConfig,
    frame_size: tuple[int, int],
    track_ids: Sequence[int | None] | None = None,
) -> list[PersonAssessment]:
    """Judge every person in one frame.

    Args:
        detections: All detections for the frame, people and items mixed.
        config: Bands, thresholds and the list of required items.
        frame_size: ``(width, height)``, needed for the frame-edge guards.
        track_ids: Optional identity per *person* detection, in the order the
            people appear in ``detections``. ``None`` for single-image use.

    Returns:
        One assessment per person detection, in input order.
    """
    persons = [d for d in detections if d.label is PPEClass.PERSON]
    if not persons:
        return []

    ids: list[int | None] = list(track_ids) if track_ids is not None else [None] * len(persons)
    if len(ids) != len(persons):
        msg = f"got {len(ids)} track ids for {len(persons)} people"
        raise ValueError(msg)

    # Assign each required class across all people at once, so the one-to-one
    # constraint holds globally rather than per person.
    assignments: dict[PPEClass, dict[int, AssociationMatch]] = {
        item: assign_one_to_one(persons, detections, item, config) for item in config.required
    }

    assessments: list[PersonAssessment] = []
    for index, person in enumerate(persons):
        matches: dict[PPEClass, AssociationMatch] = {}
        missing: set[PPEClass] = set()
        skip_reasons: list[SkipReason] = []

        for item in config.required:
            reason = evaluability(person, item, config, frame_size)
            if reason is not None:
                skip_reasons.append(reason)
                continue
            match = assignments[item].get(index)
            if match is None:
                missing.add(item)
            else:
                matches[item] = match

        # Judged on nothing at all: report why rather than declaring compliance.
        evaluated = len(skip_reasons) < len(config.required)
        assessments.append(
            PersonAssessment(
                person=person,
                matches=matches,
                missing=frozenset(missing),
                track_id=ids[index],
                evaluated=evaluated,
                skip_reason=skip_reasons[0] if skip_reasons and not evaluated else None,
            )
        )
    return assessments


def candidate_matches(
    person: Detection,
    items: Sequence[Detection],
    item_class: PPEClass,
    config: AssociationConfig,
) -> list[AssociationMatch]:
    """Score one person against every item of one class, best first.

    Returns only pairs that clear the band, containment and offset gates.
    """
    rule = config.rules.get(item_class)
    if rule is None:
        return []

    band = person.box.band(rule.band_top, rule.band_bottom)
    out: list[AssociationMatch] = []

    for item in items:
        if item.label is not item_class or item.confidence < rule.min_confidence:
            continue
        score = _score(item.box, person.box, band, rule)
        if score is not None:
            out.append(AssociationMatch(item=item, score=score, rule=item_class.value))

    out.sort(key=lambda m: m.score, reverse=True)
    return out


def _score(item: BBox, person: BBox, band: BBox, rule: ItemRule) -> float | None:
    """Score one pair, or ``None`` if it fails a gate.

    The gates are hard constraints -- a helmet at waist height is not a weak
    match, it is not a match. The score only ranks what survives them.
    """
    _, item_cy = item.centre
    if not band.y1 <= item_cy <= band.y2:
        return None

    contained = containment(item, person)
    if contained < rule.min_containment:
        return None

    offset = centre_offset(item, person)
    if offset > rule.max_centre_offset:
        return None

    band_fit = vertical_band_overlap(item, band)
    centredness = 1.0 - min(offset / rule.max_centre_offset, 1.0) if rule.max_centre_offset else 1.0

    return (
        rule.weight_containment * contained
        + rule.weight_band * band_fit
        + rule.weight_centre * centredness
    )


def assign_one_to_one(
    persons: Sequence[Detection],
    items: Sequence[Detection],
    item_class: PPEClass,
    config: AssociationConfig,
) -> dict[int, AssociationMatch]:
    """Greedy one-to-one assignment of items to people.

    Returns a mapping of person index to its winning match. Each item is
    consumed at most once, which is what stops two overlapping workers from
    both claiming the same helmet.

    Greedy rather than optimal (Hungarian) on purpose: with the handful of
    candidates that survive the gates the two agree in practice, and a greedy
    pass is small enough to read and reason about in a review.
    """
    scored: list[tuple[float, int, int, AssociationMatch]] = []
    for person_index, person in enumerate(persons):
        for match in candidate_matches(person, items, item_class, config):
            item_index = next(i for i, d in enumerate(items) if d is match.item)
            scored.append((match.score, person_index, item_index, match))

    # Sort by score, then by index, so ties resolve deterministically rather
    # than by whatever order the detector happened to emit.
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))

    assigned: dict[int, AssociationMatch] = {}
    taken_items: set[int] = set()
    for _, person_index, item_index, match in scored:
        if person_index in assigned or item_index in taken_items:
            continue
        assigned[person_index] = match
        taken_items.add(item_index)
    return assigned


def evaluability(
    person: Detection,
    item_class: PPEClass,
    config: AssociationConfig,
    frame_size: tuple[int, int],
) -> SkipReason | None:
    """Whether this person can be honestly judged for this item.

    Returns the reason to skip, or ``None`` when the person is judgeable.
    """
    if person.confidence < config.min_person_confidence:
        return SkipReason.LOW_CONFIDENCE
    if person.box.height < config.min_person_height_px:
        return SkipReason.TOO_SMALL

    rule = config.rules.get(item_class)
    if rule is None or not config.skip_when_cropped:
        return None

    # If the region a worn item would occupy is largely out of shot, we have no
    # evidence either way. Saying "no helmet" there would be a claim about
    # pixels that do not exist.
    band = person.box.band(rule.band_top, rule.band_bottom)
    if visible_fraction(band, frame_size) < config.min_band_visibility:
        return (
            SkipReason.HEAD_OUT_OF_FRAME
            if item_class is PPEClass.HELMET
            else SkipReason.TORSO_OUT_OF_FRAME
        )
    return None
