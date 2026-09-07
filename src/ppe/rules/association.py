"""Associating PPE items with people. Pure functions, implemented in Phase 3.

The algorithm, so the tests can be written against it first:

1. **Filter.** Drop items and people below their configured confidence floors.
2. **Score every (person, item) pair** of a given class. A pair is a
   *candidate* only if it clears all three gates: the item's centre lies in
   the person's configured vertical band, ``containment`` is at least
   ``min_containment``, and ``centre_offset`` is at most ``max_centre_offset``.
   The score blends containment with proximity to the centreline.
3. **Assign greedily, one-to-one.** Sort candidate pairs by score descending
   and consume them, skipping any pair whose person or item is already taken.
   A per-person ``argmax`` would hand the same helmet to two overlapping
   workers and silently hide a violation -- this is the single most important
   line in the module and it has a dedicated test.
4. **Guard before accusing.** A person whose required band falls outside the
   frame, or who is too small to judge, is returned with ``evaluated=False``
   and a :class:`~ppe.core.types.SkipReason`. Absence of evidence is not
   recorded as evidence of absence.
"""

from __future__ import annotations

from collections.abc import Sequence

from ppe.config.schema import AssociationConfig
from ppe.core.types import AssociationMatch, Detection, PersonAssessment, PPEClass, SkipReason


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
    raise NotImplementedError


def candidate_matches(
    person: Detection,
    items: Sequence[Detection],
    item_class: PPEClass,
    config: AssociationConfig,
) -> list[AssociationMatch]:
    """Score one person against every item of one class, best first.

    Returns only pairs that clear the band, containment and offset gates.
    """
    raise NotImplementedError


def assign_one_to_one(
    persons: Sequence[Detection],
    items: Sequence[Detection],
    item_class: PPEClass,
    config: AssociationConfig,
) -> dict[int, AssociationMatch]:
    """Greedy one-to-one assignment of items to people.

    Returns a mapping of person index to its winning match. Each item is
    consumed at most once.
    """
    raise NotImplementedError


def evaluability(
    person: Detection,
    item_class: PPEClass,
    config: AssociationConfig,
    frame_size: tuple[int, int],
) -> SkipReason | None:
    """Whether this person can be honestly judged for this item.

    Returns the reason to skip, or ``None`` when the person is judgeable.
    Catches the cases a naive implementation gets wrong: a person cropped at
    the top of the frame (head not visible), one occluded from the chest down
    (torso not visible), and one too small or too uncertain to assess.
    """
    raise NotImplementedError
