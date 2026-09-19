"""Tests for face blurring.

The guarantee under test is not "the image looks fuzzy" -- it is that no
recognisable face reaches an output path, and that the detector is never
handed a degraded frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppe.config.schema import PrivacyConfig
from ppe.core.types import BBox, Detection, PPEClass
from ppe.privacy.blur import anonymise, blur_regions, head_regions


def frame(value: int = 0) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def noisy_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def person(x1=200.0, y1=100.0, x2=280.0, y2=340.0) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), PPEClass.PERSON, 0.9)


class TestHeadRegions:
    def test_the_head_is_the_top_of_the_person(self) -> None:
        config = PrivacyConfig(head_band_bottom=0.25)
        [head] = head_regions([person()], config)
        assert head.y1 == 100.0
        assert head.y2 == pytest.approx(160.0)
        assert (head.x1, head.x2) == (200.0, 280.0)

    def test_ppe_items_have_no_heads(self) -> None:
        helmet = Detection(BBox(0.0, 0.0, 10.0, 10.0), PPEClass.HELMET, 0.8)
        assert head_regions([helmet], PrivacyConfig()) == []

    def test_scales_with_the_person(self) -> None:
        near = head_regions([person(0.0, 0.0, 100.0, 400.0)], PrivacyConfig())[0]
        far = head_regions([person(0.0, 0.0, 10.0, 40.0)], PrivacyConfig())[0]
        assert near.height > far.height


class TestBlurRegions:
    def test_the_region_is_changed(self) -> None:
        original = noisy_frame()
        blurred = blur_regions(original, [BBox(200.0, 100.0, 280.0, 160.0)], PrivacyConfig())
        assert not np.array_equal(original[100:160, 200:280], blurred[100:160, 200:280])

    def test_pixels_outside_the_region_are_untouched(self) -> None:
        original = noisy_frame()
        blurred = blur_regions(original, [BBox(200.0, 100.0, 280.0, 160.0)], PrivacyConfig())
        assert np.array_equal(original[300:400, 400:500], blurred[300:400, 400:500])

    def test_it_never_mutates_the_input(self) -> None:
        """The detector must keep seeing the original frame."""
        original = noisy_frame()
        snapshot = original.copy()
        blur_regions(original, [BBox(200.0, 100.0, 280.0, 160.0)], PrivacyConfig())
        assert np.array_equal(original, snapshot)

    def test_blurring_destroys_local_detail(self) -> None:
        """Variance within the region must collapse, or nothing was hidden."""
        original = noisy_frame()
        region = BBox(200.0, 100.0, 300.0, 200.0)
        blurred = blur_regions(original, [region], PrivacyConfig())
        before = original[100:200, 200:300].std()
        after = blurred[100:200, 200:300].std()
        assert after < before / 2

    def test_pixelate_also_destroys_detail(self) -> None:
        original = noisy_frame()
        config = PrivacyConfig(method="pixelate")
        blurred = blur_regions(original, [BBox(200.0, 100.0, 300.0, 200.0)], config)
        assert blurred[100:200, 200:300].std() < original[100:200, 200:300].std()

    def test_a_region_off_the_frame_edge_is_clipped_not_crashed(self) -> None:
        blurred = blur_regions(noisy_frame(), [BBox(-50.0, -50.0, 60.0, 60.0)], PrivacyConfig())
        assert blurred.shape == (480, 640, 3)

    def test_a_region_entirely_outside_is_ignored(self) -> None:
        original = noisy_frame()
        blurred = blur_regions(original, [BBox(900.0, 900.0, 950.0, 950.0)], PrivacyConfig())
        assert np.array_equal(original, blurred)

    def test_no_regions_leaves_the_frame_alone(self) -> None:
        original = noisy_frame()
        assert np.array_equal(original, blur_regions(original, [], PrivacyConfig()))

    def test_output_stays_uint8(self) -> None:
        blurred = blur_regions(noisy_frame(), [BBox(10.0, 10.0, 90.0, 90.0)], PrivacyConfig())
        assert blurred.dtype == np.uint8


class TestAnonymise:
    def test_it_is_on_by_default(self) -> None:
        """PDPL posture: blurring must not require opting in."""
        assert PrivacyConfig().blur_faces is True
        original = noisy_frame()
        out = anonymise(original, [person()], PrivacyConfig())
        assert not np.array_equal(original[100:160, 200:280], out[100:160, 200:280])

    def test_it_can_be_turned_off_explicitly(self) -> None:
        config = PrivacyConfig(blur_faces=False)
        original = noisy_frame()
        assert np.array_equal(original, anonymise(original, [person()], config))

    def test_a_frame_with_nobody_in_it_is_unchanged(self) -> None:
        original = noisy_frame()
        assert np.array_equal(original, anonymise(original, [], PrivacyConfig()))

    def test_strict_mode_raises_rather_than_leaking(self) -> None:
        """Emitting an unblurred face because blurring crashed is the one
        outcome this module exists to prevent."""

        class Exploding(list):
            def __iter__(self):
                raise RuntimeError("blur backend died")

        config = PrivacyConfig(strict=True)
        with pytest.raises(RuntimeError):
            anonymise(noisy_frame(), Exploding([person()]), config)

    def test_non_strict_mode_still_fails_closed(self) -> None:
        class Exploding(list):
            def __iter__(self):
                raise RuntimeError("blur backend died")

        config = PrivacyConfig(strict=False)
        out = anonymise(noisy_frame(), Exploding([person()]), config)
        assert out.max() == 0, "a blank frame leaks nothing; the original would leak faces"
