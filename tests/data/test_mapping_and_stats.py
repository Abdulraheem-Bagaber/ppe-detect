"""Tests for dataset mappings and the dataset statistics report."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from ppe.core.errors import ConfigError
from ppe.core.types import PPEClass
from ppe.data.mapping import CLASS_ORDER, DatasetMapping, load_mapping
from ppe.data.stats import SplitStats, collect_split_stats, format_report

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SAFETY = REPO_ROOT / "configs" / "datasets" / "worker_safety.yaml"

BASE = {
    "name": "t",
    "source_url": "https://example.invalid",
    "licence": "CC BY 4.0",
    "attribution": "t",
}
FULL_MAP = {"person": PPEClass.PERSON, "helmet": PPEClass.HELMET, "vest": PPEClass.VEST}


class TestCommittedMapping:
    def test_the_worker_safety_mapping_loads(self) -> None:
        mapping = load_mapping(WORKER_SAFETY)
        assert mapping.name == "worker-safety"

    def test_it_drops_the_negative_classes(self) -> None:
        """If these ever get mapped instead of dropped, the design is broken."""
        mapping = load_mapping(WORKER_SAFETY)
        assert "no-helmet" in mapping.drop
        assert "no-vest" in mapping.drop
        assert not any(name.startswith("no-") for name in mapping.class_map)

    def test_it_records_the_licence_for_attribution(self) -> None:
        mapping = load_mapping(WORKER_SAFETY)
        assert mapping.licence == "CC BY 4.0"
        assert mapping.attribution.strip()


class TestMappingValidation:
    def test_rejects_a_class_both_mapped_and_dropped(self) -> None:
        with pytest.raises(ValueError, match="both mapped and dropped"):
            DatasetMapping(**BASE, class_map=FULL_MAP, drop=["helmet"])

    def test_rejects_a_source_missing_one_of_our_classes(self) -> None:
        """A helmet-only dataset cannot train this model on its own."""
        partial = {"person": PPEClass.PERSON, "helmet": PPEClass.HELMET}
        with pytest.raises(ValueError, match="no mapping for"):
            DatasetMapping(**BASE, class_map=partial)

    def test_rejects_non_standard_split_names(self) -> None:
        with pytest.raises(ValueError, match="train/val/test"):
            DatasetMapping(**BASE, class_map=FULL_MAP, splits={"train": "train"})

    def test_target_index_follows_our_canonical_order(self) -> None:
        mapping = DatasetMapping(**BASE, class_map=FULL_MAP)
        assert mapping.target_index("person") == 0
        assert mapping.target_index("helmet") == 1
        assert mapping.target_index("vest") == 2

    def test_dropped_class_maps_to_none(self) -> None:
        mapping = DatasetMapping(**BASE, class_map=FULL_MAP, drop=["no-vest"])
        assert mapping.target_index("no-vest") is None

    def test_unknown_class_raises_rather_than_returning_none(self) -> None:
        """None means 'drop'; unknown must be distinguishable from that."""
        mapping = DatasetMapping(**BASE, class_map=FULL_MAP)
        with pytest.raises(KeyError):
            mapping.target_index("gloves")

    def test_missing_file_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_mapping(tmp_path / "nope.yaml")

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "m.yaml"
        path.write_text(
            yaml.safe_dump(
                {**BASE, "class_map": {k: v.value for k, v in FULL_MAP.items()}, "oops": 1}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            load_mapping(path)


def _write_split(root: Path, rows_per_image: list[list[str]]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i, rows in enumerate(rows_per_image):
        (root / "images" / f"i{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (root / "labels" / f"i{i}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )


class TestCollectSplitStats:
    def test_counts_instances_and_images(self, tmp_path: Path) -> None:
        _write_split(
            tmp_path / "train",
            [
                ["0 0.5 0.5 0.2 0.8", "1 0.5 0.2 0.06 0.05"],
                ["0 0.3 0.5 0.2 0.8"],
            ],
        )
        stats = collect_split_stats(tmp_path / "train", "train", (640, 640))
        assert stats.images == 2
        assert stats.instances == Counter({"person": 2, "helmet": 1})
        assert stats.images_per_class == Counter({"person": 2, "helmet": 1})
        assert stats.total_instances == 3

    def test_empty_label_file_counts_as_a_background_image(self, tmp_path: Path) -> None:
        _write_split(tmp_path / "train", [[]])
        stats = collect_split_stats(tmp_path / "train", "train", (640, 640))
        assert stats.background_images == 1
        assert stats.total_instances == 0

    def test_person_aspect_ratio_reflects_a_full_body_box(self, tmp_path: Path) -> None:
        # 0.1 wide, 0.4 tall on a square image -> 4x taller than wide.
        _write_split(tmp_path / "train", [["0 0.5 0.5 0.1 0.4"]])
        stats = collect_split_stats(tmp_path / "train", "train", (640, 640))
        assert stats.median_person_aspect() == pytest.approx(4.0)

    def test_no_person_boxes_gives_no_aspect_ratio(self, tmp_path: Path) -> None:
        _write_split(tmp_path / "train", [["1 0.5 0.2 0.06 0.05"]])
        assert (
            collect_split_stats(tmp_path / "train", "train", (640, 640)).median_person_aspect()
            is None
        )


class TestReport:
    def _stats(self, ratios: list[float], instances: Counter[str]) -> SplitStats:
        return SplitStats(
            split="train",
            images=100,
            instances=instances,
            person_aspect_ratios=ratios,
        )

    def test_flags_head_boxes_masquerading_as_people(self) -> None:
        """The check that would have caught SHWD before a day was wasted on it."""
        report = format_report(
            [self._stats([1.0] * 10, Counter({"person": 10, "helmet": 900, "vest": 900}))]
        )
        assert "HEAD boxes" in report
        assert "[!!]" in report

    def test_accepts_full_body_boxes(self) -> None:
        report = format_report(
            [self._stats([2.6] * 10, Counter({"person": 10, "helmet": 900, "vest": 900}))]
        )
        assert "full-body" in report
        assert "HEAD boxes" not in report

    def test_warns_when_a_required_class_is_sparse(self) -> None:
        """SH17 has 530 vests; this is the warning that ruled it out."""
        report = format_report(
            [self._stats([2.6] * 10, Counter({"person": 13802, "helmet": 927, "vest": 530}))]
        )
        assert "only 530 vest instances" in report

    def test_flags_a_missing_class_outright(self) -> None:
        report = format_report([self._stats([2.6], Counter({"person": 10, "helmet": 900}))])
        assert "no vest instances" in report

    def test_report_lists_every_class_in_canonical_order(self) -> None:
        report = format_report([self._stats([2.6], Counter({"person": 1}))])
        header = report.splitlines()[0]
        positions = [header.index(c.value) for c in CLASS_ORDER]
        assert positions == sorted(positions)
