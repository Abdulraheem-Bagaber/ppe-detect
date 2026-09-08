"""Tests for dataset conversion.

The label rewriting is pure text-to-text, so the failure modes that actually
corrupt a dataset can be pinned down here without a single image existing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ppe.core.errors import ConfigError
from ppe.core.types import PPEClass
from ppe.data.convert import (
    LabelFormatError,
    convert_split,
    parse_label_row,
    read_source_names,
    remap_label_text,
    write_data_yaml,
)
from ppe.data.mapping import CLASS_INDEX, DatasetMapping

# Deliberately in a different order from ours, and with the negative classes
# interleaved -- this is what a real Roboflow export looks like.
SOURCE_NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]


@pytest.fixture
def mapping() -> DatasetMapping:
    return DatasetMapping(
        name="test-set",
        source_url="https://example.invalid/ds",
        licence="CC BY 4.0",
        attribution="Test",
        class_map={"person": PPEClass.PERSON, "helmet": PPEClass.HELMET, "vest": PPEClass.VEST},
        drop=["no-helmet", "no-vest"],
    )


class TestParseLabelRow:
    def test_parses_a_well_formed_row(self) -> None:
        assert parse_label_row("3 0.5 0.5 0.2 0.8") == (3, (0.5, 0.5, 0.2, 0.8))

    @pytest.mark.parametrize(
        "row",
        ["3 0.5 0.5 0.2", "3 0.5 0.5 0.2 0.8 0.1", "3"],
        ids=["too-few", "too-many", "bare"],
    )
    def test_rejects_wrong_field_count(self, row: str) -> None:
        with pytest.raises(LabelFormatError, match="expected 5 fields"):
            parse_label_row(row)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(LabelFormatError, match="non-numeric"):
            parse_label_row("3 0.5 0.5 x 0.8")

    def test_rejects_unnormalised_coordinates(self) -> None:
        """Absolute pixels in a YOLO label file is a silent dataset-wide bug."""
        with pytest.raises(LabelFormatError, match="normalised"):
            parse_label_row("3 320.0 240.0 100.0 200.0")


class TestRemapLabelText:
    def test_remaps_to_our_canonical_indices(self, mapping: DatasetMapping) -> None:
        text = "\n".join(
            [
                "3 0.5 0.5 0.2 0.8",  # person
                "0 0.5 0.2 0.06 0.05",  # helmet
                "4 0.5 0.45 0.15 0.2",  # vest
            ]
        )
        out, kept, dropped = remap_label_text(text, SOURCE_NAMES, mapping)
        rows = out.strip().splitlines()
        assert [r.split()[0] for r in rows] == [
            str(CLASS_INDEX[PPEClass.PERSON]),
            str(CLASS_INDEX[PPEClass.HELMET]),
            str(CLASS_INDEX[PPEClass.VEST]),
        ]
        assert kept == {"person": 1, "helmet": 1, "vest": 1}
        assert not dropped

    def test_geometry_passes_through_byte_for_byte(self, mapping: DatasetMapping) -> None:
        """No reformatting: a float round trip must not shift a box."""
        out, _, _ = remap_label_text("3 0.123456789 0.5 0.2 0.80000001", SOURCE_NAMES, mapping)
        assert out.strip().split()[1:] == ["0.123456789", "0.5", "0.2", "0.80000001"]

    def test_drops_the_negative_classes(self, mapping: DatasetMapping) -> None:
        """The whole point: compliance is never read off a label."""
        text = "3 0.5 0.5 0.2 0.8\n1 0.5 0.2 0.06 0.05\n2 0.5 0.45 0.15 0.2\n"
        out, kept, dropped = remap_label_text(text, SOURCE_NAMES, mapping)
        assert out == "0 0.5 0.5 0.2 0.8\n"
        assert kept == {"person": 1}
        assert dropped == {"no-helmet": 1, "no-vest": 1}

    def test_all_dropped_yields_an_empty_file_not_a_blank_line(
        self, mapping: DatasetMapping
    ) -> None:
        out, kept, dropped = remap_label_text("1 0.5 0.2 0.06 0.05\n", SOURCE_NAMES, mapping)
        assert out == ""
        assert not kept
        assert dropped == {"no-helmet": 1}

    def test_blank_lines_are_ignored(self, mapping: DatasetMapping) -> None:
        out, kept, _ = remap_label_text("\n3 0.5 0.5 0.2 0.8\n\n", SOURCE_NAMES, mapping)
        assert out == "0 0.5 0.5 0.2 0.8\n"
        assert kept == {"person": 1}

    def test_empty_file_stays_empty(self, mapping: DatasetMapping) -> None:
        assert remap_label_text("", SOURCE_NAMES, mapping)[0] == ""

    def test_unknown_source_class_is_refused_not_guessed(self, mapping: DatasetMapping) -> None:
        """A class nobody mapped is a decision nobody made."""
        names = [*SOURCE_NAMES, "gloves"]
        with pytest.raises(LabelFormatError, match="neither mapped nor dropped"):
            remap_label_text("5 0.5 0.5 0.2 0.8", names, mapping)

    def test_class_index_beyond_the_source_list_is_refused(self, mapping: DatasetMapping) -> None:
        with pytest.raises(LabelFormatError, match="outside the source class list"):
            remap_label_text("99 0.5 0.5 0.2 0.8", SOURCE_NAMES, mapping)

    def test_reordering_the_source_classes_cannot_corrupt_labels(
        self, mapping: DatasetMapping
    ) -> None:
        """Mapping by name, not index: a re-export that reorders is harmless."""
        shuffled = ["vest", "person", "helmet", "no-vest", "no-helmet"]
        out, kept, _ = remap_label_text("1 0.5 0.5 0.2 0.8", shuffled, mapping)
        assert out == "0 0.5 0.5 0.2 0.8\n"
        assert kept == {"person": 1}


def _make_source(root: Path, *, splits: tuple[str, ...] = ("train", "valid", "test")) -> Path:
    """Build a miniature Roboflow-style export on disk."""
    (root).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(SOURCE_NAMES), "names": SOURCE_NAMES}), encoding="utf-8"
    )
    for split in splits:
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(2):
            (images / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (labels / f"img{i}.txt").write_text(
                "3 0.5 0.5 0.2 0.8\n0 0.5 0.2 0.06 0.05\n1 0.4 0.2 0.05 0.05\n",
                encoding="utf-8",
            )
    return root


class TestConvertSplit:
    def test_converts_a_split_end_to_end(self, tmp_path: Path, mapping: DatasetMapping) -> None:
        source = _make_source(tmp_path / "src")
        target = tmp_path / "out"
        result = convert_split(source, target, "train", "train", SOURCE_NAMES, mapping)

        assert result.images == 2
        assert result.instances_kept == {"person": 2, "helmet": 2}
        assert result.instances_dropped == {"no-helmet": 2}
        assert (target / "train" / "images" / "img0.jpg").exists()
        assert (target / "train" / "labels" / "img0.txt").read_text(encoding="utf-8") == (
            "0 0.5 0.5 0.2 0.8\n1 0.5 0.2 0.06 0.05\n"
        )

    def test_image_without_a_label_file_becomes_a_background_image(
        self, tmp_path: Path, mapping: DatasetMapping
    ) -> None:
        source = _make_source(tmp_path / "src")
        (source / "train" / "labels" / "img1.txt").unlink()
        result = convert_split(source, tmp_path / "out", "train", "train", SOURCE_NAMES, mapping)
        assert result.missing_label_files == ["img1.jpg"]
        assert result.images_without_labels == 1
        assert (tmp_path / "out" / "train" / "labels" / "img1.txt").read_text() == ""

    def test_link_only_skips_copying_images(self, tmp_path: Path, mapping: DatasetMapping) -> None:
        source = _make_source(tmp_path / "src")
        target = tmp_path / "out"
        convert_split(source, target, "train", "train", SOURCE_NAMES, mapping, copy_images=False)
        assert not list((target / "train" / "images").iterdir())
        assert list((target / "train" / "labels").iterdir())

    def test_missing_split_directory_is_reported_clearly(
        self, tmp_path: Path, mapping: DatasetMapping
    ) -> None:
        source = _make_source(tmp_path / "src", splits=("train",))
        with pytest.raises(ConfigError, match="missing image directory"):
            convert_split(source, tmp_path / "out", "test", "test", SOURCE_NAMES, mapping)

    def test_a_bad_label_row_names_the_offending_file(
        self, tmp_path: Path, mapping: DatasetMapping
    ) -> None:
        source = _make_source(tmp_path / "src")
        (source / "train" / "labels" / "img0.txt").write_text("3 0.5 0.5\n", encoding="utf-8")
        with pytest.raises(LabelFormatError, match=r"img0\.txt"):
            convert_split(source, tmp_path / "out", "train", "train", SOURCE_NAMES, mapping)


class TestSourceNames:
    def test_reads_a_list(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path / "src")
        assert read_source_names(source) == SOURCE_NAMES

    def test_reads_an_index_keyed_mapping(self, tmp_path: Path) -> None:
        """Some exports write names as {0: person, 1: helmet} instead of a list."""
        root = tmp_path / "src"
        root.mkdir()
        (root / "data.yaml").write_text(
            yaml.safe_dump({"names": {1: "helmet", 0: "person"}}), encoding="utf-8"
        )
        assert read_source_names(root) == ["person", "helmet"]

    def test_missing_data_yaml_explains_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"data\.yaml"):
            read_source_names(tmp_path)


class TestDataYaml:
    def test_declares_our_canonical_class_order(
        self, tmp_path: Path, mapping: DatasetMapping
    ) -> None:
        path = write_data_yaml(tmp_path, mapping)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["names"] == ["person", "helmet", "vest"]
        assert payload["nc"] == 3

    def test_carries_the_licence_forward(self, tmp_path: Path, mapping: DatasetMapping) -> None:
        """Attribution should travel with the data, not live only in the README."""
        payload = yaml.safe_load(write_data_yaml(tmp_path, mapping).read_text(encoding="utf-8"))
        assert payload["licence"] == "CC BY 4.0"
        assert payload["source"] == "https://example.invalid/ds"
