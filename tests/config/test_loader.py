"""Tests for configuration loading.

The committed ``configs/default.yaml`` is a deliverable, not a sample: if it
stops parsing, every later phase is running on undeclared defaults. So it is
loaded and checked here rather than mocked away.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppe.config import AppConfig, load_config
from ppe.core.errors import ConfigError
from ppe.core.types import PPEClass

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class TestDefaultConfigFile:
    def test_the_committed_default_parses(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        assert isinstance(config, AppConfig)

    def test_required_ppe_is_helmet_and_vest(self) -> None:
        """Scope guard: two PPE classes done properly, no gloves or boots."""
        config = load_config(DEFAULT_CONFIG)
        assert set(config.association.required) == {PPEClass.HELMET, PPEClass.VEST}

    def test_helmet_band_is_the_upper_region(self) -> None:
        rule = load_config(DEFAULT_CONFIG).association.rules[PPEClass.HELMET]
        assert rule.band_top == 0.0
        assert rule.band_bottom <= 0.5, "a waist-height helmet must fall outside the band"

    def test_privacy_is_on_by_default(self) -> None:
        """PDPL posture: blurring must fail closed, not require opting in."""
        config = load_config(DEFAULT_CONFIG)
        assert config.privacy.blur_faces is True
        assert config.privacy.strict is True

    def test_class_map_covers_every_class(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        assert set(config.model.class_map.values()) == set(PPEClass)


class TestLoaderBehaviour:
    def test_missing_explicit_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file_falls_back_to_schema_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        assert load_config(empty) == AppConfig()

    def test_malformed_yaml_is_reported_as_a_config_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("model: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="could not parse YAML"):
            load_config(bad)

    def test_non_mapping_root_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config(bad)

    def test_unknown_key_fails_loudly(self, tmp_path: Path) -> None:
        """A typo must not be silently ignored into a default."""
        typo = tmp_path / "typo.yaml"
        typo.write_text("debounce:\n  confirm_framez: 20\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(typo)

    def test_yaml_overrides_schema_defaults(self, tmp_path: Path) -> None:
        override = tmp_path / "override.yaml"
        override.write_text("debounce:\n  confirm_frames: 42\n", encoding="utf-8")
        assert load_config(override).debounce.confirm_frames == 42

    def test_environment_overrides_yaml(self, tmp_path: Path, monkeypatch) -> None:
        """The only case env overrides matter is when the key is *also* in the file."""
        override = tmp_path / "override.yaml"
        override.write_text("debounce:\n  confirm_frames: 42\n", encoding="utf-8")
        monkeypatch.setenv("PPE_DEBOUNCE", '{"confirm_frames": 7}')
        assert load_config(override).debounce.confirm_frames == 7

    def test_nested_delimiter_env_form_works(self, monkeypatch) -> None:
        """The form documented in the README and used in the Docker image."""
        monkeypatch.setenv("PPE_DEBOUNCE__CONFIRM_FRAMES", "20")
        assert load_config(DEFAULT_CONFIG).debounce.confirm_frames == 20


class TestSchemaValidation:
    def test_rejects_an_inverted_band(self, tmp_path: Path) -> None:
        bad = tmp_path / "band.yaml"
        bad.write_text(
            "association:\n"
            "  rules:\n"
            "    helmet: {band_top: 0.8, band_bottom: 0.2}\n"
            "    vest: {band_top: 0.15, band_bottom: 0.75}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="band_top"):
            load_config(bad)

    def test_rejects_a_required_item_with_no_rule(self, tmp_path: Path) -> None:
        bad = tmp_path / "norule.yaml"
        bad.write_text(
            "association:\n"
            "  required: [helmet, vest]\n"
            "  rules:\n"
            "    helmet: {band_top: 0.0, band_bottom: 0.35}\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="without an association rule"):
            load_config(bad)

    def test_person_cannot_be_required_ppe(self, tmp_path: Path) -> None:
        bad = tmp_path / "person.yaml"
        bad.write_text("association:\n  required: [person]\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad)

    def test_rejects_a_degenerate_zone_polygon(self, tmp_path: Path) -> None:
        bad = tmp_path / "zone.yaml"
        bad.write_text(
            "zones:\n  zones:\n    - zone_id: gate\n      polygon: [[0, 0], [0, 0], [0, 0]]\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="distinct vertices"):
            load_config(bad)

    def test_config_is_frozen(self) -> None:
        """Nothing may retune thresholds at runtime behind the operator's back."""
        config = load_config(DEFAULT_CONFIG)
        with pytest.raises((AttributeError, TypeError, ValueError)):
            config.debounce.confirm_frames = 1  # type: ignore[misc]
