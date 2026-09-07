"""Shared fixtures.

Deliberately thin in Phase 0. Geometry fixtures for the hard association cases
(overlapping workers, a carried helmet, an edge-cropped person) arrive with the
Phase 3 tests, which are written before the association code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppe.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def default_config() -> AppConfig:
    """The committed default configuration, as later phases will see it."""
    return load_config(REPO_ROOT / "configs" / "default.yaml")
