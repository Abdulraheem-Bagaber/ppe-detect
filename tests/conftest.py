"""Shared fixtures.

The important thing in here is :func:`isolate_from_environment`, which makes
the suite independent of whatever ``PPE_*`` variables happen to be set in the
shell that launched it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ppe.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Config reads these with HIGHER precedence than constructor arguments, which
# is deliberate -- a container must be able to retune a deployment without
# editing a mounted file. The side effect is that a test constructing an
# AppConfig directly can have it silently overridden by the ambient shell.
ENV_PREFIX = "PPE_"


@pytest.fixture(autouse=True)
def isolate_from_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every ``PPE_*`` variable for the duration of each test.

    Without this the suite passes or fails depending on the shell it is run
    from. The concrete case that prompted it: ``make viewer`` exports
    ``PPE_MODEL__WEIGHTS`` so the server can find the trained model, PowerShell
    keeps that set for the rest of the session, and a later ``make check`` in
    the same terminal saw two tests fail -- a fixture asked for a deliberately
    missing model, the environment pointed at a real one, and the service
    reported "ok" where the test expected "degraded".

    Nothing was wrong with the code. The tests were reading the developer's
    shell, which is a property no test should have.
    """
    for name in [k for k in os.environ if k.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def default_config() -> AppConfig:
    """The committed default configuration, as later phases will see it."""
    return load_config(REPO_ROOT / "configs" / "default.yaml")
