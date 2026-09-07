"""Smoke tests: the package is installed and importable."""

from __future__ import annotations

import importlib


def test_package_imports_and_reports_a_version() -> None:
    import ppe

    assert ppe.__version__


def test_importing_the_rule_layer_does_not_load_torch() -> None:
    """Rule-layer tests must stay fast; loading torch takes seconds."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "import ppe.rules.association, ppe.rules.geometry, ppe.rules.zones, ppe.rules.debounce;"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m);"
        "assert 'cv2' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_stub_module_is_importable() -> None:
    """Stubs raise NotImplementedError when *called*, not when imported."""
    for name in [
        "ppe.core.types",
        "ppe.core.protocols",
        "ppe.core.errors",
        "ppe.config.schema",
        "ppe.config.loader",
        "ppe.rules.geometry",
        "ppe.rules.association",
        "ppe.rules.zones",
        "ppe.rules.debounce",
        "ppe.detection.base",
        "ppe.detection.factory",
        "ppe.tracking.bytetrack",
        "ppe.pipeline.runner",
        "ppe.pipeline.sources",
        "ppe.privacy.blur",
        "ppe.api.schemas",
        "ppe.cli",
    ]:
        assert importlib.import_module(name) is not None
