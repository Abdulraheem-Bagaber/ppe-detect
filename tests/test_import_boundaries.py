"""The architecture, expressed as a test.

The central claim of this project is that the compliance rule is a separate,
inspectable, tunable layer rather than something baked into model weights. That
claim is only true if the rule layer genuinely cannot reach the model. A code
review convention decays; a failing build does not.

The check is static -- it parses the AST rather than importing anything -- so
it holds even for modules whose imports sit behind a lazy function call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ppe"

# Packages the rule layer must never reach, and why.
FORBIDDEN_IN_RULES = {
    "ultralytics": "AGPL-3.0 detector; the rule layer must be model-agnostic",
    "torch": "pulling the ML runtime into pure geometry",
    "torchvision": "pulling the ML runtime into pure geometry",
    "cv2": "video I/O has no business inside a compliance rule",
    "supervision": "imports cv2 transitively",
    "onnxruntime": "inference runtime",
    "fastapi": "transport concerns",
    "uvicorn": "transport concerns",
}

# core/ is the shared vocabulary: it must depend on nothing but stdlib + numpy.
ALLOWED_IN_CORE_ROOTS = {"numpy", "ppe"}


def _module_files(package_dir: Path) -> list[Path]:
    return sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_roots(path: Path) -> set[str]:
    """Top-level names of every module imported anywhere in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _module_files(SRC / "rules"), ids=lambda p: p.name)
def test_rule_layer_imports_nothing_from_the_model_stack(path: Path) -> None:
    offenders = _imported_roots(path) & FORBIDDEN_IN_RULES.keys()
    assert not offenders, (
        f"{path.relative_to(SRC)} imports {sorted(offenders)}. "
        + "; ".join(f"{name}: {FORBIDDEN_IN_RULES[name]}" for name in sorted(offenders))
        + ". If the rule layer genuinely needs this, the boundary is wrong -- "
        "say so rather than adding the import."
    )


@pytest.mark.parametrize("path", _module_files(SRC / "core"), ids=lambda p: p.name)
def test_core_depends_on_nothing_but_stdlib_and_numpy(path: Path) -> None:
    import sys

    external = {
        root
        for root in _imported_roots(path)
        if root not in sys.stdlib_module_names and root not in ALLOWED_IN_CORE_ROOTS
    }
    assert not external, (
        f"{path.relative_to(SRC)} imports {sorted(external)}; "
        "ppe.core is the shared vocabulary and must stay dependency-free."
    )


ULTRALYTICS_IMPORT_SITE = "detection/ultralytics_detector.py"


def test_ultralytics_is_confined_to_one_module() -> None:
    """The AGPL boundary. One permitted import site, so the obligation is legible.

    Asserted as a subset rather than an exact match: the site is still a stub in
    Phase 0 and imports nothing yet. What must never change is that no *other*
    module reaches for it.
    """
    importers = {
        path.relative_to(SRC).as_posix()
        for path in _module_files(SRC)
        if "ultralytics" in _imported_roots(path)
    }
    stray = importers - {ULTRALYTICS_IMPORT_SITE}
    assert not stray, (
        f"ultralytics is imported by {sorted(stray)}. It is AGPL-3.0 and must stay "
        f"behind the Detector protocol in {ULTRALYTICS_IMPORT_SITE} alone."
    )


def test_detection_package_init_stays_import_free() -> None:
    """``import ppe.detection.base`` must not drag torch into the process."""
    roots = _imported_roots(SRC / "detection" / "__init__.py")
    assert not roots, (
        f"src/ppe/detection/__init__.py imports {sorted(roots)}; keep it empty so "
        "importing any submodule stays cheap and side-effect free."
    )
