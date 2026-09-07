"""Run a dev tool from the project virtualenv, whatever the ambient PATH is.

pre-commit hooks with ``language: system`` inherit whichever PATH git happens
to hand them. That is one interpreter when git invokes the hook from a shell
and a different one when the hook runs under ``uv run``, so ``uv run mypy``
inside a hook is quietly unreliable -- it works until it doesn't.

Resolving ``.venv`` by path removes the ambiguity: the venv is a fixed
location relative to the repo root, and pre-commit always runs from there.

    python scripts/dev_tool.py mypy
    python scripts/dev_tool.py pytest tests/test_import_boundaries.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIRS = ("Scripts", "bin")  # Windows, then POSIX


def resolve(tool: str) -> Path:
    """Locate an executable inside the project virtualenv."""
    venv = Path(os.environ.get("VIRTUAL_ENV", REPO_ROOT / ".venv"))
    for bin_dir in BIN_DIRS:
        for suffix in (".exe", ""):
            candidate = venv / bin_dir / f"{tool}{suffix}"
            if candidate.is_file():
                return candidate
    msg = (
        f"{tool} not found in {venv}. Run `uv sync --dev` "
        f"(or `.\\make.ps1 install`) to create the environment."
    )
    raise SystemExit(msg)


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: dev_tool.py <tool> [args...]")
    return subprocess.run([str(resolve(argv[0])), *argv[1:]], cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
