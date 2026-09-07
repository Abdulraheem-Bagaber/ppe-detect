"""Command-line entry point. Fleshed out from Phase 2 onward."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="PPE compliance detection: detect, associate, and report violations.",
)


@app.command()
def version() -> None:
    """Print the package version."""
    from ppe import __version__

    typer.echo(__version__)


@app.command()
def check_config(path: str = "configs/default.yaml") -> None:
    """Validate a config file and print the resolved values."""
    from ppe.config import load_config

    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
