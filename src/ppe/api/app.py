"""FastAPI application factory. Implemented in Phase 6."""

from __future__ import annotations

from fastapi import FastAPI

from ppe.config.schema import AppConfig


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the ASGI app.

    Takes config as an argument rather than reading a global, so tests can
    construct an app with any settings they like.
    """
    raise NotImplementedError
