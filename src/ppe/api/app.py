"""FastAPI application factory.

Takes config as an argument rather than reading a global, so tests can build
an app with any settings they like and the privacy posture can be asserted
directly.

The detector is loaded once at startup and shared. Loading it per request
would make the first request of every burst pathologically slow and would
misrepresent every latency number in the benchmark table.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ppe.config import load_config
from ppe.config.schema import AppConfig
from ppe.core.errors import PPEError


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the ASGI app."""
    settings = config if config is not None else load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = settings
        app.state.detector = None
        app.state.detector_error = None
        try:
            from ppe.detection.factory import build_detector

            app.state.detector = build_detector(settings.model)
        except PPEError as exc:
            # A missing model must not stop the service from starting: /health
            # needs to come up and say what is wrong. Analysis endpoints then
            # fail loudly per request rather than silently returning nothing.
            app.state.detector_error = str(exc)
        yield

    app = FastAPI(
        title="ppe-detect",
        description=(
            "PPE compliance detection. Faces are blurred before any frame leaves this service."
        ),
        version=_version(),
        lifespan=lifespan,
    )

    from ppe.api.routes import router

    app.include_router(router)
    return app


def _version() -> str:
    from ppe import __version__

    return __version__
