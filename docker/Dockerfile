# Multi-stage, CPU-only. The runtime image carries no build toolchain, no
# training dependencies and no dataset.
#
# CPU-only is deliberate rather than a limitation: the served path is ONNX
# Runtime, the benchmark table reports CPU numbers, and a GPU image would be a
# different artefact with a different licence surface.

# ---- builder -------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the version that wrote uv.lock. An older uv cannot read a
# newer lockfile, and the resolver version is a reproducibility input
# exactly like the pinned packages are.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first, so a source change does not re-resolve the world.
COPY pyproject.toml uv.lock README.md ./
# NO extras. The base dependency set is exactly the serving path: ONNX
# Runtime behind the Detector protocol, with no ultralytics and no torch.
# That is the protocol boundary paying for itself twice -- it strips the
# training stack out of the image, and it keeps AGPL-licensed code out of
# the deployed artefact entirely.
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# The served backend is ONNX. Training happens outside the image, with
# `uv sync --extra cpu` or `--extra gpu`; see the README.

# ---- runtime -------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# libGL and libglib are what opencv-python links against. The alternative is
# opencv-python-headless, but ultralytics pins the full package, so the
# libraries come along rather than the dependency being fought.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ppe

WORKDIR /app

COPY --from=builder --chown=ppe:ppe /app/.venv /app/.venv
COPY --chown=ppe:ppe src/ ./src/
COPY --chown=ppe:ppe configs/ ./configs/

# Models are mounted, never baked in: they are large, they change on their own
# schedule, and an image is not a good place to version weights.
RUN mkdir -p /app/models /app/data && chown ppe:ppe /app/models /app/data

USER ppe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "ppe.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
