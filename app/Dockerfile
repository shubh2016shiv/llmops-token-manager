# ============================================================
# Multi-stage Dockerfile for LLM Token Manager
# ============================================================
# Stage 1: uv-binary — the `uv` tool itself, pulled as a binary
# Stage 2: builder    — resolves and installs deps via `uv sync --frozen`
# Stage 3: runtime     — minimal, hardened production image
#
# Enterprise standards applied:
#   - Multi-stage build (no build tools, no uv binary in final image)
#   - Pinned base image tags
#   - Non-root user with fixed UID
#   - pipefail on all shell RUN commands
#   - Only production deps installed (--no-dev)
#   - Only application code copied (tests/docs/schema excluded)
#   - HEALTHCHECK baked into the image
#   - Layer ordering optimized for cache hits
#
# ------------------------------------------------------------
# WHY `uv sync --frozen`, NOT `pip install -r requirements.txt`
# ------------------------------------------------------------
# This project's real dependency declaration is pyproject.toml + uv.lock.
# requirements.txt was a hand-maintained second manifest that drifted from
# it: `aiobreaker`, `coredis`, and `pyyaml` are all declared in pyproject.toml
# and imported by live code (circuit breaker, rate limiter, YAML config
# loading), but were missing from requirements.txt. Every container built
# from it crashed at import time with ModuleNotFoundError — confirmed
# against a running stack.
#
# `--frozen` makes that class of bug impossible by construction: it installs
# EXACTLY what uv.lock resolved, and FAILS the build if uv.lock does not
# match pyproject.toml, instead of silently installing an incomplete set.
# There is now exactly one dependency manifest read by this build, matching
# the pattern already used in ../../llm_services/Dockerfile.
#
# requirements.txt / requirements-dev.txt are no longer read by this
# Dockerfile. See infra/docs/04-decisions.md ADR-010.
# ============================================================

FROM python:3.11.9-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# UV_PROJECT_ENVIRONMENT is the mechanism that actually matters here. `uv sync`
# ignores VIRTUAL_ENV for placement and defaults to creating `.venv` next to
# pyproject.toml regardless of --python — verified by building this stage and
# finding uvicorn installed at /build/.venv/bin, not /opt/venv/bin, while
# /opt/venv silently stayed at its post-`python -m venv` empty state. This is
# the one variable that redirects the sync target itself.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Created once here so both `builder` and `runtime` below start from the
# identical, empty venv layer — builder fills it, runtime just receives it.
RUN python -m venv "${VIRTUAL_ENV}"


# -------------------- uv binary ------------------------------
# Pulled from astral's own distroless image rather than pip-installed, so
# `uv` never depends on the Python environment it is about to build.
FROM ghcr.io/astral-sh/uv:0.7.3 AS uv-binary


# -------------------- Builder Stage ----------------------------
FROM base AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /build

# System packages needed to compile any dependency without a pre-built
# linux/amd64 wheel. libpq-dev is for psycopg2/asyncpg.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-binary /uv /uvx /bin/

# Only the manifests — this layer stays cache-stable across app-code changes.
COPY pyproject.toml uv.lock ./

# --frozen:              fail rather than silently re-resolve if uv.lock is
#                         out of sync with pyproject.toml (the guarantee
#                         this whole switch exists for)
# --no-dev:               exclude the dev dependency group (pytest, ruff, ...)
# --no-install-project:   install only third-party deps; app/ is COPY'd
#                         below, not installed as an editable package
RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project \
    --python "${VIRTUAL_ENV}/bin/python"


# -------------------- Runtime Stage -----------------------------
FROM base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Create non-root user (fixed UID for security scanners)
RUN useradd --create-home --uid 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Install runtime system libraries only (no compilers, no headers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The fully resolved virtual environment from the builder stage — this
# replaces the empty one inherited from `base`.
COPY --from=builder /opt/venv /opt/venv

# Copy ONLY the application package (not tests, docs, schema, scripts)
COPY --chown=appuser:appuser app/ ./app/

# Drop privileges
USER appuser

# Expose application port
EXPOSE 8000

# Built-in health check (complementary to docker-compose healthcheck)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Default command (overridable via docker-compose or CLI)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
