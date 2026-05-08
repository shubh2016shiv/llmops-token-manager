# ============================================================
# Multi-stage Dockerfile for LLM Token Manager
# ============================================================
# Stage 1: Builder — compiles only production dependencies
# Stage 2: Runtime — minimal, hardened production image
#
# Enterprise standards applied:
#   - Multi-stage build (no build tools in final image)
#   - Pinned base image tags
#   - Non-root user with fixed UID
#   - pipefail on all shell RUN commands
#   - Only production deps installed (dev tools excluded)
#   - Only application code copied (tests/docs/schema excluded)
#   - HEALTHCHECK baked into the image
#   - Layer ordering optimized for cache hits
# ============================================================

# -------------------- Builder Stage --------------------------
FROM python:3.11.9-slim AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /build

# Install only the system packages needed to compile Python wheels.
# Most of our deps ship pre-built wheels for linux/amd64, so we
# only need libpq-dev headers for psycopg2.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy production dependency manifest (this layer is cache-stable)
COPY requirements.txt .

# Install production Python packages into an isolated prefix
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# -------------------- Runtime Stage --------------------------
FROM python:3.11.9-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

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

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

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
