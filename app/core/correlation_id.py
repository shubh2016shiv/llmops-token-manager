"""
Correlation ID Middleware.

Enterprise-grade request tracing for FastAPI.

Goal:
- Accept an inbound correlation id header (when present), otherwise generate one.
- Make the correlation id available to logs for the duration of the request.
- Return the correlation id on every response for downstream propagation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

if TYPE_CHECKING:
    from fastapi import Request, Response

CORRELATION_ID_HEADER = "X-Correlation-Id"


def generate_correlation_id() -> str:
    """Generate a stable, URL-safe correlation id."""
    return uuid4().hex


def get_or_create_correlation_id(request: Request) -> str:
    """
    Read correlation id from request headers or generate a new one.

    Treat empty/whitespace-only values as missing.
    """
    raw = request.headers.get(CORRELATION_ID_HEADER)
    if raw is None:
        return generate_correlation_id()

    normalized = raw.strip()
    return normalized or generate_correlation_id()


async def correlation_id_middleware(request: Request, call_next) -> Response:
    """
    FastAPI middleware that propagates a correlation id.

    - Binds correlation_id into loguru context, so any log line emitted during the
      request can include it via `{extra[correlation_id]}`.
    - Always sets `X-Correlation-Id` on the response.
    """
    correlation_id = get_or_create_correlation_id(request)

    with logger.contextualize(correlation_id=correlation_id):
        response: Response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
