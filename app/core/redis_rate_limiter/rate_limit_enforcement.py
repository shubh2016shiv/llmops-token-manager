"""FastAPI dependency and HTTP 429 translation for rate-limit decisions."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from limits import parse
from loguru import logger

from app.core.exceptions import RateLimitExceededError
from app.core.redis_rate_limiter.moving_window_limiter import rate_limiter_manager
from app.models.redis_rate_limit_models import (
    RateLimitedErrorDetail,
    RateLimitedResponse,
    RateLimitRule,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


MIN_RETRY_AFTER_SECONDS = 1


def _resolve_transient_storage_errors() -> tuple[type[BaseException], ...]:
    """Include errors raised by both socket operations and the coredis driver."""
    builtin_errors: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    try:
        from coredis.exceptions import ConnectionError as CoredisConnectionError
        from coredis.exceptions import TimeoutError as CoredisTimeoutError
    except ImportError:
        return builtin_errors
    return (*builtin_errors, CoredisConnectionError, CoredisTimeoutError)


_TRANSIENT_STORAGE_ERRORS = _resolve_transient_storage_errors()


async def _rate_limit_exceeded_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Translate an exhausted bucket into a structured HTTP 429 response."""
    if not isinstance(exc, RateLimitExceededError):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=exc.payload,
        headers={"Retry-After": str(exc.retry_after)},
    )


def register_rate_limit_exception_handler(app: FastAPI) -> None:
    """Register the application's shared HTTP 429 handler."""
    app.add_exception_handler(RateLimitExceededError, _rate_limit_exceeded_handler)


def _retry_after_seconds(reset_at: float) -> int:
    """Round up so clients never retry before the actual reset time."""
    return max(MIN_RETRY_AFTER_SECONDS, math.ceil(reset_at - time.time()))


def rate_limit_dependency(
    *,
    rule: RateLimitRule,
    key_fn: Callable[[Request], Awaitable[str]],
) -> Callable[[Request], Awaitable[None]]:
    """Build a request dependency for a validated endpoint rate-limit rule."""
    limit_item = parse(rule.limit)

    async def _dependency(request: Request) -> None:
        limiter = rate_limiter_manager.limiter
        key = await key_fn(request)
        try:
            allowed = await limiter.hit(limit_item, rule.key_namespace, key)
            if allowed:
                return
            reset_at, remaining = await limiter.get_window_stats(
                limit_item, rule.key_namespace, key
            )
            retry_after = _retry_after_seconds(reset_at)
            response = RateLimitedResponse(
                details=RateLimitedErrorDetail(
                    rule=rule.name,
                    retry_after_seconds=retry_after,
                    remaining=int(remaining),
                ),
            )
            raise RateLimitExceededError(
                payload=response.to_payload(),
                retry_after=retry_after,
            )
        except RateLimitExceededError:
            raise
        except _TRANSIENT_STORAGE_ERRORS:
            logger.warning(
                "Rate limiter unavailable for rule={}; failing open",
                rule.name,
            )
        except Exception:
            logger.exception("Unexpected rate limiter failure for rule={}", rule.name)
            raise

    return _dependency
