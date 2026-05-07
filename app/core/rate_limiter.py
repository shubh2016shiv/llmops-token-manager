"""
Rate Limiting.

Enterprise-grade rate limiting integration for FastAPI.

Design goals:
- Use a battle-tested library (`limits`) instead of bespoke algorithms.
- Centralize configuration and keying logic (single source of truth).
- Make rules explicit per-endpoint via FastAPI dependencies.
- Support distributed enforcement via Redis; allow Memory storage for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from limits import parse
from limits.aio.storage import MemoryStorage
from limits.aio.strategies import MovingWindowRateLimiter
from limits.storage import storage_from_string
from loguru import logger

from app.core.config_manager import settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# `limits` provides identical APIs for sync/async.
# Use `limits.aio` for async strategies.


@dataclass(frozen=True)
class RateLimitRule:
    """Rate limit rule definition."""

    name: str
    limit: str  # e.g. "10/minute"
    key_namespace: str  # e.g. "auth_login"


class RateLimitExceededError(Exception):
    """Structured rate-limit exception with response-ready payload."""

    def __init__(self, payload: dict, retry_after: int):
        self.payload = payload
        self.retry_after = retry_after
        super().__init__(payload.get("message", "Rate limit exceeded"))


# Backward-compatible alias for existing imports.
RateLimitExceeded = RateLimitExceededError


def register_rate_limit_exception_handler(app: FastAPI) -> None:
    """Register a standardized top-level 429 payload for rate limit errors."""

    @app.exception_handler(RateLimitExceededError)
    async def _rate_limit_exceeded_handler(
        request: Request, exc: RateLimitExceededError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=exc.payload,
            headers={"Retry-After": str(exc.retry_after)},
        )


def _redis_dsn() -> str:
    """
    Build an async Redis DSN for `limits`.

    `limits` expects an `async+redis://...` DSN for async redis storage.
    """
    auth = f":{settings.redis_password}@" if settings.redis_password else ""
    return f"async+redis://{auth}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"


@lru_cache(maxsize=1)
def get_rate_limit_storage():
    """
    Storage backend for rate limiting.

    Enterprise/testing pattern:
    - Use Redis in production for distributed enforcement.
    - Allow opting into in-memory storage for tests via env var.
    """
    if os.getenv("RATE_LIMIT_STORAGE", "").lower() == "memory":
        return MemoryStorage()
    return storage_from_string(_redis_dsn())


@lru_cache(maxsize=1)
def get_rate_limiter() -> MovingWindowRateLimiter:
    """Create the async moving-window limiter."""
    return MovingWindowRateLimiter(get_rate_limit_storage())


def get_client_ip(request: Request) -> str:
    """
    Best-effort client IP extraction.

    Enterprise note:
    - Prefer X-Forwarded-For when behind a trusted proxy/load balancer.
    - Fall back to Starlette's request.client.host.
    """
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        # Take the first (original) IP.
        first = xff.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _retry_after_seconds(reset_at) -> int:
    """Compute Retry-After seconds from limits window stats reset value."""
    if reset_at is None:
        return 1
    # `limits` may return a unix timestamp or a datetime-like object.
    try:
        reset_ts = float(reset_at)
    except (TypeError, ValueError):
        try:
            reset_ts = float(reset_at.timestamp())
        except Exception:
            return 1
    return max(1, int(reset_ts - time.time()))


def rate_limit_dependency(
    *,
    rule: RateLimitRule,
    key_fn: Callable[[Request], Awaitable[str]],
) -> Callable[[Request], Awaitable[None]]:
    """
    Build a FastAPI dependency that enforces a given rate limit rule.

    Failure mode policy (enterprise default):
    - Fail-open for availability: if Redis/storage is unavailable, allow request
      but log an error so it can be observed/alerted on.
    """
    limit_item = parse(rule.limit)

    async def _dependency(request: Request) -> None:
        limiter = get_rate_limiter()
        key = await key_fn(request)

        try:
            allowed = await limiter.hit(limit_item, rule.key_namespace, key)
            if allowed:
                return

            # Compute Retry-After (best effort)
            remaining, reset_at = await limiter.get_window_stats(
                limit_item, rule.key_namespace, key
            )
            retry_after = _retry_after_seconds(reset_at)

            raise RateLimitExceededError(
                payload={
                    "error": "RATE_LIMITED",
                    "message": "Too many requests. Please retry later.",
                    "details": {
                        "rule": rule.name,
                        "retry_after_seconds": retry_after,
                        "remaining": remaining,
                    },
                },
                retry_after=retry_after,
            )

        except RateLimitExceededError:
            raise
        except Exception as e:
            # Fail-open with observability.
            logger.error(f"Rate limiter failure for rule={rule.name}: {e}")
            return

    return _dependency


# ---------------------------------------------------------------------------
# Prebuilt key functions and dependencies for auth endpoints
# ---------------------------------------------------------------------------


async def login_rate_limit_key(request: Request) -> str:
    """
    Key for login: ip + username (if available).

    Reads JSON body safely (Starlette caches request body).
    """
    ip = get_client_ip(request)
    username: str | None = None
    try:
        body = await request.json()
        username = body.get("username") if isinstance(body, dict) else None
    except Exception:
        username = None

    if username:
        return f"{ip}:{username}"
    return ip


async def ip_only_key(request: Request) -> str:
    """Key by client IP only."""
    return get_client_ip(request)


def auth_login_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """Dependency enforcing login rate limiting."""
    rule = RateLimitRule(
        name="auth_login",
        limit=f"{settings.rate_limit_login_per_minute}/minute",
        key_namespace="auth_login",
    )
    return rate_limit_dependency(rule=rule, key_fn=login_rate_limit_key)


def auth_token_generate_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """Dependency enforcing token generation rate limiting."""
    rule = RateLimitRule(
        name="auth_token_generate",
        limit=f"{settings.rate_limit_token_generate_per_minute}/minute",
        key_namespace="auth_token_generate",
    )
    return rate_limit_dependency(rule=rule, key_fn=ip_only_key)


def auth_token_refresh_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """Dependency enforcing token refresh rate limiting."""
    rule = RateLimitRule(
        name="auth_token_refresh",
        limit=f"{settings.rate_limit_token_refresh_per_minute}/minute",
        key_namespace="auth_token_refresh",
    )
    return rate_limit_dependency(rule=rule, key_fn=ip_only_key)
