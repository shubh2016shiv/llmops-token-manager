"""Rate-limit policies for active HTTP endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.redis_rate_limiter.rate_limit_enforcement import rate_limit_dependency
from app.core.redis_rate_limiter.rate_limit_keys import ip_only_key
from app.models.redis_rate_limit_models import RateLimitRule

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request


def auth_token_generate_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """Limit development token generation per verified client address."""
    return rate_limit_dependency(
        rule=RateLimitRule(
            name="auth_token_generate",
            limit=f"{settings.rate_limit_token_generate_per_minute}/minute",
            key_namespace="auth_token_generate",
        ),
        key_fn=ip_only_key,
    )


def auth_token_refresh_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """Limit token refresh independently of token generation."""
    return rate_limit_dependency(
        rule=RateLimitRule(
            name="auth_token_refresh",
            limit=f"{settings.rate_limit_token_refresh_per_minute}/minute",
            key_namespace="auth_token_refresh",
        ),
        key_fn=ip_only_key,
    )


def token_acquire_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """Limit acquisition without trusting the caller-supplied service header."""
    return rate_limit_dependency(
        rule=RateLimitRule(
            name="token_acquire",
            limit=f"{settings.rate_limit_token_acquire_per_minute}/minute",
            key_namespace="token_acquire",
        ),
        key_fn=ip_only_key,
    )
