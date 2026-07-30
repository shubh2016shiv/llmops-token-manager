"""
The concrete per-endpoint rate-limit definitions.

----
What this module is
-------------------
This is the "recipe book." Each function here is a recipe that says:
"Protect endpoint X with limit Y, keyed by Z."

The actual enforcement logic lives in `rate_limit_enforcement.py`. The key
functions live in `rate_limit_keys.py`. This module just PAIRS them together
for specific endpoints — it's glue code, not algorithm code.

----
How a route uses one of these
------------------------------
In your route file, you attach the dependency to the route decorator:

    @router.post("/token/generate",
                 dependencies=[Depends(auth_token_generate_rate_limiter())])
    async def generate_token(...):
        ...

When FastAPI starts up:
1. It calls `auth_token_generate_rate_limiter()` ONCE.
2. That function creates a RateLimitRule + calls rate_limit_dependency().
3. rate_limit_dependency() parses the rule, builds the inner _dependency
   function, and returns it.
4. FastAPI caches the returned function.

On every request to /token/generate:
1. FastAPI calls the cached _dependency(request).
2. _dependency extracts the caller's IP, checks Redis, and either
   allows (returns silently) or blocks (raises RateLimitExceededError → 429).

----
Adding a new rate-limited endpoint
------------------------------------
1. Add a new factory function here following the same pattern:
   - Create a RateLimitRule with a unique name and namespace.
   - Call rate_limit_dependency(rule=..., key_fn=...) with the right key fn.
2. Attach it to your route with Depends(your_new_limiter()).
3. That's it — the enforcement, 429 handling, and fail-open behavior all
   come for free from the shared infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.redis_rate_limiter.rate_limit_enforcement import rate_limit_dependency
from app.core.redis_rate_limiter.rate_limit_keys import (
    ip_only_key,
    service_id_key,
)
from app.models.redis_rate_limit_models import RateLimitRule

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request


def auth_token_generate_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """
    Rate limit for the /token/generate endpoint.

    Keyed by IP only (the user hasn't authenticated yet — there's no user ID
    to key on). The limit comes from settings so it can be tuned per
    environment without code changes.

    Default: 30 requests per minute per IP.
    """
    rule = RateLimitRule(
        name="auth_token_generate",
        limit=f"{settings.rate_limit_token_generate_per_minute}/minute",
        key_namespace="auth_token_generate",
    )
    return rate_limit_dependency(rule=rule, key_fn=ip_only_key)


def auth_token_refresh_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """
    Rate limit for the /token/refresh endpoint.

    Same pattern as token generation: IP-keyed, configurable limit.
    Separated into its own rule so token refresh and token generation
    have independent budgets — hammering /token/generate doesn't block
    legitimate /token/refresh calls.

    Default: 60 requests per minute per IP.
    """
    rule = RateLimitRule(
        name="auth_token_refresh",
        limit=f"{settings.rate_limit_token_refresh_per_minute}/minute",
        key_namespace="auth_token_refresh",
    )
    return rate_limit_dependency(rule=rule, key_fn=ip_only_key)


def token_acquire_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    """
    Rate limit for the /tokens/acquire endpoint.

    Keyed by SERVICE x IP (see service_id_key in rate_limit_keys.py).
    This is the internal endpoint that upstream microservices call to
    reserve token capacity before sending LLM requests.

    Each microservice gets its own budget, identified by the X-Service-Id
    header. This prevents one misbehaving service from exhausting the
    shared rate-limit budget and starving all other services.

    Default: 500 requests per minute per service×IP pair. This is
    intentionally generous — legitimate microservices under normal load
    won't hit it, but a runaway retry loop will be caught.

    If X-Service-Id is missing, the caller is bucketed as "unknown" and
    still rate-limited, but in a separate pool from identified services.
    """
    rule = RateLimitRule(
        name="token_acquire",
        limit=f"{settings.rate_limit_token_acquire_per_minute}/minute",
        key_namespace="token_acquire",
    )
    return rate_limit_dependency(rule=rule, key_fn=service_id_key)
