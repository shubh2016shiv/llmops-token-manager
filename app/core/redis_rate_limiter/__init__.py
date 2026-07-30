"""
Redis-backed HTTP rate limiting for FastAPI endpoints.

This package is the guardian at the gate: it decides whether each incoming
HTTP request is within its rate-limit budget, and rejects over-budget callers
with a clean HTTP 429 ("Too Many Requests") response.

----
Reading order (each file builds on the previous one)
-----------------------------------------------------
    1. moving_window_limiter.py
       The limiter itself — how it's built, what "moving window" means,
       why it uses a separate Redis driver from the rest of the app.

    2. rate_limit_enforcement.py
       Where FastAPI and the limiter actually meet. The dependency factory
       that checks limits per-request, and the 429 exception handler.

    3. rate_limit_keys.py
       How we identify callers: by IP, by IP+username, by service+IP.
       Includes the trust model for X-Forwarded-For.

    4. endpoint_limiters.py
       The concrete rules: "this endpoint gets 30/minute, keyed by IP."
       Glue code — pairs rules with key functions for specific routes.

----
How it's wired at startup (app/app.py lifespan)
------------------------------------------------
    from app.core.redis_rate_limiter import (
        rate_limiter_manager,
        register_rate_limit_exception_handler,
    )

    # 1. Install the 429 error translator (once, at boot).
    register_rate_limit_exception_handler(app)

    # 2. Build the rate limiter inside the serving event loop.
    rate_limiter_manager.initialize()

----
How a route uses it
--------------------
    from app.core.redis_rate_limiter import auth_token_refresh_rate_limiter

    @router.post("/token/refresh",
                 dependencies=[Depends(auth_token_refresh_rate_limiter())])
    async def refresh_token(...):
        ...

That's it. The dependency runs BEFORE the route handler. Under budget? Route
runs normally. Over budget? 429 is returned, route never executes.

----
What this package owns vs what lives elsewhere
-----------------------------------------------
    Owns:     Request rate limiting (the decision + the 429 response).
    Does NOT: Rule/response Pydantic models → app/models/redis_rate_limit_models.py
    Does NOT: RateLimitExceededError class → app/core/exceptions.py
    Does NOT: General Redis client (redis-py) → app/core/redis.py

This package holds its own coredis connection pool, separate from the main
app Redis client. They share the same settings (host/port/password) but use
different drivers because `limits` only speaks coredis.
"""

from app.core.redis_rate_limiter.endpoint_limiters import (
    auth_token_generate_rate_limiter,
    auth_token_refresh_rate_limiter,
    token_acquire_rate_limiter,
)
from app.core.redis_rate_limiter.moving_window_limiter import (
    RateLimiterManager,
    rate_limiter_manager,
)
from app.core.redis_rate_limiter.rate_limit_enforcement import (
    rate_limit_dependency,
    register_rate_limit_exception_handler,
)
from app.core.redis_rate_limiter.rate_limit_keys import (
    get_client_ip,
    ip_only_key,
    service_id_key,
)

__all__ = [
    "RateLimiterManager",
    "auth_token_generate_rate_limiter",
    "auth_token_refresh_rate_limiter",
    "get_client_ip",
    "ip_only_key",
    "rate_limit_dependency",
    "rate_limiter_manager",
    "register_rate_limit_exception_handler",
    "service_id_key",
    "token_acquire_rate_limiter",
]
