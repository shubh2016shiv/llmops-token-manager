# redis_rate_limiter

Redis-backed HTTP rate limiting for FastAPI endpoints, using the `limits`
library's moving-window strategy.

## The 30-second mental model

You run a coffee shop. Each customer gets 3 lattes per hour — not to be mean,
but so one person doesn't empty the machine. This package is the barista who
counts: "You've had 3 already? Come back in 20 minutes."

Instead of lattes we count HTTP requests. Instead of a notepad we use Redis
(a super-fast in-memory counter). Instead of a barista we have an async
function that FastAPI calls before every route handler.

## How a request flows through the system

```
CLIENT → FastAPI → Depends(rate_limiter()) → Redis check
                       │
              ┌────────┴────────┐
              ▼                 ▼
         UNDER LIMIT        OVER LIMIT
         route runs         429 response
```

The rate limiter runs BEFORE your route handler. If the caller is under their
limit, the handler executes normally. If over, a 429 is returned and the
handler never runs. No try/except needed in your route code.

## What this package owns

| Concern | Owned? | Location |
|---|---|---|
| Request rate limiting logic | ✅ Here | This package |
| Moving-window limiter lifecycle | ✅ Here | `moving_window_limiter.py` |
| Per-request enforcement + 429 handler | ✅ Here | `rate_limit_enforcement.py` |
| Caller identification (IP, service ID) | ✅ Here | `rate_limit_keys.py` |
| Per-endpoint rule definitions | ✅ Here | `endpoint_limiters.py` |
| Pydantic models (rule, 429 payload) | ❌ | `app/models/redis_rate_limit_models.py` |
| `RateLimitExceededError` exception | ❌ | `app/core/exceptions.py` |
| General Redis client (redis-py) | ❌ | `app/core/redis.py` |

This package uses a **separate coredis connection pool** from the main app
Redis client. They share the same settings (host, port, password) but need
different drivers because `limits` only speaks coredis. That's why the
codebase has two Redis clients — they're for different purposes.

## Files (read in this order)

| File | What you'll learn |
|---|---|
| `moving_window_limiter.py` | What a moving window is, how the limiter is built, why lifecycle management beats `@lru_cache` |
| `rate_limit_enforcement.py` | The dependency factory pattern, transient vs permanent errors, fail-open under CAP |
| `rate_limit_keys.py` | How we identify callers, the X-Forwarded-For trust model, choosing the right key granularity |
| `endpoint_limiters.py` | How rules and key functions are paired for specific endpoints |
| `__init__.py` | The public API — import from here |

## Quick start

```python
from app.core.redis_rate_limiter import (
    rate_limiter_manager,
    register_rate_limit_exception_handler,
    auth_token_refresh_rate_limiter,
)

# In app lifespan startup:
register_rate_limit_exception_handler(app)
rate_limiter_manager.initialize()

# On a route:
@router.post("/token/refresh",
             dependencies=[Depends(auth_token_refresh_rate_limiter())])
async def refresh(...): ...
```

## Adding a new protected endpoint

Add one factory to `endpoint_limiters.py`:

```python
def my_new_rate_limiter() -> Callable[[Request], Awaitable[None]]:
    rule = RateLimitRule(
        name="my_new_endpoint",
        limit=f"{settings.my_new_limit_per_minute}/minute",
        key_namespace="my_new_endpoint",
    )
    return rate_limit_dependency(rule=rule, key_fn=ip_only_key)
```

Then attach it to your route. The enforcement, 429 handling, and fail-open
behavior all come for free from the shared infrastructure.

## Failure modes

- **Redis down transiently**: fail-open (AP under CAP). Requests are allowed
  through unchecked. Logged at WARNING — self-heals when Redis returns.
- **Redis driver missing/incompatible**: caught at startup — app refuses to
  boot rather than silently degrading.
- **Unexpected error during enforcement**: fail-open, logged at ERROR with
  full traceback. Still allows requests (availability first) but pages on-call.
