"""
Enforcing a rate-limit rule on a request, and rejecting over-limit with a 429.

----
The big picture: how a rate-limited request flows through the system
---------------------------------------------------------------------
Every rate-limited request follows the same 6-step path:

    1. CLIENT sends GET /api/something
              │
    2. FASTAPI sees `dependencies=[Depends(some_rate_limiter())]` on the route.
       It calls some_rate_limiter() ONCE (at startup) and caches the returned
       dependency function. Then, on EVERY request to this route, FastAPI calls
       that dependency function BEFORE the actual route handler.
              │
    3. DEPENDENCY (rate_limit_dependency) runs:
       - Figures out WHO is calling (IP, username, service ID — via key_fn).
       - Asks Redis: "has this caller hit their limit?"
              │
       ┌──────┴──────┐
       ▼              ▼
    4a. UNDER LIMIT           4b. OVER LIMIT
       Return silently.       Build a 429 payload and
       FastAPI proceeds        raise RateLimitExceededError.
       to the route handler.        │
                                    ▼
    5. EXCEPTION HANDLER catches RateLimitExceededError
       and turns it into an HTTP 429 JSON response with a Retry-After header.
              │
    6. CLIENT receives 429, reads Retry-After, waits, retries.

The key insight: the route handler never runs for over-limit requests. The
dependency acts as a gatekeeper — it either waves the request through or
slams the door before the handler ever sees it.

----
Fail-open: why we let requests through when Redis is down
----------------------------------------------------------
When Redis is temporarily unreachable (network blip, Redis restarting), the
rate limiter can't check the counter. You have two choices:

    FAIL-CLOSED:  Block EVERY request (return 429).
                  Pro: strict rate enforcement, even during outages.
                  Con: your API is completely down if Redis blips.

    FAIL-OPEN:    Allow ALL requests (let them through unchecked).
                  Pro: your API stays up; Redis outage doesn't cascade.
                  Con: rate limits temporarily stop working.

We choose FAIL-OPEN for rate limiting. A payment system would choose the
opposite (you'd rather block payments than double-charge), but for rate
limiting, availability is more important than strict enforcement. Under the
CAP theorem, this is the AP choice (Availability over Consistency during a
Partition).

----
Transient vs permanent errors: not all failures are equal
----------------------------------------------------------
The original code caught ALL exceptions in one `except Exception` block and
treated everything as fail-open. This masked a critical bug: when the wrong
version of the coredis driver was installed, EVERY rate limit check threw
RuntimeError, which was silently caught and converted to "allow the request."
The rate limiter appeared to work (no 500s!) but actually did nothing.

The fix distinguishes two classes of errors:

    TRANSIENT (caught, fail-open, logged at WARNING):
      - Redis connection refused (it'll be back)
      - Redis timeout (network congestion)
      - Redis mid-failover (temporary)
      These are EXPECTED in production. Failing open is the right call.

    PERMANENT (caught, fail-open, logged at ERROR with full traceback):
      - Unexpected exception types (bugs, driver incompatibilities)
      - These should page the on-call engineer. We still fail-open to
        avoid a hard 500, but we scream loudly about it.
"""

from __future__ import annotations

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


# Retry-After must be at least 1 second (HTTP spec requires a positive integer).
MIN_RETRY_AFTER_SECONDS = 1


def _resolve_transient_storage_errors() -> tuple[type[BaseException], ...]:
    """
    Return the exception types that mean "Redis is temporarily unavailable".

    coredis (the async Redis driver used by `limits`) has its own exception
    hierarchy that does NOT inherit from Python's built-in ConnectionError or
    TimeoutError. If we only caught the built-ins we would miss every real Redis
    outage, so coredis' connection/timeout types are appended when available.

    The import is guarded so the module still loads when coredis isn't installed
    (e.g. test environments using in-memory storage), in which case only the
    built-in types apply.
    """
    builtin_transient_errors: tuple[type[BaseException], ...] = (
        ConnectionError,  # built-in: socket can't connect
        TimeoutError,  # built-in / asyncio: operation timed out
        OSError,  # built-in: low-level OS error
    )
    try:
        from coredis.exceptions import ConnectionError as CoredisConnectionError
        from coredis.exceptions import TimeoutError as CoredisTimeoutError
    except ImportError:
        return builtin_transient_errors
    return (
        *builtin_transient_errors,
        CoredisConnectionError,  # coredis: BusyLoadingError, ProtocolError, etc.
        CoredisTimeoutError,  # coredis: Redis didn't respond in time
    )


# Assigned exactly once: the resolution logic lives in the function above so the
# constant has a single, unambiguous definition.
_TRANSIENT_STORAGE_ERRORS: tuple[type[BaseException], ...] = (
    _resolve_transient_storage_errors()
)


async def _rate_limit_exceeded_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Render a RateLimitExceededError as an HTTP 429 JSON response.

    Registered by register_rate_limit_exception_handler() below. Starlette types
    every exception handler as accepting the base Exception, so this narrows to
    our own type; anything else is re-raised untouched because it is not ours to
    render. `request` is unused but required by the handler signature.

    Returns a 429 whose body is the payload the raiser built, plus the standard
    Retry-After header telling the client how long to wait.
    """
    if not isinstance(exc, RateLimitExceededError):
        raise exc
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=exc.payload,
        headers={"Retry-After": str(exc.retry_after)},
    )


def register_rate_limit_exception_handler(app: FastAPI) -> None:
    """
    Wire up the 429 error handler with FastAPI.

    After calling this ONCE during app startup, any code anywhere in the app
    can `raise RateLimitExceededError(payload=..., retry_after=...)` and
    FastAPI will automatically convert it into a proper HTTP 429 JSON response
    with a `Retry-After` header. No try/except needed in individual routes.

    This is a standard FastAPI pattern called an "exception handler" — you
    register a function that says "whenever you see this exception type, call
    me to build the HTTP response." FastAPI walks its handler registry on
    every uncaught exception and dispatches to the matching handler.

    The handler is a module-level function registered explicitly (rather than a
    nested function using the @app.exception_handler decorator) so it can be
    unit-tested on its own and is visibly referenced at its registration site.
    """
    app.add_exception_handler(RateLimitExceededError, _rate_limit_exceeded_handler)


def _retry_after_seconds(reset_at: float) -> int:
    """
    Convert the rate-limit window's reset time into "seconds from now".

    `reset_at` is the reset_time field of the `limits` WindowStats NamedTuple,
    which the library declares as a float Unix timestamp (seconds since the
    epoch). Earlier revisions of this function also handled None and
    datetime-like inputs; the library ships type information (py.typed) and
    guarantees a float, so those branches were unreachable and were removed
    rather than left as decoration.

    Returns the whole seconds until reset, floored at MIN_RETRY_AFTER_SECONDS
    so an already-elapsed window still yields a valid positive Retry-After.
    """
    seconds_until_reset = int(reset_at - time.time())
    return max(MIN_RETRY_AFTER_SECONDS, seconds_until_reset)


def rate_limit_dependency(
    *,
    rule: RateLimitRule,
    key_fn: Callable[[Request], Awaitable[str]],
) -> Callable[[Request], Awaitable[None]]:
    """
    Build a FastAPI dependency that enforces ONE specific rate limit rule.

    This function is called a "dependency factory" because instead of being
    the dependency itself, it BUILDS and RETURNS a dependency function. The
    built function is configured with a specific rule and key function, then
    handed to FastAPI via `Depends(...)`.

    Usage in a route definition:
        @router.get("/api/endpoint",
                    dependencies=[Depends(some_rate_limiter())])
        async def endpoint(): ...

    The flow in detail:

    1. AT STARTUP (route registration time):
       `some_rate_limiter()` calls this function with a rule and key_fn.
       This function parses the rule string ("5/minute" -> structured object),
       defines an inner async function `_dependency`, and RETURNS it.
       FastAPI caches this returned function.

    2. ON EVERY REQUEST:
       FastAPI calls the cached `_dependency(request)` BEFORE the route handler.
       `_dependency` asks Redis: "is this caller over their limit?"
       - Under limit -> returns None silently -> route handler runs.
       - Over limit -> raises RateLimitExceededError -> 429 response.
       - Redis unreachable -> fail-open (returns None, logs warning).

    Keyword-only arguments (the * in the function signature) force callers
    to write `rate_limit_dependency(rule=..., key_fn=...)` instead of
    `rate_limit_dependency(my_rule, my_fn)`. This prevents accidentally
    swapping the two arguments — a bug that would pass silently and produce
    baffling behavior at runtime.
    """
    # Convert "5/minute" into a structured limit object ONCE, at setup time.
    # This object knows: max_hits=5, time_window=60_seconds.
    limit_item = parse(rule.limit)

    # --- The inner function: this IS the dependency ---
    # It "closes over" (captures) rule, key_fn, and limit_item from the outer
    # scope. This is a closure — a function that remembers the variables from
    # where it was defined, even after the outer function returns.
    async def _dependency(request: Request) -> None:
        # Get the lifecycle-managed limiter. If the app's startup code forgot
        # to call rate_limiter_manager.initialize(), this raises RuntimeError
        # immediately — a loud, unmissable failure at request time, not a
        # silent degradation for weeks.
        limiter = rate_limiter_manager.limiter

        # Ask the key function: "who is making this request?"
        # The key function reads the request (IP, headers, body) and returns
        # a string like "192.168.1.5" or "192.168.1.5:alice" or "ms-gateway:10.0.0.1".
        # This string is the "bucket" — all requests with the same key share
        # the same rate-limit budget.
        key = await key_fn(request)

        try:
            # --- THE MOMENT OF TRUTH ---
            # `limiter.hit(...)` does three things atomically in Redis:
            #   1. Increments the counter for `key` in the current window.
            #   2. Removes counters from expired windows.
            #   3. Returns True (still under limit) or False (over limit).
            # This is a single O(1) operation — it's very fast.
            allowed = await limiter.hit(limit_item, rule.key_namespace, key)
            if allowed:
                # Still has requests left in this window. Let them through.
                return

            # --- OVER THE LIMIT ---
            # Ask the limiter for the current window stats so we can tell the
            # client how long to wait. This does NOT increment the counter —
            # it's a read-only peek at the current state.
            reset_at, remaining = await limiter.get_window_stats(
                limit_item, rule.key_namespace, key
            )
            retry_after = _retry_after_seconds(reset_at)

            # Build a structured 429 payload using our Pydantic models.
            # This keeps the 429 response shape consistent across ALL
            # rate-limited endpoints — clients can rely on a contract.
            response = RateLimitedResponse(
                details=RateLimitedErrorDetail(
                    rule=rule.name,
                    retry_after_seconds=retry_after,
                    remaining=int(remaining),
                ),
            )
            # Raising this exception triggers the handler we registered
            # earlier (register_rate_limit_exception_handler), which turns
            # it into a proper HTTP 429 JSON response.
            raise RateLimitExceededError(
                payload=response.to_payload(),
                retry_after=retry_after,
            )

        except RateLimitExceededError:
            # Re-raise our own exception so it reaches the 429 handler.
            # We catch it here only to stop it from falling into the
            # transient/unexpected handlers below.
            raise
        except _TRANSIENT_STORAGE_ERRORS as transient_error:
            # Redis briefly unreachable (network blip, restarting, mid-failover).
            # This is EXPECTED in production — fail open (AP under CAP).
            # Logged at WARNING: on-call should know it happened, but it's
            # not an emergency — it self-heals when Redis returns.
            logger.warning(
                f"Rate limiter storage transiently unavailable for "
                f"rule={rule.name}; failing open: {transient_error}"
            )
            return
        except Exception as unexpected_error:
            # Something went wrong that we did NOT expect: a bug, a driver
            # incompatibility, a malformed response from Redis. We still
            # fail-open to avoid a hard 500, but we log at ERROR with a full
            # traceback — this SHOULD page the on-call engineer because it
            # means rate limiting has been silently disabled until someone
            # investigates.
            logger.error(
                f"Unexpected rate limiter failure for rule={rule.name}; "
                f"failing open: {unexpected_error}",
                exc_info=True,
            )
            return

    # Return the inner function to FastAPI. It will be called on every request
    # to the route that uses this dependency.
    return _dependency
