"""
The Redis-backed moving-window rate limiter and its lifecycle.

----
What problem does this module solve?
------------------------------------
Imagine you run a coffee shop. You want to limit each customer to 3 lattes
per hour — not because you're mean, but because one person shouldn't empty
the machine before anyone else gets served.

This module is the machine that counts the lattes. Every time someone orders,
it checks: "Have you had 3 already in the last hour? No? Here you go. Yes?
Come back later."

But instead of lattes, we count HTTP requests. And instead of a notepad behind
the counter, we use Redis — a super-fast in-memory database that can count
things atomically (meaning two baristas can't accidentally serve the same
customer a 4th latte because they both checked the notepad at the same time).

----
The moving window: what does "3 per minute" actually mean?
----------------------------------------------------------
A naive approach would be "reset the counter every 60 seconds." That creates
a problem: if you send 3 requests at second 59 and 3 more at second 61
(just 2 seconds apart), both bursts pass — you effectively got 6 requests
in 2 seconds.

The moving window fixes this. For a "3 per minute" rule, it looks back over
the LAST 60 seconds from right now, not a fixed clock minute. Every request
slides the window forward. This means you can never squeeze more than your
allowance into any 60-second slice, no matter how you time the bursts.

----
Relationship to app/core/redis.py
----------------------------------
You might wonder: "Doesn't the app already have a Redis connection in
app/core/redis.py? Why a second one here?"

The app's main Redis client uses `redis-py` (the `redis.asyncio` driver).
But the `limits` library — the battle-tested rate-limiting engine we depend
on — only speaks to Redis through a DIFFERENT driver called `coredis`. These
are two separate Python libraries that both talk to the same Redis server,
but they can't share a connection pool.

Think of it like two delivery drivers working for the same restaurant: they
both deliver food to the same address, but each drives their own car. Same
destination, separate vehicles.

Both read the same `settings.redis_*` configuration (host, port, password),
so they point at the same Redis instance — just through different driver
libraries with independent connection pools.

----
Lifecycle: why a manager class instead of a global singleton?
--------------------------------------------------------------
In the original code, the limiter was created with `@lru_cache` — a module-level
singleton that was built once and reused forever. This broke in subtle ways:

Python's async code runs inside an "event loop" — think of it as the conductor
of an orchestra, coordinating which instrument plays when. A coredis connection
pool binds itself to whichever event loop first touches it. If a second event
loop later tries to use the same pool (which happens during testing, or when
hot-reloading in development), the pool is still attached to the first — now
dead — loop, and every Redis call fails with a cryptic error deep inside coredis.

The `RateLimiterManager` class fixes this by owning the limiter's entire
lifecycle: it's created once during app startup (inside the correct event loop)
and torn down during shutdown. This mirrors how `app.core.redis.RedisManager`
works — same pattern, same reliability.
"""

from __future__ import annotations

import os
from urllib.parse import quote

from limits.aio.storage import MemoryStorage
from limits.aio.strategies import MovingWindowRateLimiter
from limits.storage import storage_from_string
from loguru import logger

from app.core.config import settings


def _redis_dsn() -> str:
    """
    Build the connection string that tells the `limits` library where Redis lives.

    A DSN (Data Source Name) is just a URL for databases. This one looks like:

        async+redis://:mypassword@localhost:6379/0
        ──────┬────── ──┬── ───┬─── ─┬─ ┬
              │         │      │     │  └─ database number (Redis has 0-15)
              │         │      │     └──── port (6379 is Redis's default)
              │         │      └────────── host (where Redis is running)
              │         └───────────────── password (empty if no auth)
              └─────────────────────────── protocol prefix

    The "async+" prefix is the critical part: it tells `limits` to use the
    asynchronous (non-blocking) coredis driver. Without it, `limits` uses a
    synchronous driver that FREEZES the entire server on every rate check —
    fine for scripts, catastrophic for a web server handling many users.

    The password is URL-encoded so that special characters like @, :, or /
    don't get confused with the DSN's own delimiters. Without encoding, a
    password like "p@ss:word" would break the URL: the @ would be read as
    "user info separator" and the : as "port separator", pointing the limiter
    at completely the wrong address.
    """
    if settings.redis_password:
        # quote(..., safe='') means "encode EVERYTHING that isn't a letter or
        # digit" — no characters slip through unencoded.
        auth = f":{quote(settings.redis_password, safe='')}@"
    else:
        # No password configured — don't include the ":password@" section at all.
        auth = ""
    return (
        f"async+redis://{auth}"
        f"{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    )


def _build_rate_limit_storage():
    """
    Build the storage backend — the "notepad" where the limiter writes counts.

    There are two backends to choose from:

    ┌──────────────────────┬──────────────────────────────────────┐
    │ MemoryStorage        │ A Python dictionary inside the app's │
    │                      │ own process. Fast, zero setup, but   │
    │                      │ vanishes on restart and can't be     │
    │                      │ shared across multiple servers.      │
    │                      │ Used for: unit tests, local dev.     │
    ├──────────────────────┼──────────────────────────────────────┤
    │ Redis (via coredis)  │ A real Redis server. Survives        │
    │                      │ restarts, shared across all server   │
    │                      │ instances, persisted to disk.        │
    │                      │ Used for: staging, production.       │
    └──────────────────────┴──────────────────────────────────────┘

    Set RATE_LIMIT_STORAGE=memory in your environment to use MemoryStorage.
    Otherwise, the real Redis backend is built from _redis_dsn().
    """
    if os.getenv("RATE_LIMIT_STORAGE", "").lower() == "memory":
        return MemoryStorage()
    return storage_from_string(_redis_dsn())


class RateLimiterManager:
    """
    The caretaker of the rate limiter — creates it, holds it, releases it.

    Think of this as the "on/off switch" for rate limiting. The app's startup
    code calls `initialize()` to build the limiter, and shutdown code calls
    `close()` to release it. Between those two calls, any part of the app can
    access the limiter through the `.limiter` property.

    This is NOT a singleton or a global cache. It's a managed resource with an
    explicit lifecycle — the same pattern used for database connections, Redis
    clients, and any other resource that needs setup and teardown.

    Why not just `@lru_cache` on a function?
    ........................................
    The original code used `@lru_cache(maxsize=1)` on `get_rate_limiter()`.
    This cached the limiter forever — but coredis connection pools bind to a
    SPECIFIC async event loop. When tests create a fresh event loop, or when
    dev hot-reload restarts the app, the cached pool is still attached to the
    old (dead) loop, and every Redis call crashes silently.

    `RateLimiterManager` fixes this: `initialize()` builds a fresh limiter
    inside the current event loop, and `close()` discards it so a new one can
    be built later.

    Failure semantics
    -----------------
    `initialize()` constructs the storage object but does NOT open a network
    connection yet — that happens lazily on the first `limiter.hit()` call.
    However, it DOES validate that coredis is installed and the DSN is
    well-formed. If coredis is missing or the DSN is unparseable,
    `initialize()` raises immediately at startup — the app refuses to boot
    rather than silently degrading later.

    Transient Redis outages (network blips, brief downtime) are NOT caught
    here — they're handled per-request in `rate_limit_enforcement.py`, where
    the fail-open logic lives.
    """

    def __init__(self) -> None:
        # Start empty — nothing exists until `initialize()` is called.
        self._storage = None
        self._limiter: MovingWindowRateLimiter | None = None

    def initialize(self) -> None:
        """
        Build the storage and the moving-window limiter on top of it.

        Safe to call multiple times — second call is a no-op. Call this once
        during the FastAPI lifespan startup (in app/app.py).
        """
        if self._limiter is not None:
            logger.warning("Rate limiter already initialized; skipping re-init")
            return

        self._storage = _build_rate_limit_storage()
        backend_name = type(self._storage).__name__
        if isinstance(self._storage, MemoryStorage):
            logger.info(
                "Rate limiter initialized with in-memory storage "
                "(non-distributed — intended for tests/single-process only)"
            )
        else:
            logger.info(
                f"Rate limiter initialized with {backend_name} at "
                f"{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
            )
        # Wrap the storage in a MovingWindowRateLimiter — this is the object
        # that actually implements the sliding-window counting algorithm.
        self._limiter = MovingWindowRateLimiter(self._storage)

    async def close(self) -> None:
        """
        Release the limiter and its underlying storage/connection pool.

        After this, accessing `.limiter` raises RuntimeError — a loud failure
        that says "you forgot to initialize." This is intentional: a silent
        broken limiter (like the old @lru_cache approach) hides bugs.
        """
        if self._limiter is None:
            return
        logger.info("Closing rate limiter storage")
        self._storage = None
        self._limiter = None

    @property
    def limiter(self) -> MovingWindowRateLimiter:
        """
        The live, ready-to-use rate limiter.

        Raises RuntimeError if accessed before `initialize()` was called.
        This is the "fail loud" pattern: if the wiring is wrong (someone
        forgot to call initialize in the lifespan), every request gets a 500
        immediately. That's better than silently skipping rate limits for
        weeks because of a one-line omission in app startup.
        """
        if self._limiter is None:
            raise RuntimeError(
                "Rate limiter not initialized. Call "
                "rate_limiter_manager.initialize() (done in the app lifespan) "
                "before handling requests."
            )
        return self._limiter


# The single, module-level instance that the rest of the app uses.
# It starts empty — the FastAPI lifespan in app/app.py calls initialize().
rate_limiter_manager = RateLimiterManager()
