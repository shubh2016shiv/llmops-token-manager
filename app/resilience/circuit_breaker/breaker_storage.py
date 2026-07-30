"""
Circuit breaker storage — decides WHERE each breaker's state is kept.

THE ONE QUESTION THIS FILE ANSWERS
----------------------------------
A breaker's state (CLOSED/OPEN/HALF_OPEN, plus its failure count) has to live
somewhere. There are two choices, and picking the right one per dependency is the
whole point of this file:

  • CircuitMemoryStorage — state lives in THIS process's memory. Fast, zero
    dependencies, but each server replica has its own independent view.
  • CircuitRedisStorage  — state lives in Redis, SHARED across every replica. One
    replica trips the breaker → all replicas see it open at once.

`build_breaker_storage(name)` at the bottom makes that choice per breaker. It also
owns the synchronous Redis client that the Redis-backed storage needs.

WHY A *SYNCHRONOUS* REDIS CLIENT (this surprises people)
--------------------------------------------------------
The rest of the app uses async Redis. But aiobreaker's `CircuitRedisStorage` reads
and writes state using a *blocking, synchronous* redis client. So this file keeps
its own dedicated sync client, separate from the app's async one. That is not a
mistake or duplication — it is what the storage layer requires.

Author: Engineering Team
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import aiobreaker
from aiobreaker.storage import CircuitMemoryStorage, CircuitRedisStorage
from loguru import logger
import redis

# TYPE_CHECKING import: `CircuitBreakerStorage` is only needed as a type
# annotation on build_breaker_storage's return. Importing it under TYPE_CHECKING
# means it's used by the type-checker but never imported at runtime — slightly
# cheaper, and it avoids a needless hard dependency on the library's internal path.
if TYPE_CHECKING:
    from aiobreaker.storage.base import CircuitBreakerStorage

from app.core.config import settings

# ---------------------------------------------------------------------------
# Synchronous Redis client — a lazily-built, thread-safe singleton.
# ---------------------------------------------------------------------------
# We want exactly ONE shared sync client (opening a new Redis connection pool per
# breaker call would be wasteful). These two module-level variables implement that
# singleton: the client itself (initially None = "not built yet") and a lock that
# guards its construction.
_synchronous_redis_client: redis.Redis | None = None
_synchronous_redis_client_lock = threading.Lock()


def build_synchronous_redis_client() -> redis.Redis:
    """
    Construct (once) a synchronous Redis client for aiobreaker's storage layer.

    Uses the "double-checked locking" pattern for a thread-safe lazy singleton:
      1. Check without the lock (the fast path — almost always already built).
      2. If not built, take the lock and check AGAIN before building.
    The second check matters: two threads could both pass step 1 at the same time;
    only one should build the client. Whichever thread gets the lock first builds
    it; the second thread sees it's now non-None and skips construction.
    """
    # `global` is required because we REASSIGN the module-level variable below
    # (rebinding a name, not just mutating an object). Without `global`, Python
    # would treat `_synchronous_redis_client` as a new local variable.
    global _synchronous_redis_client  # noqa: PLW0603

    # 1st check — no lock. If already built, return immediately (the common case).
    if _synchronous_redis_client is not None:
        return _synchronous_redis_client

    # Slow path: serialize construction so only one thread builds the client.
    with _synchronous_redis_client_lock:
        # 2nd check — now holding the lock. Another thread may have built it while
        # we waited for the lock; if so, don't build a second one.
        if _synchronous_redis_client is None:
            # A connection POOL (not a single connection) so concurrent breaker
            # reads/writes don't serialize on one socket. Timeouts/keepalive/health
            # checks keep it from hanging forever on a dead Redis.
            connection_pool = redis.ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                max_connections=settings.redis_max_connections,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
            )
            _synchronous_redis_client = redis.Redis(connection_pool=connection_pool)

    return _synchronous_redis_client


def close_synchronous_redis_client() -> None:
    """
    Best-effort shutdown for the synchronous Redis client and its pool.

    Called on app shutdown. "Best-effort" = we log but never raise if cleanup
    fails, because failing to close a client during shutdown shouldn't crash the
    shutdown itself.
    """
    global _synchronous_redis_client  # noqa: PLW0603
    # Swap the singleton out to None *under the lock*, then do the slow close()
    # OUTSIDE the lock. This keeps the lock held only for the quick swap, and
    # guarantees a concurrent build() sees None and rebuilds a fresh client rather
    # than handing back one we're in the middle of closing.
    with _synchronous_redis_client_lock:
        client = _synchronous_redis_client
        _synchronous_redis_client = None
    if client is None:
        return

    try:
        client.close()
    except Exception:
        logger.exception(
            "[CircuitBreaker] Failed closing synchronous Redis client cleanly"
        )

    try:
        client.connection_pool.disconnect()
    except Exception:
        logger.exception(
            "[CircuitBreaker] Failed disconnecting synchronous Redis connection pool"
        )


# ---------------------------------------------------------------------------
# Storage factory — the per-breaker "local vs shared" decision.
# ---------------------------------------------------------------------------


def build_breaker_storage(breaker_name: str) -> CircuitBreakerStorage:
    """
    Build the storage backend for a named circuit breaker.

    Mapping:
    - `postgres`: `CircuitMemoryStorage` (local, no Redis dependency)
    - `redis`   : `CircuitRedisStorage`  (distributed, fail-closed fallback)
    - `rabbitmq`: `CircuitRedisStorage`  (distributed, fail-closed fallback)

    WHY postgres is different: its state is kept in local memory so that DB
    protection does NOT depend on Redis being up. If the DB breaker stored its
    state in Redis, a Redis outage would also break the database breaker — coupling
    two unrelated failures. Local memory keeps them independent. (Trade-off: each
    replica has its own view of the DB breaker.)
    """
    # PostgreSQL breaker: local, in-memory, starts CLOSED (healthy).
    if breaker_name == "postgres":
        return CircuitMemoryStorage(aiobreaker.CircuitBreakerState.CLOSED)

    # Redis / RabbitMQ breakers: shared state in Redis so all replicas agree.
    redis_client = build_synchronous_redis_client()
    return CircuitRedisStorage(
        # Normal starting state when storage is readable.
        state=aiobreaker.CircuitBreakerState.CLOSED,
        redis_object=redis_client,
        # Each breaker gets its own Redis key namespace ("cb:redis", "cb:rabbitmq")
        # so they never read each other's state.
        namespace=f"cb:{breaker_name}",
        # THE FAIL-CLOSED CHOICE: if we cannot even READ the breaker's state from
        # Redis, assume OPEN (blocked). We'd rather wrongly block for a moment than
        # wrongly send traffic at a dependency we can't verify is healthy. This is
        # the deliberate opposite of backpressure's fail-OPEN philosophy.
        fallback_circuit_state=aiobreaker.CircuitBreakerState.OPEN,
    )
