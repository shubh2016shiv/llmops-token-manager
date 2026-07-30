"""
Service registry — one shared RedisTokenCounterService per process.

Opening a Redis connection pool is expensive, and the whole app should share one
counter service (and its pool) rather than each caller making its own. This module
provides that shared instance via the standard lazy, thread-safe singleton pattern,
plus a `create_...` for tests/special cases and a `close_...` for shutdown.

Author: Engineering Team
Last Updated: 2026-07-24
"""

from __future__ import annotations

import threading

import redis.asyncio as aioredis

from app.core.config import settings
from app.resilience.redis_token_counter.counter_service import (
    RedisTokenCounterService,
)

# The shared instance (None until first use) and a lock guarding its creation.
_shared_redis_token_counter_service: RedisTokenCounterService | None = None
_shared_redis_token_counter_service_lock = threading.Lock()


def create_redis_token_counter_service() -> RedisTokenCounterService:
    """Create a fresh Redis token counter service instance (its own client/pool)."""
    # `decode_responses=True` -> Redis returns str, so the service reads plain
    # numbers/strings back from Lua rather than bytes.
    redis_client = aioredis.from_url(
        settings.redis_token_counter_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_token_counter_max_connections,
    )
    return RedisTokenCounterService(redis_client)


def get_shared_redis_token_counter_service() -> RedisTokenCounterService:
    """Return the process-local shared Redis token counter service."""
    # Double-checked locking: cheap check first, then lock + re-check only if we
    # actually need to build it, so concurrent callers can't create two instances.
    global _shared_redis_token_counter_service
    if _shared_redis_token_counter_service is None:
        with _shared_redis_token_counter_service_lock:
            if _shared_redis_token_counter_service is None:
                _shared_redis_token_counter_service = (
                    create_redis_token_counter_service()
                )
    return _shared_redis_token_counter_service


async def close_shared_redis_token_counter_service() -> None:
    """Close and clear the process-local shared Redis token counter service."""
    # Swap the singleton out under the lock (quick), then close it OUTSIDE the lock
    # (the await could be slow) so we don't hold the lock during I/O.
    global _shared_redis_token_counter_service
    service_to_close: RedisTokenCounterService | None
    with _shared_redis_token_counter_service_lock:
        service_to_close = _shared_redis_token_counter_service
        _shared_redis_token_counter_service = None

    if service_to_close is not None:
        await service_to_close.close()
