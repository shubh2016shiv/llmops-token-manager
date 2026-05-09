"""
Redis token counter service registry - service construction and shared lifecycle.

Architecture:
-------------
    ┌──────────────────────────────┐     ┌──────────────────────────────┐
    │ API / worker callers         │────▶│ service_registry.py          │
    │ acquire shared service       │     │ create/get/close helpers     │
    └──────────────────────────────┘     └──────────────┬───────────────┘
                                                        │
                                                        ▼
                                         ┌──────────────────────────────┐
                                         │ RedisTokenCounterService     │
                                         │ one shared client per        │
                                         │ process                      │
                                         └──────────────────────────────┘

Dependencies:
    - app/core/config.py - Redis connection configuration
    - redis.asyncio - Redis client construction

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

import threading

import redis.asyncio as aioredis

from app.core.config import settings
from app.resilience.redis_token_counter.counter_service import (
    RedisTokenCounterService,
)

_shared_redis_token_counter_service: RedisTokenCounterService | None = None
_shared_redis_token_counter_service_lock = threading.Lock()


def create_redis_token_counter_service() -> RedisTokenCounterService:
    """Create a fresh Redis token counter service instance."""
    redis_client = aioredis.from_url(
        settings.redis_token_counter_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_token_counter_max_connections,
    )
    return RedisTokenCounterService(redis_client)


def get_shared_redis_token_counter_service() -> RedisTokenCounterService:
    """Return the process-local shared Redis token counter service."""
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
    global _shared_redis_token_counter_service
    service_to_close: RedisTokenCounterService | None
    with _shared_redis_token_counter_service_lock:
        service_to_close = _shared_redis_token_counter_service
        _shared_redis_token_counter_service = None

    if service_to_close is not None:
        await service_to_close.close()
