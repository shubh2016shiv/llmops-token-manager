"""
Storage helpers for circuit breaker state backends.

This module provides a shared synchronous Redis client and a factory that
returns the appropriate aiobreaker storage implementation per breaker type.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import aiobreaker
from aiobreaker.storage import CircuitMemoryStorage, CircuitRedisStorage
from loguru import logger
import redis

if TYPE_CHECKING:
    from aiobreaker.storage.base import CircuitBreakerStorage

from app.core.config import settings

# ---------------------------------------------------------------------------
# Synchronous Redis client - singleton with thread-safe double-checked locking
# ---------------------------------------------------------------------------

_synchronous_redis_client: redis.Redis | None = None
_synchronous_redis_client_lock = threading.Lock()


def build_synchronous_redis_client() -> redis.Redis:
    """
    Construct (once) a synchronous Redis client for aiobreaker's storage layer.

    Reuses existing Redis host/port/credentials from application settings.
    Each breaker is isolated via `CircuitRedisStorage` namespaces.
    """
    global _synchronous_redis_client  # noqa: PLW0603

    if _synchronous_redis_client is not None:
        return _synchronous_redis_client

    with _synchronous_redis_client_lock:
        if _synchronous_redis_client is None:
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
    """Best-effort shutdown for the synchronous Redis client and its pool."""
    global _synchronous_redis_client  # noqa: PLW0603
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
# Storage factory
# ---------------------------------------------------------------------------


def build_breaker_storage(breaker_name: str) -> CircuitBreakerStorage:
    """
    Build the storage backend for a named circuit breaker.

    Mapping:
    - `postgres`: `CircuitMemoryStorage` (local, no Redis dependency)
    - `redis`: `CircuitRedisStorage` (distributed, fail-closed fallback)
    - `rabbitmq`: `CircuitRedisStorage` (distributed, fail-closed fallback)

    PostgreSQL uses in-memory storage so DB breaker state is decoupled from
    Redis availability. Redis and RabbitMQ breakers share state across replicas
    via Redis, with a fail-closed fallback.
    """
    if breaker_name == "postgres":
        return CircuitMemoryStorage(aiobreaker.CircuitBreakerState.CLOSED)

    redis_client = build_synchronous_redis_client()
    return CircuitRedisStorage(
        state=aiobreaker.CircuitBreakerState.CLOSED,
        redis_object=redis_client,
        namespace=f"cb:{breaker_name}",
        fallback_circuit_state=aiobreaker.CircuitBreakerState.OPEN,
    )
