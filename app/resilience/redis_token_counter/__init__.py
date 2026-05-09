"""
Redis token counter package - public API for Redis fast-path token accounting.

Architecture:
-------------
    External callers import from this package root only.

    Public surface:
    - `RedisTokenCounterService` for Redis-backed counter operations
    - `TokenReservationResult` and `CounterReconciliationResult` for explicit outcomes
    - shared-service lifecycle helpers from `service_registry.py`

    Internal composition:
    - `counter_service.py` implements Redis operations and Lua execution
    - `service_registry.py` manages one shared service per process
    - `counter_results.py` defines public enums
    - `lua_script_definitions.py` stores atomic Lua source strings

Dependencies:
    - app/core/config.py - Redis connection settings
    - app/resilience/circuit_breaker - Redis circuit breaker integration

Author: Engineering Team
Last Updated: 2026-05-09
"""

from app.resilience.redis_token_counter.counter_results import (
    CounterReconciliationResult,
    TokenReservationResult,
)
from app.resilience.redis_token_counter.counter_service import (
    RedisTokenCounterService,
)
from app.resilience.redis_token_counter.service_registry import (
    close_shared_redis_token_counter_service,
    create_redis_token_counter_service,
    get_shared_redis_token_counter_service,
)

__all__ = [
    "CounterReconciliationResult",
    "TokenReservationResult",
    "RedisTokenCounterService",
    "create_redis_token_counter_service",
    "get_shared_redis_token_counter_service",
    "close_shared_redis_token_counter_service",
]
