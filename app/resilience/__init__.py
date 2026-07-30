"""
Public API for the Token Manager resilience layer.

This package implements the 5-layer resilience stack for the Token Manager
hot path:

    Layer 0 - Per-service rate limiting (extended in app/core/rate_limiter.py)
    Layer 1 - Back pressure (fail-fast 503 on saturation)       <- backpressure/
    Layer 2 - Circuit breakers (DB / Redis / RabbitMQ)          <- circuit_breaker/
    Layer 3 - Redis fast path (Lua atomic token counter)        <- redis_token_counter/
    Layer 4 - Queue absorption + DLQ                            <- token_queue/
              Celery maintenance + reconciliation               <- token_maintenance/

Design principles:
- This package root exposes shared resilience utilities, not every subsystem API.
- Redis token counter behavior is intentionally imported from
  `app.resilience.redis_token_counter`, not re-exported here.
- Backpressure is isolated as a dedicated package so Layer 1 admission control
  can evolve or be removed without touching unrelated resilience layers.
- All classes/functions follow existing project patterns (Depends(), loguru, settings).
"""

from app.resilience.backpressure import backpressure_dependency
from app.resilience.circuit_breaker import (
    CircuitBreakerState,
    get_db_circuit_breaker,
    get_redis_circuit_breaker,
    get_rmq_circuit_breaker,
)
from app.resilience.token_queue import TokenAllocationPublisher

__all__ = [
    "get_db_circuit_breaker",
    "get_redis_circuit_breaker",
    "get_rmq_circuit_breaker",
    "CircuitBreakerState",
    "backpressure_dependency",
    "TokenAllocationPublisher",
]
