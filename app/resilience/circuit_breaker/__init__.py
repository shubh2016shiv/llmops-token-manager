"""
Public API for the circuit breaker sub-package.

Sub-package structure:
- `breaker_state.py`: `CircuitBreakerState` enum.
- `breaker_listener.py`: listener that emits structured breaker logs.
- `breaker_storage.py`: synchronous Redis client and storage factory.
- `breaker_registry.py`: thread-safe breaker registry and factory functions.

All public symbols are re-exported here so existing callers can keep using
`from app.resilience.circuit_breaker import get_db_circuit_breaker`.
"""

import aiobreaker

from app.resilience.circuit_breaker.breaker_listener import CircuitBreakerListener
from app.resilience.circuit_breaker.breaker_registry import (
    _circuit_breaker_registry,
    create_circuit_breaker,
    get_circuit_breaker_states,
    get_db_circuit_breaker,
    get_redis_circuit_breaker,
    get_rmq_circuit_breaker,
)
from app.resilience.circuit_breaker.breaker_state import CircuitBreakerState

# Expose breaker_storage as a submodule attribute so tests can set/reset the
# canonical `_synchronous_redis_client` variable without aliasing pitfalls.
import app.resilience.circuit_breaker.breaker_storage as breaker_storage
from app.resilience.circuit_breaker.breaker_storage import (
    build_breaker_storage,
    build_synchronous_redis_client,
    close_synchronous_redis_client,
)


def close_circuit_breaker_redis_client() -> None:
    """Backward-compatible alias for synchronous breaker Redis cleanup."""
    close_synchronous_redis_client()


__all__ = [
    "aiobreaker",
    "CircuitBreakerState",
    "CircuitBreakerListener",
    "get_db_circuit_breaker",
    "get_redis_circuit_breaker",
    "get_rmq_circuit_breaker",
    "get_circuit_breaker_states",
    "close_circuit_breaker_redis_client",
    "close_synchronous_redis_client",
    "breaker_storage",
    "build_synchronous_redis_client",
    "build_breaker_storage",
    "create_circuit_breaker",
    "_circuit_breaker_registry",
]
