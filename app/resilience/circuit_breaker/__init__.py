"""
Public API for the circuit breaker sub-package.

See README.md in this folder for the full mental model (the three states, why this
folder is a "factory around aiobreaker", where the breakers are consumed, and the
fail-closed design). This file is just the front door: it re-exports the symbols
other packages import so they never depend on the internal file layout.

Sub-package structure:
- `breaker_state.py`   : `CircuitBreakerState` enum (our normalized state names).
- `breaker_listener.py`: listener that emits structured breaker logs.
- `breaker_storage.py` : synchronous Redis client + per-breaker storage factory.
- `breaker_registry.py`: thread-safe breaker registry, factories, state readout.

The everyday public surface is small — most callers only need:
    from app.resilience.circuit_breaker import get_db_circuit_breaker
    from app.resilience.circuit_breaker import CircuitBreakerState
The remaining exports below are lower-level lifecycle/factory helpers and a couple
of internals surfaced deliberately for tests (see the note on `breaker_storage`).
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
# canonical `_synchronous_redis_client` variable (and monkeypatch the storage
# factory) on the ONE real module object, without import-aliasing pitfalls.
import app.resilience.circuit_breaker.breaker_storage as breaker_storage
from app.resilience.circuit_breaker.breaker_storage import (
    build_breaker_storage,
    build_synchronous_redis_client,
    close_synchronous_redis_client,
)


def close_circuit_breaker_redis_client() -> None:
    """
    Backward-compatible alias for synchronous breaker Redis cleanup.

    Older shutdown code imported this name; it simply forwards to the current
    `close_synchronous_redis_client`. Kept so existing callers don't break.
    """
    close_synchronous_redis_client()


__all__ = [
    # Re-exported library handle (so callers can catch aiobreaker.CircuitBreakerError
    # without importing aiobreaker themselves).
    "aiobreaker",
    # Everyday public API:
    "CircuitBreakerState",
    "CircuitBreakerListener",
    "get_db_circuit_breaker",
    "get_redis_circuit_breaker",
    "get_rmq_circuit_breaker",
    "get_circuit_breaker_states",
    # Lifecycle / cleanup:
    "close_circuit_breaker_redis_client",
    "close_synchronous_redis_client",
    # Lower-level factory internals (used by wiring code and tests):
    "breaker_storage",
    "build_synchronous_redis_client",
    "build_breaker_storage",
    "create_circuit_breaker",
    "_circuit_breaker_registry",
]
