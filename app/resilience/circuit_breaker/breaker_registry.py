"""
Thread-safe breaker registry and factory functions.

This module owns singleton circuit breaker instances and exposes helpers for
creating dependency-specific breakers and inspecting their current states.
"""

from __future__ import annotations

from datetime import timedelta
import threading

import aiobreaker
from aiobreaker.storage import CircuitMemoryStorage
from loguru import logger

from app.core.config import settings

# Imported as module (not from-import) so monkeypatching
# breaker_storage.build_breaker_storage reaches this module.
from app.resilience.circuit_breaker import breaker_storage
from app.resilience.circuit_breaker.breaker_listener import (
    _CIRCUIT_BREAKER_LISTENER,
)
from app.resilience.circuit_breaker.breaker_state import CircuitBreakerState

# ---------------------------------------------------------------------------
# Registry - thread-safe singleton cache with double-checked locking
# ---------------------------------------------------------------------------

_circuit_breaker_registry: dict[str, aiobreaker.CircuitBreaker] = {}
_circuit_breaker_registry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal factory
# ---------------------------------------------------------------------------


def create_circuit_breaker(
    breaker_name: str,
    failure_threshold: int,
    recovery_timeout_seconds: int,
) -> aiobreaker.CircuitBreaker:
    """
    Create or return a cached thread-safe singleton circuit breaker.

    Args:
        breaker_name: unique identifier ("postgres", "redis", "rabbitmq")
        failure_threshold: consecutive failures before opening the circuit
        recovery_timeout_seconds: seconds in OPEN before HALF_OPEN probe

    Returns:
        A configured aiobreaker.CircuitBreaker instance (singleton per name).
    """
    if breaker_name in _circuit_breaker_registry:
        return _circuit_breaker_registry[breaker_name]

    with _circuit_breaker_registry_lock:
        if breaker_name in _circuit_breaker_registry:
            return _circuit_breaker_registry[breaker_name]

        try:
            storage = breaker_storage.build_breaker_storage(breaker_name)
        except Exception:
            # Broad by intent: if storage wiring is broken for any reason,
            # create a fail-closed local breaker instead of failing open.
            logger.exception(
                f"[CircuitBreaker:{breaker_name}] Failed to build storage; "
                "falling back to local OPEN state storage"
            )
            storage = CircuitMemoryStorage(aiobreaker.CircuitBreakerState.OPEN)

        circuit_breaker = aiobreaker.CircuitBreaker(
            fail_max=failure_threshold,
            timeout_duration=timedelta(seconds=recovery_timeout_seconds),
            state_storage=storage,
            listeners=[_CIRCUIT_BREAKER_LISTENER],
            name=breaker_name,
        )
        _circuit_breaker_registry[breaker_name] = circuit_breaker
        logger.info(
            f"[CircuitBreaker:{breaker_name}] Registered "
            f"(threshold={failure_threshold}, "
            f"reset={recovery_timeout_seconds}s)"
        )
        return circuit_breaker


# ---------------------------------------------------------------------------
# Public factory functions - one per dependency
# ---------------------------------------------------------------------------


def get_db_circuit_breaker() -> aiobreaker.CircuitBreaker:
    """
    Circuit breaker protecting PostgreSQL (local in-memory state).

    Decoupled from Redis so DB protection survives a Redis outage.
    """
    return create_circuit_breaker(
        breaker_name="postgres",
        failure_threshold=settings.cb_db_failure_threshold,
        recovery_timeout_seconds=settings.cb_db_recovery_timeout,
    )


def get_redis_circuit_breaker() -> aiobreaker.CircuitBreaker:
    """
    Circuit breaker protecting Redis operations.

    State is distributed across replicas via Redis-backed storage.
    Fails closed (OPEN) when breaker storage is unreachable.
    """
    return create_circuit_breaker(
        breaker_name="redis",
        failure_threshold=settings.cb_redis_failure_threshold,
        recovery_timeout_seconds=settings.cb_redis_recovery_timeout,
    )


def get_rmq_circuit_breaker() -> aiobreaker.CircuitBreaker:
    """
    Circuit breaker protecting RabbitMQ publish calls.

    State is distributed across replicas via Redis-backed storage.
    Fails closed (OPEN) when breaker storage is unreachable.
    """
    return create_circuit_breaker(
        breaker_name="rabbitmq",
        failure_threshold=settings.cb_rmq_failure_threshold,
        recovery_timeout_seconds=settings.cb_rmq_recovery_timeout,
    )


# ---------------------------------------------------------------------------
# State introspection
# ---------------------------------------------------------------------------


def get_circuit_breaker_states() -> dict[str, str]:
    """
    Return current state of all registered circuit breakers.

    Values are normalized through the CircuitBreakerState enum.
    Redis/RMQ state is distributed across replicas; DB state is local in-memory.
    """
    states: dict[str, str] = {}
    for breaker_name, circuit_breaker in _circuit_breaker_registry.items():
        # aiobreaker.CircuitBreakerState is an Enum whose .name gives 'CLOSED',
        # 'OPEN', 'HALF_OPEN'. Map to our lowercase convention ('closed',
        # 'open', 'half-open') so callers are insulated from the library enum.
        raw_name = circuit_breaker.current_state.name.lower().replace("_", "-")
        try:
            states[breaker_name] = CircuitBreakerState(raw_name).value
        except ValueError:
            states[breaker_name] = raw_name
    return states
