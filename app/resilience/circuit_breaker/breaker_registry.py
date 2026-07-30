"""
Circuit breaker registry — the factory that builds and shares the three breakers.

WHAT THIS FILE OWNS
-------------------
This is the heart of the sub-package. It:
  1. Builds each named breaker exactly once and CACHES it (so the whole app shares
     one breaker per dependency — see "why a singleton" below).
  2. Exposes one `get_*_circuit_breaker()` helper per dependency, each wired with
     that dependency's configured failure threshold and recovery timeout.
  3. Reports every breaker's current state for health endpoints.

WHY A SINGLETON PER BREAKER (this is important)
-----------------------------------------------
A breaker's whole job is to COUNT failures across many calls. If two parts of the
app each created their own `redis` breaker object, each would count failures
separately — neither would ever reach the threshold, and the breaker would never
trip. So there must be exactly one shared `redis` breaker, one shared `postgres`
breaker, etc. The registry dict below is what guarantees that.

Author: Engineering Team
"""

from __future__ import annotations

from datetime import timedelta
import threading

import aiobreaker
from aiobreaker.storage import CircuitMemoryStorage
from loguru import logger

from app.core.config import settings

# NOTE (subtle but deliberate): we import the storage *module* and call
# `breaker_storage.build_breaker_storage(...)` below, instead of doing
# `from ...breaker_storage import build_breaker_storage`. Why? So tests can
# monkeypatch `breaker_storage.build_breaker_storage` and have THIS module pick up
# the patched version. With a `from`-import we'd have captured our own reference to
# the original function at import time, and the patch wouldn't reach us.
from app.resilience.circuit_breaker import breaker_storage
from app.resilience.circuit_breaker.breaker_listener import (
    _CIRCUIT_BREAKER_LISTENER,
)
from app.resilience.circuit_breaker.breaker_state import CircuitBreakerState

# ---------------------------------------------------------------------------
# Registry — thread-safe singleton cache with double-checked locking
# ---------------------------------------------------------------------------

# name -> the one shared breaker for that dependency. Populated lazily on first use.
_circuit_breaker_registry: dict[str, aiobreaker.CircuitBreaker] = {}
# Guards construction so two threads can't build the same breaker simultaneously.
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

    Uses "double-checked locking" (same pattern as the Redis client in
    breaker_storage): check the cache without the lock first (fast path), and only
    take the lock — and check again — when we actually need to build. The second
    check prevents two threads that both missed the cache from each building a
    breaker; only the first through the lock builds, the second reuses it.

    Args:
        breaker_name: unique identifier ("postgres", "redis", "rabbitmq")
        failure_threshold: consecutive failures before opening the circuit
        recovery_timeout_seconds: seconds in OPEN before HALF_OPEN probe

    Returns:
        A configured aiobreaker.CircuitBreaker instance (singleton per name).
    """
    # 1st check — no lock. Almost always a hit after startup; avoids lock contention.
    if breaker_name in _circuit_breaker_registry:
        return _circuit_breaker_registry[breaker_name]

    with _circuit_breaker_registry_lock:
        # 2nd check — under the lock. Another thread may have built it while we
        # waited; if so, return theirs instead of building a duplicate.
        if breaker_name in _circuit_breaker_registry:
            return _circuit_breaker_registry[breaker_name]

        # Build the storage backend (local vs Redis) for this breaker.
        try:
            storage = breaker_storage.build_breaker_storage(breaker_name)
        except Exception:
            # Broad `except` ON PURPOSE, and it FAILS CLOSED. If storage wiring is
            # broken for ANY reason (Redis down at startup, bad config, etc.), we do
            # NOT want a breaker that defaults to "closed/healthy" and lets traffic
            # flow at a dependency we can't protect. Instead we build a local
            # breaker whose state starts OPEN — assume the worst until proven
            # otherwise. (Mirror image of backpressure's fail-open default.)
            logger.exception(
                f"[CircuitBreaker:{breaker_name}] Failed to build storage; "
                "falling back to local OPEN state storage"
            )
            storage = CircuitMemoryStorage(aiobreaker.CircuitBreakerState.OPEN)

        # Construct the actual aiobreaker breaker with our configuration:
        #   fail_max         — trip after this many consecutive failures.
        #   timeout_duration — how long to stay OPEN before a HALF_OPEN probe
        #                      (a timedelta; we convert seconds -> timedelta here).
        #   state_storage    — the backend chosen above (local or Redis).
        #   listeners        — our logging listener, so transitions are observable.
        #   name             — used in logs and to key the registry.
        circuit_breaker = aiobreaker.CircuitBreaker(
            fail_max=failure_threshold,
            timeout_duration=timedelta(seconds=recovery_timeout_seconds),
            state_storage=storage,
            listeners=[_CIRCUIT_BREAKER_LISTENER],
            name=breaker_name,
        )
        # Cache it so every future caller shares this exact instance.
        _circuit_breaker_registry[breaker_name] = circuit_breaker
        logger.info(
            f"[CircuitBreaker:{breaker_name}] Registered "
            f"(threshold={failure_threshold}, "
            f"reset={recovery_timeout_seconds}s)"
        )
        return circuit_breaker


# ---------------------------------------------------------------------------
# Public factory functions — one per dependency
# ---------------------------------------------------------------------------
# Each just calls create_circuit_breaker with that dependency's tuned settings.
# Callers use these (never create_circuit_breaker directly) so the name/threshold/
# timeout for each dependency live in exactly one place.


def get_db_circuit_breaker() -> aiobreaker.CircuitBreaker:
    """
    Circuit breaker protecting PostgreSQL (local in-memory state).

    Decoupled from Redis so DB protection survives a Redis outage. Higher failure
    threshold than the others (DB is the durable fallback — tolerate a few blips
    before cutting it off).
    """
    return create_circuit_breaker(
        breaker_name="postgres",
        failure_threshold=settings.cb_db_failure_threshold,
        recovery_timeout_seconds=settings.cb_db_recovery_timeout,
    )


def get_redis_circuit_breaker() -> aiobreaker.CircuitBreaker:
    """
    Circuit breaker protecting Redis operations.

    State is distributed across replicas via Redis-backed storage (one replica
    tripping it opens it for all). Fails closed (OPEN) when breaker storage is
    unreachable.
    """
    return create_circuit_breaker(
        breaker_name="redis",
        failure_threshold=settings.cb_redis_failure_threshold,
        recovery_timeout_seconds=settings.cb_redis_recovery_timeout,
    )


def get_rmq_circuit_breaker() -> aiobreaker.CircuitBreaker:
    """
    Circuit breaker protecting RabbitMQ publish calls.

    State is distributed across replicas via Redis-backed storage. Fails closed
    (OPEN) when breaker storage is unreachable.
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
    Return the current state of every registered breaker.

    E.g. {"postgres": "closed", "redis": "open"}. Used by health/diagnostics
    endpoints. Only reports breakers that have actually been created (are in
    the registry). Redis/RMQ state reflects the shared distributed view; DB
    state is this replica's local view.
    """
    states: dict[str, str] = {}
    for breaker_name, circuit_breaker in _circuit_breaker_registry.items():
        # aiobreaker reports names like "CLOSED" / "OPEN" / "HALF_OPEN". Normalize
        # to our convention: lowercase, hyphenated ("half-open").
        raw_name = circuit_breaker.current_state.name.lower().replace("_", "-")
        try:
            # Round-trip through OUR enum so the output is guaranteed to be one of
            # our canonical values (insulating callers from the library).
            states[breaker_name] = CircuitBreakerState(raw_name).value
        except ValueError:
            # Defensive: if a future aiobreaker version introduces a state our enum
            # doesn't know, don't crash the health endpoint — pass the raw name
            # through so at least it's visible.
            states[breaker_name] = raw_name
    return states
