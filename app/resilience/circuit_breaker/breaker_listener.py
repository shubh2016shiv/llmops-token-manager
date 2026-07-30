"""
Circuit breaker logging listener — your observability window into every breaker.

THE PATTERN: an "observer"/"listener" hook
------------------------------------------
aiobreaker doesn't log anything itself. Instead it lets you register a *listener*
object, and it CALLS the listener's methods when things happen: a state change, a
failed call, a successful call. You don't call these methods — aiobreaker does,
from inside `call_async`. We subclass its listener base and fill in the three
hooks with structured log lines.

This is the ONLY place breaker activity becomes visible in production. When you're
debugging "why did requests start failing at 2am?", the `CLOSED -> OPEN` line this
file emits is the smoking gun.

WHERE IT IS ATTACHED
--------------------
The single instance at the bottom (`_CIRCUIT_BREAKER_LISTENER`) is passed to every
breaker in breaker_registry.create_circuit_breaker(..., listeners=[...]). One
shared instance is fine because the listener holds no state — every method is
handed the specific breaker it concerns.

Author: Engineering Team
"""

from __future__ import annotations

import aiobreaker
from loguru import logger


class CircuitBreakerListener(aiobreaker.CircuitBreakerListener):
    """Emit structured log lines for state changes and call outcomes."""

    def state_change(
        self,
        circuit_breaker: aiobreaker.CircuitBreaker,
        old_state: aiobreaker.CircuitBreakerState,
        new_state: aiobreaker.CircuitBreakerState,
    ) -> None:
        """
        Called by aiobreaker whenever a breaker moves between states.

        Logged at WARNING because a transition is always operationally
        interesting — CLOSED->OPEN means a dependency just got cut off;
        OPEN->HALF_OPEN->CLOSED means it recovered. We include the running
        failure count and threshold so the log explains *why* it moved.
        """
        logger.warning(
            f"[CircuitBreaker:{circuit_breaker.name}] "
            f"State transition: {old_state.name} -> {new_state.name} | "
            f"failures={circuit_breaker.fail_counter} "
            f"threshold={circuit_breaker.fail_max}"
        )

    def failure(
        self,
        circuit_breaker: aiobreaker.CircuitBreaker,
        exception: Exception,
    ) -> None:
        """
        Called when a protected call raised an exception (a counted failure).

        The `count/threshold` (e.g. 2/3) lets you watch a breaker creep toward
        tripping in the logs before it actually opens.
        """
        logger.error(
            f"[CircuitBreaker:{circuit_breaker.name}] Failure recorded "
            f"({circuit_breaker.fail_counter}/{circuit_breaker.fail_max}): "
            f"{type(exception).__name__}: {exception}"
        )

    def success(
        self,
        circuit_breaker: aiobreaker.CircuitBreaker,
    ) -> None:
        """
        Called when a protected call succeeded (which resets the failure count).

        Logged at DEBUG because successes are the common case and would be noise
        at higher levels — but they're invaluable when tracing a HALF_OPEN probe
        that recovered the breaker.
        """
        logger.debug(
            f"[CircuitBreaker:{circuit_breaker.name}] "
            f"Success (state={circuit_breaker.current_state.name})"
        )


# Module-level singleton. It is stateless (every hook receives the breaker it
# concerns), so one shared instance is safely attached to all three breakers.
_CIRCUIT_BREAKER_LISTENER = CircuitBreakerListener()
