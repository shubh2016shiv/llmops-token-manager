"""
Structured logging listener for circuit breaker events.

This listener records state transitions, failures, and successes for every
registered breaker.
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
        """Log a breaker state transition with current failure counters."""
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
        """Log an operation failure captured by the breaker."""
        logger.error(
            f"[CircuitBreaker:{circuit_breaker.name}] Failure recorded "
            f"({circuit_breaker.fail_counter}/{circuit_breaker.fail_max}): "
            f"{type(exception).__name__}: {exception}"
        )

    def success(
        self,
        circuit_breaker: aiobreaker.CircuitBreaker,
    ) -> None:
        """Log a successful protected call."""
        logger.debug(
            f"[CircuitBreaker:{circuit_breaker.name}] "
            f"Success (state={circuit_breaker.current_state.name})"
        )


# Module-level singleton; stateless and safe to share.
_CIRCUIT_BREAKER_LISTENER = CircuitBreakerListener()
