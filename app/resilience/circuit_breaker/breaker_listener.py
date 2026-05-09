"""
Structured logging listener for circuit breaker events.

This listener records state transitions, failures, and successes for every
registered breaker.
"""

from __future__ import annotations

from loguru import logger
import pybreaker


class CircuitBreakerListener(pybreaker.CircuitBreakerListener):
    """Emit structured log lines for state changes and call outcomes."""

    def state_change(
        self,
        circuit_breaker: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState,
        new_state: pybreaker.CircuitBreakerState,
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
        circuit_breaker: pybreaker.CircuitBreaker,
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
        circuit_breaker: pybreaker.CircuitBreaker,
    ) -> None:
        """Log a successful protected call."""
        logger.debug(
            f"[CircuitBreaker:{circuit_breaker.name}] "
            f"Success (state={circuit_breaker.current_state})"
        )


# Module-level singleton; stateless and safe to share.
_CIRCUIT_BREAKER_LISTENER = CircuitBreakerListener()
