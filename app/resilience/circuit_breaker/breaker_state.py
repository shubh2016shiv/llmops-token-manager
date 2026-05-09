"""
Circuit breaker state enum definitions.

The values mirror pybreaker state strings so callers can compare normalized
state values without depending on raw literals.
"""

from __future__ import annotations

from enum import Enum


class CircuitBreakerState(str, Enum):
    """Canonical circuit breaker states matching pybreaker's lifecycle."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"
