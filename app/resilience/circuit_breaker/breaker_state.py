"""
Circuit breaker state enum definitions.

The values mirror aiobreaker state names (lowercased, underscores replaced
with hyphens) so callers can compare normalized state values without
depending on the library enum directly.
"""

from __future__ import annotations

from enum import Enum


class CircuitBreakerState(str, Enum):
    """Canonical circuit breaker states matching aiobreaker's lifecycle."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"
