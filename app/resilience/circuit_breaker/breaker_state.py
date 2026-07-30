"""
Circuit breaker state enum — our own normalized names for the three states.

WHY THIS TINY FILE EXISTS
-------------------------
The `aiobreaker` library has its own state enum (`CircuitBreakerState.CLOSED`,
`.OPEN`, `.HALF_OPEN`). We could compare against that library enum everywhere —
but then every part of our codebase (health endpoints, the backpressure probe,
logs) would be coupled to a third-party library's exact naming. If aiobreaker
ever renamed a state, our whole app would break.

So we define OUR OWN enum with stable string values we control, and normalize the
library's names into it at the one boundary where we read them (see
breaker_registry.get_circuit_breaker_states and the backpressure circuit_state
probe). Callers compare against THIS enum and stay insulated from the library.

Author: Engineering Team
"""

from __future__ import annotations

from enum import Enum


# `str, Enum` (a "string enum"): each member IS a real string, so
# `CircuitBreakerState.OPEN == "open"` is True and it serializes straight to
# "open" in JSON — no `.value` needed at call sites. That's why the values are
# lowercase, hyphenated strings rather than bare enum members.
class CircuitBreakerState(str, Enum):
    """Canonical circuit breaker states matching aiobreaker's lifecycle."""

    # The values deliberately use lowercase + hyphens ("half-open", not
    # "HALF_OPEN"). aiobreaker reports "HALF_OPEN"; we normalize with
    # `.lower().replace("_", "-")` when reading, so everything downstream sees
    # this friendly, URL/JSON-safe form.
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"
