"""
Circuit-state probe — gauge #3 of the backpressure evaluator.

WHAT THIS MEASURES
------------------
A "circuit breaker" is a separate safety device (Layer 2) wrapped around the
database. After repeated DB failures it "opens" and short-circuits further calls,
giving the database time to recover. This probe reads a READ-ONLY snapshot of the
DB breaker so the evaluator can ask one question: "is the breaker open right now?"
If it is, the database is effectively unavailable and we should reject fast.

This is gauge #3 (checked last) because it is the most severe, most specific
signal — "the DB is actually down" — as opposed to the earlier gauges that catch
saturation *before* it becomes an outage.

STRICTLY A READER
-----------------
This probe only *reads* breaker state. It never trips, resets, or mutates the
breaker — that is the circuit_breaker package's job. We take a snapshot and leave.

    ┌───────────────────┐     ┌──────────────────────────────┐
    │ evaluator.py      │────▶│ probes/circuit_state.py      │
    │ (gauge #3 check)  │     │ snapshot DB breaker state    │
    └───────────────────┘     └──────────────┬───────────────┘
                                             │ read-only introspection
                                             ▼
                              ┌────────────────────────────────┐
                              │ app/resilience/circuit_breaker │
                              └────────────────────────────────┘

Author: Engineering Team
Last Updated: 2026-07-23
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

# The typed snapshot we hand back to the evaluator — a small, validated record.
from app.models.resilience_models import CircuitBreakerSnapshot

# Accessor for the shared DB circuit-breaker instance (registry-managed).
from app.resilience.circuit_breaker import get_db_circuit_breaker


def read_db_circuit_breaker_snapshot() -> CircuitBreakerSnapshot:
    """
    Return a read-only snapshot of the DB circuit breaker's current state.

    Note: unlike the other two probes this does NOT return `None` on trouble — the
    breaker object is always present in-process, so there is no "unknown" case to
    fail open on. The evaluator inspects `.state` to decide whether to reject.
    """
    db_circuit_breaker = get_db_circuit_breaker()

    # `opened_at` lives in the breaker's private state storage and may be a raw
    # unix timestamp, a datetime, or None depending on the backend. We read it
    # defensively via getattr and normalize it just below.
    raw_opened_at = getattr(db_circuit_breaker._state_storage, "opened_at", None)  # noqa: SLF001
    opened_at = _coerce_opened_at(raw_opened_at)

    breaker_name = db_circuit_breaker.name

    # Recovery timeout = how long the breaker stays OPEN before it probes the DB
    # again. We surface it as the client's Retry-After when the breaker is open.
    #
    # Type note: aiobreaker annotates the `timeout_duration` *setter* as a
    # datetime (the reopen-at moment), which makes the static checker infer the
    # property as datetime. At RUNTIME the getter always returns the timedelta
    # stored in __init__, so we cast to timedelta and read total_seconds().
    recovery_timeout_seconds = int(
        cast("timedelta", db_circuit_breaker.timeout_duration).total_seconds()
    )

    # Normalize the state name to a lowercase, hyphenated string
    # (e.g. "HALF_OPEN" → "half-open") for a stable, client-friendly value.
    state_str = db_circuit_breaker.current_state.name.lower().replace("_", "-")

    return CircuitBreakerSnapshot(
        name=breaker_name,
        state=state_str,
        failure_count=db_circuit_breaker.fail_counter,
        recovery_timeout_seconds=recovery_timeout_seconds,
        opened_at=opened_at,
    )


def _coerce_opened_at(raw_opened_at: object) -> datetime | None:
    """
    Normalize the breaker's stored "opened at" value into a UTC datetime.

    Different storage backends record this differently, so we accept all shapes:
      • None            → never opened (or unknown)     → None
      • datetime        → already a datetime            → pass through
      • int/float       → a unix timestamp              → convert to UTC datetime
      • anything else   → unrecognized                  → None (safe default)
    """
    if raw_opened_at is None:
        return None
    if isinstance(raw_opened_at, datetime):
        return raw_opened_at
    if isinstance(raw_opened_at, (int, float)):
        return datetime.fromtimestamp(raw_opened_at, tz=timezone.utc)
    return None
