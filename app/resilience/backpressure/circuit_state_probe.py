"""
Circuit state probe - Layer 1 read-only breaker health snapshot.

Architecture:
-------------
    ┌───────────────────┐     ┌──────────────────────────┐
    │ evaluator.py      │────▶│ circuit_state_probe.py   │
    │ Layer 1 ordering  │     │ DB breaker state reader  │
    └───────────────────┘     └──────────────┬───────────┘
                                             │
                                             ▼
                              ┌────────────────────────────────┐
                              │ app/resilience/circuit_breaker │
                              │ state introspection only       │
                              └────────────────────────────────┘

Dependencies:
    - app/models/resilience_models.py - CircuitBreakerSnapshot
    - app/resilience/circuit_breaker - DB breaker registry

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

from app.models.resilience_models import CircuitBreakerSnapshot
from app.resilience.circuit_breaker import get_db_circuit_breaker


def read_db_circuit_breaker_snapshot() -> CircuitBreakerSnapshot:
    """Return a read-only snapshot of the DB circuit breaker state."""
    db_circuit_breaker = get_db_circuit_breaker()
    raw_opened_at = getattr(db_circuit_breaker._state_storage, "opened_at", None)  # noqa: SLF001
    opened_at = _coerce_opened_at(raw_opened_at)
    breaker_name = db_circuit_breaker.name
    # aiobreaker's timeout_duration setter is annotated as datetime (the
    # reopen-at timestamp), making pyright infer the property type as datetime.
    # At runtime the getter always returns a timedelta as stored in __init__.
    recovery_timeout_seconds = int(
        cast("timedelta", db_circuit_breaker.timeout_duration).total_seconds()
    )
    state_str = db_circuit_breaker.current_state.name.lower().replace("_", "-")
    return CircuitBreakerSnapshot(
        name=breaker_name,
        state=state_str,
        failure_count=db_circuit_breaker.fail_counter,
        recovery_timeout_seconds=recovery_timeout_seconds,
        opened_at=opened_at,
    )


def _coerce_opened_at(raw_opened_at: object) -> datetime | None:
    """Convert pybreaker's storage timestamp into a UTC datetime."""
    if raw_opened_at is None:
        return None
    if isinstance(raw_opened_at, datetime):
        return raw_opened_at
    if isinstance(raw_opened_at, (int, float)):
        return datetime.fromtimestamp(raw_opened_at, tz=timezone.utc)
    return None
