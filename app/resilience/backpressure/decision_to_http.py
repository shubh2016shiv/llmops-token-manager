"""
Decision-to-HTTP mapper - convert Layer 1 decisions into FastAPI 503 responses.

Architecture:
-------------
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │ dependency.py / guard.py │────▶│ decision_to_http.py      │
    │ typed decision available │     │ HTTPException conversion │
    └──────────────────────────┘     └──────────────────────────┘

Dependencies:
    - fastapi - HTTPException and status codes
    - app/models/resilience_models.py - BackpressureDecision contract

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from app.resilience.backpressure.constants import (
    DB_CIRCUIT_BREAKER_OPEN_REASON,
    DB_POOL_SATURATED_REASON,
    QUEUE_DEPTH_EXCEEDED_REASON,
)

if TYPE_CHECKING:
    from app.models.resilience_models import BackpressureDecision


def raise_for_backpressure_decision(decision: BackpressureDecision) -> None:
    """Raise `HTTPException(503)` when Layer 1 rejects the current request."""
    if not decision.should_reject_request:
        return

    retry_after_seconds = decision.retry_after_seconds
    if retry_after_seconds is None:
        raise ValueError("retry_after_seconds is required for rejection decisions")

    detail = {
        "error": "SERVICE_UNAVAILABLE",
        "message": _message_for_reason(decision.reason),
        "retry_after_seconds": retry_after_seconds,
        "reason": decision.reason,
    }
    if decision.queue_depth is not None:
        detail["queue_depth"] = decision.queue_depth
    if decision.pool_utilization_pct is not None:
        detail["pool_utilization_pct"] = decision.pool_utilization_pct
    if decision.circuit_breaker_name is not None:
        detail["circuit_breaker_name"] = decision.circuit_breaker_name

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
        headers={
            "Retry-After": str(retry_after_seconds),
            "X-Backpressure-Reason": decision.reason or "backpressure_rejection",
        },
    )


def _message_for_reason(reason: str | None) -> str:
    """Return a client-facing message for a backpressure reason code."""
    if reason == QUEUE_DEPTH_EXCEEDED_REASON:
        return (
            "System is temporarily at capacity. "
            "Please retry after the indicated interval."
        )
    if reason == DB_POOL_SATURATED_REASON:
        return "Database connection pool is saturated. Please retry in a few seconds."
    if reason == DB_CIRCUIT_BREAKER_OPEN_REASON:
        return (
            "Database is temporarily unavailable. "
            "Please retry after the indicated interval."
        )
    return "Service is temporarily unavailable. Please retry later."
