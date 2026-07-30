"""
HTTP response translator — turns a backpressure verdict into an HTTP 503.

WHERE THIS SITS IN THE FLOW
---------------------------
The evaluator produces a typed `BackpressureDecision` — a pure data verdict that
knows nothing about the web. THIS module is the ONLY place that knows about HTTP.
It reads the verdict and, if it says "reject", raises FastAPI's HTTPException(503)
with the right status, headers, and body. Swap web frameworks tomorrow and this is
the single file you would rewrite.

    ┌──────────────────────────┐     ┌──────────────────────────┐
    │ dependency.py            │────▶│ http_response.py         │
    │ (has a typed decision)   │     │ decision → HTTP 503      │
    └──────────────────────────┘     └──────────────────────────┘

WHY 503 (and not 500, 429, etc.)
--------------------------------
503 Service Unavailable is the correct semantic for "healthy service, temporarily
overloaded, try again later". We pair it with a `Retry-After` header so clients
(and load balancers) know *when* to retry instead of hammering us immediately.

Author: Engineering Team
Last Updated: 2026-07-23
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, status

# Reason codes are defined once in constants.py; we map each to a human message.
from app.resilience.backpressure.constants import (
    DB_CIRCUIT_BREAKER_OPEN_REASON,
    DB_POOL_SATURATED_REASON,
    QUEUE_DEPTH_EXCEEDED_REASON,
)

# Type-only import: the decision contract. Under TYPE_CHECKING so there is no
# runtime import cost and no risk of an import cycle.
if TYPE_CHECKING:
    from app.models.resilience_models import BackpressureDecision


def raise_for_backpressure_decision(decision: BackpressureDecision) -> None:
    """
    Raise `HTTPException(503)` when the verdict rejects the request; else do nothing.

    Named after the `requests.Response.raise_for_status()` idiom: call it
    unconditionally, and it only raises when there is something to raise for.
    """
    # The happy path: the system is healthy, so there is nothing to do. The
    # request continues to the real token-allocation handler.
    if not decision.should_reject_request:
        return

    # A rejection MUST carry a Retry-After. The decision model already enforces
    # this invariant, but we re-check here so a hand-built decision can never
    # produce a 503 with no guidance for the client. This is the one place in the
    # module that raises loudly on bad input rather than failing open.
    retry_after_seconds = decision.retry_after_seconds
    if retry_after_seconds is None:
        raise ValueError("retry_after_seconds is required for rejection decisions")

    # --- Build the JSON body. ----------------------------------------------
    # `error` + `message` are for humans/clients; `reason` is a stable machine
    # code clients can branch on without parsing prose.
    detail = {
        "error": "SERVICE_UNAVAILABLE",
        "message": _message_for_reason(decision.reason),
        "retry_after_seconds": retry_after_seconds,
        "reason": decision.reason,
    }
    # Attach whichever diagnostic the triggering gauge provided. Each is optional
    # because only the gauge that fired knows its own number.
    if decision.queue_depth is not None:
        detail["queue_depth"] = decision.queue_depth
    if decision.pool_utilization_pct is not None:
        detail["pool_utilization_pct"] = decision.pool_utilization_pct
    if decision.circuit_breaker_name is not None:
        detail["circuit_breaker_name"] = decision.circuit_breaker_name

    # --- Raise the 503. -----------------------------------------------------
    # `Retry-After` is the standard header clients/proxies honor. The custom
    # `X-Backpressure-Reason` header lets ops and dashboards see the cause without
    # reading the body.
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
        headers={
            "Retry-After": str(retry_after_seconds),
            "X-Backpressure-Reason": decision.reason or "backpressure_rejection",
        },
    )


def _message_for_reason(reason: str | None) -> str:
    """Map a machine reason code to a friendly, client-facing message."""
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
    # Fallback for any unknown/unset reason — still a valid, honest 503 message.
    return "Service is temporarily unavailable. Please retry later."
