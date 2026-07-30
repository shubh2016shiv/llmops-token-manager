"""
Backpressure evaluator — the decision brain (Layer 1 admission control).

WHAT THIS FILE IS
-----------------
This is where "the rule" lives: read the three health gauges IN PRIORITY ORDER
and, the moment one reads "red", stop and return a rejection verdict. If all three
are healthy, return an "accept" verdict. That's the entire policy.

Crucially, this file produces a *verdict as data* (`BackpressureDecision`) — it
does NOT touch HTTP. Deciding "what is true" (here) is kept separate from "how to
tell the client" (http_response.py). That separation is what lets this logic be
unit-tested with no web server, and is why the tests can swap the probes out.

THE ORDER MATTERS (and is deliberate)
-------------------------------------
    1. Queue depth   — broadest early-warning signal; catches trouble first.
    2. DB pool        — mid-severity; connections running out.
    3. Circuit breaker— most severe/specific: the DB is effectively down.
Whichever fires first decides the `reason` and the `Retry-After`. We check cheap,
broad signals before narrow, severe ones, and we short-circuit on the first hit.

    dependency.py ──▶ evaluate_backpressure()
                          │  reads gauges via the probes package
                          ├─▶ probes.read_queue_depth() ............ gauge #1
                          ├─▶ probes.read_db_pool_utilization_pct(). gauge #2
                          └─▶ probes.read_db_circuit_breaker_...() .. gauge #3
                          │
                          ▼
                   BackpressureDecision  (typed verdict, no HTTP)

Author: Engineering Team
Last Updated: 2026-07-23
"""

from __future__ import annotations

from loguru import logger

# Thresholds and retry tuning (settings.bp_*) — the numbers we compare gauges to.
from app.core.config import settings

# The typed verdict this module returns.
from app.models.resilience_models import BackpressureDecision

# Stable machine reason codes, shared with http_response.py.
from app.resilience.backpressure.constants import (
    DB_CIRCUIT_BREAKER_OPEN_REASON,
    DB_POOL_SATURATED_REASON,
    QUEUE_DEPTH_EXCEEDED_REASON,
)

# The three gauge readers, imported as bare names from the probes package.
# NOTE: they are bound as module-level names here on purpose — the unit tests
# monkeypatch `evaluator.read_queue_depth` etc. to drive each branch in isolation.
from app.resilience.backpressure.probes import (
    estimate_queue_retry_after_seconds,
    read_db_circuit_breaker_snapshot,
    read_db_pool_utilization_pct,
    read_queue_depth,
)

# Enum of breaker states; we compare the snapshot's state against OPEN.
from app.resilience.circuit_breaker import CircuitBreakerState


async def evaluate_backpressure() -> BackpressureDecision:
    """
    Evaluate the three health gauges in order and return a typed admission verdict.

    Returns a rejection `BackpressureDecision` for the first gauge that is "red",
    or an accept decision (`should_reject_request=False`) if all are healthy.
    """
    # ---- Gauge #1: QUEUE DEPTH ---------------------------------------------
    # `read_queue_depth()` returns None when the signal is unknown (fail-open).
    # We only reject when we have a real number AND it exceeds the configured max.
    queue_depth = await read_queue_depth()
    if queue_depth is not None and queue_depth > settings.bp_max_queue_depth:
        # Estimate how long the client should wait (excess ÷ drain rate, clamped).
        retry_after_seconds = estimate_queue_retry_after_seconds(queue_depth)
        logger.bind(
            reason=QUEUE_DEPTH_EXCEEDED_REASON,
            queue_depth=queue_depth,
            retry_after_seconds=retry_after_seconds,
        ).warning("Backpressure rejected request due to queue depth")
        return BackpressureDecision(
            should_reject_request=True,
            reason=QUEUE_DEPTH_EXCEEDED_REASON,
            retry_after_seconds=retry_after_seconds,
            queue_depth=queue_depth,
        )

    # ---- Gauge #2: DB CONNECTION POOL --------------------------------------
    # Reject when utilization is at/above the saturation threshold (default 90%).
    pool_utilization_pct = read_db_pool_utilization_pct()
    if (
        pool_utilization_pct is not None
        and pool_utilization_pct >= settings.bp_db_pool_saturation_pct
    ):
        logger.bind(
            reason=DB_POOL_SATURATED_REASON,
            pool_utilization_pct=pool_utilization_pct,
            retry_after_seconds=settings.bp_db_pool_retry_after_seconds,
        ).warning("Backpressure rejected request due to DB pool saturation")
        return BackpressureDecision(
            should_reject_request=True,
            reason=DB_POOL_SATURATED_REASON,
            # Fixed wait: pool pressure clears on its own timescale, so there is
            # no backlog to "drain" — a small constant retry is the right hint.
            retry_after_seconds=settings.bp_db_pool_retry_after_seconds,
            pool_utilization_pct=pool_utilization_pct,
        )

    # ---- Gauge #3: DB CIRCUIT BREAKER --------------------------------------
    # Most severe signal: if the breaker is OPEN the DB is effectively down.
    db_circuit_breaker_snapshot = read_db_circuit_breaker_snapshot()
    if db_circuit_breaker_snapshot.state == CircuitBreakerState.OPEN.value:
        logger.bind(
            reason=DB_CIRCUIT_BREAKER_OPEN_REASON,
            circuit_breaker_name=db_circuit_breaker_snapshot.name,
            retry_after_seconds=db_circuit_breaker_snapshot.recovery_timeout_seconds,
        ).warning("Backpressure rejected request due to DB breaker state")
        return BackpressureDecision(
            should_reject_request=True,
            reason=DB_CIRCUIT_BREAKER_OPEN_REASON,
            # Wait exactly as long as the breaker will stay open before it retries.
            retry_after_seconds=db_circuit_breaker_snapshot.recovery_timeout_seconds,
            circuit_breaker_name=db_circuit_breaker_snapshot.name,
        )

    # ---- All gauges healthy: ADMIT the request. ----------------------------
    return BackpressureDecision(should_reject_request=False)
