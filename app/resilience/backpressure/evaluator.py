"""
Backpressure evaluator - ordered Layer 1 admission decisions.

Architecture:
-------------
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │ dependency.py / backpressure_gate.py │────▶│ evaluator.py │
    │ request admission        │     │ ordered Layer 1 checks   │
    └──────────────────────────┘     └───────┬────────┬─────────┘
                                             │        │
                                             ▼        ▼
                                   ┌──────────────┐  ┌────────────────┐
                                   │ queue_depth  │  │ pool / breaker │
                                   │ probe        │  │ probes         │
                                   └──────────────┘  └────────────────┘

Dependencies:
    - app/core/config.py - thresholds and retry values
    - app/models/resilience_models.py - BackpressureDecision contract
    - app/resilience/backpressure/*_probe.py - signal readers

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings
from app.models.resilience_models import BackpressureDecision
from app.resilience.backpressure.circuit_state_probe import (
    read_db_circuit_breaker_snapshot,
)
from app.resilience.backpressure.constants import (
    DB_CIRCUIT_BREAKER_OPEN_REASON,
    DB_POOL_SATURATED_REASON,
    QUEUE_DEPTH_EXCEEDED_REASON,
)
from app.resilience.backpressure.db_connection_pool_probe import (
    read_db_pool_utilization_pct,
)
from app.resilience.backpressure.token_queue_depth_probe import (
    estimate_queue_retry_after_seconds,
    read_queue_depth,
)
from app.resilience.circuit_breaker import CircuitBreakerState


async def evaluate_backpressure() -> BackpressureDecision:
    """Evaluate Layer 1 system health and return a typed admission decision."""
    queue_depth = await read_queue_depth()
    if queue_depth is not None and queue_depth > settings.bp_max_queue_depth:
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
            retry_after_seconds=settings.bp_db_pool_retry_after_seconds,
            pool_utilization_pct=pool_utilization_pct,
        )

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
            retry_after_seconds=db_circuit_breaker_snapshot.recovery_timeout_seconds,
            circuit_breaker_name=db_circuit_breaker_snapshot.name,
        )

    return BackpressureDecision(should_reject_request=False)
