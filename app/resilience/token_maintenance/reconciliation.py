"""
Token maintenance reconciliation - orchestration for Redis and PostgreSQL drift repair.

Architecture:
-------------
    ┌────────────────────────────────────┐     ┌────────────────────────────────────┐
    │ token_maintenance/tasks.py         │────▶│ reconciliation.py                  │
    │ Celery task wrappers               │     │ drift detection + lock + repair    │
    └────────────────────────────────────┘     └────────────────┬───────────────────┘
                                                                │
                                      ┌─────────────────────────┴────────────────────┐
                                      │ TokenMaintenancePersistence + Redis counter   │
                                      └───────────────────────────────────────────────┘

Dependencies:
    - app/core/config.py - reconciliation interval and warning threshold
    - app/persistence/token_maintenance_persistence.py - authoritative PG snapshots
    - app/resilience/redis_token_counter - shared Redis counter service

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from app.core.config import settings
from app.persistence.token_maintenance_persistence import TokenMaintenancePersistence
from app.resilience.redis_token_counter import (
    CounterReconciliationResult,
    get_shared_redis_token_counter_service,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.models.resilience_models import (
        CounterSeedRecord,
        InvalidActiveDeploymentRecord,
    )

RECONCILIATION_LOCK_KEY = "token:lock:reconcile"
DRIFT_BUCKET_ZERO = "0"
DRIFT_BUCKET_ONE_TO_TEN = "1-10"
DRIFT_BUCKET_ELEVEN_TO_ONE_HUNDRED = "11-100"
DRIFT_BUCKET_ONE_HUNDRED_ONE_TO_ONE_THOUSAND = "101-1000"
DRIFT_BUCKET_LARGER = "1001+"


async def reconcile_async() -> None:
    """Reconcile Redis token counters against PostgreSQL ground truth."""
    persistence = TokenMaintenancePersistence()
    invalid_active_deployments = (
        await persistence.list_invalid_active_models_without_capacity()
    )
    _log_invalid_active_deployments(invalid_active_deployments)

    token_counter_service = get_shared_redis_token_counter_service()
    lock_ttl_seconds = max(1, settings.celery_reconcile_interval_secs - 15)
    lock_acquired = await token_counter_service.redis_client.set(
        RECONCILIATION_LOCK_KEY,
        "1",
        nx=True,
        ex=lock_ttl_seconds,
    )
    if not lock_acquired:
        logger.info(
            "Token maintenance reconciliation skipped"
            " — another run still holds the lock",
            lock_key=RECONCILIATION_LOCK_KEY,
            lock_ttl_seconds=lock_ttl_seconds,
        )
        return

    try:
        seed_records = await persistence.list_active_deployment_capacity_snapshots()
        await _reconcile_seed_records(seed_records)
    finally:
        await token_counter_service.redis_client.delete(RECONCILIATION_LOCK_KEY)


async def _reconcile_seed_records(seed_records: Sequence[CounterSeedRecord]) -> None:
    """Reconcile one authoritative deployment snapshot set into Redis."""
    token_counter_service = get_shared_redis_token_counter_service()
    unchanged_count = 0
    delta_applied_count = 0
    reseeded_partial_count = 0
    initialized_missing_count = 0
    missing_counter_snapshot_count = 0
    drift_bucket_counts: dict[str, int] = {
        DRIFT_BUCKET_ZERO: 0,
        DRIFT_BUCKET_ONE_TO_TEN: 0,
        DRIFT_BUCKET_ELEVEN_TO_ONE_HUNDRED: 0,
        DRIFT_BUCKET_ONE_HUNDRED_ONE_TO_ONE_THOUSAND: 0,
        DRIFT_BUCKET_LARGER: 0,
    }

    for seed_record in seed_records:
        current_counter = await token_counter_service.get_counter(
            seed_record.llm_model_name,
            seed_record.api_endpoint_url,
        )
        if current_counter is None:
            missing_counter_snapshot_count += 1
        else:
            current_allocated, _ = current_counter
            drift_magnitude = abs(seed_record.allocated_tokens - current_allocated)
            _increment_drift_bucket(drift_bucket_counts, drift_magnitude)
            if drift_magnitude >= settings.celery_reconcile_drift_warning_threshold:
                logger.warning(
                    "Token maintenance detected large Redis/PostgreSQL"
                    " drift before correction",
                    llm_model_name=seed_record.llm_model_name,
                    api_endpoint_url=seed_record.api_endpoint_url,
                    postgres_allocated_tokens=seed_record.allocated_tokens,
                    redis_allocated_tokens=current_allocated,
                    drift_magnitude=drift_magnitude,
                )

        reconcile_result = await token_counter_service.reconcile_counter(
            model_name=seed_record.llm_model_name,
            api_endpoint_url=seed_record.api_endpoint_url,
            allocated_tokens_from_db=seed_record.allocated_tokens,
            max_tokens_from_db=seed_record.max_tokens,
        )
        if reconcile_result == CounterReconciliationResult.UNCHANGED:
            unchanged_count += 1
        elif reconcile_result == CounterReconciliationResult.DELTA_APPLIED:
            delta_applied_count += 1
        elif reconcile_result == CounterReconciliationResult.RESEEDED_PARTIAL:
            reseeded_partial_count += 1
        elif reconcile_result == CounterReconciliationResult.INITIALIZED_MISSING:
            initialized_missing_count += 1

    logger.info(
        "Token maintenance reconciliation summary",
        scanned=len(seed_records),
        unchanged=unchanged_count,
        delta_applied=delta_applied_count,
        reseeded_partial=reseeded_partial_count,
        initialized_missing=initialized_missing_count,
        missing_snapshot=missing_counter_snapshot_count,
        drift_buckets=drift_bucket_counts,
    )


def _increment_drift_bucket(
    drift_bucket_counts: dict[str, int],
    drift_magnitude: int,
) -> None:
    """Increment the configured drift histogram bucket for one deployment."""
    if drift_magnitude == 0:
        drift_bucket_counts[DRIFT_BUCKET_ZERO] += 1
    elif drift_magnitude <= 10:
        drift_bucket_counts[DRIFT_BUCKET_ONE_TO_TEN] += 1
    elif drift_magnitude <= 100:
        drift_bucket_counts[DRIFT_BUCKET_ELEVEN_TO_ONE_HUNDRED] += 1
    elif drift_magnitude <= 1000:
        drift_bucket_counts[DRIFT_BUCKET_ONE_HUNDRED_ONE_TO_ONE_THOUSAND] += 1
    else:
        drift_bucket_counts[DRIFT_BUCKET_LARGER] += 1


def _log_invalid_active_deployments(
    invalid_active_deployments: Sequence[InvalidActiveDeploymentRecord],
) -> None:
    """Log invalid active deployment rows before maintenance continues."""
    for invalid_record in invalid_active_deployments:
        logger.error(
            "Active deployment is missing max_tokens and is excluded"
            " from maintenance capacity flows",
            llm_provider=invalid_record.llm_provider,
            llm_model_name=invalid_record.llm_model_name,
            api_endpoint_url=invalid_record.api_endpoint_url,
            deployment_name=invalid_record.deployment_name,
            deployment_region=invalid_record.deployment_region,
        )
