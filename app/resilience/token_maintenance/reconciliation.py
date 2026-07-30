"""
Reconciliation job — keep the fast Redis token counters honest vs PostgreSQL.

WHAT THIS IS (the crown jewel of token_maintenance)
---------------------------------------------------
The token manager keeps the SAME fact in two places:
  • Redis      — a fast in-memory counter, so it can reserve tokens in ~1ms.
  • PostgreSQL — the durable, authoritative record (the source of truth).

Two copies always drift: a crash between the Redis update and the PG write, a
failed async persist, an expired key. Left alone, the drift compounds until the
counters are simply wrong. This job runs on a timer, reads the PG truth, and
corrects Redis toward it. See ./PRODUCTION_PATTERNS.md §5 for the general pattern.

It is designed to be **idempotent**: it sets Redis TOWARD the PG value ("it should
be 500"), never applies a delta ("add 30"). Running it twice therefore lands on the
same answer — which is what makes it safe to run from any replica under a lock.

Author: Engineering Team
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from app.core.config import settings
from app.persistence.token_maintenance import TokenMaintenancePersistence
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

# The Redis key used as a distributed lock so only ONE replica reconciles per tick
# (see ./PRODUCTION_PATTERNS.md §2 for why this is needed with multiple replicas).
RECONCILIATION_LOCK_KEY = "token:lock:reconcile"

# Drift-histogram buckets. Instead of logging every deployment's exact drift, we
# tally how many fall into each magnitude band — a compact health signal that tells
# you at a glance whether the system is calm (mostly 0) or misbehaving (many large).
DRIFT_BUCKET_ZERO = "0"
DRIFT_BUCKET_ONE_TO_TEN = "1-10"
DRIFT_BUCKET_ELEVEN_TO_ONE_HUNDRED = "11-100"
DRIFT_BUCKET_ONE_HUNDRED_ONE_TO_ONE_THOUSAND = "101-1000"
DRIFT_BUCKET_LARGER = "1001+"


async def reconcile_async() -> None:
    """Reconcile Redis token counters against PostgreSQL ground truth (one run)."""
    persistence = TokenMaintenancePersistence()

    # First surface any active deployments that are misconfigured (missing capacity);
    # these are logged and excluded so they can't corrupt the capacity math.
    invalid_active_deployments = (
        await persistence.list_invalid_active_models_without_capacity()
    )
    _log_invalid_active_deployments(invalid_active_deployments)

    token_counter_service = get_shared_redis_token_counter_service()

    # --- Distributed lock: run once across all replicas ---------------------
    # `set(key, "1", nx=True, ex=ttl)` = "create this key ONLY if it doesn't exist,
    # and auto-expire it after ttl seconds." Redis makes this atomic, so when N
    # replicas fire at once, exactly one gets True and proceeds; the rest get None
    # and skip this tick. The TTL is set BELOW the run interval so the lock is always
    # released (even if the holder crashes) before the next scheduled run.
    lock_ttl_seconds = max(1, settings.reconcile_interval_secs - 15)
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

    # `finally` guarantees we release the lock even if reconciliation raises, so a
    # crash mid-run doesn't block the next run for the full TTL.
    try:
        seed_records = await persistence.list_active_deployment_capacity_snapshots()
        await _reconcile_seed_records(seed_records)
    finally:
        await token_counter_service.redis_client.delete(RECONCILIATION_LOCK_KEY)


async def _reconcile_seed_records(seed_records: Sequence[CounterSeedRecord]) -> None:
    """Reconcile one authoritative deployment snapshot set into Redis."""
    token_counter_service = get_shared_redis_token_counter_service()

    # Counters for the end-of-run summary log (how much work each category did).
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
        # 1. Read what Redis currently thinks is allocated for this deployment.
        current_counter = await token_counter_service.get_counter(
            seed_record.llm_model_name,
            seed_record.api_endpoint_url,
        )
        if current_counter is None:
            # Redis has no counter yet (fresh/expired) — nothing to measure drift
            # against; the reconcile step below will initialize it.
            missing_counter_snapshot_count += 1
        else:
            # 2. Measure drift = |PG truth − Redis value| and bucket it. A large
            #    drift is logged individually as a warning so it's actionable.
            current_allocated, _ = current_counter
            drift_magnitude = abs(seed_record.allocated_tokens - current_allocated)
            _increment_drift_bucket(drift_bucket_counts, drift_magnitude)
            if drift_magnitude >= settings.reconcile_drift_warning_threshold:
                logger.warning(
                    "Token maintenance detected large Redis/PostgreSQL"
                    " drift before correction",
                    llm_model_name=seed_record.llm_model_name,
                    api_endpoint_url=seed_record.api_endpoint_url,
                    postgres_allocated_tokens=seed_record.allocated_tokens,
                    redis_allocated_tokens=current_allocated,
                    drift_magnitude=drift_magnitude,
                )

        # 3. Repair: push Redis toward the PG truth. This is the idempotent write —
        #    it targets the correct VALUE, so re-running lands on the same result.
        #    The service reports what it did (unchanged / delta / reseed / init).
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

    # One structured summary line per run — this IS the job's observability.
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
    """Increment the drift-histogram bucket that this deployment's drift falls into."""
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
