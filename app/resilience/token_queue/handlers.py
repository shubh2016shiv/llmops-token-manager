"""
Token queue handlers - persistence and DLQ side effects for the raw consumer.

Architecture:
-------------
    ┌──────────────────────────┐
    │ consumer.py callbacks    │
    └──────────┬───────────────┘
               ├──────────────▶ persist work payload to PostgreSQL
               └──────────────▶ alert, reconcile, and log DLQ payloads

Dependencies:
    - app/models/resilience_models.py - payload validation
    - app/persistence/llm_token_allocations.py - durable writes
    - app/resilience/redis_token_counter - counter reconciliation side effects

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from loguru import logger

from app.models.resilience_models import DlqPayload, TokenAllocationPersistPayload
from app.persistence.allocations import LLMTokenAllocationPersistence
from app.resilience.redis_token_counter import get_shared_redis_token_counter_service

# The consumer runs in plain (sync) Kombu callbacks, but our persistence code is
# async. We keep ONE event loop per thread and reuse it, rather than spinning up a
# fresh loop per message (which would be wasteful and lose connection state).
_CONSUMER_LOOP_STATE = threading.local()


def _run_on_consumer_loop(coroutine: Any) -> Any:
    """Run one coroutine on a persistent per-thread event loop (sync bridge)."""
    event_loop = getattr(_CONSUMER_LOOP_STATE, "event_loop", None)
    if event_loop is None or event_loop.is_closed():
        event_loop = asyncio.new_event_loop()
        _CONSUMER_LOOP_STATE.event_loop = event_loop
    return event_loop.run_until_complete(coroutine)


def persist_allocation_message(
    payload: dict[str, Any],
) -> TokenAllocationPersistPayload:
    """Persist one validated work payload to PostgreSQL (the happy-path write)."""
    # Validate the raw dict into the typed contract, then run the async DB write on
    # this thread's loop. Any exception propagates up to the consumer's retry logic.
    persist_payload = TokenAllocationPersistPayload.model_validate(payload)
    _run_on_consumer_loop(_persist_allocation_async(persist_payload))
    return persist_payload


async def _persist_allocation_async(payload: TokenAllocationPersistPayload) -> None:
    """Persist the typed fast-path payload (capacity already reserved in Redis)."""
    # The consumer runs as its own process/loop (no FastAPI lifespan), so ensure
    # the shared DB engine is bound to this loop. initialize() is idempotent.
    from app.core.database import db_manager

    await db_manager.initialize()
    persistence = LLMTokenAllocationPersistence()
    await persistence.create_reserved_allocation(
        token_request_identifier=payload.token_request_id,
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
        deployment_id=payload.deployment_id,
        provider_name=payload.provider_name,
        model_name=payload.model_name,
        deployment_key=payload.deployment_key,
        api_endpoint_url=payload.api_endpoint_url,
        token_count=payload.token_count,
        allocation_status=payload.allocation_status.value,
        deployment_name=payload.deployment_name,
        provider_deployment_name=payload.provider_deployment_name,
        cloud_provider=payload.cloud_provider,
        cloud_region=payload.cloud_region,
        request_metadata=payload.request_context,
        temperature=payload.temperature,
        top_p=payload.top_p,
        seed=payload.seed,
        expiration_timestamp=payload.expires_at,
    )


def process_dlq_alert(
    payload: dict[str, Any],
    *,
    headers: dict[str, Any] | None,
) -> DlqPayload:
    """
    Handle a terminally-failed message: COMPENSATE, then ALERT.

    Two responsibilities:
      1. Release the Redis tokens that were reserved for this allocation. It never
         persisted, so those tokens must be given back or Redis leaks capacity.
      2. Log loudly (critical) with the full payload so a human can investigate.
    """
    # A message can arrive here two ways: explicitly published to the DLQ by us, or
    # broker-dead-lettered (which lacks our enrichment). Normalize both shapes.
    dlq_payload = _normalize_dlq_payload(payload, headers=headers)
    # 1. COMPENSATE: undo the Redis reservation for this failed allocation.
    _run_on_consumer_loop(_release_reserved_tokens_async(dlq_payload))
    # 2. ALERT: this needs manual review.
    logger.critical(
        "[TokenQueue:DLQ] Manual review required "
        f"token_request_id={dlq_payload.token_request_id} "
        f"message_id={dlq_payload.message_id} "
        f"reason={dlq_payload.dlq_reason} "
        f"retry_attempts={dlq_payload.retry_attempts} "
        f"headers={headers or {}}"
    )
    logger.error(
        "[TokenQueue:DLQ] Full payload for manual review "
        f"payload={dlq_payload.model_dump(mode='json')}"
    )
    return dlq_payload


def _normalize_dlq_payload(
    payload: dict[str, Any],
    *,
    headers: dict[str, Any] | None,
) -> DlqPayload:
    """Build a DLQ payload even when broker-routed messages lack enrichment."""
    if "dlq_reason" in payload:
        return DlqPayload.model_validate(payload)

    persist_payload = TokenAllocationPersistPayload.model_validate(payload)
    retry_attempts = int((headers or {}).get("x-token-retry-attempt", 0))
    reason = str((headers or {}).get("x-token-retry-reason", "broker_dead_lettered"))
    return DlqPayload(
        **persist_payload.model_dump(exclude={"message_id"}),
        message_id=persist_payload.message_id or persist_payload.token_request_id,
        dlq_reason=reason,
        dlq_routed_by="explicit",
        retry_attempts=retry_attempts,
    )


async def _release_reserved_tokens_async(payload: DlqPayload) -> None:
    """
    Release the reserved Redis tokens for a terminally failed allocation.

    This calls back into the redis_token_counter fast path (release_tokens), closing
    the loop: Layer 3 reserved the tokens, and here Layer 4 gives them back when the
    durable write never happened. `None` means the release couldn't be applied
    (e.g. Redis breaker open) — logged for follow-up, not retried here.
    """
    shared_token_counter_service = get_shared_redis_token_counter_service()
    released = await shared_token_counter_service.release_tokens(
        payload.model_name,
        payload.api_endpoint_url,
        payload.token_count,
    )
    if released is None:
        logger.warning(
            "[TokenQueue:DLQ] Redis release could not be applied during DLQ handling "
            f"token_request_id={payload.token_request_id}"
        )
