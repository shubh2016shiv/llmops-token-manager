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
from app.persistence.llm_token_allocations import LLMTokenAllocationPersistence
from app.resilience.redis_token_counter import get_shared_redis_token_counter_service

_CONSUMER_LOOP_STATE = threading.local()


def _run_on_consumer_loop(coroutine: Any) -> Any:
    """Run one coroutine on a persistent event loop for the current thread."""
    event_loop = getattr(_CONSUMER_LOOP_STATE, "event_loop", None)
    if event_loop is None or event_loop.is_closed():
        event_loop = asyncio.new_event_loop()
        _CONSUMER_LOOP_STATE.event_loop = event_loop
    return event_loop.run_until_complete(coroutine)


def persist_allocation_message(
    payload: dict[str, Any],
) -> TokenAllocationPersistPayload:
    """Persist one validated work payload to PostgreSQL."""
    persist_payload = TokenAllocationPersistPayload.model_validate(payload)
    _run_on_consumer_loop(_persist_allocation_async(persist_payload))
    return persist_payload


async def _persist_allocation_async(payload: TokenAllocationPersistPayload) -> None:
    """Persist the typed payload using the shared persistence service."""
    persistence = LLMTokenAllocationPersistence()
    await persistence.create_token_allocation(
        token_request_identifier=payload.token_request_id,
        user_id=payload.user_id,
        llm_provider=payload.llm_provider,
        llm_model_name=payload.llm_model_name,
        token_count=payload.token_count,
        api_endpoint_url=payload.api_endpoint_url,
        allocation_status=payload.allocation_status.value,
        deployment_name=payload.deployment_name,
        cloud_provider_name=payload.cloud_provider,
        deployment_region=payload.deployment_region,
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
    """Emit alerting side effects for a DLQ message and reconcile Redis drift."""
    dlq_payload = _normalize_dlq_payload(payload, headers=headers)
    _run_on_consumer_loop(_release_reserved_tokens_async(dlq_payload))
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
        **persist_payload.model_dump(),
        message_id=persist_payload.message_id or persist_payload.token_request_id,
        dlq_reason=reason,
        dlq_routed_by="explicit",
        retry_attempts=retry_attempts,
    )


async def _release_reserved_tokens_async(payload: DlqPayload) -> None:
    """Release the reserved Redis tokens for a terminally failed allocation."""
    shared_token_counter_service = get_shared_redis_token_counter_service()
    released = await shared_token_counter_service.release_tokens(
        payload.llm_model_name,
        payload.api_endpoint_url,
        payload.token_count,
    )
    if released is None:
        logger.warning(
            "[TokenQueue:DLQ] Redis release could not be applied during DLQ handling "
            f"token_request_id={payload.token_request_id}"
        )
