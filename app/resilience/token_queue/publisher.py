"""
Token queue publisher - typed Kombu publishing for work, retry, and DLQ messages.

Architecture:
-------------
    ┌──────────────────────────┐
    │ API / worker callers     │
    └──────────┬───────────────┘
               ▼
    ┌──────────────────────────┐
    │ TokenAllocationPublisher │
    │ validation + CB + pool   │
    └──────────┬───────────────┘
               ▼
    ┌──────────────────────────┐
    │ RabbitMQ exchanges       │
    │ work / retry / DLQ       │
    └──────────────────────────┘

Dependencies:
    - app/models/resilience_models.py - payload validation
    - app/resilience/circuit_breaker - RabbitMQ breaker
    - app/resilience/token_queue/topology.py - queues and pool-backed connection

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from typing import Any
import uuid

from kombu import Producer, Queue, pools
from kombu.exceptions import KombuError
from loguru import logger
import pybreaker

from app.core.config import settings
from app.models.resilience_models import DlqPayload, TokenAllocationPersistPayload
from app.resilience.circuit_breaker import get_rmq_circuit_breaker
from app.resilience.token_queue.topology import (
    TOKEN_ALLOCATION_DLQ,
    TOKEN_ALLOCATION_QUEUE,
    TOKEN_BROKER_CONNECTION,
    TOKEN_DLX,
    TOKEN_EXCHANGE,
    TOKEN_MESSAGE_ID_HEADER,
    TOKEN_RETRY_ATTEMPT_HEADER,
    TOKEN_RETRY_REASON_HEADER,
    get_retry_stage,
)


class TokenAllocationPublisher:
    """Publish typed token allocation messages to RabbitMQ."""

    def __init__(self) -> None:
        """Initialize the publisher with the RabbitMQ circuit breaker."""
        self._rmq_cb = get_rmq_circuit_breaker()

    def publish_allocation_request(
        self, payload: TokenAllocationPersistPayload | dict[str, Any]
    ) -> str:
        """Publish a validated token allocation persistence request."""
        persist_payload = TokenAllocationPersistPayload.model_validate(payload)
        message_id = persist_payload.message_id or persist_payload.token_request_id
        message_payload = persist_payload.model_copy(
            update={"message_id": message_id}
        ).model_dump(mode="json")

        try:
            self._rmq_cb.call(
                self._publish_sync,
                message_payload,
                message_id,
                queue=TOKEN_ALLOCATION_QUEUE,
                routing_key=settings.rabbitmq_token_allocate_routing_key,
                exchange=TOKEN_EXCHANGE,
                headers={TOKEN_RETRY_ATTEMPT_HEADER: 0},
            )
            logger.info(
                f"[TokenQueue] Published allocation request "
                f"msg_id={message_id} model={persist_payload.llm_model_name}"
            )
            return message_id
        except pybreaker.CircuitBreakerError:
            logger.warning(
                f"[TokenQueue] RMQ circuit breaker OPEN - "
                f"caller should use DB fallback for msg_id={message_id}"
            )
            raise

    def publish_retry_request(
        self,
        payload: TokenAllocationPersistPayload | dict[str, Any],
        attempt: int,
        reason: str,
    ) -> None:
        """Publish a failed work message into the next TTL-backed retry stage."""
        persist_payload = TokenAllocationPersistPayload.model_validate(payload)
        retry_stage = get_retry_stage(attempt)
        message_id = (
            persist_payload.message_id
            or persist_payload.token_request_id
            or str(uuid.uuid4())
        )
        retry_payload = persist_payload.model_copy(
            update={"message_id": message_id}
        ).model_dump(mode="json")
        headers = {
            TOKEN_RETRY_ATTEMPT_HEADER: attempt,
            TOKEN_RETRY_REASON_HEADER: reason,
        }

        self._rmq_cb.call(
            self._publish_sync,
            retry_payload,
            message_id,
            queue=retry_stage.queue,
            routing_key=retry_stage.routing_key,
            exchange=TOKEN_EXCHANGE,
            headers=headers,
        )
        logger.warning(
            f"[TokenQueue] Scheduled retry attempt={attempt} "
            f"delay={retry_stage.delay_seconds}s msg_id={message_id}"
        )

    def publish_dlq_notification(
        self,
        payload: TokenAllocationPersistPayload | dict[str, Any],
        reason: str,
        retry_attempts: int = 0,
    ) -> None:
        """Publish a validated dead-letter notification message."""
        persist_payload = TokenAllocationPersistPayload.model_validate(payload)
        message_id = (
            persist_payload.message_id
            or persist_payload.token_request_id
            or str(uuid.uuid4())
        )
        dlq_payload = DlqPayload(
            **persist_payload.model_dump(),
            message_id=message_id,
            dlq_reason=reason,
            dlq_routed_by="explicit",
            retry_attempts=retry_attempts,
        ).model_dump(mode="json")

        try:
            self._rmq_cb.call(
                self._publish_sync,
                dlq_payload,
                message_id,
                queue=TOKEN_ALLOCATION_DLQ,
                routing_key=settings.rabbitmq_token_allocate_dead_routing_key,
                exchange=TOKEN_DLX,
                headers={TOKEN_RETRY_ATTEMPT_HEADER: retry_attempts},
            )
            logger.warning(
                f"[TokenQueue:DLQ] Routed message to DLQ: "
                f"reason={reason} msg_id={message_id}"
            )
        except (
            pybreaker.CircuitBreakerError,
            KombuError,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as exc:
            logger.error(f"[TokenQueue:DLQ] Failed to route to DLQ: {exc}")
            raise

    @staticmethod
    def _publish_sync(
        payload: dict[str, Any],
        message_id: str,
        *,
        queue: Queue,
        routing_key: str,
        exchange: Any,
        headers: dict[str, Any] | None = None,
    ) -> None:
        """Publish a single JSON message using a pooled Kombu connection."""
        publish_headers = {TOKEN_MESSAGE_ID_HEADER: message_id, **(headers or {})}
        with (
            pools.connections[TOKEN_BROKER_CONNECTION].acquire(block=True) as conn,
            conn.channel() as channel,
        ):
            producer = Producer(channel)
            producer.publish(
                payload,
                exchange=exchange,
                routing_key=routing_key,
                declare=[queue],
                retry=True,
                retry_policy={
                    "interval_start": 0,
                    "interval_step": 0.5,
                    "interval_max": 2,
                    "max_retries": settings.rabbitmq_token_queue_delivery_limit,
                },
                content_type="application/json",
                content_encoding="utf-8",
                headers=publish_headers,
                correlation_id=message_id,
                delivery_mode=2,
            )
