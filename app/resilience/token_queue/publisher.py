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

import aiobreaker
from amqp.exceptions import AMQPError
from kombu import Producer, Queue, pools
from kombu.exceptions import KombuError
from loguru import logger

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


class TokenPublishError(Exception):
    """The allocation could not be handed to RabbitMQ; use the DB fallback."""


class TokenAllocationPublisher:
    """
    Publish typed token allocation messages to RabbitMQ.

    Three publish paths, all guarded by the RabbitMQ circuit breaker:
      - publish_allocation_request : the hot-path handoff (work queue)
      - publish_retry_request      : a failed message into the next delayed stage
      - publish_dlq_notification   : a terminally-failed message into the DLQ
    """

    def __init__(self) -> None:
        """Initialize the publisher with the RabbitMQ circuit breaker."""
        # The shared 'rmq' breaker (from the circuit_breaker module). Every publish
        # runs through it, so if RabbitMQ is down we fail fast instead of hanging.
        self._rmq_cb = get_rmq_circuit_breaker()

    def publish_allocation_request(
        self, payload: TokenAllocationPersistPayload | dict[str, Any]
    ) -> str:
        """Publish a validated token allocation persistence request (the handoff)."""
        # 1. Validate the payload and pin a stable message_id (falls back to the
        #    token_request_id) so retries/DLQ can be correlated to this request.
        persist_payload = TokenAllocationPersistPayload.model_validate(payload)
        message_id = persist_payload.message_id or persist_payload.token_request_id
        message_payload = persist_payload.model_copy(
            update={"message_id": message_id}
        ).model_dump(mode="json")

        try:
            # 2. Publish through the breaker. attempt header = 0 (first delivery).
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
                f"msg_id={message_id} model={persist_payload.model_name}"
            )
            return message_id
        except aiobreaker.CircuitBreakerError:
            # 3. Breaker OPEN = RabbitMQ unhealthy. We re-raise so the caller
            #    (token_acquisition_service) can fall back to a SYNCHRONOUS DB write
            #    instead of losing the allocation.
            logger.warning(
                f"[TokenQueue] RMQ circuit breaker OPEN - "
                f"caller should use DB fallback for msg_id={message_id}"
            )
            raise
        except (AMQPError, KombuError, ConnectionError, TimeoutError, OSError) as exc:
            logger.warning(
                "[TokenQueue] RabbitMQ publish failed; caller should use DB fallback "
                f"for msg_id={message_id}: {exc}"
            )
            raise TokenPublishError("RabbitMQ allocation publish failed") from exc

    def publish_retry_request(
        self,
        payload: TokenAllocationPersistPayload | dict[str, Any],
        attempt: int,
        reason: str,
    ) -> None:
        """
        Publish a failed work message into the next TTL-backed retry stage.

        The message is sent to the `retry.{delay}s` parking-lot queue; it will sit
        there for the delay, then dead-letter back to the work queue (see
        topology._build_retry_stages). `attempt` selects which delay stage to use.
        """
        persist_payload = TokenAllocationPersistPayload.model_validate(payload)
        # Pick the retry queue/routing-key for this attempt number.
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

        try:
            self._rmq_cb.call(
                self._publish_sync,
                retry_payload,
                message_id,
                queue=retry_stage.queue,
                routing_key=retry_stage.routing_key,
                exchange=TOKEN_EXCHANGE,
                headers=headers,
            )
        except aiobreaker.CircuitBreakerError:
            # Deliberately NOT logged here (unlike the branch below): the current
            # caller (consumer._on_work_message) treats an open breaker as its own
            # distinct case — pause-and-requeue rather than log-and-requeue — and
            # logs it there with that context. Re-raise bare so that branch still
            # matches on the exact exception type.
            raise
        except (AMQPError, KombuError, ConnectionError, TimeoutError, OSError) as exc:
            # Same exception classes publish_allocation_request and
            # publish_dlq_notification already handle — added here for the same
            # reason: a caller other than the current consumer (which happens to
            # catch bare Exception around this call) would otherwise see an
            # unlabeled raw exception instead of a logged, attributable failure.
            logger.warning(
                f"[TokenQueue] Retry publish failed for attempt={attempt} "
                f"msg_id={message_id}: {exc}"
            )
            raise
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
            aiobreaker.CircuitBreakerError,
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
        """
        Publish a single JSON message using a pooled Kombu connection.

        This is the low-level "put bytes on the broker" step shared by all three
        publish paths. It runs synchronously inside the circuit breaker's `.call`.
        """
        publish_headers = {TOKEN_MESSAGE_ID_HEADER: message_id, **(headers or {})}
        # Borrow a connection from the shared pool (cheap) and open a channel on it.
        # Both are returned/closed automatically by the `with` block.
        with pools.connections[TOKEN_BROKER_CONNECTION].acquire(block=True) as conn:
            # A pooled connection may have been closed by an earlier failed
            # publish. Reconnect it before opening a channel.
            conn.ensure_connection(max_retries=1)
            with conn.channel() as channel:
                producer = Producer(channel)
                producer.publish(
                    payload,
                    exchange=exchange,
                    routing_key=routing_key,
                    declare=[queue],  # ensure the target queue exists before publishing
                    retry=True,  # retry the PUBLISH itself on a transient broker hiccup
                    retry_policy={
                        "interval_start": 0,
                        "interval_step": 0.5,
                        "interval_max": 2,
                        "max_retries": settings.rabbitmq_token_queue_delivery_limit,
                    },
                    serializer="json",
                    headers=publish_headers,
                    correlation_id=message_id,
                    delivery_mode=2,  # persistent across broker restarts
                )
