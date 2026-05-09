"""
Token queue topology - exchanges, queues, retry stages, and startup declarations.

Architecture:
-------------
    ┌─────────────────────┐
    │ token.allocation    │
    │ direct exchange     │
    └───────┬─────────────┘
            │
            ├── token.allocate ───────────────▶ token.allocation.work
            └── token.allocate.retry.* ───────▶ token.allocation.retry.{delay}s

    Retry queues use TTL and dead-letter back to the work routing key.
    Terminal failures are published to:

    ┌────────────────────────┐
    │ token.allocation.dlx   │
    └──────────┬─────────────┘
               └── token.allocate.dead ───────▶ token.allocation.dlq

Dependencies:
    - app/core/config.py - queue names, limits, heartbeat, retry schedule
    - kombu - exchanges, queues, and pooled broker connections

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from kombu import Connection, Exchange, Queue, pools
from loguru import logger

from app.core.config import settings

TOKEN_RETRY_ATTEMPT_HEADER = "x-token-retry-attempt"
TOKEN_RETRY_REASON_HEADER = "x-token-retry-reason"
TOKEN_MESSAGE_ID_HEADER = "message_id"

TOKEN_EXCHANGE = Exchange(
    settings.rabbitmq_token_exchange_name,
    type=settings.rabbitmq_token_exchange_type,
    durable=True,
    delivery_mode=2,
)

TOKEN_DLX = Exchange(
    settings.rabbitmq_token_dlx_name,
    type=settings.rabbitmq_token_exchange_type,
    durable=True,
    delivery_mode=2,
)

TOKEN_ALLOCATION_QUEUE = Queue(
    settings.rabbitmq_token_work_queue_name,
    exchange=TOKEN_EXCHANGE,
    routing_key=settings.rabbitmq_token_allocate_routing_key,
    durable=True,
    queue_arguments={
        "x-queue-type": "quorum",
        "x-message-ttl": settings.rabbitmq_token_queue_message_ttl_ms,
        "x-dead-letter-exchange": settings.rabbitmq_token_dlx_name,
        "x-dead-letter-routing-key": settings.rabbitmq_token_allocate_dead_routing_key,
        "x-delivery-limit": settings.rabbitmq_token_queue_delivery_limit,
    },
)

TOKEN_ALLOCATION_DLQ = Queue(
    settings.rabbitmq_token_dlq_queue_name,
    exchange=TOKEN_DLX,
    routing_key=settings.rabbitmq_token_allocate_dead_routing_key,
    durable=True,
    queue_arguments={"x-queue-type": "quorum"},
)


@dataclass(frozen=True)
class RetryStage:
    """Metadata for one RabbitMQ TTL-backed retry queue."""

    attempt: int
    delay_seconds: int
    routing_key: str
    queue: Queue


class BrokerChannelProtocol(Protocol):
    """Minimal channel protocol needed for queue declaration lifecycle."""

    def close(self) -> None:
        """Close the broker channel."""
        ...


class BrokerConnectionProtocol(Protocol):
    """Minimal connection protocol needed for queue declaration lifecycle."""

    def channel(self) -> BrokerChannelProtocol:
        """Open a broker channel."""
        ...


def _build_retry_stages() -> tuple[RetryStage, ...]:
    """Build the configured retry-stage queue set."""
    retry_stages: list[RetryStage] = []
    for attempt, delay_seconds in enumerate(
        settings.token_queue_retry_schedule_seconds,
        start=1,
    ):
        queue_name = f"{settings.rabbitmq_token_exchange_name}.retry.{delay_seconds}s"
        routing_key = (
            f"{settings.rabbitmq_token_allocate_routing_key}.retry.{delay_seconds}s"
        )
        queue = Queue(
            queue_name,
            exchange=TOKEN_EXCHANGE,
            routing_key=routing_key,
            durable=True,
            queue_arguments={
                "x-queue-type": "quorum",
                "x-message-ttl": delay_seconds * 1000,
                "x-dead-letter-exchange": settings.rabbitmq_token_exchange_name,
                "x-dead-letter-routing-key": (
                    settings.rabbitmq_token_allocate_routing_key
                ),
            },
        )
        retry_stages.append(
            RetryStage(
                attempt=attempt,
                delay_seconds=delay_seconds,
                routing_key=routing_key,
                queue=queue,
            )
        )
    return tuple(retry_stages)


TOKEN_RETRY_STAGES = _build_retry_stages()
TOKEN_RETRY_QUEUES = tuple(stage.queue for stage in TOKEN_RETRY_STAGES)
ALL_TOKEN_QUEUES = (TOKEN_ALLOCATION_QUEUE, TOKEN_ALLOCATION_DLQ, *TOKEN_RETRY_QUEUES)

TOKEN_BROKER_CONNECTION = Connection(
    settings.broker_url,
    heartbeat=settings.rabbitmq_token_heartbeat_seconds,
)
pools.set_limit(settings.token_queue_connection_pool_limit)


def get_retry_stage(attempt: int) -> RetryStage:
    """Return the configured retry stage for a 1-based attempt number."""
    for stage in TOKEN_RETRY_STAGES:
        if stage.attempt == attempt:
            return stage
    raise ValueError(f"Unknown retry attempt '{attempt}' for token queue")


def get_max_retry_attempts() -> int:
    """Return the configured number of retry stages before DLQ routing."""
    return len(TOKEN_RETRY_STAGES)


def declare_token_queues() -> None:
    """Declare token allocation exchanges and queues at startup."""
    try:
        with Connection(
            settings.broker_url,
            heartbeat=settings.rabbitmq_token_heartbeat_seconds,
        ) as conn:
            channel = cast("BrokerConnectionProtocol", conn).channel()
            try:
                TOKEN_EXCHANGE.declare(channel=channel)
                TOKEN_DLX.declare(channel=channel)
                for queue in ALL_TOKEN_QUEUES:
                    queue.declare(channel=channel)
            finally:
                channel.close()
        queue_names = [queue.name for queue in ALL_TOKEN_QUEUES]
        logger.info(f"[TokenQueue] Queues declared: {queue_names}")
    except Exception as exc:
        logger.error(f"[TokenQueue] Failed to declare queues at startup: {exc}")
        raise
