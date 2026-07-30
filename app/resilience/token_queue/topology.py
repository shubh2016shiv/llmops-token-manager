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

# Message headers that carry retry state alongside every message body.
TOKEN_RETRY_ATTEMPT_HEADER = "x-token-retry-attempt"  # which attempt this is
TOKEN_RETRY_REASON_HEADER = "x-token-retry-reason"  # why the last attempt failed
TOKEN_MESSAGE_ID_HEADER = "message_id"

# The normal-traffic exchange (the "router" for work + retry messages).
# delivery_mode=2 = persistent: messages survive a broker restart.
TOKEN_EXCHANGE = Exchange(
    settings.rabbitmq_token_exchange_name,
    type=settings.rabbitmq_token_exchange_type,
    durable=True,
    delivery_mode=2,
)

# The dead-letter exchange (DLX): routes terminally-failed messages to the DLQ.
TOKEN_DLX = Exchange(
    settings.rabbitmq_token_dlx_name,
    type=settings.rabbitmq_token_exchange_type,
    durable=True,
    delivery_mode=2,
)

# The WORK queue — where the API publishes allocations to be persisted.
TOKEN_ALLOCATION_QUEUE = Queue(
    settings.rabbitmq_token_work_queue_name,
    exchange=TOKEN_EXCHANGE,
    routing_key=settings.rabbitmq_token_allocate_routing_key,
    durable=True,
    queue_arguments={
        # quorum = replicated across broker nodes -> a node dying loses no messages.
        "x-queue-type": "quorum",
        # A message that sits here too long expires and dead-letters (safety net).
        "x-message-ttl": settings.rabbitmq_token_queue_message_ttl_ms,
        # Where expired / rejected messages go: the DLX -> DLQ.
        "x-dead-letter-exchange": settings.rabbitmq_token_dlx_name,
        "x-dead-letter-routing-key": settings.rabbitmq_token_allocate_dead_routing_key,
        # Quorum queues also route to the DLX after this many failed deliveries.
        "x-delivery-limit": settings.rabbitmq_token_queue_delivery_limit,
    },
)

# The DEAD-LETTER queue — terminal failures land here for the DLQ consumer to
# alert on and compensate for (release the Redis reservation). No retries beyond.
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
    """
    Build one delayed-retry "parking-lot" queue per configured delay.

    THE DELAYED-RETRY TRICK (no scheduler needed):
    Each retry queue has NO consumer. A failed message published here just SITS
    until its `x-message-ttl` expires, at which point RabbitMQ dead-letters it —
    and because the dead-letter target is the MAIN exchange + work routing key, it
    lands back on the work queue for another attempt. So "wait N seconds, then
    retry" is achieved purely by a message expiring in a queue. Elegant and
    scheduler-free.
    """
    retry_stages: list[RetryStage] = []
    # One stage per delay in the configured schedule, e.g. [30, 120, 600] seconds.
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
                # Hold the message for exactly this delay (ms), then expire it...
                "x-message-ttl": delay_seconds * 1000,
                # ...and on expiry, dead-letter it BACK to the work queue to retry.
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
    """Declare token allocation exchanges and queues at startup (idempotent)."""
    try:
        with Connection(
            settings.broker_url,
            heartbeat=settings.rabbitmq_token_heartbeat_seconds,
        ) as conn:
            channel = cast("BrokerConnectionProtocol", conn).channel()
            try:
                # ORDER MATTERS: exchanges first, then queues — a queue binds to an
                # exchange, so the exchange must already exist. (The test suite
                # asserts this exchange-before-queue order.)
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
