"""
Queue depth publisher - worker-side RabbitMQ telemetry for Layer 1.

Architecture:
-------------
    ┌────────────────────────────┐     ┌────────────────────────────┐
    │ token_worker.py beat task  │────▶│ queue_depth_publisher.py   │
    │ scheduler / dispatcher     │     │ queue depth snapshot write │
    └────────────────────────────┘     └──────────────┬─────────────┘
                                                      │
                     ┌─────────────────────┐          │          ┌──────────────────┐
                     │ RabbitMQ work queue │◀─────────┘─────────▶│ Redis depth key  │
                     └─────────────────────┘                     └──────────────────┘

Dependencies:
    - app/core/config.py - queue names and publish interval
    - app/core/redis.py - Redis client access
    - kombu - queue depth sampling via SimpleQueue.qsize()

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from typing import Protocol, cast

from kombu import Connection
from kombu.simple import SimpleQueue
from loguru import logger

from app.core.config import settings
from app.core.redis import redis_manager
from app.resilience.backpressure.constants import (
    QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER,
    QUEUE_DEPTH_REDIS_KEY,
)


class BrokerChannelContextProtocol(Protocol):
    """Context-manager protocol for the Kombu channel object used here."""

    def __enter__(self) -> object:
        """Enter the channel context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit the channel context."""
        ...


class BrokerConnectionContextProtocol(Protocol):
    """Context-manager protocol for the Kombu connection object used here."""

    def __enter__(self) -> BrokerConnectionContextProtocol:
        """Enter the connection context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit the connection context."""
        ...

    def channel(self) -> BrokerChannelContextProtocol:
        """Open a channel context manager."""
        ...


async def publish_queue_depth_snapshot() -> None:
    """Publish the current RabbitMQ work-queue depth into Redis with a short TTL."""
    try:
        queue_depth = _read_current_work_queue_depth()
    except Exception as exc:
        logger.error(f"[BackPressure] Queue depth publish sampling failed: {exc}")
        return

    try:
        ttl_seconds = (
            settings.bp_queue_depth_publish_interval_secs
            * QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER
        )
        await redis_manager.client.set(
            QUEUE_DEPTH_REDIS_KEY,
            queue_depth,
            ex=ttl_seconds,
        )
        logger.debug(
            f"[BackPressure] Published queue depth={queue_depth} "
            f"ttl={ttl_seconds}s key={QUEUE_DEPTH_REDIS_KEY}"
        )
    except Exception as exc:
        logger.error(f"[BackPressure] Queue depth publish write failed: {exc}")


def _read_current_work_queue_depth() -> int:
    """Read the current RabbitMQ work-queue depth using Kombu's queue helper."""
    connection = cast(
        "BrokerConnectionContextProtocol",
        Connection(settings.broker_url, heartbeat=10),
    )
    with connection:
        channel = connection.channel()
        with channel as active_channel:
            queue = SimpleQueue(active_channel, settings.rabbitmq_token_work_queue_name)
            return int(queue.qsize())
