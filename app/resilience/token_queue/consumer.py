"""
Token queue consumer - long-running raw Kombu consumer for work and DLQ traffic.

Architecture:
-------------
    ┌──────────────────────────┐
    │ token_queue consumer     │
    │ ConsumerMixin loop       │
    └──────────┬───────────────┘
               ├──────────────▶ work queue callback
               │               persist or schedule retry / DLQ
               └──────────────▶ DLQ callback
                               alert + reconcile + manual review logging

Dependencies:
    - app/resilience/token_queue/handlers.py - business side effects
    - app/resilience/token_queue/publisher.py - retry and DLQ publishing
    - app/resilience/token_queue/topology.py - queue declarations and connection

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from typing import TYPE_CHECKING, Any

import aiobreaker
from kombu.mixins import ConsumerMixin
from loguru import logger

from app.core.config import settings
from app.resilience.token_queue.handlers import (
    persist_allocation_message,
    process_dlq_alert,
)
from app.resilience.token_queue.publisher import TokenAllocationPublisher
from app.resilience.token_queue.topology import (
    TOKEN_ALLOCATION_DLQ,
    TOKEN_ALLOCATION_QUEUE,
    TOKEN_BROKER_CONNECTION,
    TOKEN_RETRY_ATTEMPT_HEADER,
    get_max_retry_attempts,
)

if TYPE_CHECKING:
    from kombu import Consumer


class TokenQueueConsumerService(ConsumerMixin):
    """Run the Layer 4 raw RabbitMQ consumer for work and DLQ queues."""

    def __init__(self, *, prefetch_count: int | None = None) -> None:
        """Construct the raw consumer service with a shared publisher."""
        self.connection = TOKEN_BROKER_CONNECTION.clone()
        self._publisher = TokenAllocationPublisher()
        self._prefetch_count = (
            prefetch_count
            if prefetch_count is not None
            else settings.token_queue_consumer_prefetch_count
        )

    def get_consumers(
        self, consumer_cls: type[Consumer], channel: object
    ) -> list[Consumer]:
        """
        Register two consumers: one for the work queue, one for the DLQ.

        `consumer_cls` is already bound to the channel by ConsumerMixin, so the
        channel must NOT be passed again. Prefetch (QoS) is applied on each
        consumer rather than passed to the constructor.
        """
        # Work consumer: pull up to `prefetch_count` unacked messages at once
        # (higher throughput). Each message runs through _on_work_message.
        work_consumer = consumer_cls(
            queues=[TOKEN_ALLOCATION_QUEUE],
            callbacks=[self._on_work_message],
            accept=["json"],
        )
        work_consumer.qos(prefetch_count=self._prefetch_count)

        # DLQ consumer: prefetch 1 — terminal failures are rare and each does
        # alerting/compensation, so we handle them one at a time.
        dlq_consumer = consumer_cls(
            queues=[TOKEN_ALLOCATION_DLQ],
            callbacks=[self._on_dlq_message],
            accept=["json"],
        )
        dlq_consumer.qos(prefetch_count=1)

        return [work_consumer, dlq_consumer]

    def _on_work_message(self, body: dict[str, Any], message: Any) -> None:
        """
        Persist a work message, or transition it to retry / DLQ. THE decision tree.

        Outcomes (each ends by ack-ing or requeueing the message so nothing is lost):
          • persist succeeds            -> ack (done)
          • persist fails, retries left -> publish to next retry stage, then ack
          • persist fails, retries out  -> publish to DLQ, then ack
          • can't publish (breaker open)-> backoff + requeue (try again later)
        """
        # How many times has this message already been attempted? (0 on first delivery)
        retry_attempt = int((message.headers or {}).get(TOKEN_RETRY_ATTEMPT_HEADER, 0))
        try:
            # --- Happy path: write to PostgreSQL and acknowledge. ---
            persist_payload = persist_allocation_message(body)
            logger.info(
                "[TokenQueue] Persisted allocation "
                f"token_request_id={persist_payload.token_request_id} "
                f"retry_attempt={retry_attempt}"
            )
            message.ack()  # tell RabbitMQ it's safely handled -> drop it from the queue
        except Exception as exc:
            logger.error(
                "[TokenQueue] Persistence failure "
                f"retry_attempt={retry_attempt} error={exc}"
            )
            # --- Failure branch A: still have retry stages left -> schedule retry. ---
            if retry_attempt < get_max_retry_attempts():
                next_attempt = retry_attempt + 1
                try:
                    # Send to the next delayed retry queue (it comes back later).
                    self._publisher.publish_retry_request(
                        body,
                        attempt=next_attempt,
                        reason=str(exc),
                    )
                except aiobreaker.CircuitBreakerError as publish_exc:
                    # Broker breaker open -> can't publish the retry. Don't lose the
                    # message: pause and requeue so it's redelivered later.
                    self._backoff_requeue(
                        message=message,
                        retry_attempt=next_attempt,
                        error=publish_exc,
                    )
                    return
                except Exception as publish_exc:
                    # Any other publish error -> requeue this original message.
                    logger.error(
                        "[TokenQueue] Retry publish failed; requeueing work message "
                        f"attempt={next_attempt} error={publish_exc}"
                    )
                    message.reject(requeue=True)
                    return
                # Retry safely enqueued -> ack THIS delivery (its clone lives on).
                message.ack()
                return

            # --- Failure branch B: retries exhausted -> route to the DLQ. ---
            try:
                self._publisher.publish_dlq_notification(
                    body,
                    reason=str(exc),
                    retry_attempts=retry_attempt,
                )
            except aiobreaker.CircuitBreakerError as publish_exc:
                # Same "don't lose it" handling if the breaker blocks the DLQ publish.
                self._backoff_requeue(
                    message=message,
                    retry_attempt=retry_attempt,
                    error=publish_exc,
                )
                return
            except Exception as publish_exc:
                logger.error(
                    "[TokenQueue] DLQ publish failed; requeueing work message "
                    f"retry_attempt={retry_attempt} error={publish_exc}"
                )
                message.reject(requeue=True)
                return
            # DLQ notification enqueued -> ack this delivery.
            message.ack()

    def _on_dlq_message(self, body: dict[str, Any], message: Any) -> None:
        """
        Process terminal DLQ side effects and acknowledge the message.

        process_dlq_alert both ALERTS a human and RELEASES the Redis reservation
        for this terminally-failed allocation (so capacity isn't leaked). If that
        handling itself fails, requeue so we don't drop a message that still needs
        its compensation applied.
        """
        try:
            process_dlq_alert(body, headers=message.headers or {})
        except Exception as exc:
            logger.exception(f"[TokenQueue:DLQ] Alert handling failed: {exc}")
            message.reject(requeue=True)
            return
        message.ack()

    def _backoff_requeue(
        self,
        *,
        message: Any,
        retry_attempt: int,
        error: aiobreaker.CircuitBreakerError,
    ) -> None:
        """Pause briefly before requeueing when the broker breaker is open."""
        breaker_state = getattr(self._publisher._rmq_cb, "current_state", "unknown")
        logger.warning(
            "[TokenQueue] Broker circuit breaker blocked retry/DLQ publish; "
            f"requeueing after backoff attempt={retry_attempt} state={breaker_state} "
            f"error={error}"
        )
        time.sleep(settings.token_queue_consumer_requeue_backoff_seconds)
        message.reject(requeue=True)


def run_token_queue_consumer(
    *,
    concurrency: int | None = None,
    prefetch_count: int | None = None,
) -> None:
    """Declare queues and start one or more long-running raw queue consumers."""
    from app.resilience.token_queue.topology import declare_token_queues

    process_count = (
        concurrency
        if concurrency is not None
        else settings.token_queue_consumer_concurrency
    )
    resolved_prefetch_count = (
        prefetch_count
        if prefetch_count is not None
        else settings.token_queue_consumer_prefetch_count
    )
    declare_token_queues()
    if process_count <= 1:
        logger.info(
            "[TokenQueue] Starting raw queue consumer "
            f"prefetch={resolved_prefetch_count}"
        )
        TokenQueueConsumerService(prefetch_count=resolved_prefetch_count).run()
        return

    logger.info(
        "[TokenQueue] Starting raw queue consumer pool "
        f"processes={process_count} prefetch={resolved_prefetch_count}"
    )
    process_context = multiprocessing.get_context("spawn")
    worker_processes = [
        process_context.Process(
            target=_run_single_consumer_process,
            args=(resolved_prefetch_count, worker_index),
        )
        for worker_index in range(process_count)
    ]
    for worker_process in worker_processes:
        worker_process.start()

    try:
        for worker_process in worker_processes:
            worker_process.join()
    except KeyboardInterrupt:
        logger.warning("[TokenQueue] Interrupt received; stopping consumer pool")
        for worker_process in worker_processes:
            worker_process.terminate()
        for worker_process in worker_processes:
            worker_process.join()


def _run_single_consumer_process(prefetch_count: int, worker_index: int) -> None:
    """Start one raw queue consumer process."""
    logger.info(
        "[TokenQueue] Consumer worker starting "
        f"index={worker_index} prefetch={prefetch_count}"
    )
    TokenQueueConsumerService(prefetch_count=prefetch_count).run()


def _parse_cli_arguments() -> argparse.Namespace:
    """Parse CLI overrides for token queue consumer process layout."""
    parser = argparse.ArgumentParser(description="Run the token queue consumer")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override the number of consumer processes to spawn",
    )
    parser.add_argument(
        "--prefetch-count",
        type=int,
        default=None,
        help="Override the per-process work-queue prefetch count",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_arguments = _parse_cli_arguments()
    run_token_queue_consumer(
        concurrency=cli_arguments.concurrency,
        prefetch_count=cli_arguments.prefetch_count,
    )
