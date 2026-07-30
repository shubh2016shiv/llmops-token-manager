"""
Queue-depth publisher — the background half of backpressure ("Flow B").

WHAT THIS DOES AND WHY IT IS SEPARATE
-------------------------------------
The request-time evaluator (evaluator.py) needs to know how many token-allocation
jobs are waiting in the RabbitMQ work queue. But it CANNOT measure that cheaply on
every request: the queue lives in the message broker, and asking the broker its
size means a network round-trip — far too slow for a per-request admission check.

So we split the work in two:
    • Flow B (this file)  — a background task runs on a timer, measures the real
                            queue length ONCE, and writes that number into Redis.
    • Flow A (the probe)  — reads that pre-computed number from Redis instantly
                            (see probes/queue_depth.py).

This file is Flow B. It is invoked on a schedule from the token-maintenance task
runner (app/resilience/token_maintenance), NOT from any web request. The two
flows never call each other; they meet only at the shared Redis key defined in
constants.py (QUEUE_DEPTH_REDIS_KEY).

    ┌────────────────────────────┐     ┌────────────────────────────┐
    │ token_maintenance task     │────▶│ publisher.py               │
    │ scheduler (every N secs)   │     │ measure RMQ → write Redis  │
    └────────────────────────────┘     └──────────────┬─────────────┘
              measures │                               │ writes (with TTL)
                       ▼                               ▼
             ┌─────────────────────┐          ┌──────────────────────────┐
             │ RabbitMQ work queue │          │ Redis key                │
             │ (source of truth)   │          │ token_alloc:queue_depth  │
             └─────────────────────┘          └──────────────────────────┘

Author: Engineering Team
Last Updated: 2026-07-23
"""

from __future__ import annotations

# kombu is our RabbitMQ client. `Connection` opens a link to the broker;
# `SimpleQueue.qsize()` reports how many messages are currently sitting in a
# named queue — that count IS our "queue depth" signal.
from kombu import Connection
from kombu.simple import SimpleQueue
from loguru import logger

# settings.* holds the tunable knobs (broker URL, queue name, publish interval).
from app.core.config import settings

# redis_manager.client is the async Redis connection we write the depth into.
from app.core.redis import redis_manager

# Shared names, defined once so Flow A (reader) and Flow B (writer) cannot drift:
#   QUEUE_DEPTH_REDIS_KEY            — the exact Redis key both sides use.
#   QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER — how many publish-intervals the value
#                                        should live before Redis auto-expires it.
from app.resilience.backpressure.constants import (
    QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER,
    QUEUE_DEPTH_REDIS_KEY,
)


async def publish_queue_depth_snapshot() -> None:
    """
    Measure the current RabbitMQ work-queue depth and publish it into Redis.

    This is the single public function of Flow B. It runs in two guarded steps so
    that a failure in either half is logged and swallowed rather than crashing the
    scheduled task — the whole backpressure subsystem is designed to "fail open"
    (a missing signal must never take the service down).

    Step 1 — SAMPLE: read the queue length from the broker.
    Step 2 — PUBLISH: write that length into Redis with a short time-to-live.
    """
    # ---- Step 1: SAMPLE the broker -----------------------------------------
    # If sampling throws (broker unreachable, auth error, etc.) we log and return.
    # We do NOT write anything to Redis in that case; the previously published
    # value is left to expire on its own (see the TTL note in Step 2), after which
    # the reader sees "unknown" rather than a stale lie.
    try:
        queue_depth = _read_current_work_queue_depth()
    except Exception as exc:
        logger.error(f"[BackPressure] Queue depth publish sampling failed: {exc}")
        return

    # ---- Step 2: PUBLISH to Redis ------------------------------------------
    try:
        # TTL = publish interval × multiplier. Example with defaults:
        #   interval = 5s, multiplier = 3  →  TTL = 15s.
        # Meaning: the number stays valid for ~3 publish cycles. If this publisher
        # stops running (crash, deploy, scaling to zero), the key EXPIRES after
        # TTL and the reader treats the gauge as "unknown". That is deliberate:
        # an expired key is safer than a frozen, ever-more-wrong queue depth.
        ttl_seconds = (
            settings.bp_queue_depth_publish_interval_secs
            * QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER
        )
        # `set(..., ex=ttl_seconds)` writes the value AND its expiry atomically.
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
        # A write failure is non-fatal: the old value keeps serving until it
        # expires, and the next scheduled run will try again.
        logger.error(f"[BackPressure] Queue depth publish write failed: {exc}")


def _read_current_work_queue_depth() -> int:
    """
    Open a short-lived broker connection and read the work queue's message count.

    Both `Connection` and the channel it opens are context managers, so combining
    them into one `with` statement guarantees they are always closed — even if an
    error is raised — and we never leak a broker connection from a health probe.
    """
    # 1. Open a connection to the broker. `heartbeat` keeps this (very short-lived)
    #    link healthy for the moment we hold it.
    # 2. A channel is the lightweight session we talk to the queue over.
    with (
        Connection(
            settings.broker_url,
            heartbeat=settings.rabbitmq_token_heartbeat_seconds,
        ) as connection,
        connection.channel() as channel,
    ):
        # 3. Bind to the named work queue and ask how many messages it holds.
        queue = SimpleQueue(channel, settings.rabbitmq_token_work_queue_name)
        # `qsize()` is the raw depth signal; `int(...)` normalizes the type.
        return int(queue.qsize())
