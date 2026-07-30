"""
Queue-depth probe — gauge #1 of the backpressure evaluator.

WHAT THIS MEASURES
------------------
"Queue depth" = how many token-allocation jobs are currently waiting in the
RabbitMQ work queue. A growing backlog is the earliest, broadest sign that the
workers are falling behind and the system is heading toward overload. That is why
the evaluator checks this gauge FIRST.

TWO RESPONSIBILITIES (both tiny, both pure reads)
-------------------------------------------------
    1. read_queue_depth()                 — read the latest depth number.
    2. estimate_queue_retry_after_seconds — turn a depth into a "come back in N
                                            seconds" hint for a rejected client.

WHERE THE NUMBER COMES FROM (the two-flow split)
------------------------------------------------
This probe does NOT contact RabbitMQ. Measuring the broker on every request would
be far too slow. Instead, a background task (publisher.py, "Flow B") measures the
queue every few seconds and writes the number into a Redis key. This probe just
reads that pre-computed number — a single fast Redis GET.

    publisher.py (Flow B) ──writes──▶ Redis[token_alloc:queue_depth]
                                       ◀──reads── this probe

FAIL-OPEN
---------
If Redis is unreachable, the key is missing, or its value is malformed, we return
`None` ("unknown"). The evaluator treats "unknown" as "do not block". A broken
protector must never reject legitimate traffic.

Author: Engineering Team
Last Updated: 2026-07-23
"""

from __future__ import annotations

from loguru import logger

# settings.* holds the tunable thresholds used by the Retry-After estimate below.
from app.core.config import settings

# redis_manager.client is the async Redis connection we read the depth from.
from app.core.redis import redis_manager

# The exact Redis key. Defined once in constants.py so this reader and the
# publisher (writer) can never disagree about the spelling.
from app.resilience.backpressure.constants import QUEUE_DEPTH_REDIS_KEY


async def read_queue_depth() -> int | None:
    """
    Return the latest published work-queue depth, or `None` on any fail-open path.

    Three distinct "unknown" outcomes, all mapped to None so the evaluator can
    uniformly treat them as "don't block":
      • Redis call itself failed   → connection/timeout error.
      • Key is absent              → publisher hasn't run yet, or its value expired.
      • Value present but garbage  → someone wrote a non-integer; ignore it.
    """
    # --- Read attempt: a single Redis GET on the shared key. ---------------
    try:
        raw_queue_depth = await redis_manager.client.get(QUEUE_DEPTH_REDIS_KEY)
    except Exception as exc:
        # Redis unreachable / timing out → fail open.
        logger.error(f"[BackPressure] Queue depth probe failed (fail-open): {exc}")
        return None

    # --- Missing key: the publisher hasn't written yet, or the TTL expired. -
    # An expired key is EXPECTED and healthy behavior (see publisher.py's TTL
    # note): it means "no recent measurement", which we report as unknown.
    if raw_queue_depth is None:
        return None

    # --- Parse: Redis stores strings/bytes; convert to int. ----------------
    try:
        return int(raw_queue_depth)
    except (TypeError, ValueError) as exc:
        # A malformed payload should never reject traffic — log and fail open.
        logger.warning(
            f"[BackPressure] Ignoring malformed queue depth payload "
            f"{raw_queue_depth!r}: {exc}"
        )
        return None


def estimate_queue_retry_after_seconds(queue_depth: int) -> int:
    """
    Estimate how long a rejected client should wait, from the current queue depth.

    THE MODEL: the queue is "healthy" up to a safe depth (a fraction of the max).
    Anything above that is *excess* that must drain before we are comfortable. We
    assume a roughly constant drain rate, so:

        wait ≈ excess_depth / drain_rate,  clamped to [1, cap] seconds.

    Worked example with defaults (max=10000, safe_ratio=0.8, drain=400, cap=60):
        depth = 12000
        safe_depth    = 10000 * 0.8      = 8000
        excess_depth  = 12000 - 8000     = 4000
        drain_seconds = 4000 / 400       = 10
        result        = clamp(10, 1..60) = 10   → "retry after 10 seconds"
    """
    # "Healthy" ceiling: e.g. 80% of the max queue depth.
    safe_depth = int(settings.bp_max_queue_depth * settings.bp_queue_safe_depth_ratio)
    # How far above healthy we are (never negative).
    excess_depth = max(0, queue_depth - safe_depth)
    # Time to drain that excess at the assumed worker drain rate.
    drain_seconds = int(excess_depth / settings.bp_drain_rate_per_second)
    # Clamp: at least 1s (never tell a client "retry in 0s"), at most the config
    # cap (never hand out an absurdly long wait, even for a huge backlog).
    return min(max(1, drain_seconds), settings.bp_retry_after_cap_seconds)
