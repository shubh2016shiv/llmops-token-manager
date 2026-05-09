"""
Queue depth probe - Redis-backed Layer 1 queue saturation signal.

Architecture:
-------------
    ┌───────────────────┐     ┌──────────────────────────┐
    │ evaluator.py      │────▶│ queue_depth_probe.py     │
    │ Layer 1 ordering  │     │ Redis queue-depth reader │
    └───────────────────┘     └─────────────┬────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ app/core/redis.py   │
                                  │ published queue key │
                                  └─────────────────────┘

Dependencies:
    - app/core/config.py - queue thresholds and retry tuning
    - app/core/redis.py - Redis client access

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings
from app.core.redis import redis_manager
from app.resilience.backpressure.constants import QUEUE_DEPTH_REDIS_KEY


async def read_queue_depth() -> int | None:
    """Return the latest published work-queue depth, or `None` on fail-open paths."""
    try:
        raw_queue_depth = await redis_manager.client.get(QUEUE_DEPTH_REDIS_KEY)
    except Exception as exc:
        logger.error(f"[BackPressure] Queue depth probe failed (fail-open): {exc}")
        return None

    if raw_queue_depth is None:
        return None

    try:
        return int(raw_queue_depth)
    except (TypeError, ValueError) as exc:
        logger.warning(
            f"[BackPressure] Ignoring malformed queue depth payload "
            f"{raw_queue_depth!r}: {exc}"
        )
        return None


def estimate_queue_retry_after_seconds(queue_depth: int) -> int:
    """Estimate queue drain time using config-backed thresholds and caps."""
    safe_depth = int(settings.bp_max_queue_depth * settings.bp_queue_safe_depth_ratio)
    excess_depth = max(0, queue_depth - safe_depth)
    drain_seconds = int(excess_depth / settings.bp_drain_rate_per_second)
    return min(max(1, drain_seconds), settings.bp_retry_after_cap_seconds)
