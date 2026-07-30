"""
DB connection-pool probe — gauge #2 of the backpressure evaluator.

WHAT THIS MEASURES
------------------
Every database query needs a connection from a fixed-size pool. When almost all
connections are already checked out, new work has to queue up waiting for one to
free — a classic early sign of saturation. This probe reports pool pressure as a
single percentage (0–100): what fraction of the pool is currently in use.

    utilization% = (connections currently checked out / total pool size) × 100

The evaluator compares that number against `settings.bp_db_pool_saturation_pct`
(default 90) and rejects the request if we are at/above it.

THE ONE RULE EVERY PROBE FOLLOWS: FAIL OPEN
-------------------------------------------
If this probe cannot read the pool for ANY reason, it returns `None`, meaning
"I don't know." The evaluator treats "I don't know" as "do not block." We would
rather occasionally miss saturation than let a bug in the *protector* reject
legitimate traffic. So every failure path here returns `None`, never raises.

    ┌───────────────────┐     ┌──────────────────────────────┐
    │ evaluator.py      │────▶│ probes/db_pool.py            │
    │ (gauge #2 check)  │     │ read pool → utilization %    │
    └───────────────────┘     └──────────────┬───────────────┘
                                             │ reads live stats
                                             ▼
                                   ┌─────────────────────┐
                                   │ app/core/database.py│
                                   │ db_manager.pool     │
                                   └─────────────────────┘

Author: Engineering Team
Last Updated: 2026-07-23
"""

from __future__ import annotations

from loguru import logger

# db_manager.pool is the live SQLAlchemy connection pool. Its `.size()` and
# `.checkedout()` methods give the two numbers we need to compute utilization.
from app.core.database import db_manager


def read_db_pool_utilization_pct() -> int | None:
    """
    Return DB pool utilization as an integer percent, or None if unavailable.

    Percent is 0-100. Returns None if the pool is unavailable or unreadable
    (fail-open — see module docstring).
    """
    try:
        # The pool may legitimately be absent (e.g. before startup wiring, or in
        # a process that never opened a DB pool). "No pool" → "unknown" → None.
        pool = db_manager.pool
        if pool is None:
            return None

        # Read the two live counters straight off the SQLAlchemy pool:
        #   size()       → total connections the pool is configured to hold.
        #   checkedout() → how many of those are currently in use.
        pool_size = pool.size()
        checked_out_connections = pool.checkedout()

        # Guard against a zero/negative size: dividing by it would raise, and a
        # pool that reports size 0 tells us nothing useful → treat as "unknown".
        if pool_size <= 0:
            return None

        # Compute the percentage and truncate to an int (e.g. 7/10 → 70).
        return int((checked_out_connections / pool_size) * 100)
    except Exception as exc:
        # ANY failure reading the pool → log and fail open with None.
        logger.error(f"[BackPressure] Pool probe failed (fail-open): {exc}")
        return None
