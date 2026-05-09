"""
Pool probe - Layer 1 database pool saturation signal.

Architecture:
-------------
    ┌───────────────────┐     ┌──────────────────────┐
    │ evaluator.py      │────▶│ pool_probe.py        │
    │ Layer 1 ordering  │     │ public pool accessor │
    └───────────────────┘     └────────────┬─────────┘
                                           │
                                           ▼
                                  ┌─────────────────────┐
                                  │ app/core/database.py│
                                  │ db_manager.pool     │
                                  └─────────────────────┘

Dependencies:
    - app/core/database.py - public pool accessor

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from typing import Protocol, cast

from loguru import logger

from app.core.database import db_manager


class DatabasePoolStatsProtocol(Protocol):
    """Minimal SQLAlchemy pool stats interface used by Layer 1 backpressure."""

    def size(self) -> int:
        """Return the configured pool size."""
        ...

    def checkedout(self) -> int:
        """Return the number of currently checked-out connections."""
        ...


def read_db_pool_utilization_pct() -> int | None:
    """Return DB pool utilization percent, or `None` when unavailable."""
    try:
        raw_pool = db_manager.pool
        if raw_pool is None:
            return None

        pool = cast("DatabasePoolStatsProtocol", raw_pool)
        pool_size = pool.size()
        checked_out_connections = pool.checkedout()
        if pool_size <= 0:
            return None

        return int((checked_out_connections / pool_size) * 100)
    except Exception as exc:
        logger.error(f"[BackPressure] Pool probe failed (fail-open): {exc}")
        return None
