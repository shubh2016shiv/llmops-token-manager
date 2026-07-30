"""
Scheduler — the single declaration of WHICH maintenance jobs run and HOW OFTEN.

This is the "background world" timetable. It answers one question in one place:
"what periodic jobs exist, what do they do, and at what interval?" A new developer
should be able to read `build_maintenance_schedule()` and understand the entire
background surface of the token manager at a glance.

IMPORTANT — this module is DECLARATION ONLY. It does not start any loops or run any
job by itself. Wiring these jobs into an actual running scheduler (an in-process
asyncio loop started at app startup) is a deliberate, separate step. Keeping the
"what/when" (here) apart from the "actually run it" (a future runner) means this
file stays trivially readable and testable — it's just a list.

Author: Engineering Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.config import settings

# Queue-depth publishing is OWNED by the backpressure package — it writes the Redis
# key that the backpressure guard reads. token_maintenance only decides WHEN it
# runs, so we reference the real function directly here rather than wrapping it in a
# pointless local module. (This is intentional: no indirection, no duplicated logic.)
from app.resilience.backpressure.publisher import publish_queue_depth_snapshot
from app.resilience.token_maintenance.cleanup import cleanup_expired_allocations
from app.resilience.token_maintenance.reconciliation import reconcile_async

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True)
class MaintenanceJob:
    """One periodic background job: what it's called, what to run, how often, why."""

    name: str
    run: Callable[[], Awaitable[object]]
    interval_seconds: int
    purpose: str


def build_maintenance_schedule() -> list[MaintenanceJob]:
    """
    Return the canonical list of maintenance jobs and their intervals.

    Reading this list IS the documentation of the background world:
      • reconciliation      every ~60s  — keep Redis counters honest vs PostgreSQL
      • queue_depth_publish every ~5s   — feed Gate 1 (backpressure) its gauge
      • cleanup             every ~300s — delete expired allocation rows

    Building the list does not start anything; it is the single source of truth a
    runner (or a test) consumes to know what to schedule.
    """
    return [
        MaintenanceJob(
            name="reconciliation",
            run=reconcile_async,
            interval_seconds=settings.reconcile_interval_secs,
            purpose="keep Redis token counters honest against PostgreSQL",
        ),
        MaintenanceJob(
            name="queue_depth_publish",
            run=publish_queue_depth_snapshot,
            interval_seconds=settings.bp_queue_depth_publish_interval_secs,
            purpose="feed Gate 1 (backpressure) its queue-depth gauge",
        ),
        MaintenanceJob(
            name="cleanup",
            run=cleanup_expired_allocations,
            interval_seconds=settings.cleanup_interval_secs,
            purpose="delete expired token-allocation rows from PostgreSQL",
        ),
    ]
