"""
Token maintenance package — the background world.

While the rest of the token manager runs per-request, this package runs on timers:
it keeps the fast Redis token counters honest (reconciliation), feeds the
backpressure queue-depth gauge, and deletes expired rows (cleanup). See README.md
in this folder for the full picture and PRODUCTION_PATTERNS.md for the concepts.

    reconciliation.py  — keep Redis counters honest vs PostgreSQL (the crown jewel)
    cleanup.py         — delete expired allocation rows
    scheduler.py       — the schedule: which jobs run, how often (declaration only)
    health.py          — readiness reporting for startup / the /health endpoint

(Queue-depth publishing itself lives in backpressure/publisher.py; this package
only schedules it.)
"""

from __future__ import annotations

from app.resilience.token_maintenance.cleanup import cleanup_expired_allocations
from app.resilience.token_maintenance.health import (
    inspect_token_maintenance_runtime,
    verify_token_maintenance_readiness,
)
from app.resilience.token_maintenance.reconciliation import reconcile_async
from app.resilience.token_maintenance.scheduler import (
    MaintenanceJob,
    build_maintenance_schedule,
)

__all__ = [
    "reconcile_async",
    "cleanup_expired_allocations",
    "MaintenanceJob",
    "build_maintenance_schedule",
    "verify_token_maintenance_readiness",
    "inspect_token_maintenance_runtime",
]
