"""
Token maintenance schedule registry - beat metadata and route ownership for Layer 4.

Architecture:
-------------
    ┌────────────────────────────────────┐     ┌────────────────────────────────────┐
    │ llm_client_request_queue.py        │────▶│ schedule_registry.py               │
    │ Celery app ownership               │     │ pure route/schedule metadata       │
    └────────────────────────────────────┘     └────────────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ token_maintenance/tasks.py                                                  │
    └──────────────────────────────────────────────────────────────────────────────┘

Dependencies:
    - app/core/config.py - queue name and periodic intervals
    - app/resilience/backpressure/constants.py - queue-depth task identifiers

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.resilience.backpressure.constants import (
    QUEUE_DEPTH_PUBLISH_SCHEDULE_NAME,
    QUEUE_DEPTH_PUBLISH_TASK_NAME,
)

RECONCILE_TASK_NAME = "app.resilience.token_maintenance.reconcile_redis_postgres"
CLEANUP_TASK_NAME = "app.resilience.token_maintenance.cleanup_expired_allocations"
RECONCILE_BEAT_NAME = "reconcile-redis-postgres"
CLEANUP_BEAT_NAME = "cleanup-expired-allocations"

MAINTENANCE_TASK_ROUTES: dict[str, dict[str, str]] = {
    RECONCILE_TASK_NAME: {"queue": settings.celery_token_maintenance_queue_name},
    QUEUE_DEPTH_PUBLISH_TASK_NAME: {
        "queue": settings.celery_token_maintenance_queue_name
    },
    CLEANUP_TASK_NAME: {"queue": settings.celery_token_maintenance_queue_name},
}

_beat_registered = False


def build_beat_schedule() -> dict[str, dict[str, Any]]:
    """Return the canonical beat schedule for Layer 4 maintenance tasks."""
    return {
        RECONCILE_BEAT_NAME: {
            "task": RECONCILE_TASK_NAME,
            "schedule": settings.celery_reconcile_interval_secs,
            "options": {"queue": settings.celery_token_maintenance_queue_name},
        },
        QUEUE_DEPTH_PUBLISH_SCHEDULE_NAME: {
            "task": QUEUE_DEPTH_PUBLISH_TASK_NAME,
            "schedule": settings.bp_queue_depth_publish_interval_secs,
            "options": {"queue": settings.celery_token_maintenance_queue_name},
        },
        CLEANUP_BEAT_NAME: {
            "task": CLEANUP_TASK_NAME,
            "schedule": settings.celery_cleanup_interval_secs,
            "options": {"queue": settings.celery_token_maintenance_queue_name},
        },
    }


def register_beat_schedule(celery_app: Any) -> None:
    """Register the maintenance beat schedule idempotently on one Celery app."""
    global _beat_registered
    if _beat_registered:
        return
    celery_app.conf.beat_schedule = {
        **getattr(celery_app.conf, "beat_schedule", {}),
        **build_beat_schedule(),
    }
    _beat_registered = True
