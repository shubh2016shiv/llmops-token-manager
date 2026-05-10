"""
Token maintenance tasks - Celery entrypoints for Layer 4 periodic operations.

Architecture:
-------------
    ┌────────────────────────────────────┐     ┌────────────────────────────────────┐
    │ Celery beat / workers              │────▶│ token_maintenance/tasks.py         │
    │ periodic execution                 │     │ task lifecycle wrappers            │
    └────────────────────────────────────┘     └──────────────┬─────────────────────┘
                                                              │
                               ┌──────────────────────────────┴──────────────────────┐
                               │ reconciliation.py + queue_depth_publisher.py + PG   │
                               └──────────────────────────────────────────────────────┘

Dependencies:
    - app/llm_client_provisioning/llm_client_request_queue.py - shared Celery app
    - app/persistence/token_maintenance.py - cleanup persistence
    - app/resilience/backpressure/queue_depth_publisher.py - Layer 1 telemetry write
    - app/resilience/token_maintenance/reconciliation.py - reconciliation orchestration

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.core.config import settings
from app.llm_client_provisioning.llm_client_request_queue import celery_app
from app.persistence.token_maintenance import TokenMaintenancePersistence
from app.resilience.backpressure.constants import QUEUE_DEPTH_PUBLISH_TASK_NAME
from app.resilience.backpressure.queue_depth_publisher import (
    publish_queue_depth_snapshot,
)
from app.resilience.token_maintenance.reconciliation import (
    reconcile_async as _reconcile_async,
)
from app.resilience.token_maintenance.schedule_registry import (
    CLEANUP_TASK_NAME,
    RECONCILE_TASK_NAME,
)

task_logger = get_task_logger(__name__)


@celery_app.task(
    name=RECONCILE_TASK_NAME,
    queue=settings.celery_token_maintenance_queue_name,
    serializer="json",
    task_time_limit=120,
    task_soft_time_limit=90,
    ignore_result=True,
)
def reconcile_redis_postgres() -> None:
    """Reconcile Redis token counters against PostgreSQL totals."""
    task_logger.info("[token_maintenance] Starting reconciliation")
    try:
        asyncio.run(_reconcile_async())
        task_logger.info("[token_maintenance] Reconciliation complete")
    except Exception as exc:
        task_logger.error(f"[token_maintenance] Reconciliation failed: {exc}")


@celery_app.task(
    name=QUEUE_DEPTH_PUBLISH_TASK_NAME,
    queue=settings.celery_token_maintenance_queue_name,
    serializer="json",
    task_time_limit=20,
    task_soft_time_limit=10,
    ignore_result=True,
)
def publish_backpressure_queue_depth() -> None:
    """Publish the work-queue depth snapshot for Layer 1 backpressure decisions."""
    task_logger.debug("[token_maintenance] Publishing work-queue depth snapshot")
    try:
        asyncio.run(publish_queue_depth_snapshot())
    except Exception as exc:
        task_logger.error(f"[token_maintenance] Queue depth publication failed: {exc}")


@celery_app.task(
    name=CLEANUP_TASK_NAME,
    queue=settings.celery_token_maintenance_queue_name,
    serializer="json",
    task_time_limit=60,
    task_soft_time_limit=45,
    ignore_result=True,
)
def cleanup_expired_allocations() -> None:
    """Delete expired token allocations from PostgreSQL."""
    task_logger.info("[token_maintenance] Running expired allocation cleanup")
    try:
        deleted_count = asyncio.run(_cleanup_async())
        task_logger.info(
            f"[token_maintenance] Deleted {deleted_count} expired allocation(s)"
        )
    except Exception as exc:
        task_logger.error(f"[token_maintenance] Cleanup failed: {exc}")


async def _cleanup_async() -> int:
    """Delete expired allocations using the maintenance persistence service."""
    persistence = TokenMaintenancePersistence()
    return await persistence.delete_expired_allocations()
