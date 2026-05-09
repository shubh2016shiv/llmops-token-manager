"""
Application-facing health and readiness reporting for token maintenance.

This adapter turns the deterministic probe in `healthcheck.py` into the shared
`ServiceStatus` contract used by FastAPI startup and dependency health.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.service_health import ServiceStatus
from app.resilience.backpressure.constants import QUEUE_DEPTH_PUBLISH_TASK_NAME
from app.resilience.token_maintenance.healthcheck import (
    inspect_token_maintenance_runtime,
)
from app.resilience.token_maintenance.schedule_registry import (
    CLEANUP_TASK_NAME,
    RECONCILE_TASK_NAME,
)


async def verify_token_maintenance_readiness() -> ServiceStatus:
    """Verify token-maintenance runtime readiness using the shared probe."""
    connection_details = {
        "host": settings.rabbitmq_host,
        "port": str(settings.rabbitmq_port),
        "virtual_host": settings.rabbitmq_vhost,
        "queue": settings.celery_token_maintenance_queue_name,
        "required_tasks": ", ".join(
            sorted(
                {
                    RECONCILE_TASK_NAME,
                    CLEANUP_TASK_NAME,
                    QUEUE_DEPTH_PUBLISH_TASK_NAME,
                }
            )
        ),
    }

    try:
        is_ready, reason = inspect_token_maintenance_runtime(max_retries=1)
        if not is_ready:
            return ServiceStatus(
                name="Token maintenance",
                status="failed",
                error_message=reason or "Token maintenance readiness check failed",
                suggestion=(
                    "Verify token-maintenance tasks are registered, the queue name "
                    "is configured, and the Celery broker is reachable."
                ),
                connection_details=connection_details,
            )

        return ServiceStatus(
            name="Token maintenance",
            status="connected",
            connection_details=connection_details,
        )
    except Exception as exc:
        return ServiceStatus(
            name="Token maintenance",
            status="failed",
            error_message=str(exc),
            suggestion=(
                "Check token-maintenance runtime wiring and confirm the "
                "readiness probe can import tasks and reach the broker."
            ),
            connection_details=connection_details,
        )
