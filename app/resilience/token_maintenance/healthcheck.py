"""
Container and diagnostics probe for the token-maintenance runtime.

This module intentionally stays shallow so Docker health checks and ad hoc
runtime diagnostics use the same deterministic contract. FastAPI startup and
API dependency health should call `verify_token_maintenance_readiness()` from
`service_health.py`, while container orchestration can execute this module
directly with `python -m app.resilience.token_maintenance.healthcheck`.
"""

from __future__ import annotations

from app.core.celery_runtime_health import inspect_celery_runtime
from app.core.config import settings
from app.llm_client_provisioning.llm_client_request_queue import celery_app
from app.resilience.backpressure.constants import QUEUE_DEPTH_PUBLISH_TASK_NAME
from app.resilience.token_maintenance.schedule_registry import (
    CLEANUP_TASK_NAME,
    RECONCILE_TASK_NAME,
)

REQUIRED_TOKEN_MAINTENANCE_TASKS = {
    RECONCILE_TASK_NAME,
    CLEANUP_TASK_NAME,
    QUEUE_DEPTH_PUBLISH_TASK_NAME,
}


def inspect_token_maintenance_runtime(max_retries: int = 1) -> tuple[bool, str | None]:
    """Validate token-maintenance task registration and broker connectivity."""
    queue_name = settings.celery_token_maintenance_queue_name
    if not queue_name:
        return False, "Token maintenance queue name is not configured"

    return inspect_celery_runtime(
        celery_app=celery_app,
        required_task_names=REQUIRED_TOKEN_MAINTENANCE_TASKS,
        max_retries=max_retries,
        missing_tasks_prefix=("Required token maintenance tasks are not registered: "),
        broker_failure_prefix=("Token maintenance broker connectivity check failed: "),
    )


def main() -> int:
    """Return a shell-compatible exit status for container health checks."""
    is_ready, _ = inspect_token_maintenance_runtime(max_retries=1)
    return 0 if is_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
