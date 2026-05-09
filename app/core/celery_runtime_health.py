"""
Generic Celery runtime readiness helpers.

This module centralizes the small, deterministic probe shared by multiple
subsystems that need to validate task registration and broker connectivity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def inspect_celery_runtime(
    *,
    celery_app: Any,
    required_task_names: Iterable[str],
    max_retries: int = 1,
    missing_tasks_prefix: str = "Required Celery tasks are not registered: ",
    broker_failure_prefix: str = "Broker connectivity check failed: ",
) -> tuple[bool, str | None]:
    """Validate task registration and broker connectivity for a Celery app."""
    celery_app.loader.import_default_modules()

    required_tasks = set(required_task_names)
    missing_tasks = sorted(required_tasks.difference(celery_app.tasks))
    if missing_tasks:
        return False, missing_tasks_prefix + ", ".join(missing_tasks)

    try:
        with celery_app.connection() as connection:
            connection.ensure_connection(max_retries=max_retries)
    except Exception as exc:
        return False, broker_failure_prefix + str(exc)

    return True, None
