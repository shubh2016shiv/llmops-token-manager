"""
Container health check for the Celery worker.

This keeps runtime health deterministic by validating two things only:
1. The Celery application and task module import successfully.
2. The configured RabbitMQ broker is reachable from inside the worker container.
"""

from app.llm_client_provisioning.llm_client_request_queue import celery_app

REQUIRED_TASKS = {
    "app.llm_client_provisioning.llm_tasks.process_llm_request",
    "app.llm_client_provisioning.llm_tasks.process_priority_llm_request",
}


def inspect_celery_worker_readiness(max_retries: int = 1) -> tuple[bool, str | None]:
    """
    Validate the shared Celery worker readiness contract.

    This helper is intentionally reused by both Docker health checks and FastAPI
    dependency/readiness reporting so the worker status cannot drift between them.
    """
    celery_app.loader.import_default_modules()

    missing_tasks = sorted(REQUIRED_TASKS.difference(celery_app.tasks))
    if missing_tasks:
        return (
            False,
            "Required Celery tasks are not registered: " + ", ".join(missing_tasks),
        )

    try:
        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=max_retries)
    except Exception as exc:
        return False, f"Broker connectivity check failed: {exc}"

    return True, None


def main() -> int:
    """Return a shell-compatible exit status for container health checks."""
    is_ready, _ = inspect_celery_worker_readiness(max_retries=1)
    return 0 if is_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
