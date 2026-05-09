from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.core.config import settings
from app.llm_client_provisioning.llm_client_request_queue import celery_app
from app.resilience.backpressure.constants import QUEUE_DEPTH_PUBLISH_TASK_NAME
from app.resilience.token_maintenance import schedule_registry
from app.resilience.token_maintenance.healthcheck import (
    inspect_token_maintenance_runtime,
)
from app.resilience.token_maintenance.schedule_registry import (
    CLEANUP_TASK_NAME,
    MAINTENANCE_TASK_ROUTES,
    RECONCILE_TASK_NAME,
)
from app.resilience.token_queue.healthcheck import (
    inspect_token_queue_consumer_readiness,
)


def test_token_maintenance_tasks_are_registered() -> None:
    celery_app.loader.import_default_modules()

    assert RECONCILE_TASK_NAME in celery_app.tasks
    assert CLEANUP_TASK_NAME in celery_app.tasks
    assert QUEUE_DEPTH_PUBLISH_TASK_NAME in celery_app.tasks


def test_schedule_registry_exports_single_route_owner() -> None:
    assert schedule_registry.MAINTENANCE_TASK_ROUTES is MAINTENANCE_TASK_ROUTES
    assert MAINTENANCE_TASK_ROUTES[RECONCILE_TASK_NAME]["queue"] == (
        settings.celery_token_maintenance_queue_name
    )
    assert not hasattr(schedule_registry, "celery_app")


def test_register_beat_schedule_is_idempotent(monkeypatch) -> None:
    fake_celery_app = SimpleNamespace(conf=SimpleNamespace(beat_schedule={}))
    monkeypatch.setattr(schedule_registry, "_beat_registered", False)

    schedule_registry.register_beat_schedule(fake_celery_app)
    first_schedule = dict(fake_celery_app.conf.beat_schedule)
    schedule_registry.register_beat_schedule(fake_celery_app)

    assert fake_celery_app.conf.beat_schedule == first_schedule


def test_token_healthchecks_boot_without_runtime_errors(monkeypatch) -> None:
    class _FakeConnection:
        def __enter__(self) -> _FakeConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def ensure_connection(self, *, max_retries: int) -> None:
            assert max_retries == 1

    monkeypatch.setattr(
        "app.resilience.token_queue.healthcheck.TOKEN_BROKER_CONNECTION",
        type("Broker", (), {"clone": staticmethod(lambda: _FakeConnection())})(),
    )
    monkeypatch.setattr(
        "app.resilience.token_maintenance.healthcheck.celery_app",
        type(
            "FakeCeleryApp",
            (),
            {
                "tasks": {
                    RECONCILE_TASK_NAME: object(),
                    CLEANUP_TASK_NAME: object(),
                    QUEUE_DEPTH_PUBLISH_TASK_NAME: object(),
                },
                "loader": type(
                    "Loader",
                    (),
                    {"import_default_modules": staticmethod(lambda: None)},
                )(),
                "connection": staticmethod(lambda: _FakeConnection()),
            },
        )(),
    )

    queue_ok, queue_reason = inspect_token_queue_consumer_readiness(max_retries=1)
    worker_ok, worker_reason = inspect_token_maintenance_runtime(max_retries=1)

    assert queue_ok is True
    assert queue_reason is None
    assert worker_ok is True
    assert worker_reason is None


def test_token_maintenance_runtime_fails_when_task_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.resilience.token_maintenance.healthcheck.celery_app",
        type(
            "FakeCeleryApp",
            (),
            {
                "tasks": {
                    RECONCILE_TASK_NAME: object(),
                    CLEANUP_TASK_NAME: object(),
                },
                "loader": type(
                    "Loader",
                    (),
                    {"import_default_modules": staticmethod(lambda: None)},
                )(),
                "connection": staticmethod(lambda: None),
            },
        )(),
    )

    is_ready, reason = inspect_token_maintenance_runtime(max_retries=1)

    assert is_ready is False
    assert QUEUE_DEPTH_PUBLISH_TASK_NAME in (reason or "")


def test_token_maintenance_runtime_fails_when_broker_check_raises(
    monkeypatch,
) -> None:
    class _BrokenConnection:
        def __enter__(self) -> _BrokenConnection:
            raise RuntimeError("broker down")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "app.resilience.token_maintenance.healthcheck.celery_app",
        type(
            "FakeCeleryApp",
            (),
            {
                "tasks": {
                    RECONCILE_TASK_NAME: object(),
                    CLEANUP_TASK_NAME: object(),
                    QUEUE_DEPTH_PUBLISH_TASK_NAME: object(),
                },
                "loader": type(
                    "Loader",
                    (),
                    {"import_default_modules": staticmethod(lambda: None)},
                )(),
                "connection": staticmethod(lambda: _BrokenConnection()),
            },
        )(),
    )

    is_ready, reason = inspect_token_maintenance_runtime(max_retries=1)

    assert is_ready is False
    assert "broker down" in (reason or "")


def test_token_maintenance_probe_main_returns_shell_safe_exit_code(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.resilience.token_maintenance.healthcheck.inspect_token_maintenance_runtime",
        lambda max_retries=1: (True, None),
    )

    from app.resilience.token_maintenance import healthcheck as maintenance_healthcheck

    assert maintenance_healthcheck.main() == 0


def test_cleanup_async_delegates_to_maintenance_persistence(monkeypatch) -> None:
    class _FakePersistence:
        async def delete_expired_allocations(self) -> int:
            return 7

    from app.resilience.token_maintenance import tasks as maintenance_tasks

    monkeypatch.setattr(
        maintenance_tasks,
        "TokenMaintenancePersistence",
        lambda: _FakePersistence(),
    )

    deleted_count = asyncio.run(maintenance_tasks._cleanup_async())

    assert deleted_count == 7


def test_cleanup_task_logs_deleted_count(monkeypatch) -> None:
    logged_messages: list[str] = []

    class _FakeLogger:
        def info(self, message: str) -> None:
            logged_messages.append(message)

        def error(self, message: str) -> None:
            logged_messages.append(message)

    from app.resilience.token_maintenance import tasks as maintenance_tasks

    monkeypatch.setattr(maintenance_tasks, "task_logger", _FakeLogger())
    monkeypatch.setattr(
        maintenance_tasks,
        "_cleanup_async",
        lambda: 5,
    )
    monkeypatch.setattr(
        maintenance_tasks.asyncio,
        "run",
        lambda coroutine: 5,
    )

    maintenance_tasks.cleanup_expired_allocations()

    assert any(
        "Deleted 5 expired allocation(s)" in message for message in logged_messages
    )


def test_cleanup_task_logs_failure_without_raising(monkeypatch) -> None:
    logged_messages: list[str] = []

    class _FakeLogger:
        def info(self, message: str) -> None:
            logged_messages.append(message)

        def error(self, message: str) -> None:
            logged_messages.append(message)

    from app.resilience.token_maintenance import tasks as maintenance_tasks

    monkeypatch.setattr(maintenance_tasks, "task_logger", _FakeLogger())

    def _raise_cleanup_failure(coroutine: object) -> int:
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()
        raise RuntimeError("cleanup crashed")

    monkeypatch.setattr(maintenance_tasks.asyncio, "run", _raise_cleanup_failure)

    maintenance_tasks.cleanup_expired_allocations()

    assert any(
        "Cleanup failed: cleanup crashed" in message for message in logged_messages
    )
