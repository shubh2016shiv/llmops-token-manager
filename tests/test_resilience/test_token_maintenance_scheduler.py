"""
Tests for the token-maintenance schedule, health probe, and cleanup job.

Replaces the old Celery-registration tests: there is no Celery beat/worker anymore,
so we test the plain schedule declaration, the readiness probe, and the cleanup job.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.resilience.token_maintenance import cleanup as cleanup_module
from app.resilience.token_maintenance import health as health_module
from app.resilience.token_maintenance.scheduler import (
    MaintenanceJob,
    build_maintenance_schedule,
)


def test_schedule_declares_the_three_background_jobs() -> None:
    names = {job.name for job in build_maintenance_schedule()}
    assert names == {"reconciliation", "queue_depth_publish", "cleanup"}


def test_schedule_intervals_come_from_settings() -> None:
    by_name = {job.name: job for job in build_maintenance_schedule()}
    assert (
        by_name["reconciliation"].interval_seconds == settings.reconcile_interval_secs
    )
    assert (
        by_name["queue_depth_publish"].interval_seconds
        == settings.bp_queue_depth_publish_interval_secs
    )
    assert by_name["cleanup"].interval_seconds == settings.cleanup_interval_secs


def test_schedule_jobs_are_callables() -> None:
    for job in build_maintenance_schedule():
        assert callable(job.run)


def test_runtime_is_ready_for_a_valid_schedule() -> None:
    is_ready, reason = health_module.inspect_token_maintenance_runtime()
    assert is_ready is True
    assert reason is None


def test_runtime_not_ready_when_an_interval_is_non_positive(monkeypatch) -> None:
    broken_schedule = [
        MaintenanceJob(
            name="reconciliation",
            run=lambda: None,
            interval_seconds=0,
            purpose="x",
        )
    ]
    monkeypatch.setattr(
        health_module, "build_maintenance_schedule", lambda: broken_schedule
    )

    is_ready, reason = health_module.inspect_token_maintenance_runtime()

    assert is_ready is False
    assert "reconciliation" in (reason or "")


def test_health_main_returns_zero_when_ready() -> None:
    assert health_module.main() == 0


def test_cleanup_delegates_to_persistence(monkeypatch) -> None:
    class _FakePersistence:
        async def delete_expired_allocations(self) -> int:
            return 7

    monkeypatch.setattr(
        cleanup_module, "TokenMaintenancePersistence", lambda: _FakePersistence()
    )

    deleted = asyncio.run(cleanup_module.cleanup_expired_allocations())

    assert deleted == 7
