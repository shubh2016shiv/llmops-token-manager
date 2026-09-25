"""Lifecycle regressions for the in-process maintenance runner."""

import asyncio

from app.resilience.token_maintenance.runner import MaintenanceRunner
from app.resilience.token_maintenance.scheduler import MaintenanceJob


def test_runner_starts_once_runs_jobs_and_closes_idempotently(monkeypatch):
    """A second start cannot duplicate a periodic job or leak tasks."""
    calls = 0
    completed = asyncio.Event()

    async def job():
        nonlocal calls
        calls += 1
        completed.set()

    monkeypatch.setattr(
        "app.resilience.token_maintenance.runner.build_maintenance_schedule",
        lambda: [
            MaintenanceJob(
                name="test",
                run=job,
                interval_seconds=1,
                purpose="verify lifecycle",
            )
        ],
    )

    async def exercise():
        runner = MaintenanceRunner()
        runner.start()
        runner.start()
        await asyncio.wait_for(completed.wait(), timeout=3)
        await runner.close()
        count_at_close = calls
        await runner.close()
        await asyncio.sleep(0)
        return count_at_close

    count_at_close = asyncio.run(exercise())

    assert count_at_close > 0
    assert calls == count_at_close


def test_runner_rejects_invalid_schedule(monkeypatch):
    """Startup fails if a configured interval could spin or never run."""
    monkeypatch.setattr(
        "app.resilience.token_maintenance.runner.build_maintenance_schedule",
        lambda: [
            MaintenanceJob(
                name="invalid",
                run=lambda: asyncio.sleep(0),
                interval_seconds=0,
                purpose="invalid configuration",
            )
        ],
    )

    async def exercise():
        runner = MaintenanceRunner()
        try:
            runner.start()
        except ValueError:
            return True
        finally:
            await runner.close()
        return False

    assert asyncio.run(exercise()) is True
