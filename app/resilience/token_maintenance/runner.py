"""
Run the declared maintenance schedule during the application lifespan.

The schedule owns which jobs exist; this runner owns only task creation and
shutdown. Each job is sequential within its own loop, so a slow invocation
cannot create an unbounded backlog of the same maintenance operation.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from app.resilience.token_maintenance.scheduler import (
    MaintenanceJob,
    build_maintenance_schedule,
)


class MaintenanceRunner:
    """Own one cancellable task per periodic maintenance job."""

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        """Start the schedule once in the serving event loop."""
        if self._tasks:
            return
        jobs = build_maintenance_schedule()
        if not jobs or any(job.interval_seconds <= 0 for job in jobs):
            raise ValueError("Maintenance schedule requires positive job intervals")
        self._tasks = [
            asyncio.create_task(self._run_job(job), name=f"maintenance:{job.name}")
            for job in jobs
        ]

    async def close(self) -> None:
        """Cancel and join every task; repeated shutdown is safe."""
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_job(self, job: MaintenanceJob) -> None:
        while True:
            await asyncio.sleep(job.interval_seconds)
            try:
                await asyncio.wait_for(job.run(), timeout=job.interval_seconds)
            except asyncio.TimeoutError:
                logger.warning("Maintenance job timed out", job=job.name)
            except Exception:
                # Jobs retry on their next scheduled tick. Never log exception
                # text here: provider/DB exceptions may contain credentials.
                logger.warning("Maintenance job failed", job=job.name)


maintenance_runner = MaintenanceRunner()
