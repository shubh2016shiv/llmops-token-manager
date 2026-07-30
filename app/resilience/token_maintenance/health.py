"""
Health — readiness reporting for the token-maintenance background world.

Two audiences, one source of truth:
  • verify_token_maintenance_readiness() -> ServiceStatus
        used by FastAPI startup and the /health endpoint.
  • main() / `python -m app.resilience.token_maintenance.health`
        a shell-safe probe for container health checks.

Since maintenance now runs in-process (there is no Celery worker/broker to probe),
"ready" means something simple and honest: the job schedule is well-formed and every
interval is a positive number. If the schedule can't even be built (e.g. a missing
setting), that's what "not ready" reports.

Author: Engineering Team
"""

from __future__ import annotations

from app.core.service_health import ServiceStatus
from app.resilience.token_maintenance.scheduler import build_maintenance_schedule


def inspect_token_maintenance_runtime() -> tuple[bool, str | None]:
    """
    Return (is_ready, reason).

    Ready when the maintenance schedule is valid: at least one job, and every
    interval strictly positive.
    """
    schedule = build_maintenance_schedule()
    if not schedule:
        return False, "No maintenance jobs are scheduled"
    for job in schedule:
        if job.interval_seconds <= 0:
            return False, f"Job '{job.name}' has a non-positive interval"
    return True, None


async def verify_token_maintenance_readiness() -> ServiceStatus:
    """Adapt the probe into the shared ServiceStatus used by startup and /health."""
    try:
        schedule = build_maintenance_schedule()
        # Report the schedule itself as the "connection details" — for maintenance,
        # the meaningful state is "which jobs run and how often", not a socket.
        connection_details = {
            "mode": "in-process scheduler",
            "scheduled_jobs": ", ".join(sorted(job.name for job in schedule)),
            **{
                f"{job.name}_interval_secs": str(job.interval_seconds)
                for job in schedule
            },
        }

        is_ready, reason = inspect_token_maintenance_runtime()
        if not is_ready:
            return ServiceStatus(
                name="Token maintenance",
                status="failed",
                error_message=reason or "Token maintenance readiness check failed",
                suggestion=(
                    "Verify the maintenance job intervals are configured "
                    "(settings.reconcile_interval_secs, "
                    "settings.bp_queue_depth_publish_interval_secs, "
                    "settings.cleanup_interval_secs)."
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
                "Check the maintenance scheduler wiring and that all "
                "*_interval_secs settings are present."
            ),
        )


def main() -> int:
    """Exit status for `python -m app.resilience.token_maintenance.health`."""
    is_ready, _ = inspect_token_maintenance_runtime()
    return 0 if is_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
