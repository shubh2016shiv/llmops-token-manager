"""
Token maintenance package - Layer 4 periodic maintenance public API.

The package root stays intentionally light so callers can import submodules
without triggering the Celery runtime or creating circular imports during
Celery app initialization.
"""

# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MAINTENANCE_TASK_ROUTES",
    "build_beat_schedule",
    "register_beat_schedule",
    "REQUIRED_TOKEN_MAINTENANCE_TASKS",
    "inspect_token_maintenance_runtime",
    "verify_token_maintenance_readiness",
    "reconcile_async",
    "reconcile_redis_postgres",
    "publish_backpressure_queue_depth",
    "cleanup_expired_allocations",
    "_reconcile_async",
    "_cleanup_async",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "MAINTENANCE_TASK_ROUTES": (
        "app.resilience.token_maintenance.schedule_registry",
        "MAINTENANCE_TASK_ROUTES",
    ),
    "build_beat_schedule": (
        "app.resilience.token_maintenance.schedule_registry",
        "build_beat_schedule",
    ),
    "register_beat_schedule": (
        "app.resilience.token_maintenance.schedule_registry",
        "register_beat_schedule",
    ),
    "REQUIRED_TOKEN_MAINTENANCE_TASKS": (
        "app.resilience.token_maintenance.healthcheck",
        "REQUIRED_TOKEN_MAINTENANCE_TASKS",
    ),
    "inspect_token_maintenance_runtime": (
        "app.resilience.token_maintenance.healthcheck",
        "inspect_token_maintenance_runtime",
    ),
    "verify_token_maintenance_readiness": (
        "app.resilience.token_maintenance.service_health",
        "verify_token_maintenance_readiness",
    ),
    "reconcile_async": (
        "app.resilience.token_maintenance.reconciliation",
        "reconcile_async",
    ),
    "reconcile_redis_postgres": (
        "app.resilience.token_maintenance.tasks",
        "reconcile_redis_postgres",
    ),
    "publish_backpressure_queue_depth": (
        "app.resilience.token_maintenance.tasks",
        "publish_backpressure_queue_depth",
    ),
    "cleanup_expired_allocations": (
        "app.resilience.token_maintenance.tasks",
        "cleanup_expired_allocations",
    ),
    "_reconcile_async": (
        "app.resilience.token_maintenance.tasks",
        "_reconcile_async",
    ),
    "_cleanup_async": (
        "app.resilience.token_maintenance.tasks",
        "_cleanup_async",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily load public token-maintenance exports on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
