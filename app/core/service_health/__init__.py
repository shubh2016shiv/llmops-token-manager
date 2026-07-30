"""
Infrastructure connectivity probes and startup console reporting.

Each infra dependency owns its own probe module (database, Redis, RabbitMQ)
so new dependencies can be added independently, and the pattern is reusable
in other projects. Console reporting is a separate concern, consumed by the
probes' results but not owning connectivity logic itself.
"""

from app.core.service_health.database_connectivity_probe import (
    verify_database_connectivity,
)
from app.core.service_health.rabbitmq_connectivity_probe import (
    verify_rabbitmq_connectivity,
)
from app.core.service_health.redis_connectivity_probe import (
    verify_redis_connectivity,
)
from app.core.service_health.service_status_console_report import (
    display_service_info,
    display_startup_failure,
)
from app.models.service_health_models import ServiceStatus

__all__ = [
    "ServiceStatus",
    "verify_database_connectivity",
    "verify_redis_connectivity",
    "verify_rabbitmq_connectivity",
    "display_startup_failure",
    "display_service_info",
]
