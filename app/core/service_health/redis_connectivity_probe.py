"""Verifies live connectivity to the Redis cache."""

from app.core.config import settings
from app.core.redis import redis_manager
from app.models.service_health_models import ServiceStatus


async def verify_redis_connectivity() -> ServiceStatus:
    """Verify Redis connectivity with detailed error reporting."""
    try:
        if not await redis_manager.ping():
            return ServiceStatus(
                name="Redis",
                status="failed",
                error_message="Redis server did not respond to ping",
                suggestion="Check if Redis server is running and accessible",
                connection_details={
                    "host": settings.redis_host,
                    "port": str(settings.redis_port),
                    "database": str(settings.redis_db),
                },
            )
        return ServiceStatus(
            name="Redis",
            status="connected",
            connection_details={
                "host": settings.redis_host,
                "port": str(settings.redis_port),
                "database": str(settings.redis_db),
            },
        )
    except ConnectionRefusedError:
        return ServiceStatus(
            name="Redis",
            status="failed",
            error_message=(
                "Connection refused - Redis is not running or not accessible"
            ),
            suggestion=(
                "Start Redis server with: redis-server or check if it's running on "
                f"{settings.redis_host}:{settings.redis_port}"
            ),
            connection_details={
                "host": settings.redis_host,
                "port": str(settings.redis_port),
                "database": str(settings.redis_db),
            },
        )
    except Exception as e:
        return ServiceStatus(
            name="Redis",
            status="failed",
            error_message=str(e),
            suggestion="Check Redis configuration in .env file and verify credentials",
            connection_details={
                "host": settings.redis_host,
                "port": str(settings.redis_port),
            },
        )
