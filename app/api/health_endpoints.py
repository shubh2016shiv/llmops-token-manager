"""
Health Check Endpoints
---------------------
Health monitoring endpoints for service and dependencies.
Provides status checks for database, Redis, RabbitMQ,
and token-maintenance runtime readiness.
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from loguru import logger

from app.core.config import settings
from app.core.service_health import (
    ServiceStatus,
    verify_database_connectivity,
    verify_rabbitmq_connectivity,
    verify_redis_connectivity,
)
from app.models.response_models import DependencyHealth, HealthStatus
from app.resilience.token_maintenance.health import (
    verify_token_maintenance_readiness,
)

router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.
    Returns service status and version information.

    Returns:
        HealthStatus: Service health status

    """
    logger.debug("Health check requested")

    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.app_version,
    )


@router.get("/dependencies", response_model=DependencyHealth, status_code=200)
async def check_dependencies() -> DependencyHealth:
    """
    Check health of all service dependencies.
    Tests connectivity to PostgreSQL, Redis, RabbitMQ,
    and token-maintenance runtime readiness.

    Returns a 200 OK status with the health status of each component.
    If any component is unhealthy, the overall status will be 'unhealthy'
    but the endpoint will still return a 200 status code with detailed information.

    This follows industry best practices for health check endpoints:
    - Always return a 200 status with detailed health information
    - Let monitoring systems determine criticality based on the response content
    - Provide component-level granularity for targeted troubleshooting

    Returns:
        DependencyHealth: Health status of each infrastructure component

    """
    logger.debug("Dependency health check requested")

    # Check PostgreSQL database
    postgresql_healthy = await _check_database()

    # Check Redis cache
    redis_healthy = await _check_redis()

    # Check RabbitMQ message broker
    rabbitmq_healthy = await _check_rabbitmq()

    # Check token-maintenance runtime readiness
    token_maintenance_healthy = await _check_token_maintenance()

    # Determine overall health status
    all_healthy = (
        postgresql_healthy
        and redis_healthy
        and rabbitmq_healthy
        and token_maintenance_healthy
    )
    status = "healthy" if all_healthy else "unhealthy"

    # Log appropriate message based on health status
    if not all_healthy:
        logger.warning(
            f"Infrastructure health check detected issues: "
            "postgresql="
            f"{postgresql_healthy}, redis={redis_healthy}, rabbitmq={rabbitmq_healthy}, "
            f"token_maintenance={token_maintenance_healthy}"
        )
    else:
        logger.info("All infrastructure components healthy")

    # Return health status for all components
    # Always return a 200 status code with detailed component health information
    return DependencyHealth(
        postgresql=postgresql_healthy,
        redis=redis_healthy,
        rabbitmq=rabbitmq_healthy,
        token_maintenance=token_maintenance_healthy,
        status=status,
        timestamp=datetime.now(timezone.utc),
    )


async def _check_database() -> bool:
    """
    Check PostgreSQL database connectivity.

    Returns:
        bool: True if database is accessible

    """
    result = await verify_database_connectivity()
    return _is_service_connected(result, "Database")


async def _check_redis() -> bool:
    """
    Check Redis connectivity.

    Returns:
        bool: True if Redis is accessible

    """
    result = await verify_redis_connectivity()
    return _is_service_connected(result, "Redis")


async def _check_rabbitmq() -> bool:
    """
    Check RabbitMQ broker connectivity.

    RabbitMQ remains a real, live dependency of the token manager
    (token_queue_consumer depends on it) even though the LLM-job Celery
    app that used to own this check was decoupled into llm_gateway. This
    probes RabbitMQ directly instead, same pattern as _check_database/_check_redis.

    Returns:
        bool: True if RabbitMQ is accessible

    """
    result = await verify_rabbitmq_connectivity()
    return _is_service_connected(result, "RabbitMQ")


async def _check_token_maintenance() -> bool:
    """
    Check token-maintenance runtime readiness.

    Returns:
        bool: True if the token-maintenance runtime is ready

    """
    result = await verify_token_maintenance_readiness()
    return _is_service_connected(result, "Token maintenance")


def _is_service_connected(result: ServiceStatus, service_name: str) -> bool:
    """
    Normalize dependency-health logging for a typed service status result.

    This keeps the endpoint contract focused on booleans while startup
    diagnostics remain the single source of truth for rich connection details.
    """
    is_connected = result.status == "connected"
    if not is_connected:
        error_message = result.error_message or "Unknown health check failure"
        logger.error(f"{service_name} health check failed: {error_message}")
    return is_connected
