"""
Health Check Endpoints
---------------------
Health monitoring endpoints for service and dependencies.
Provides status checks for database, Redis, and RabbitMQ.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from loguru import logger

from app.models.response_models import HealthStatus, DependencyHealth
from app.core.config_manager import settings
from app.core.database_connection import db_manager
from app.core.redis_connection import redis_manager
from sqlalchemy import text


router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("/", response_model=HealthStatus)
async def health_check():
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
async def check_dependencies():
    """
    Check health of all service dependencies.
    Tests connectivity to PostgreSQL, Redis, and RabbitMQ.

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

    # Determine overall health status
    all_healthy = postgresql_healthy and redis_healthy and rabbitmq_healthy
    status = "healthy" if all_healthy else "unhealthy"

    # Log appropriate message based on health status
    if not all_healthy:
        logger.warning(
            f"Infrastructure health check detected issues: "
            f"postgresql={postgresql_healthy}, redis={redis_healthy}, rabbitmq={rabbitmq_healthy}"
        )
    else:
        logger.info("All infrastructure components healthy")

    # Return health status for all components
    # Always return a 200 status code with detailed component health information
    return DependencyHealth(
        postgresql=postgresql_healthy,
        redis=redis_healthy,
        rabbitmq=rabbitmq_healthy,
        status=status,
        timestamp=datetime.now(timezone.utc),
    )


async def _check_database() -> bool:
    """
    Check PostgreSQL database connectivity.

    Returns:
        bool: True if database is accessible
    """
    try:
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def _check_redis() -> bool:
    """
    Check Redis connectivity.

    Returns:
        bool: True if Redis is accessible
    """
    try:
        return await redis_manager.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


async def _check_rabbitmq() -> bool:
    """
    Check RabbitMQ broker connectivity via Celery.

    Returns:
        bool: True if RabbitMQ broker is accessible
    """
    try:
        from app.workers.celery_app import celery_app

        # Check broker connection directly instead of worker inspection
        # This verifies RabbitMQ server is accessible, not worker availability
        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=3)
            # If we can establish a connection, RabbitMQ broker is working
            return True
    except Exception as e:
        logger.error(f"RabbitMQ broker health check failed: {e}")
        return False
