"""
Startup Diagnostics Module.

Handles service connectivity verification and error reporting during
application startup.
Provides clear, actionable error messages when infrastructure services
are unavailable.
"""

from dataclasses import dataclass

from loguru import logger
from sqlalchemy import text

from app.core.config_manager import settings
from app.core.database_connection import db_manager
from app.core.redis_connection import redis_manager
from app.llm_client_provisioning.celery_healthcheck import (
    inspect_celery_worker_readiness,
)
from app.llm_client_provisioning.queue_contract import (
    LLM_REQUESTS_QUEUE,
    PRIORITY_REQUESTS_QUEUE,
)


@dataclass
class ServiceStatus:
    """Track service connection status with detailed error information."""

    name: str
    status: str  # "connected", "failed", "skipped"
    error_message: str | None = None
    suggestion: str | None = None
    connection_details: dict[str, str] | None = None


def display_startup_failure(failed_services: list[ServiceStatus]):
    """Display formatted startup failure message."""
    border = "═" * 80
    print("\n" + border)
    print("[FATAL ERROR] APPLICATION STARTUP FAILED")
    print(border)

    for service in failed_services:
        print(f"\n[FATAL ERROR] {service.name}: {service.status.upper()}")
        print(f"   Error: {service.error_message}")

        if service.connection_details:
            print("   Connection Details:")
            for key, value in service.connection_details.items():
                print(f"     • {key}: {value}")

        if service.suggestion:
            print(f"   >> Suggestion: {service.suggestion}")

    print("\n" + border)
    print("Please fix the issues above and restart the application.")
    print(border + "\n")


def display_service_info():
    """Display service connection information when all services are healthy."""
    # Display service URLs in formatted tables
    # Use print for properly aligned tables, with consistent width
    border_line = "═" * 80
    header_line = "─" * 80

    print("\n" + border_line)
    print("SERVICE ENDPOINTS & CONNECTION INFORMATION")
    print(border_line)

    # FastAPI URLs table - properly aligned with fixed width columns
    # Show localhost URLs for local access (works reliably on Windows)
    local_api_base = f"http://localhost:{settings.fastapi_port}"
    print("FASTAPI SERVICE")
    print(header_line)
    print(f"{'Service':<20} | {'URL':<57}")
    print(f"{header_line}")
    print(f"{'Main API':<20} | {local_api_base + '/':<57}")
    print(f"{'API Documentation':<20} | {local_api_base + '/api/docs':<57}")
    print(f"{'ReDoc Interface':<20} | {local_api_base + '/api/redoc':<57}")
    print(f"{'OpenAPI Schema':<20} | {local_api_base + '/api/openapi.json':<57}")
    print(f"{'Health Check':<20} | {local_api_base + '/api/v1/health':<57}")
    print(header_line)

    # Database connection table - properly aligned
    print("\nPOSTGRESQL DATABASE")
    print(header_line)
    print(f"{'Parameter':<20} | {'Value':<57}")
    print(f"{header_line}")
    print(f"{'Host':<20} | {settings.database_host:<57}")
    print(f"{'Port':<20} | {str(settings.database_port):<57}")
    print(f"{'Database':<20} | {settings.database_name:<57}")
    pool_value = (
        f"{settings.database_pool_size} connections "
        f"(+ {settings.database_max_overflow} overflow)"
    )
    print(f"{'Connection Pool':<20} | {pool_value:<57}")
    print(header_line)

    # Redis connection table - properly aligned
    print("\nREDIS CACHE")
    print(header_line)
    print(f"{'Parameter':<20} | {'Value':<57}")
    print(f"{header_line}")
    print(f"{'Host':<20} | {settings.redis_host:<57}")
    print(f"{'Port':<20} | {str(settings.redis_port):<57}")
    print(f"{'Database':<20} | {str(settings.redis_db):<57}")
    print(f"{'Max Connections':<20} | {str(settings.redis_max_connections):<57}")
    print(header_line)

    print("\nRABBITMQ BROKER")
    print(header_line)
    print(f"{'Parameter':<20} | {'Value':<57}")
    print(f"{header_line}")
    print(f"{'Host':<20} | {settings.rabbitmq_host:<57}")
    print(f"{'Port':<20} | {str(settings.rabbitmq_port):<57}")
    print(f"{'Virtual Host':<20} | {settings.rabbitmq_vhost:<57}")
    print(f"{'Broker URL Source':<20} | {'Environment / derived settings':<57}")
    print(header_line)

    print("\nCELERY WORKER READINESS")
    print(header_line)
    print(f"{'Parameter':<20} | {'Value':<57}")
    print(f"{header_line}")
    print(
        f"{'Startup Critical':<20} | "
        f"{str(settings.require_celery_worker_on_startup):<57}"
    )
    print(f"{'Primary Queue':<20} | {LLM_REQUESTS_QUEUE:<57}")
    print(f"{'Priority Queue':<20} | {PRIORITY_REQUESTS_QUEUE:<57}")
    print(f"{'Result Backend':<20} | {settings.celery_result_backend:<57}")
    print(header_line)
    print(border_line + "\n")

    # Still log a simple message for the log file
    logger.info("Service endpoints and connection information displayed")


async def verify_database_connectivity() -> ServiceStatus:
    """Verify database connectivity with detailed error reporting."""
    try:
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            if result.scalar() != 1:
                return ServiceStatus(
                    name="PostgreSQL",
                    status="failed",
                    error_message="Connection test query failed",
                    suggestion="Check database permissions and query execution",
                    connection_details={
                        "host": settings.database_host,
                        "port": str(settings.database_port),
                        "database": settings.database_name,
                    },
                )
            return ServiceStatus(
                name="PostgreSQL",
                status="connected",
                connection_details={
                    "host": settings.database_host,
                    "port": str(settings.database_port),
                    "database": settings.database_name,
                },
            )
    except ConnectionRefusedError:
        return ServiceStatus(
            name="PostgreSQL",
            status="failed",
            error_message=(
                "Connection refused - PostgreSQL is not running or not accessible"
            ),
            suggestion=(
                "Start PostgreSQL server or check if it's running on "
                f"{settings.database_host}:{settings.database_port}"
            ),
            connection_details={
                "host": settings.database_host,
                "port": str(settings.database_port),
                "database": settings.database_name,
            },
        )
    except Exception as e:
        return ServiceStatus(
            name="PostgreSQL",
            status="failed",
            error_message=str(e),
            suggestion=(
                "Check database configuration in .env file and verify credentials"
            ),
            connection_details={
                "host": settings.database_host,
                "port": str(settings.database_port),
            },
        )


async def verify_redis_connectivity() -> ServiceStatus:
    """Verify Redis connectivity with detailed error reporting."""
    try:
        # Use the ping method directly from the redis_manager
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


async def verify_rabbitmq_connectivity() -> ServiceStatus:
    """Verify RabbitMQ connectivity with detailed error reporting."""
    try:
        from app.llm_client_provisioning.llm_client_request_queue import celery_app

        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=3)
            return ServiceStatus(
                name="RabbitMQ",
                status="connected",
                connection_details={
                    "host": settings.rabbitmq_host,
                    "port": str(settings.rabbitmq_port),
                    "virtual_host": settings.rabbitmq_vhost,
                    "component": "Celery broker",
                },
            )
    except ConnectionRefusedError:
        return ServiceStatus(
            name="RabbitMQ",
            status="failed",
            error_message=(
                "Connection refused - RabbitMQ is not running or not accessible"
            ),
            suggestion=(
                "Start RabbitMQ server or check whether the configured "
                "broker host and port are reachable"
            ),
            connection_details={
                "host": settings.rabbitmq_host,
                "port": str(settings.rabbitmq_port),
                "virtual_host": settings.rabbitmq_vhost,
                "component": "Celery broker",
            },
        )
    except Exception as e:
        return ServiceStatus(
            name="RabbitMQ",
            status="failed",
            error_message=str(e),
            suggestion="Check RabbitMQ configuration and verify broker settings",
            connection_details={
                "host": settings.rabbitmq_host,
                "port": str(settings.rabbitmq_port),
                "virtual_host": settings.rabbitmq_vhost,
            },
        )


async def verify_celery_worker_readiness() -> ServiceStatus:
    """Verify Celery worker readiness using the shared worker health contract."""
    try:
        is_ready, reason = inspect_celery_worker_readiness(max_retries=1)
        if not is_ready:
            return ServiceStatus(
                name="Celery worker",
                status="failed",
                error_message=reason or "Celery worker readiness check failed",
                suggestion=(
                    "Start a Celery worker and verify it can import tasks "
                    "and reach the configured broker"
                ),
                connection_details={
                    "host": settings.rabbitmq_host,
                    "port": str(settings.rabbitmq_port),
                    "queues": f"{LLM_REQUESTS_QUEUE}, {PRIORITY_REQUESTS_QUEUE}",
                },
            )

        return ServiceStatus(
            name="Celery worker",
            status="connected",
            connection_details={
                "host": settings.rabbitmq_host,
                "port": str(settings.rabbitmq_port),
                "queues": f"{LLM_REQUESTS_QUEUE}, {PRIORITY_REQUESTS_QUEUE}",
            },
        )
    except Exception as e:
        return ServiceStatus(
            name="Celery worker",
            status="failed",
            error_message=str(e),
            suggestion=(
                "Check Celery worker configuration and confirm the worker "
                "health contract can run successfully"
            ),
            connection_details={
                "host": settings.rabbitmq_host,
                "port": str(settings.rabbitmq_port),
                "queues": f"{LLM_REQUESTS_QUEUE}, {PRIORITY_REQUESTS_QUEUE}",
            },
        )
