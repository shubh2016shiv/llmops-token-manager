"""Verifies live connectivity to the RabbitMQ broker."""

import asyncio

from kombu import Connection
from kombu.exceptions import KombuError

from app.core.config import settings
from app.models.service_health_models import ServiceStatus


async def verify_rabbitmq_connectivity() -> ServiceStatus:
    """Verify RabbitMQ connectivity with detailed error reporting."""
    connection_details = {
        "host": settings.rabbitmq_host,
        "port": str(settings.rabbitmq_port),
        "vhost": settings.rabbitmq_vhost,
    }
    connection = Connection(
        settings.broker_url,
        heartbeat=settings.rabbitmq_token_heartbeat_seconds,
    )
    try:
        await asyncio.to_thread(connection.ensure_connection, max_retries=1)
        return ServiceStatus(
            name="RabbitMQ",
            status="connected",
            connection_details=connection_details,
        )
    except (KombuError, OSError) as e:
        # Kombu wraps socket errors (e.g. ConnectionRefusedError) in
        # OperationalError, chaining the original as __cause__.
        if isinstance(e.__cause__, ConnectionRefusedError):
            return ServiceStatus(
                name="RabbitMQ",
                status="failed",
                error_message=(
                    "Connection refused - RabbitMQ is not running or not accessible"
                ),
                suggestion=(
                    "Start RabbitMQ server or check if it's running on "
                    f"{settings.rabbitmq_host}:{settings.rabbitmq_port}"
                ),
                connection_details=connection_details,
            )
        return ServiceStatus(
            name="RabbitMQ",
            status="failed",
            error_message=str(e),
            suggestion=(
                "Check RabbitMQ configuration in .env file and verify credentials"
            ),
            connection_details=connection_details,
        )
    finally:
        await asyncio.to_thread(connection.release)
