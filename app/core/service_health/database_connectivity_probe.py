"""Verifies live connectivity to the PostgreSQL database."""

from sqlalchemy import text

from app.core.config import settings
from app.core.database import db_manager
from app.models.service_health_models import ServiceStatus


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
