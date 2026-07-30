"""Renders human-readable console reports of service status for local operators."""

from loguru import logger

from app.core.config import settings
from app.models.service_health_models import ServiceStatus


def display_startup_failure(failed_services: list[ServiceStatus]) -> None:
    """Display formatted startup failure message."""
    border = "=" * 80
    print("\n" + border)
    print("[FATAL ERROR] APPLICATION STARTUP FAILED")
    print(border)

    for service in failed_services:
        print(f"\n[FATAL ERROR] {service.name}: {service.status.upper()}")
        print(f"   Error: {service.error_message}")

        if service.connection_details:
            print("   Connection Details:")
            for key, value in service.connection_details.items():
                print(f"     - {key}: {value}")

        if service.suggestion:
            print(f"   >> Suggestion: {service.suggestion}")

    print("\n" + border)
    print("Please fix the issues above and restart the application.")
    print(border + "\n")


def display_service_info() -> None:
    """Display generic service connection information for the local runtime."""
    border_line = "=" * 80
    header_line = "-" * 80

    print("\n" + border_line)
    print("SERVICE ENDPOINTS & CONNECTION INFORMATION")
    print(border_line)

    local_api_base = f"http://localhost:{settings.fastapi_port}"
    print("FASTAPI SERVICE")
    print(header_line)
    print(f"{'Service':<20} | {'URL':<57}")
    print(header_line)
    print(f"{'Main API':<20} | {local_api_base + '/':<57}")
    print(f"{'API Documentation':<20} | {local_api_base + '/api/docs':<57}")
    print(f"{'ReDoc Interface':<20} | {local_api_base + '/api/redoc':<57}")
    print(f"{'OpenAPI Schema':<20} | {local_api_base + '/api/openapi.json':<57}")
    print(f"{'Health Check':<20} | {local_api_base + '/api/v1/health':<57}")
    print(header_line)

    print("\nPOSTGRESQL DATABASE")
    print(header_line)
    print(f"{'Parameter':<20} | {'Value':<57}")
    print(header_line)
    print(f"{'Host':<20} | {settings.database_host:<57}")
    print(f"{'Port':<20} | {str(settings.database_port):<57}")
    print(f"{'Database':<20} | {settings.database_name:<57}")
    pool_value = (
        f"{settings.database_pool_size} connections "
        f"(+ {settings.database_max_overflow} overflow)"
    )
    print(f"{'Connection Pool':<20} | {pool_value:<57}")
    print(header_line)

    print("\nREDIS CACHE")
    print(header_line)
    print(f"{'Parameter':<20} | {'Value':<57}")
    print(header_line)
    print(f"{'Host':<20} | {settings.redis_host:<57}")
    print(f"{'Port':<20} | {str(settings.redis_port):<57}")
    print(f"{'Database':<20} | {str(settings.redis_db):<57}")
    print(f"{'Max Connections':<20} | {str(settings.redis_max_connections):<57}")
    print(header_line)

    print(border_line + "\n")
    logger.info("Service endpoints and connection information displayed")
