"""Unit tests for FastAPI lifespan dependency startup policy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.app import app as fastapi_app
from app.app import lifespan
from app.core.service_health import ServiceStatus


@pytest.mark.asyncio
@patch("app.app.display_provisioning_service_info")
@patch("app.app.display_service_info")
@patch("app.app.verify_token_maintenance_readiness", new_callable=AsyncMock)
@patch("app.app.verify_celery_worker_readiness", new_callable=AsyncMock)
@patch("app.app.verify_rabbitmq_connectivity", new_callable=AsyncMock)
@patch("app.app.verify_redis_connectivity", new_callable=AsyncMock)
@patch("app.app.verify_database_connectivity", new_callable=AsyncMock)
@patch("app.app.redis_manager")
@patch("app.app.db_manager")
@patch("app.app.settings")
async def test_lifespan_allows_worker_degradation_by_default(
    mock_settings,
    mock_db_manager,
    mock_redis_manager,
    mock_verify_database,
    mock_verify_redis,
    mock_verify_rabbitmq,
    mock_verify_celery_worker,
    mock_verify_token_maintenance,
    mock_display_service_info,
    mock_display_provisioning_service_info,
):
    """FastAPI should keep booting when only the worker is down by default."""
    mock_settings.app_name = "LLM Token Manager"
    mock_settings.app_version = "1.0.0"
    mock_settings.debug = False
    mock_settings.require_celery_worker_on_startup = False

    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.close = AsyncMock()
    mock_redis_manager.initialize = MagicMock()
    mock_redis_manager.close = AsyncMock()

    mock_verify_database.return_value = ServiceStatus("PostgreSQL", "connected")
    mock_verify_redis.return_value = ServiceStatus("Redis", "connected")
    mock_verify_rabbitmq.return_value = ServiceStatus("RabbitMQ", "connected")
    mock_verify_celery_worker.return_value = ServiceStatus(
        "Celery worker", "failed", error_message="worker unavailable"
    )
    mock_verify_token_maintenance.return_value = ServiceStatus(
        "Token maintenance", "connected"
    )

    async with lifespan(fastapi_app):
        pass

    mock_display_service_info.assert_called_once()
    mock_display_provisioning_service_info.assert_called_once()


@pytest.mark.asyncio
@patch("app.app.display_startup_failure")
@patch("app.app.verify_token_maintenance_readiness", new_callable=AsyncMock)
@patch("app.app.verify_celery_worker_readiness", new_callable=AsyncMock)
@patch("app.app.verify_rabbitmq_connectivity", new_callable=AsyncMock)
@patch("app.app.verify_redis_connectivity", new_callable=AsyncMock)
@patch("app.app.verify_database_connectivity", new_callable=AsyncMock)
@patch("app.app.redis_manager")
@patch("app.app.db_manager")
@patch("app.app.settings")
@patch("os._exit", side_effect=RuntimeError("startup halted"))
async def test_lifespan_can_require_celery_worker_on_startup(
    mock_os_exit,
    mock_settings,
    mock_db_manager,
    mock_redis_manager,
    mock_verify_database,
    mock_verify_redis,
    mock_verify_rabbitmq,
    mock_verify_celery_worker,
    mock_verify_token_maintenance,
    mock_display_startup_failure,
):
    """FastAPI should fail startup when the worker is required and unavailable."""
    mock_settings.app_name = "LLM Token Manager"
    mock_settings.app_version = "1.0.0"
    mock_settings.debug = False
    mock_settings.require_celery_worker_on_startup = True

    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.close = AsyncMock()
    mock_redis_manager.initialize = MagicMock()
    mock_redis_manager.close = AsyncMock()

    mock_verify_database.return_value = ServiceStatus("PostgreSQL", "connected")
    mock_verify_redis.return_value = ServiceStatus("Redis", "connected")
    mock_verify_rabbitmq.return_value = ServiceStatus("RabbitMQ", "connected")
    worker_status = ServiceStatus(
        "Celery worker", "failed", error_message="worker unavailable"
    )
    mock_verify_celery_worker.return_value = worker_status
    mock_verify_token_maintenance.return_value = ServiceStatus(
        "Token maintenance", "connected"
    )

    with pytest.raises(RuntimeError, match="startup halted"):
        async with lifespan(fastapi_app):
            pass

    mock_display_startup_failure.assert_called_once()
    failed_services = mock_display_startup_failure.call_args.args[0]
    assert failed_services == [worker_status]
    mock_os_exit.assert_called_once_with(1)
