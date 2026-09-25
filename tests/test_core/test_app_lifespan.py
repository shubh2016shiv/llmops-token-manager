"""Application lifespan checks for maintenance ownership and startup policy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.app import app as fastapi_app
from app.app import lifespan
from app.core.service_health import ServiceStatus


@pytest.mark.asyncio
@patch("app.app.maintenance_runner")
@patch("app.app._seed_token_counters", new_callable=AsyncMock)
@patch("app.app.declare_token_queues")
@patch("app.app.display_service_info")
@patch("app.app.verify_token_maintenance_readiness", new_callable=AsyncMock)
@patch("app.app.verify_redis_connectivity", new_callable=AsyncMock)
@patch("app.app.verify_database_connectivity", new_callable=AsyncMock)
@patch("app.app.rate_limiter_manager")
@patch("app.app.redis_manager")
@patch("app.app.db_manager")
async def test_lifespan_starts_and_stops_maintenance(
    mock_db_manager,
    mock_redis_manager,
    mock_rate_limiter,
    mock_verify_database,
    mock_verify_redis,
    mock_verify_maintenance,
    mock_display_service_info,
    mock_declare_queues,
    mock_seed,
    mock_runner,
):
    """A serving app owns periodic jobs for exactly its lifespan."""
    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.close = AsyncMock()
    mock_redis_manager.initialize = MagicMock()
    mock_redis_manager.close = AsyncMock()
    mock_rate_limiter.close = AsyncMock()
    mock_runner.close = AsyncMock()
    mock_verify_database.return_value = ServiceStatus(
        name="PostgreSQL", status="connected"
    )
    mock_verify_redis.return_value = ServiceStatus(name="Redis", status="connected")
    mock_verify_maintenance.return_value = ServiceStatus(
        name="Token maintenance", status="connected"
    )

    async with lifespan(fastapi_app):
        mock_runner.start.assert_called_once_with()

    mock_runner.close.assert_awaited_once_with()
    mock_display_service_info.assert_called_once()
    mock_declare_queues.assert_called_once()
    mock_seed.assert_awaited_once()


@pytest.mark.asyncio
@patch("app.app.display_startup_failure")
@patch("app.app.verify_token_maintenance_readiness", new_callable=AsyncMock)
@patch("app.app.verify_redis_connectivity", new_callable=AsyncMock)
@patch("app.app.verify_database_connectivity", new_callable=AsyncMock)
@patch("app.app.rate_limiter_manager")
@patch("app.app.redis_manager")
@patch("app.app.db_manager")
async def test_lifespan_rejects_unavailable_maintenance(
    mock_db_manager,
    mock_redis_manager,
    mock_rate_limiter,
    mock_verify_database,
    mock_verify_redis,
    mock_verify_maintenance,
    mock_display_startup_failure,
):
    """A service cannot declare readiness when reconciliation cannot run."""
    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.close = AsyncMock()
    mock_redis_manager.initialize = MagicMock()
    mock_redis_manager.close = AsyncMock()
    mock_rate_limiter.close = AsyncMock()
    mock_verify_database.return_value = ServiceStatus(
        name="PostgreSQL", status="connected"
    )
    mock_verify_redis.return_value = ServiceStatus(name="Redis", status="connected")
    failed = ServiceStatus(name="Token maintenance", status="failed")
    mock_verify_maintenance.return_value = failed

    with pytest.raises(RuntimeError, match="Required startup dependencies"):
        async with lifespan(fastapi_app):
            pass

    mock_display_startup_failure.assert_called_once_with([failed])
    mock_rate_limiter.initialize.assert_called_once()
