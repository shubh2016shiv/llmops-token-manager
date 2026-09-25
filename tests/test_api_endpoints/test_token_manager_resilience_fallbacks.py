"""
Resilience Fallback Tests for Token Manager
============================================
Tests targeting the resilience mechanisms introduced in the service layer:

1. Redis COUNTER_MISS → DB path fallback inside TokenAcquisitionService
2. Redis release deferred → X-Redis-Counter-Reconcile header via the release endpoint

These tests go below the endpoint boundary so the fallback logic is exercised
directly, not hidden behind a top-level service mock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.token_manager_endpoints import (
    get_token_release_service,
    router,
)
from app.models.request_models import TokenAllocationClientRequest
from app.services.token_release_service import TokenReleaseService


def test_acquire_tokens_service_falls_back_to_db_when_redis_counter_miss(
    mock_developer_user,
) -> None:
    """When Redis returns COUNTER_MISS (counter not seeded), service uses the DB path."""
    from app.resilience.redis_token_counter import TokenReservationResult
    from app.services.token_acquisition_service import TokenAcquisitionService

    user_id = mock_developer_user.user_id

    deployment_config = {
        "api_endpoint_url": "https://api.openai.com/v1",
        "deployment_id": uuid4(),
        "token_capacity_limit": 100_000,
        "token_lock_duration_seconds": 70,
        "deployment_name": None,
        "cloud_provider": None,
        "cloud_region": "eastus2",
        "temperature": 0.7,
        "seed": 42,
    }
    allocation_persistence = MagicMock()
    allocation_persistence.create_allocation_with_capacity_check = AsyncMock(
        return_value={
            "token_request_id": "req_db_fallback",
            "user_id": user_id,
            "tenant_id": mock_developer_user.tenant_id,
            "provider_name": "openai",
            "model_name": "gpt-4o",
            "token_count": 42,
            "allocation_status": "ACQUIRED",
            "allocated_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
            "deployment_name": None,
            "api_endpoint_url": "https://api.openai.com/v1",
            "cloud_region": "eastus2",
        }
    )

    redis_counter = MagicMock()
    redis_counter.reserve_tokens = AsyncMock(
        return_value=TokenReservationResult.COUNTER_MISS
    )
    publisher = MagicMock()
    load_balancer = MagicMock()
    load_balancer.choose_least_loaded = AsyncMock(return_value=deployment_config)

    db_circuit_breaker = MagicMock()

    async def call_async(func, *args, **kwargs):
        return await func(*args, **kwargs)

    db_circuit_breaker.call_async = AsyncMock(side_effect=call_async)
    service = TokenAcquisitionService(
        allocation_persistence=allocation_persistence,
        load_balancer=load_balancer,
        redis_counter=redis_counter,
        publisher=publisher,
        db_circuit_breaker=db_circuit_breaker,
    )

    request = TokenAllocationClientRequest(
        llm_provider="openai",
        model_name="gpt-4o",
        input_data="hello",
        request_context={"source": "test"},
    )

    with patch(
        "app.services.token_acquisition_service.estimate_tokens",
        return_value=SimpleNamespace(total_tokens=42),
    ):
        result = asyncio.run(
            service.acquire_tokens(user_id, mock_developer_user.tenant_id, request)
        )

    assert result.token_request_id == "req_db_fallback"
    assert result.allocation_status == "ACQUIRED"
    load_balancer.choose_least_loaded.assert_awaited_once_with(
        mock_developer_user.tenant_id, "openai", "gpt-4o", None
    )
    # DB path was used via the atomic capacity-check insert.
    allocation_persistence.create_allocation_with_capacity_check.assert_awaited_once()
    # Fast path was skipped — RMQ publisher must not have been called
    publisher.publish_allocation_request.assert_not_called()


def test_release_tokens_sets_reconcile_header_when_redis_release_deferred(
    mock_developer_user,
) -> None:
    """When execute_release returns True (Redis deferred), endpoint sets the reconcile header."""
    from app.auth.auth_dependencies import get_current_user

    allocation_payload = {
        "token_request_id": "req_release_1",
        "user_id": str(mock_developer_user.user_id),
        "tenant_id": str(mock_developer_user.tenant_id),
        "llm_model_name": "gpt-4o",
        "api_endpoint_url": "https://api.openai.com/v1",
        "token_count": 42,
    }

    mock_service = MagicMock()
    mock_service.fetch_allocation_for_release = AsyncMock(
        return_value=allocation_payload
    )
    mock_service.execute_release = AsyncMock(return_value=True)  # Redis deferred

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: mock_developer_user
    app.dependency_overrides[get_token_release_service] = lambda: mock_service

    client = TestClient(app)
    response = client.put(
        "/api/v1/tokens/release",
        json={"token_request_id": "req_release_1"},
    )

    assert response.status_code == 200
    assert response.json()["token_request_id"] == "req_release_1"
    assert response.headers.get("X-Redis-Counter-Reconcile") == "pending"


def test_release_service_does_not_release_redis_for_waiting_allocation() -> None:
    """WAITING rows carry endpoint metadata but have not reserved Redis tokens."""
    allocation_persistence = MagicMock()
    allocation_persistence.delete_allocation = AsyncMock(return_value=True)

    redis_counter = MagicMock()
    redis_counter.release_tokens = AsyncMock(return_value=0)

    service = TokenReleaseService(
        allocation_persistence=allocation_persistence,
        redis_counter=redis_counter,
        db_circuit_breaker=MagicMock(),
    )

    allocation = {
        "token_request_id": "req_waiting",
        "allocation_status": "WAITING",
        "llm_model_name": "gpt-4o",
        "api_endpoint_url": "https://api.openai.com/v1",
        "token_count": 42,
    }

    redis_deferred = asyncio.run(service.execute_release("req_waiting", allocation))

    assert redis_deferred is False
    allocation_persistence.delete_allocation.assert_awaited_once_with("req_waiting")
    redis_counter.release_tokens.assert_not_awaited()


def test_acquired_release_with_missing_counter_metadata_is_deferred() -> None:
    """Incomplete DB metadata must not be reported as a clean Redis release."""
    allocation_persistence = MagicMock()
    allocation_persistence.delete_allocation = AsyncMock(return_value=True)
    redis_counter = MagicMock()
    redis_counter.release_tokens = AsyncMock()
    service = TokenReleaseService(
        allocation_persistence=allocation_persistence,
        redis_counter=redis_counter,
        db_circuit_breaker=MagicMock(),
    )

    deferred = asyncio.run(
        service.execute_release(
            "req_incomplete",
            {"allocation_status": "ACQUIRED", "token_count": 42},
        )
    )

    assert deferred is True
    redis_counter.release_tokens.assert_not_awaited()
