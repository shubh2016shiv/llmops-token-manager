from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import Response

from app.api import token_manager_endpoints as token_module
from app.models.request_models import TokenAllocationClientRequest, TokenReleaseRequest
from app.models.response_models import TokenAllocationResponse


class _PassThroughDbBreaker:
    async def call_async(self, func, *args, **kwargs):
        return await func(*args, **kwargs)


def test_acquire_tokens_falls_back_to_db_when_redis_breaker_errors(
    monkeypatch, mock_developer_user
) -> None:
    user_payload = {
        "user_id": mock_developer_user.user_id,
        "status": "active",
    }
    users_service = SimpleNamespace(get_user_by_id=AsyncMock(return_value=user_payload))

    allocation_service = MagicMock()
    allocation_service.get_least_loaded_deployment = AsyncMock(
        return_value=(
            0,
            {
                "api_endpoint_url": "https://api.openai.com/v1",
                "max_tokens": 100000,
            },
        )
    )

    request = TokenAllocationClientRequest(
        llm_provider="openai",
        llm_model_name="gpt-4o",
        input_data="hello",
        request_context={"source": "test"},
    )

    expected_response = TokenAllocationResponse(
        token_request_id="req_db_fallback",
        user_id=mock_developer_user.user_id,
        llm_provider="openai",
        llm_model_name="gpt-4o",
        deployment_name=None,
        cloud_provider=None,
        api_endpoint_url="https://api.openai.com/v1",
        deployment_region=None,
        token_count=42,
        allocation_status="ACQUIRED",
        allocated_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        request_context={"source": "test"},
        temperature=None,
        top_p=None,
        seed=None,
    )

    monkeypatch.setattr(
        token_module, "estimate_tokens", lambda *_: SimpleNamespace(total_tokens=42)
    )
    monkeypatch.setattr(
        token_module, "LLMTokenAllocationPersistence", lambda: allocation_service
    )
    monkeypatch.setattr(
        token_module, "get_db_circuit_breaker", lambda: _PassThroughDbBreaker()
    )
    monkeypatch.setattr(
        token_module._shared_token_counter_service,
        "reserve_tokens",
        AsyncMock(return_value=token_module.TokenReservationResult.COUNTER_MISS),
    )
    db_fallback_mock = AsyncMock(return_value=expected_response)
    monkeypatch.setattr(token_module, "_db_acquire_fallback", db_fallback_mock)

    result = asyncio.run(
        token_module.acquire_tokens(
            request=request,
            current_user=mock_developer_user,
            users_service=users_service,
            _rate_limit=None,
            _backpressure=None,
        )
    )

    assert result == expected_response
    db_fallback_mock.assert_awaited_once()


def test_release_tokens_sets_reconcile_header_when_redis_release_deferred(
    monkeypatch, mock_developer_user
) -> None:
    allocation_payload = {
        "token_request_id": "req_release_1",
        "user_id": str(mock_developer_user.user_id),
        "llm_model_name": "gpt-4o",
        "api_endpoint_url": "https://api.openai.com/v1",
        "token_count": 42,
    }

    allocation_service = MagicMock()
    allocation_service.get_allocation_by_request_id = AsyncMock(
        return_value=allocation_payload
    )
    allocation_service.delete_allocation = AsyncMock(return_value=True)

    monkeypatch.setattr(
        token_module, "LLMTokenAllocationPersistence", lambda: allocation_service
    )
    monkeypatch.setattr(
        token_module, "get_db_circuit_breaker", lambda: _PassThroughDbBreaker()
    )
    monkeypatch.setattr(
        token_module._shared_token_counter_service,
        "release_tokens",
        AsyncMock(return_value=None),
    )

    response = Response()
    result = asyncio.run(
        token_module.release_tokens(
            request=TokenReleaseRequest(token_request_id="req_release_1"),
            response=response,
            current_user=mock_developer_user,
        )
    )

    assert result.token_request_id == "req_release_1"
    assert response.headers.get("X-Redis-Counter-Reconcile") == "pending"
