"""A failed queue handoff must not strand a Redis token reservation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.request_models import TokenAllocationClientRequest
from app.resilience.redis_token_counter import TokenReservationResult
from app.resilience.token_queue.publisher import TokenPublishError
from app.services.token_acquisition_service import TokenAcquisitionService


@pytest.mark.asyncio
async def test_publish_failure_rolls_back_reservation_then_uses_db_path() -> None:
    publisher = MagicMock()
    publisher.publish_allocation_request.side_effect = TokenPublishError("disconnected")
    redis_counter = MagicMock()
    redis_counter.reserve_tokens = AsyncMock(
        return_value=TokenReservationResult.ALLOCATED
    )
    redis_counter.release_tokens = AsyncMock(return_value=0)
    service = TokenAcquisitionService(
        allocation_persistence=MagicMock(),
        load_balancer=MagicMock(),
        redis_counter=redis_counter,
        publisher=publisher,
        db_circuit_breaker=MagicMock(),
    )
    deployment = {
        "api_endpoint_url": "https://example.test/v1",
        "deployment_id": uuid4(),
        "deployment_key": "openai-deployment",
    }
    service._choose_deployment = AsyncMock(return_value=deployment)
    db_response = object()
    service._create_db_allocation = AsyncMock(return_value=db_response)
    request = TokenAllocationClientRequest(
        llm_provider="openai",
        model_name="gpt-4o",
        input_data="hello",
        requested_completion_tokens=10,
    )

    result = await service.acquire_tokens(uuid4(), uuid4(), request)

    assert result is db_response
    redis_counter.release_tokens.assert_awaited_once()
    service._create_db_allocation.assert_awaited_once()


@pytest.mark.asyncio
async def test_unexpected_publish_error_rolls_back_without_hiding_bug() -> None:
    publisher = MagicMock()
    publisher.publish_allocation_request.side_effect = RuntimeError("programming bug")
    redis_counter = MagicMock()
    redis_counter.release_tokens = AsyncMock(return_value=0)
    service = TokenAcquisitionService(
        allocation_persistence=MagicMock(),
        load_balancer=MagicMock(),
        redis_counter=redis_counter,
        publisher=publisher,
        db_circuit_breaker=MagicMock(),
    )
    request = TokenAllocationClientRequest(
        llm_provider="openai", model_name="gpt-4o", input_data="hello"
    )

    with pytest.raises(RuntimeError, match="programming bug"):
        await service._handle_fast_path(
            uuid4(),
            uuid4(),
            request,
            10,
            {
                "api_endpoint_url": "https://example.test/v1",
                "deployment_id": uuid4(),
                "deployment_key": "openai-deployment",
            },
        )

    redis_counter.release_tokens.assert_awaited_once()
