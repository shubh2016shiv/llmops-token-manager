"""Current allocation persistence contracts, without removed repository APIs."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.persistence.allocations import LLMTokenAllocationPersistence
from app.persistence.queries.allocation_queries import (
    CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL,
    TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL,
)


def _repository(*rows: dict[str, object] | None):
    session = MagicMock()
    results = []
    for row in rows:
        result = MagicMock()
        result.mappings.return_value.one_or_none.return_value = row
        results.append(result)
    session.execute = AsyncMock(side_effect=results)

    @asynccontextmanager
    async def get_session():
        yield session

    manager = MagicMock()
    manager.get_session = get_session
    return LLMTokenAllocationPersistence(manager), session


def _reservation() -> dict[str, object]:
    return {
        "token_request_identifier": "req_1",
        "tenant_id": uuid4(),
        "user_id": uuid4(),
        "deployment_id": uuid4(),
        "provider_name": "openai",
        "model_name": "gpt-4o",
        "deployment_key": "test-route",
        "api_endpoint_url": "https://example.test",
        "token_count": 100,
    }


@pytest.mark.asyncio
async def test_identical_reserved_message_is_idempotent():
    values = _reservation()
    existing = {
        "token_request_id": values["token_request_identifier"],
        **{
            key: values[key]
            for key in (
                "tenant_id",
                "user_id",
                "deployment_id",
                "provider_name",
                "model_name",
                "deployment_key",
                "api_endpoint_url",
                "token_count",
            )
        },
        "allocation_status": "ACQUIRED",
    }
    repository, session = _repository(None, existing)
    assert await repository.create_reserved_allocation(**values) == existing
    assert session.execute.await_count == 2


@pytest.mark.asyncio
async def test_conflicting_reserved_message_is_rejected():
    values = _reservation()
    existing = {
        key: values[key]
        for key in (
            "tenant_id",
            "user_id",
            "deployment_id",
            "provider_name",
            "model_name",
            "deployment_key",
            "api_endpoint_url",
            "token_count",
        )
    }
    existing["allocation_status"] = "ACQUIRED"
    existing["tenant_id"] = uuid4()
    repository, _ = _repository(None, existing)
    with pytest.raises(ValueError, match="Conflicting reserved allocation"):
        await repository.create_reserved_allocation(**values)


@pytest.mark.asyncio
async def test_reserved_allocation_rejects_invalid_capacity_before_db():
    values = _reservation()
    values["token_count"] = True
    repository, session = _repository()
    with pytest.raises(ValueError, match="must be an integer"):
        await repository.create_reserved_allocation(**values)
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_reserved_allocation_rejects_non_finite_metadata_before_db():
    values = _reservation()
    values["request_metadata"] = {"temperature": float("nan")}
    repository, session = _repository()
    with pytest.raises(ValueError, match="valid JSON"):
        await repository.create_reserved_allocation(**values)
    session.execute.assert_not_awaited()


def test_atomic_acquire_is_scoped_to_requested_deployment_identity():
    sql = CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL
    assert "td.tenant_id = :tenant_id" in sql
    assert "pc.provider_name = :provider_name" in sql
    assert "mc.model_name = :model_name" in sql
    assert "FOR UPDATE OF td" in sql


def test_atomic_retry_preserves_waiting_allocation_identity():
    sql = TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL
    assert "waiting.tenant_id = deployment.tenant_id" in sql
    assert "waiting.provider_name = deployment.provider_name" in sql
    assert "waiting.model_name = deployment.model_name" in sql
