from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.persistence.token_maintenance_persistence import TokenMaintenancePersistence


class _FakeResult:
    def __init__(self, rows: list[dict[str, object]] | None = None, rowcount: int = 0):
        self._rows = rows or []
        self.rowcount = rowcount

    def mappings(self) -> _FakeResult:
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _FakeSessionContext:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result
        self.execute = AsyncMock(return_value=result)

    async def __aenter__(self) -> _FakeSessionContext:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeDatabaseManager:
    def __init__(self, result: _FakeResult) -> None:
        self._context = _FakeSessionContext(result)

    def get_session(self) -> _FakeSessionContext:
        return self._context


@pytest.mark.asyncio
async def test_list_active_deployment_capacity_snapshots_returns_models() -> None:
    service = TokenMaintenancePersistence(
        _FakeDatabaseManager(
            _FakeResult(
                rows=[
                    {
                        "llm_model_name": "gpt-4o",
                        "api_endpoint_url": "https://endpoint-one",
                        "allocated_tokens": 500,
                        "max_tokens": 1000,
                    }
                ]
            )
        )
    )

    records = await service.list_active_deployment_capacity_snapshots()

    assert len(records) == 1
    assert records[0].llm_model_name == "gpt-4o"
    assert records[0].allocated_tokens == 500


@pytest.mark.asyncio
async def test_list_invalid_active_models_without_capacity_returns_models() -> None:
    service = TokenMaintenancePersistence(
        _FakeDatabaseManager(
            _FakeResult(
                rows=[
                    {
                        "llm_provider": "openai",
                        "llm_model_name": "gpt-4o",
                        "api_endpoint_url": "https://endpoint-one",
                        "deployment_name": "eastus-prod",
                        "deployment_region": "eastus",
                    }
                ]
            )
        )
    )

    records = await service.list_invalid_active_models_without_capacity()

    assert len(records) == 1
    assert records[0].llm_provider == "openai"
    assert records[0].deployment_region == "eastus"


@pytest.mark.asyncio
async def test_delete_expired_allocations_returns_rowcount() -> None:
    service = TokenMaintenancePersistence(_FakeDatabaseManager(_FakeResult(rowcount=4)))

    deleted_count = await service.delete_expired_allocations()

    assert deleted_count == 4
