from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.resilience import token_worker as tw
import app.resilience.redis_token_counter as redis_token_counter_package
from app.resilience.redis_token_counter import (
    service_registry as service_registry_module,
)


class _FakeQueryResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _query):
        return _FakeQueryResult(self._rows)


class _FakePersistence:
    def __init__(self, rows):
        self._rows = rows

    def get_session(self):
        return _FakeSession(self._rows)


def _snapshot_rows():
    return [
        {
            "llm_model_name": "gpt-4o",
            "api_endpoint_url": "https://endpoint-one",
            "allocated_tokens": 500,
            "max_tokens": 1000,
        }
    ]


def test_reconcile_uses_atomic_reconcile_method(monkeypatch) -> None:
    fake_counter = type("Counter", (), {})()
    fake_counter.get_counter = AsyncMock(return_value=(600, 1000))
    fake_counter.reconcile_counter = AsyncMock(
        return_value=redis_token_counter_package.CounterReconciliationResult.DELTA_APPLIED
    )
    fake_counter.seed_counter = AsyncMock()

    monkeypatch.setattr(
        tw,
        "LLMTokenAllocationPersistence",
        lambda: _FakePersistence(_snapshot_rows()),
    )
    monkeypatch.setattr(
        tw,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )

    asyncio.run(tw._reconcile_async())

    fake_counter.reconcile_counter.assert_awaited_once()
    fake_counter.seed_counter.assert_not_awaited()


def test_reconcile_is_safe_under_concurrent_runs(monkeypatch) -> None:
    fake_counter = type("Counter", (), {})()
    fake_counter.get_counter = AsyncMock(return_value=(500, 1000))
    fake_counter.reconcile_counter = AsyncMock(
        return_value=redis_token_counter_package.CounterReconciliationResult.UNCHANGED
    )

    monkeypatch.setattr(
        tw,
        "LLMTokenAllocationPersistence",
        lambda: _FakePersistence(_snapshot_rows()),
    )
    monkeypatch.setattr(
        tw,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )

    async def _run_concurrent_reconcile() -> None:
        await asyncio.gather(tw._reconcile_async(), tw._reconcile_async())

    asyncio.run(_run_concurrent_reconcile())

    assert fake_counter.reconcile_counter.await_count == 2


def test_reconcile_reuses_singleton_counter_without_pool_churn(monkeypatch) -> None:
    fake_counter = type("Counter", (), {})()
    fake_counter.get_counter = AsyncMock(return_value=(500, 1000))
    fake_counter.reconcile_counter = AsyncMock(
        return_value=redis_token_counter_package.CounterReconciliationResult.UNCHANGED
    )
    fake_counter.close = AsyncMock(return_value=None)

    build_counter_calls = {"count": 0}

    def _build_counter():
        build_counter_calls["count"] += 1
        return fake_counter

    service_registry_module._shared_redis_token_counter_service = None
    monkeypatch.setattr(
        service_registry_module,
        "create_redis_token_counter_service",
        _build_counter,
    )
    monkeypatch.setattr(
        tw,
        "LLMTokenAllocationPersistence",
        lambda: _FakePersistence(_snapshot_rows()),
    )
    monkeypatch.setattr(
        tw,
        "get_shared_redis_token_counter_service",
        redis_token_counter_package.get_shared_redis_token_counter_service,
    )

    asyncio.run(tw._reconcile_async())
    asyncio.run(tw._reconcile_async())

    assert build_counter_calls["count"] == 1
    asyncio.run(redis_token_counter_package.close_shared_redis_token_counter_service())
