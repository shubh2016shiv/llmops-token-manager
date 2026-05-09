from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.models.resilience_models import (
    CounterSeedRecord,
    InvalidActiveDeploymentRecord,
)
from app.resilience.token_maintenance import reconciliation as tr


class _FakeRedisClient:
    def __init__(self, *, lock_acquired: bool) -> None:
        self.set = AsyncMock(return_value=lock_acquired)
        self.delete = AsyncMock(return_value=1)


class _FakeCounterService:
    def __init__(
        self,
        *,
        lock_acquired: bool = True,
        current_counter: tuple[int, int] | None = (600, 1000),
    ) -> None:
        self.redis_client = _FakeRedisClient(lock_acquired=lock_acquired)
        self.get_counter = AsyncMock(return_value=current_counter)
        self.reconcile_counter = AsyncMock(
            return_value=tr.CounterReconciliationResult.DELTA_APPLIED
        )


class _FakePersistence:
    def __init__(
        self,
        seed_records: list[CounterSeedRecord],
        invalid_active_deployments: list[InvalidActiveDeploymentRecord] | None = None,
    ) -> None:
        self._seed_records = seed_records
        self._invalid_active_deployments = invalid_active_deployments or []

    async def list_invalid_active_models_without_capacity(
        self,
    ) -> list[InvalidActiveDeploymentRecord]:
        return self._invalid_active_deployments

    async def list_active_deployment_capacity_snapshots(
        self,
    ) -> list[CounterSeedRecord]:
        return self._seed_records


class _FakeLogger:
    def __init__(self) -> None:
        self.warning_calls: list[tuple[str, dict[str, object]]] = []
        self.info_calls: list[tuple[str, dict[str, object]]] = []
        self.error_calls: list[tuple[str, dict[str, object]]] = []

    def warning(self, message: str, **kwargs: object) -> None:
        self.warning_calls.append((message, kwargs))

    def info(self, message: str, **kwargs: object) -> None:
        self.info_calls.append((message, kwargs))

    def error(self, message: str, **kwargs: object) -> None:
        self.error_calls.append((message, kwargs))


def _seed_records() -> list[CounterSeedRecord]:
    return [
        CounterSeedRecord(
            llm_model_name="gpt-4o",
            api_endpoint_url="https://endpoint-one",
            allocated_tokens=500,
            max_tokens=1000,
        )
    ]


def _invalid_active_deployments() -> list[InvalidActiveDeploymentRecord]:
    return [
        InvalidActiveDeploymentRecord(
            llm_provider="openai",
            llm_model_name="gpt-4o",
            api_endpoint_url="https://endpoint-invalid",
            deployment_name="broken-deployment",
            deployment_region="eastus",
        )
    ]


def test_reconcile_uses_atomic_reconcile_method(monkeypatch) -> None:
    fake_counter = _FakeCounterService()
    monkeypatch.setattr(
        tr,
        "TokenMaintenancePersistence",
        lambda: _FakePersistence(_seed_records()),
    )
    monkeypatch.setattr(
        tr,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )

    asyncio.run(tr.reconcile_async())

    fake_counter.reconcile_counter.assert_awaited_once()


def test_reconcile_skips_when_lock_is_held(monkeypatch) -> None:
    fake_counter = _FakeCounterService(lock_acquired=False)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        tr,
        "TokenMaintenancePersistence",
        lambda: _FakePersistence(_seed_records()),
    )
    monkeypatch.setattr(
        tr,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )
    monkeypatch.setattr(tr, "logger", fake_logger)

    asyncio.run(tr.reconcile_async())

    fake_counter.reconcile_counter.assert_not_awaited()
    assert fake_logger.info_calls


def test_reconcile_warns_for_large_drift(monkeypatch) -> None:
    fake_counter = _FakeCounterService(current_counter=(900, 1000))
    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        tr,
        "TokenMaintenancePersistence",
        lambda: _FakePersistence(_seed_records()),
    )
    monkeypatch.setattr(
        tr,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )
    monkeypatch.setattr(tr, "logger", fake_logger)

    asyncio.run(tr.reconcile_async())

    assert any(
        call_kwargs.get("drift_magnitude") == 400
        for _, call_kwargs in fake_logger.warning_calls
    )


def test_reconcile_initializes_missing_counters(monkeypatch) -> None:
    fake_counter = _FakeCounterService(current_counter=None)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        tr,
        "TokenMaintenancePersistence",
        lambda: _FakePersistence(_seed_records()),
    )
    monkeypatch.setattr(
        tr,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )
    monkeypatch.setattr(tr, "logger", fake_logger)

    asyncio.run(tr.reconcile_async())

    fake_counter.reconcile_counter.assert_awaited_once()
    assert any(
        call_kwargs.get("missing_snapshot") == 1
        for _, call_kwargs in fake_logger.info_calls
    )


def test_reconcile_logs_invalid_active_deployments_and_continues(monkeypatch) -> None:
    fake_counter = _FakeCounterService()
    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        tr,
        "TokenMaintenancePersistence",
        lambda: _FakePersistence(
            _seed_records(),
            invalid_active_deployments=_invalid_active_deployments(),
        ),
    )
    monkeypatch.setattr(
        tr,
        "get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )
    monkeypatch.setattr(tr, "logger", fake_logger)

    asyncio.run(tr.reconcile_async())

    fake_counter.reconcile_counter.assert_awaited_once()
    assert any(
        call_kwargs.get("deployment_name") == "broken-deployment"
        for _, call_kwargs in fake_logger.error_calls
    )
