from __future__ import annotations

import asyncio

import pytest
from redis import exceptions as redis_exceptions

from app.models.resilience_models import BackpressureDecision
from app.resilience.backpressure import pool_probe as pool_probe_module
from app.resilience.backpressure import queue_depth_probe as queue_depth_probe_module
from app.resilience.backpressure.decision_to_http import (
    raise_for_backpressure_decision,
)


class _RedisWithMissingKey:
    async def get(self, _key: str) -> None:
        return None


class _RedisWithMalformedValue:
    async def get(self, _key: str) -> str:
        return "not-an-int"


class _RedisThatFails:
    async def get(self, _key: str) -> str:
        raise redis_exceptions.ConnectionError("redis unavailable")


class _FakePool:
    def __init__(self, size: int, checked_out: int) -> None:
        self._size = size
        self._checked_out = checked_out

    def size(self) -> int:
        return self._size

    def checkedout(self) -> int:
        return self._checked_out


class _DbManagerWithPool:
    def __init__(self, pool: object | None) -> None:
        self.pool = pool


class _RedisManagerWithClient:
    def __init__(self, client: object) -> None:
        self.client = client


class _BrokenPool:
    def size(self) -> int:
        raise RuntimeError("pool unavailable")


def test_queue_depth_probe_returns_none_when_key_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        queue_depth_probe_module,
        "redis_manager",
        _RedisManagerWithClient(_RedisWithMissingKey()),
    )

    depth = asyncio.run(queue_depth_probe_module.read_queue_depth())

    assert depth is None


def test_queue_depth_probe_returns_none_for_malformed_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        queue_depth_probe_module,
        "redis_manager",
        _RedisManagerWithClient(_RedisWithMalformedValue()),
    )

    depth = asyncio.run(queue_depth_probe_module.read_queue_depth())

    assert depth is None


def test_queue_depth_probe_fails_open_when_redis_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        queue_depth_probe_module,
        "redis_manager",
        _RedisManagerWithClient(_RedisThatFails()),
    )

    depth = asyncio.run(queue_depth_probe_module.read_queue_depth())

    assert depth is None


def test_pool_probe_returns_none_when_pool_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        pool_probe_module,
        "db_manager",
        _DbManagerWithPool(pool=None),
    )

    utilization_pct = pool_probe_module.read_db_pool_utilization_pct()

    assert utilization_pct is None


def test_pool_probe_returns_none_when_pool_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        pool_probe_module,
        "db_manager",
        _DbManagerWithPool(pool=_BrokenPool()),
    )

    utilization_pct = pool_probe_module.read_db_pool_utilization_pct()

    assert utilization_pct is None


def test_pool_probe_returns_utilization_percent(monkeypatch) -> None:
    monkeypatch.setattr(
        pool_probe_module,
        "db_manager",
        _DbManagerWithPool(pool=_FakePool(size=10, checked_out=7)),
    )

    utilization_pct = pool_probe_module.read_db_pool_utilization_pct()

    assert utilization_pct == 70


def test_http_exception_factory_emits_expected_headers() -> None:
    decision = BackpressureDecision(
        should_reject_request=True,
        reason="queue_depth_exceeded",
        retry_after_seconds=12,
        queue_depth=12001,
    )

    with pytest.raises(Exception) as exc_info:
        raise_for_backpressure_decision(decision)

    http_exception = exc_info.value
    assert http_exception.status_code == 503
    assert http_exception.headers["Retry-After"] == "12"
    assert http_exception.headers["X-Backpressure-Reason"] == "queue_depth_exceeded"
    assert http_exception.detail["queue_depth"] == 12001
