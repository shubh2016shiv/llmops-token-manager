from __future__ import annotations

from datetime import timedelta
import threading
import time
from typing import TYPE_CHECKING, Any

import aiobreaker
from aiobreaker.storage import CircuitMemoryStorage
import pytest

if TYPE_CHECKING:
    from aiobreaker.storage.base import CircuitBreakerStorage

from app.resilience import circuit_breaker as cb_module
from app.resilience.circuit_breaker.breaker_listener import CircuitBreakerListener


def test_listener_does_not_mask_failure_when_breaker_opens() -> None:
    breaker = aiobreaker.CircuitBreaker(
        fail_max=1,
        timeout_duration=timedelta(seconds=30),
        state_storage=CircuitMemoryStorage(aiobreaker.CircuitBreakerState.CLOSED),
        listeners=[CircuitBreakerListener()],
        name="rabbitmq-test",
    )

    def _fail() -> None:
        raise RuntimeError("broker unavailable")

    with pytest.raises(aiobreaker.CircuitBreakerError):
        breaker.call(_fail)

    assert breaker.current_state is aiobreaker.CircuitBreakerState.OPEN


def _reset_registry_state() -> None:
    cb_module._circuit_breaker_registry.clear()
    cb_module.breaker_storage._synchronous_redis_client = None


def test_build_storage_uses_memory_for_postgres() -> None:
    _reset_registry_state()

    storage = cb_module.build_breaker_storage("postgres")

    assert isinstance(storage, CircuitMemoryStorage)
    assert storage.state == aiobreaker.CircuitBreakerState.CLOSED


def test_build_storage_uses_redis_with_expected_namespace_and_open_fallback(
    monkeypatch,
) -> None:
    _reset_registry_state()
    captured: list[dict[str, Any]] = []

    class _DummyRedis:
        pass

    def _fake_circuit_redis_storage(
        *args: object, **kwargs: object
    ) -> CircuitMemoryStorage:
        captured.append({"args": args, "kwargs": kwargs})
        return CircuitMemoryStorage(aiobreaker.CircuitBreakerState.CLOSED)

    monkeypatch.setattr(
        cb_module.breaker_storage,
        "build_synchronous_redis_client",
        lambda: _DummyRedis(),
    )
    monkeypatch.setattr(
        cb_module.breaker_storage,
        "CircuitRedisStorage",
        _fake_circuit_redis_storage,
    )

    cb_module.build_breaker_storage("redis")
    cb_module.build_breaker_storage("rabbitmq")

    assert len(captured) == 2
    assert captured[0]["kwargs"]["namespace"] == "cb:redis"
    assert captured[1]["kwargs"]["namespace"] == "cb:rabbitmq"
    assert (
        captured[0]["kwargs"]["fallback_circuit_state"]
        == aiobreaker.CircuitBreakerState.OPEN
    )
    assert (
        captured[1]["kwargs"]["fallback_circuit_state"]
        == aiobreaker.CircuitBreakerState.OPEN
    )


def test_make_circuit_breaker_is_thread_safe(monkeypatch) -> None:
    _reset_registry_state()
    build_calls = {"count": 0}

    def _fake_build_storage(_name: str) -> CircuitBreakerStorage:
        build_calls["count"] += 1
        time.sleep(0.01)
        return CircuitMemoryStorage(aiobreaker.CircuitBreakerState.CLOSED)

    monkeypatch.setattr(
        cb_module.breaker_storage, "build_breaker_storage", _fake_build_storage
    )

    created_breakers: list[aiobreaker.CircuitBreaker] = []
    list_lock = threading.Lock()

    def _worker() -> None:
        cb = cb_module.create_circuit_breaker(
            breaker_name="redis",
            failure_threshold=3,
            recovery_timeout_seconds=10,
        )
        with list_lock:
            created_breakers.append(cb)

    threads = [threading.Thread(target=_worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert build_calls["count"] == 1
    assert len({id(cb) for cb in created_breakers}) == 1


def test_get_circuit_breaker_states_normalizes_to_public_enum(monkeypatch) -> None:
    _reset_registry_state()
    monkeypatch.setattr(
        cb_module.breaker_storage,
        "build_breaker_storage",
        lambda _name: CircuitMemoryStorage(aiobreaker.CircuitBreakerState.CLOSED),
    )
    cb = cb_module.create_circuit_breaker("redis", 3, 10)
    cb._state_storage.state = aiobreaker.CircuitBreakerState.OPEN

    states = cb_module.get_circuit_breaker_states()

    assert states["redis"] == cb_module.CircuitBreakerState.OPEN.value


def test_close_circuit_breaker_redis_client_disconnects_pool() -> None:
    _reset_registry_state()

    class _DummyPool:
        def __init__(self) -> None:
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    class _DummyClient:
        def __init__(self) -> None:
            self.closed = False
            self.connection_pool = _DummyPool()

        def close(self) -> None:
            self.closed = True

    dummy_client = _DummyClient()
    cb_module.breaker_storage._synchronous_redis_client = dummy_client  # type: ignore[assignment]

    cb_module.close_circuit_breaker_redis_client()

    assert dummy_client.closed is True
    assert dummy_client.connection_pool.disconnected is True
    assert cb_module.breaker_storage._synchronous_redis_client is None
