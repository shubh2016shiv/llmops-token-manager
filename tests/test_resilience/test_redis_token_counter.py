from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pybreaker
import redis

from app.resilience import redis_token_counter as rtc


class _NoopRedis:
    def register_script(self, _script):
        async def _runner(*args, **kwargs):
            return 1

        return _runner


def test_reserve_tokens_trips_redis_breaker_on_transport_failures(monkeypatch) -> None:
    breaker = pybreaker.CircuitBreaker(
        fail_max=2,
        reset_timeout=60,
        state_storage=pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED),
        name="redis-test",
    )
    monkeypatch.setattr(rtc, "get_redis_circuit_breaker", lambda: breaker)

    counter = rtc.RedisTokenCounter(_NoopRedis())
    counter._reserve_tokens_raw = AsyncMock(
        side_effect=redis.exceptions.ConnectionError("redis unavailable")
    )

    first = asyncio.run(counter.reserve_tokens("gpt-4", "https://endpoint", 100))
    second = asyncio.run(counter.reserve_tokens("gpt-4", "https://endpoint", 100))

    assert first == rtc.FastPathResult.FAST_PATH_MISS
    assert second == rtc.FastPathResult.FAST_PATH_MISS
    assert breaker.current_state == pybreaker.STATE_OPEN
    assert counter._reserve_tokens_raw.await_count == 2


def test_reserve_tokens_returns_miss_when_breaker_is_open(monkeypatch) -> None:
    breaker = pybreaker.CircuitBreaker(
        fail_max=2,
        reset_timeout=60,
        state_storage=pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED),
        name="redis-test",
    )
    breaker.open()
    monkeypatch.setattr(rtc, "get_redis_circuit_breaker", lambda: breaker)

    counter = rtc.RedisTokenCounter(_NoopRedis())
    counter._reserve_tokens_raw = AsyncMock(
        return_value=rtc.FastPathResult.FAST_PATH_ALLOCATED
    )

    result = asyncio.run(counter.reserve_tokens("gpt-4", "https://endpoint", 100))

    assert result == rtc.FastPathResult.FAST_PATH_MISS
    counter._reserve_tokens_raw.assert_not_awaited()


def test_seed_counter_does_not_use_circuit_breaker(monkeypatch) -> None:
    breaker_called = {"called": False}

    def _raise_if_called():
        breaker_called["called"] = True
        raise AssertionError("seed_counter should not call get_redis_circuit_breaker")

    monkeypatch.setattr(rtc, "get_redis_circuit_breaker", _raise_if_called)

    class _Pipe:
        def __init__(self) -> None:
            self.commands: list[tuple[str, int, int]] = []

        def set(self, key: str, value: int, ex: int) -> None:
            self.commands.append((key, value, ex))

        async def execute(self) -> None:
            return None

    class _SeedRedis:
        def register_script(self, _script):
            async def _runner(*args, **kwargs):
                return 1

            return _runner

        def pipeline(self) -> _Pipe:
            return _Pipe()

    counter = rtc.RedisTokenCounter(_SeedRedis())

    asyncio.run(counter.seed_counter("gpt-4", "https://endpoint", 50, 1000))

    assert breaker_called["called"] is False


def test_release_tokens_returns_none_when_breaker_is_open(monkeypatch) -> None:
    breaker = pybreaker.CircuitBreaker(
        fail_max=2,
        reset_timeout=60,
        state_storage=pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED),
        name="redis-test",
    )
    breaker.open()
    monkeypatch.setattr(rtc, "get_redis_circuit_breaker", lambda: breaker)

    counter = rtc.RedisTokenCounter(_NoopRedis())
    counter._release_tokens_raw = AsyncMock(return_value=0)

    result = asyncio.run(counter.release_tokens("gpt-4", "https://endpoint", 100))

    assert result is None
    counter._release_tokens_raw.assert_not_awaited()
