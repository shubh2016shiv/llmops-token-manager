from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pybreaker
from redis import exceptions as redis_exceptions

from app.resilience.redis_token_counter import (
    CounterReconciliationResult,
    RedisTokenCounterService,
    TokenReservationResult,
)
from app.resilience.redis_token_counter import counter_service as counter_service_module
from app.resilience.redis_token_counter.lua_script_definitions import (
    LUA_RECONCILE_COUNTER,
)


class _PassThroughBreaker:
    async def call_async(self, func, *args, **kwargs):
        return await func(*args, **kwargs)


class _OpenBreaker:
    async def call_async(self, _func, *_args, **_kwargs):
        raise pybreaker.CircuitBreakerError("open")


class _ScriptRedis:
    def __init__(self) -> None:
        self.pipeline_called = False
        self.close_count = 0

    def register_script(self, _script_text):
        async def _runner(*_args, **_kwargs):
            return 1

        return _runner

    def pipeline(self):
        self.pipeline_called = True
        raise AssertionError("pipeline should not be used for seed/reconcile scripts")

    async def close(self):
        self.close_count += 1


class _NoScriptRedis:
    def __init__(self) -> None:
        self.register_count = 0

    def register_script(self, _script_text):
        self.register_count += 1
        if self.register_count == 1:

            async def _fail(*_args, **_kwargs):
                raise redis_exceptions.NoScriptError("NOSCRIPT")

            return _fail

        async def _succeed(*_args, **_kwargs):
            return 1

        return _succeed

    async def close(self):
        return None


def test_reserve_tokens_returns_counter_miss_when_breaker_is_open(monkeypatch) -> None:
    monkeypatch.setattr(
        counter_service_module,
        "get_redis_circuit_breaker",
        lambda: _OpenBreaker(),
    )
    counter_service = RedisTokenCounterService(_ScriptRedis())

    result = asyncio.run(
        counter_service.reserve_tokens("gpt-4", "https://endpoint", 100)
    )

    assert result == TokenReservationResult.COUNTER_MISS


def test_reserve_tokens_returns_counter_miss_on_transport_error(monkeypatch) -> None:
    monkeypatch.setattr(
        counter_service_module,
        "get_redis_circuit_breaker",
        lambda: _PassThroughBreaker(),
    )
    counter_service = RedisTokenCounterService(_ScriptRedis())
    counter_service._reserve_tokens_raw = AsyncMock(
        side_effect=redis_exceptions.ConnectionError("redis unavailable")
    )

    result = asyncio.run(
        counter_service.reserve_tokens("gpt-4", "https://endpoint", 100)
    )

    assert result == TokenReservationResult.COUNTER_MISS


def test_seed_counter_is_atomic_and_avoids_pipeline(monkeypatch) -> None:
    breaker_called = {"called": False}

    def _raise_if_called():
        breaker_called["called"] = True
        raise AssertionError("seed_counter should not call get_redis_circuit_breaker")

    monkeypatch.setattr(
        counter_service_module,
        "get_redis_circuit_breaker",
        _raise_if_called,
    )
    redis_client = _ScriptRedis()
    counter_service = RedisTokenCounterService(redis_client)

    asyncio.run(counter_service.seed_counter("gpt-4", "https://endpoint", 50, 1000))

    assert breaker_called["called"] is False
    assert redis_client.pipeline_called is False


def test_noscript_recovery_retries_once(monkeypatch) -> None:
    monkeypatch.setattr(
        counter_service_module,
        "get_redis_circuit_breaker",
        lambda: _PassThroughBreaker(),
    )
    redis_client = _NoScriptRedis()
    counter_service = RedisTokenCounterService(redis_client)

    result = asyncio.run(
        counter_service._reserve_tokens_raw("gpt-4", "https://endpoint", 25)
    )

    assert result == TokenReservationResult.ALLOCATED
    assert redis_client.register_count == 2


def test_reconcile_counter_status_mapping() -> None:
    counter_service = RedisTokenCounterService(_ScriptRedis())
    counter_service._execute_lua_script = AsyncMock(return_value=2)

    result = asyncio.run(
        counter_service.reconcile_counter(
            model_name="gpt-4",
            api_endpoint_url="https://endpoint",
            allocated_tokens_from_db=123,
            max_tokens_from_db=1000,
        )
    )

    assert result == CounterReconciliationResult.RESEEDED_PARTIAL
    counter_service._execute_lua_script.assert_awaited_once()


def test_reconcile_counter_uses_atomic_script_contract() -> None:
    counter_service = RedisTokenCounterService(_ScriptRedis())
    counter_service._execute_lua_script = AsyncMock(return_value=0)

    asyncio.run(
        counter_service.reconcile_counter(
            model_name="gpt-4",
            api_endpoint_url="https://endpoint",
            allocated_tokens_from_db=100,
            max_tokens_from_db=1000,
        )
    )

    script_call = counter_service._execute_lua_script.await_args.kwargs
    assert script_call["script_name"] == counter_service_module.SCRIPT_NAME_RECONCILE
    assert script_call["args"] == [
        100,
        1000,
        counter_service_module.settings.redis_token_counter_ttl_secs,
    ]


def test_reconcile_lua_clamps_negative_incrby() -> None:
    assert "if new_allocated < 0 then" in LUA_RECONCILE_COUNTER
    assert "redis.call('SET', KEYS[1], 0)" in LUA_RECONCILE_COUNTER


def test_reconcile_lua_handles_missing_limit_without_blind_counter_set() -> None:
    assert "if limit_val == false then" in LUA_RECONCILE_COUNTER
    assert "if counter_val == false or limit_val == false then" not in (
        LUA_RECONCILE_COUNTER
    )


def test_release_tokens_returns_none_when_breaker_is_open(monkeypatch) -> None:
    monkeypatch.setattr(
        counter_service_module,
        "get_redis_circuit_breaker",
        lambda: _OpenBreaker(),
    )
    counter_service = RedisTokenCounterService(_ScriptRedis())
    counter_service._release_tokens_raw = AsyncMock(return_value=0)

    result = asyncio.run(
        counter_service.release_tokens("gpt-4", "https://endpoint", 100)
    )

    assert result is None
    counter_service._release_tokens_raw.assert_not_awaited()


def test_close_is_idempotent() -> None:
    redis_client = _ScriptRedis()
    counter_service = RedisTokenCounterService(redis_client)

    asyncio.run(counter_service.close())
    asyncio.run(counter_service.close())

    assert redis_client.close_count == 1
