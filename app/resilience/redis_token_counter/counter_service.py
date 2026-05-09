"""
Redis token counter service - atomic Redis operations for fast-path token flow.

Architecture:
-------------
    API and worker callers delegate fast-path token accounting to
    `RedisTokenCounterService`.

    The service coordinates three concerns in one place:
    - Lua scripts from `lua_script_definitions.py` for atomic Redis mutations
    - the Redis circuit breaker for fail-through behavior under infrastructure failure
    - Redis key construction, script registration, and lifecycle management

    This module owns the operational counter workflow:
    - reserve tokens atomically against the current counter and configured limit
    - release tokens defensively without ever allowing negative balances
    - seed cold counters from a database snapshot during startup
    - reconcile active counters with atomic delta correction during runtime

Dependencies:
    - app/core/config.py - TTL settings
    - app/resilience/circuit_breaker - Redis circuit breaker
    - redis.asyncio - async Redis client and script runner

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable

from loguru import logger
import pybreaker
from redis import exceptions as redis_exceptions
import redis.asyncio as aioredis

from app.core.config import settings
from app.resilience.circuit_breaker import get_redis_circuit_breaker
from app.resilience.redis_token_counter.counter_results import (
    CounterReconciliationResult,
    TokenReservationResult,
)
from app.resilience.redis_token_counter.lua_script_definitions import (
    LUA_RECONCILE_COUNTER,
    LUA_RELEASE_TOKENS,
    LUA_RESERVE_TOKENS,
    LUA_SEED_COUNTER,
)

SCRIPT_NAME_RESERVE = "reserve"
SCRIPT_NAME_RELEASE = "release"
SCRIPT_NAME_SEED = "seed"
SCRIPT_NAME_RECONCILE = "reconcile"


class RedisLuaScriptRunner(Protocol):
    """Callable Redis Lua script runner returned by `register_script`."""

    def __call__(
        self,
        *,
        keys: list[str],
        args: list[int],
    ) -> Awaitable[object]:
        """Execute one registered Redis Lua script invocation."""
        ...


class AsyncCircuitBreakerProtocol(Protocol):
    """Subset of the circuit breaker interface used by this service."""

    def call_async(
        self,
        func: object,
        *args: object,
        **kwargs: object,
    ) -> Awaitable[object]:
        """Execute one async function under circuit-breaker protection."""
        ...


SCRIPT_SOURCE_BY_NAME = {
    SCRIPT_NAME_RESERVE: LUA_RESERVE_TOKENS,
    SCRIPT_NAME_RELEASE: LUA_RELEASE_TOKENS,
    SCRIPT_NAME_SEED: LUA_SEED_COUNTER,
    SCRIPT_NAME_RECONCILE: LUA_RECONCILE_COUNTER,
}


class RedisTokenCounterService:
    """Atomic Redis token counter operations for the resilience fast path."""

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._reserve_script: RedisLuaScriptRunner | None = None
        self._release_script: RedisLuaScriptRunner | None = None
        self._seed_script: RedisLuaScriptRunner | None = None
        self._reconcile_script: RedisLuaScriptRunner | None = None
        self._closed = False

    async def __aenter__(self) -> RedisTokenCounterService:
        """Return the service for async context manager usage."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Close the underlying Redis client when exiting context."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying Redis client idempotently."""
        if self._closed:
            return
        close_result = self._redis.close()
        if inspect.isawaitable(close_result):
            await close_result
        self._closed = True

    async def reserve_tokens(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> TokenReservationResult:
        """Atomically reserve tokens for one model and endpoint."""
        try:
            reservation_result = await self._call_with_redis_circuit_breaker(
                self._reserve_tokens_raw,
                model_name,
                api_endpoint_url,
                token_count,
            )
            reservation_result = cast(
                "TokenReservationResult",
                reservation_result,
            )
            logger.debug(
                f"[FastPath] reserve model={model_name} tokens={token_count} "
                f"result={reservation_result.name}"
            )
            return reservation_result
        except pybreaker.CircuitBreakerError as exc:
            logger.warning(
                f"[FastPath] reserve_tokens short-circuited by Redis breaker: {exc}"
            )
            return TokenReservationResult.COUNTER_MISS
        except Exception as exc:
            logger.error(
                f"[FastPath] reserve_tokens failed (fail-through to DB): {exc}"
            )
            return TokenReservationResult.COUNTER_MISS

    async def release_tokens(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> int | None:
        """Release previously reserved tokens and clamp the counter at zero."""
        try:
            updated_counter_value = await self._call_with_redis_circuit_breaker(
                self._release_tokens_raw,
                model_name,
                api_endpoint_url,
                token_count,
            )
            updated_counter_value = cast("int", updated_counter_value)
            logger.debug(
                f"[FastPath] release model={model_name} tokens={token_count} "
                f"new_counter={updated_counter_value}"
            )
            return int(updated_counter_value)
        except pybreaker.CircuitBreakerError as exc:
            logger.warning(
                f"[FastPath] release_tokens short-circuited by Redis breaker: {exc}"
            )
            return None
        except Exception as exc:
            logger.error(f"[FastPath] release_tokens failed: {exc}")
            return None

    async def seed_counter(
        self,
        model_name: str,
        api_endpoint_url: str,
        current_allocated: int,
        max_limit: int,
    ) -> None:
        """Seed counter and limit keys atomically from a DB snapshot."""
        try:
            counter_key, limit_key = self._build_counter_keys(
                model_name,
                api_endpoint_url,
            )
            ttl_seconds = settings.redis_token_counter_ttl_secs
            await self._execute_lua_script(
                script_name=SCRIPT_NAME_SEED,
                keys=[counter_key, limit_key],
                args=[current_allocated, max_limit, ttl_seconds],
            )
            logger.info(
                f"[FastPath] Seeded model={model_name} "
                f"allocated={current_allocated}/{max_limit} TTL={ttl_seconds}s"
            )
        except Exception as exc:
            logger.error(f"[FastPath] seed_counter failed for {model_name}: {exc}")

    async def reconcile_counter(
        self,
        model_name: str,
        api_endpoint_url: str,
        allocated_tokens_from_db: int,
        max_tokens_from_db: int,
    ) -> CounterReconciliationResult:
        """Atomically reconcile Redis counter state to the latest DB snapshot."""
        counter_key, limit_key = self._build_counter_keys(
            model_name,
            api_endpoint_url,
        )
        ttl_seconds = settings.redis_token_counter_ttl_secs
        reconciliation_result = await self._execute_lua_script(
            script_name=SCRIPT_NAME_RECONCILE,
            keys=[counter_key, limit_key],
            args=[allocated_tokens_from_db, max_tokens_from_db, ttl_seconds],
        )
        return CounterReconciliationResult(int(reconciliation_result))

    async def get_counter(
        self,
        model_name: str,
        api_endpoint_url: str,
    ) -> tuple[int, int] | None:
        """Return allocated and limit values, or None when either key is missing."""
        try:
            counter_snapshot = await self._call_with_redis_circuit_breaker(
                self._get_counter_raw,
                model_name,
                api_endpoint_url,
            )
            return cast("tuple[int, int] | None", counter_snapshot)
        except pybreaker.CircuitBreakerError as exc:
            logger.warning(
                f"[FastPath] get_counter short-circuited by Redis breaker: {exc}"
            )
            return None
        except Exception as exc:
            logger.error(f"[FastPath] get_counter failed: {exc}")
            return None

    async def _reserve_tokens_raw(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> TokenReservationResult:
        """Reserve tokens without outer breaker handling."""
        counter_key, limit_key = self._build_counter_keys(
            model_name,
            api_endpoint_url,
        )
        reservation_result = await self._execute_lua_script(
            script_name=SCRIPT_NAME_RESERVE,
            keys=[counter_key, limit_key],
            args=[token_count],
        )
        return TokenReservationResult(int(reservation_result))

    async def _release_tokens_raw(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> int:
        """Release tokens without outer breaker handling."""
        counter_key, _ = self._build_counter_keys(model_name, api_endpoint_url)
        updated_counter_value = await self._execute_lua_script(
            script_name=SCRIPT_NAME_RELEASE,
            keys=[counter_key],
            args=[token_count],
        )
        return int(updated_counter_value)

    async def _get_counter_raw(
        self,
        model_name: str,
        api_endpoint_url: str,
    ) -> tuple[int, int] | None:
        """Read counter and limit values without outer breaker handling."""
        counter_key, limit_key = self._build_counter_keys(
            model_name,
            api_endpoint_url,
        )
        pipeline = self._redis.pipeline()
        pipeline.get(counter_key)
        pipeline.get(limit_key)
        counter_value, limit_value = await pipeline.execute()
        if counter_value is None or limit_value is None:
            return None
        return int(counter_value), int(limit_value)

    async def _execute_lua_script(
        self,
        script_name: str,
        keys: list[str],
        args: list[int],
    ) -> int:
        """Execute a registered Lua script with one NOSCRIPT retry."""
        script_runner = self._get_or_register_script(script_name)
        try:
            script_result = await script_runner(keys=keys, args=args)
            return int(cast("int | str", script_result))
        except redis_exceptions.NoScriptError:
            logger.warning(
                f"[FastPath] Redis NOSCRIPT encountered for script={script_name}; "
                "re-registering and retrying once."
            )
            script_runner = self._register_script(script_name)
            script_result = await script_runner(keys=keys, args=args)
            return int(cast("int | str", script_result))

    def _get_or_register_script(self, script_name: str) -> RedisLuaScriptRunner:
        script_attr_name = self._script_attr_name(script_name)
        registered_script = getattr(self, script_attr_name)
        if registered_script is None:
            # Benign race: concurrent coroutines may double-register the same
            # script; Redis SCRIPT LOAD is idempotent and both runners are valid.
            registered_script = self._register_script(script_name)
        return registered_script

    def _register_script(self, script_name: str) -> RedisLuaScriptRunner:
        script_source = SCRIPT_SOURCE_BY_NAME[script_name]
        script_runner = cast(
            "RedisLuaScriptRunner",
            self._redis.register_script(script_source),
        )
        setattr(self, self._script_attr_name(script_name), script_runner)
        return script_runner

    async def _call_with_redis_circuit_breaker(
        self,
        operation: object,
        *args: object,
    ) -> object:
        """
        Execute one async Redis operation under breaker protection.

        Third-party stubs for `pybreaker.call_async` are too loose for pyright,
        so this adapter narrows the awaitable contract in one place.
        """
        redis_circuit_breaker = cast(
            "AsyncCircuitBreakerProtocol",
            get_redis_circuit_breaker(),
        )
        return await redis_circuit_breaker.call_async(operation, *args)

    @staticmethod
    def _script_attr_name(script_name: str) -> str:
        if script_name == SCRIPT_NAME_RESERVE:
            return "_reserve_script"
        if script_name == SCRIPT_NAME_RELEASE:
            return "_release_script"
        if script_name == SCRIPT_NAME_SEED:
            return "_seed_script"
        if script_name == SCRIPT_NAME_RECONCILE:
            return "_reconcile_script"
        raise ValueError(f"Unknown Lua script name: {script_name}")

    @staticmethod
    def _build_counter_keys(
        model_name: str,
        api_endpoint_url: str,
    ) -> tuple[str, str]:
        """Build stable Redis counter and limit keys for a deployment."""
        endpoint_hash = hashlib.sha256(api_endpoint_url.encode()).hexdigest()[:16]
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        counter_key = f"token:counter:{safe_model_name}:{endpoint_hash}"
        limit_key = f"token:limit:{safe_model_name}:{endpoint_hash}"
        return counter_key, limit_key
