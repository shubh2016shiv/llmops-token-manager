"""
Redis token counter service — the plumbing around the Lua algorithm.

The actual logic is the Lua in `lua_script_definitions.py`. This file's job is to:
  1. build the two Redis keys for a deployment,
  2. register + run the right Lua script,
  3. wrap hot-path calls in the Redis circuit breaker, and
  4. FAIL THROUGH (return a safe "miss") so Redis trouble degrades to the DB path
     instead of hard-failing the request.

Read README.md in this folder for the end-to-end flow; read the method comments
below for the step-by-step implementation.

Author: Engineering Team
Last Updated: 2026-07-24
"""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import redis.asyncio as aioredis

import aiobreaker
from loguru import logger
from redis import exceptions as redis_exceptions

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

# Stable internal names for the four scripts (used as dict keys + attribute suffixes).
SCRIPT_NAME_RESERVE = "reserve"
SCRIPT_NAME_RELEASE = "release"
SCRIPT_NAME_SEED = "seed"
SCRIPT_NAME_RECONCILE = "reconcile"


# --- Type-checker shims (describe only the slice of each API we use) ---------
# These Protocols let tests pass lightweight fakes instead of real Redis/breaker
# objects. They carry no runtime behavior — skip past them; the logic is below.
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


class AsyncRedisClientProtocol(Protocol):
    """
    Minimal async Redis client interface consumed by RedisTokenCounterService.

    Typed as a Protocol (not the concrete aioredis.Redis) so that test doubles
    and alternative implementations satisfy the interface without subclassing.
    Methods that return pipeline objects or coroutines are typed as Any because
    the pipeline API is accessed dynamically and its shape is an aioredis detail.
    """

    def register_script(self, script: str) -> RedisLuaScriptRunner:
        """Register and return a callable Lua script runner."""
        ...

    def pipeline(self) -> Any:
        """Return a pipeline context for batched Redis commands."""
        ...

    def close(self) -> Any:
        """Close the connection; may return an awaitable."""
        ...


# Maps our internal script name -> the Lua source to register for it.
SCRIPT_SOURCE_BY_NAME = {
    SCRIPT_NAME_RESERVE: LUA_RESERVE_TOKENS,
    SCRIPT_NAME_RELEASE: LUA_RELEASE_TOKENS,
    SCRIPT_NAME_SEED: LUA_SEED_COUNTER,
    SCRIPT_NAME_RECONCILE: LUA_RECONCILE_COUNTER,
}


class RedisTokenCounterService:
    """Atomic Redis token counter operations for the resilience fast path."""

    def __init__(self, redis_client: AsyncRedisClientProtocol) -> None:
        self._redis = redis_client
        # Lazily-registered script runners (Redis remembers a script by hash; we
        # register each one once, then invoke it many times). None = not yet loaded.
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
        # aioredis .close() may return a coroutine; await only if it does.
        close_result = self._redis.close()
        if inspect.isawaitable(close_result):
            await close_result
        self._closed = True

    @property
    def redis_client(self) -> aioredis.Redis:
        """Expose the shared Redis client for coordinated maintenance operations."""
        # cast: __init__ accepts the minimal AsyncRedisClientProtocol so test
        # doubles work without subclassing aioredis.Redis; callers that need the
        # full Redis API (e.g. reconciliation lock) always pass a real Redis client.
        return cast("aioredis.Redis", self._redis)

    # -----------------------------------------------------------------------
    # PUBLIC OPERATIONS
    # Each hot-path op follows the SAME shape: run the raw Lua call under the Redis
    # circuit breaker, and on ANY failure "fail through" to a safe miss value so the
    # caller falls back to the PostgreSQL path instead of erroring.
    # -----------------------------------------------------------------------

    async def reserve_tokens(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> TokenReservationResult:
        """Atomically reserve tokens for one model and endpoint."""
        try:
            # Run the reserve Lua (via _reserve_tokens_raw) under breaker protection.
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
        except aiobreaker.CircuitBreakerError as exc:
            # Breaker is OPEN (Redis looks unhealthy) -> don't even try Redis.
            # COUNTER_MISS tells the caller "use the DB path".
            logger.warning(
                f"[FastPath] reserve_tokens short-circuited by Redis breaker: {exc}"
            )
            return TokenReservationResult.COUNTER_MISS
        except Exception as exc:
            # Any other Redis error -> also fail through to the DB path.
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
            return updated_counter_value
        except aiobreaker.CircuitBreakerError as exc:
            # Fail through: None means "couldn't update Redis" (caller handles it).
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
        # Not breaker-wrapped: seeding is startup/warm-up work, not hot-path traffic.
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
        # Not breaker-wrapped: the reconciliation job already runs under its own
        # Redis lock, and it WANTS the raw error to surface if Redis is unreachable.
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
        # The Lua returns 0/1/2/3; turn it into the typed enum for callers.
        return CounterReconciliationResult(reconciliation_result)

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
        except aiobreaker.CircuitBreakerError as exc:
            logger.warning(
                f"[FastPath] get_counter short-circuited by Redis breaker: {exc}"
            )
            return None
        except Exception as exc:
            logger.error(f"[FastPath] get_counter failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # RAW OPERATIONS — the actual Redis work, WITHOUT breaker/fail-through.
    # The public methods above add the breaker + error handling around these.
    # -----------------------------------------------------------------------

    async def _reserve_tokens_raw(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> TokenReservationResult:
        """Reserve tokens without outer breaker handling."""
        # 1. two keys for this deployment; 2. run RESERVE Lua; 3. map int -> enum.
        counter_key, limit_key = self._build_counter_keys(
            model_name,
            api_endpoint_url,
        )
        reservation_result = await self._execute_lua_script(
            script_name=SCRIPT_NAME_RESERVE,
            keys=[counter_key, limit_key],
            args=[token_count],
        )
        return TokenReservationResult(reservation_result)

    async def _release_tokens_raw(
        self,
        model_name: str,
        api_endpoint_url: str,
        token_count: int,
    ) -> int:
        """Release tokens without outer breaker handling."""
        # Release only touches the counter key (no limit needed).
        counter_key, _ = self._build_counter_keys(model_name, api_endpoint_url)
        updated_counter_value = await self._execute_lua_script(
            script_name=SCRIPT_NAME_RELEASE,
            keys=[counter_key],
            args=[token_count],
        )
        return updated_counter_value

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
        # A pipeline batches both GETs into ONE round-trip to Redis.
        pipeline = self._redis.pipeline()
        pipeline.get(counter_key)
        pipeline.get(limit_key)
        counter_value, limit_value = await pipeline.execute()
        # If either key is gone, report "no snapshot" rather than a half answer.
        if counter_value is None or limit_value is None:
            return None
        return int(counter_value), int(limit_value)

    # -----------------------------------------------------------------------
    # LUA SCRIPT EXECUTION + REGISTRATION
    # -----------------------------------------------------------------------

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
            # Redis identifies scripts by hash and runs them by hash (fast). If Redis
            # RESTARTED, it forgot the script and raises NOSCRIPT. Self-heal: register
            # the source again (which reloads it) and retry exactly once.
            logger.warning(
                f"[FastPath] Redis NOSCRIPT encountered for script={script_name}; "
                "re-registering and retrying once."
            )
            script_runner = self._register_script(script_name)
            script_result = await script_runner(keys=keys, args=args)
            return int(cast("int | str", script_result))

    def _get_or_register_script(self, script_name: str) -> RedisLuaScriptRunner:
        # Return the cached runner if we've registered it before; otherwise load it.
        script_attr_name = self._script_attr_name(script_name)
        registered_script = getattr(self, script_attr_name)
        if registered_script is None:
            # Benign race: concurrent coroutines may double-register the same
            # script; Redis SCRIPT LOAD is idempotent and both runners are valid.
            registered_script = self._register_script(script_name)
        return registered_script

    def _register_script(self, script_name: str) -> RedisLuaScriptRunner:
        # Ask Redis to register the Lua source; cache the returned callable runner.
        script_source = SCRIPT_SOURCE_BY_NAME[script_name]
        script_runner = self._redis.register_script(script_source)
        setattr(self, self._script_attr_name(script_name), script_runner)
        return script_runner

    async def _call_with_redis_circuit_breaker(
        self,
        operation: object,
        *args: object,
    ) -> object:
        """Execute one async Redis operation under breaker protection."""
        # get_redis_circuit_breaker() returns the shared 'redis' breaker; call_async
        # runs `operation` through it — raising CircuitBreakerError if it's OPEN.
        redis_circuit_breaker = cast(
            "AsyncCircuitBreakerProtocol",
            get_redis_circuit_breaker(),
        )
        return await redis_circuit_breaker.call_async(operation, *args)

    @staticmethod
    def _script_attr_name(script_name: str) -> str:
        # Map a script name to the instance attribute that caches its runner.
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
        # Hash the endpoint so the key stays short/safe no matter the URL, and
        # sanitize the model name so ':' / '/' can't corrupt the key format.
        endpoint_hash = hashlib.sha256(api_endpoint_url.encode()).hexdigest()[:16]
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        counter_key = f"token:counter:{safe_model_name}:{endpoint_hash}"
        limit_key = f"token:limit:{safe_model_name}:{endpoint_hash}"
        return counter_key, limit_key
