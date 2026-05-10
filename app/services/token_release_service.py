"""
TokenReleaseService — Token release use-case orchestration.

===========================================================

Coordinates the two-phase token release: delete the allocation from
PostgreSQL, then decrement the Redis fast-path counter.

The endpoint layer is responsible for object-level authorisation (checking
that the requester owns the allocation) before calling this service.  This
service does only the persistence work and returns a flag indicating whether
the Redis counter release was deferred (circuit breaker open).

Architecture:
-------------
    ┌────────────────────────┐
    │  release_tokens API    │
    │  (app/api/)            │
    │  ─ auth check here ─   │
    └──────────┬─────────────┘
               │  Depends()
    ┌──────────▼─────────────┐     ┌──────────────────────────────┐
    │ TokenReleaseService    │────▶│ LLMTokenAllocationPersistence│
    │ (app/services/)        │     │ (app/persistence/)            │
    │                        │────▶│ RedisTokenCounterService     │
    └────────────────────────┘     │ (app/resilience/)            │
                                   └──────────────────────────────┘

Two-step release flow:
    1. fetch_allocation_for_release  — DB lookup (CB-wrapped), for auth gate
    2. execute_release               — DB delete + Redis counter decrement

Dependencies:
    - app/persistence/llm_token_allocations.py — get/delete allocation record
    - app/resilience/redis_token_counter        — release Redis counter
    - app/resilience/circuit_breaker            — DB circuit breaker
    - app/core/exceptions.py                    — domain exceptions

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import aiobreaker

from app.core.exceptions import AllocationNotFoundError, DatabaseUnavailableError

if TYPE_CHECKING:
    from app.persistence.llm_token_allocations import LLMTokenAllocationPersistence
    from app.resilience.redis_token_counter import RedisTokenCounterService

logger = logging.getLogger(__name__)


class TokenReleaseService:
    """
    Orchestrates the token release use case.

    Provides two methods that the endpoint calls in sequence:
    1. fetch_allocation_for_release — returns the record so the endpoint
       can perform object-level auth before proceeding.
    2. execute_release — deletes the DB record and releases the Redis counter.

    Example:
        >>> service = TokenReleaseService(
        ...     allocation_persistence=LLMTokenAllocationPersistence(),
        ...     redis_counter=get_shared_redis_token_counter_service(),
        ...     db_circuit_breaker=get_db_circuit_breaker(),
        ... )
        >>> allocation = await service.fetch_allocation_for_release("req_abc123")
        >>> redis_deferred = await service.execute_release("req_abc123", allocation)
    """

    def __init__(
        self,
        allocation_persistence: LLMTokenAllocationPersistence,
        redis_counter: RedisTokenCounterService,
        db_circuit_breaker: aiobreaker.CircuitBreaker,
    ) -> None:
        """
        Initialise with injected dependencies.

        Args:
            allocation_persistence: Persistence adapter for token allocations.
            redis_counter: Redis-backed atomic token counter service.
            db_circuit_breaker: Circuit breaker wrapping DB read calls.
        """
        self._allocation_persistence = allocation_persistence
        self._redis_counter = redis_counter
        self._db_circuit_breaker = db_circuit_breaker

    # ------------------------------------------------------------------
    # Step 1: Fetch (for auth gate in endpoint)
    # ------------------------------------------------------------------

    async def fetch_allocation_for_release(
        self, token_request_id: str
    ) -> dict[str, Any] | None:
        """
        Fetch the allocation record so the endpoint can check ownership.

        Returns None if the allocation does not exist (idempotent — the
        endpoint should treat this as already released).

        Args:
            token_request_id: The unique allocation identifier to look up.

        Returns:
            Allocation dict if found, None if already released/absent.

        Raises:
            DatabaseUnavailableError: If the DB circuit breaker is open.
        """
        try:
            return cast(
                "dict[str, Any] | None",
                await self._db_circuit_breaker.call_async(
                    self._allocation_persistence.get_allocation_by_request_id,
                    token_request_id,
                ),
            )
        except aiobreaker.CircuitBreakerError as exc:
            raise DatabaseUnavailableError(
                "DB circuit breaker open; cannot fetch allocation for release"
            ) from exc

    # ------------------------------------------------------------------
    # Step 2: Execute release (after endpoint auth check passes)
    # ------------------------------------------------------------------

    async def execute_release(
        self, token_request_id: str, allocation: dict[str, Any]
    ) -> bool:
        """
        Delete the allocation record and release the Redis counter.

        Args:
            token_request_id: Allocation identifier to delete.
            allocation: The allocation dict returned by fetch_allocation_for_release
                (used to extract model/endpoint/count for Redis release).

        Returns:
            True if the Redis counter release was deferred (circuit breaker open
            or missing counter fields); False if Redis was released successfully.

        Raises:
            AllocationNotFoundError: If the DB delete finds no matching record
                (indicates a race condition between fetch and delete).
        """
        deleted = await self._allocation_persistence.delete_allocation(token_request_id)
        if not deleted:
            raise AllocationNotFoundError(token_request_id)

        redis_deferred = await self._release_redis_counter(token_request_id, allocation)
        logger.info(
            "[release] Token allocation deleted",
            extra={
                "token_request_id": token_request_id,
                "redis_deferred": redis_deferred,
            },
        )
        return redis_deferred

    # ------------------------------------------------------------------
    # Redis counter release (best-effort; deferred on CB open)
    # ------------------------------------------------------------------

    async def _release_redis_counter(
        self, token_request_id: str, allocation: dict[str, Any]
    ) -> bool:
        """
        Attempt to release the Redis fast-path counter for this allocation.

        Best-effort: if fields are missing or the Redis circuit breaker is open,
        logs a warning and returns True (deferred) so the caller can signal
        the client via a response header.

        Args:
            token_request_id: For structured log correlation.
            allocation: The full allocation record dict.

        Returns:
            True if the Redis release was deferred, False if released cleanly.
        """
        allocation_status = allocation.get("allocation_status", "")
        if allocation_status != "ACQUIRED":
            logger.debug(
                "[release] Skipping Redis counter release for non-ACQUIRED allocation",
                extra={
                    "token_request_id": token_request_id,
                    "allocation_status": allocation_status,
                },
            )
            return False

        model = allocation.get("llm_model_name", "")
        endpoint = allocation.get("api_endpoint_url", "")
        token_count: int = allocation.get("token_count", 0)

        if not model or not endpoint or not token_count:
            return False

        result = await self._redis_counter.release_tokens(model, endpoint, token_count)
        if result is None:
            logger.warning(
                "[release] Redis counter release deferred (CB open/unavailable); "
                "reconciler will correct the counter",
                extra={"token_request_id": token_request_id},
            )
            return True
        return False
