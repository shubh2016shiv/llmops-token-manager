"""
TokenAcquisitionService — Token allocation use-case orchestration.

=================================================================

Implements the full 5-layer acquisition flow, coordinating across
Redis, RabbitMQ, and PostgreSQL without embedding that logic in the
API layer or the persistence layer.

Architecture:
-------------
    ┌────────────────────────┐
    │  acquire_tokens API    │
    │  (app/api/)            │
    └──────────┬─────────────┘
               │  Depends()
    ┌──────────▼─────────────┐     ┌──────────────────────────────┐
    │ TokenAcquisitionService│────▶│ LLMTokenAllocationPersistence│
    │ (app/services/)        │     │ (app/persistence/)            │
    │                        │────▶│ UserPersistence               │
    │                        │     └──────────────────────────────┘
    │                        │     ┌──────────────────────────────┐
    │                        │────▶│ RedisTokenCounterService     │
    │                        │     │ (app/resilience/)            │
    │                        │────▶│ TokenAllocationPublisher     │
    └────────────────────────┘     └──────────────────────────────┘

Layer execution order:
    0. Rate limit          (enforced in API layer via Depends)
    1. Back pressure       (enforced in API layer via Depends)
    2. User active guard   (this service)
    3. Token estimation    (this service)
    4. Redis fast path     (this service → RedisTokenCounterService)
    4a.  ALLOCATED → RMQ publish; RMQ CB open → Redis rollback → DB path
    4b.  MISS / EXHAUSTED → DB path
    5. DB path             (this service → LLMTokenAllocationPersistence)
       Decides WAITING vs ACQUIRED; creates the allocation record.

Dependencies:
    - app/persistence/llm_token_allocations.py — create_token_allocation,
      get_least_loaded_deployment
    - app/persistence/users.py                 — get_user_by_id
    - app/resilience/redis_token_counter       — reserve / release tokens
    - app/resilience/token_queue/publisher.py  — publish to RabbitMQ
    - app/resilience/circuit_breaker           — DB circuit breaker
    - app/core/exceptions.py                   — domain exceptions
    - app/utils/token_count_estimation.py      — LiteLLM token counting

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, cast
import uuid
from uuid import UUID

import aiobreaker

from app.core.exceptions import (
    DatabaseUnavailableError,
    DeploymentConfigurationError,
    TokenLimitExceededError,
    UserInactiveError,
    UserNotFoundError,
)
from app.models.response_models import TokenAllocationResponse
from app.resilience.redis_token_counter import TokenReservationResult
from app.utils.token_count_estimation import estimate_tokens

if TYPE_CHECKING:
    from app.models.request_models import TokenAllocationClientRequest
    from app.persistence.llm_token_allocations import LLMTokenAllocationPersistence
    from app.persistence.users import UserPersistence
    from app.resilience.redis_token_counter import RedisTokenCounterService
    from app.resilience.token_queue import TokenAllocationPublisher

logger = logging.getLogger(__name__)


class TokenAcquisitionService:
    """
    Orchestrates the complete token acquisition use case.

    Coordinates user validation, token estimation, Redis fast-path
    reservation, RabbitMQ async persistence, and PostgreSQL DB fallback.
    Does not perform DB queries directly — delegates to injected adapters.

    Example:
        >>> service = TokenAcquisitionService(
        ...     allocation_persistence=LLMTokenAllocationPersistence(),
        ...     user_persistence=UserPersistence(),
        ...     redis_counter=get_shared_redis_token_counter_service(),
        ...     publisher=TokenAllocationPublisher(),
        ...     db_circuit_breaker=get_db_circuit_breaker(),
        ... )
        >>> response = await service.acquire_tokens(user_id, request)
    """

    def __init__(
        self,
        allocation_persistence: LLMTokenAllocationPersistence,
        user_persistence: UserPersistence,
        redis_counter: RedisTokenCounterService,
        publisher: TokenAllocationPublisher,
        db_circuit_breaker: aiobreaker.CircuitBreaker,
    ) -> None:
        """
        Initialise with injected dependencies.

        Args:
            allocation_persistence: Persistence adapter for token allocations.
            user_persistence: Persistence adapter for user lookups.
            redis_counter: Redis-backed atomic token counter service.
            publisher: RabbitMQ publisher for async allocation persistence.
            db_circuit_breaker: Circuit breaker wrapping DB calls.
        """
        self._allocation_persistence = allocation_persistence
        self._user_persistence = user_persistence
        self._redis_counter = redis_counter
        self._publisher = publisher
        self._db_circuit_breaker = db_circuit_breaker

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def acquire_tokens(
        self,
        user_id: UUID,
        request: TokenAllocationClientRequest,
    ) -> TokenAllocationResponse:
        """
        Acquire tokens following the full 5-layer resilience path.

        Args:
            user_id: Authenticated user's UUID (from JWT).
            request: Validated client allocation request.

        Returns:
            TokenAllocationResponse with status ACQUIRED or WAITING.

        Raises:
            UserNotFoundError: If the user does not exist in the database.
            UserInactiveError: If the user's status is not 'active'.
            DeploymentConfigurationError: If the selected deployment is missing
                required configuration fields (e.g. max_tokens).
            TokenLimitExceededError: If the single request exceeds the
                deployment's configured maximum token allocation.
            DatabaseUnavailableError: If the DB circuit breaker is open.
        """
        await self._validate_user_is_active(user_id)
        estimated_token_count = self._estimate_token_count(request)
        _total_allocated, chosen_config = await self._get_deployment_config(
            request.llm_provider.value,
            request.llm_model_name,
        )
        api_endpoint = chosen_config.get("api_endpoint_url", "")

        reservation_result = await self._redis_counter.reserve_tokens(
            model_name=request.llm_model_name,
            api_endpoint_url=api_endpoint,
            token_count=estimated_token_count,
        )

        if reservation_result == TokenReservationResult.ALLOCATED:
            fast_path_response = await self._handle_fast_path(
                user_id, request, estimated_token_count, chosen_config, api_endpoint
            )
            if fast_path_response is not None:
                return fast_path_response
            # RMQ CB was open — Redis reservation rolled back; fall to DB path

        _log_redis_miss_reason(reservation_result, request.llm_model_name)
        return await self._create_db_allocation(
            user_id, request, estimated_token_count, chosen_config
        )

    # ------------------------------------------------------------------
    # User validation
    # ------------------------------------------------------------------

    async def _validate_user_is_active(self, user_id: UUID) -> None:
        """
        Raise UserNotFoundError or UserInactiveError if the user cannot proceed.

        Args:
            user_id: The user's UUID to validate.

        Raises:
            UserNotFoundError: User record does not exist.
            UserInactiveError: User exists but status is not 'active'.
        """
        user = await self._user_persistence.get_user_by_id(user_id)
        if user is None:
            raise UserNotFoundError(str(user_id))
        if user["status"] != "active":
            raise UserInactiveError(str(user_id), user["status"])

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _estimate_token_count(self, request: TokenAllocationClientRequest) -> int:
        """
        Return estimated token count for the request's input data.

        Args:
            request: The incoming allocation request with input_data.

        Returns:
            Estimated total token count.
        """
        return estimate_tokens(request.input_data, request.llm_model_name).total_tokens

    # ------------------------------------------------------------------
    # Deployment selection (DB + circuit breaker)
    # ------------------------------------------------------------------

    async def _get_deployment_config(
        self, provider_name: str, model_name: str
    ) -> tuple[int, dict[str, Any]]:
        """
        Fetch the least-loaded deployment for the provider/model pair.

        Args:
            provider_name: The LLM provider identifier from the request.
            model_name: The logical model identifier to route.

        Returns:
            Tuple of (total_allocated_tokens, deployment_config_dict).

        Raises:
            DatabaseUnavailableError: If the DB circuit breaker is open.
            ValueError: If no active deployments are configured for the model.
        """
        try:
            return cast(
                "tuple[int, dict[str, Any]]",
                await self._db_circuit_breaker.call_async(
                    self._allocation_persistence.get_least_loaded_deployment,
                    provider_name,
                    model_name,
                ),
            )
        except aiobreaker.CircuitBreakerError as exc:
            raise DatabaseUnavailableError(
                "DB circuit breaker open; cannot select deployment"
            ) from exc

    # ------------------------------------------------------------------
    # Redis fast path
    # ------------------------------------------------------------------

    async def _handle_fast_path(
        self,
        user_id: UUID,
        request: TokenAllocationClientRequest,
        token_count: int,
        chosen_config: dict[str, Any],
        api_endpoint: str,
    ) -> TokenAllocationResponse | None:
        """
        Attempt RMQ publish after a successful Redis reservation.

        Returns the allocation response on success. Returns None if the
        RabbitMQ circuit breaker is open (Redis reservation already rolled back).

        Args:
            user_id: Authenticated user's UUID.
            request: The original allocation request.
            token_count: Estimated token count already reserved in Redis.
            chosen_config: Deployment config dict from get_least_loaded_deployment.
            api_endpoint: The selected deployment's API endpoint URL.

        Returns:
            TokenAllocationResponse on success, None on RMQ circuit breaker open.
        """
        token_request_id = f"req_{uuid.uuid4().hex}"
        max_lock_secs: int = chosen_config.get("max_token_lock_time_secs", 70)
        now = datetime.now()
        expires_at = now + timedelta(seconds=max_lock_secs)

        payload = _build_allocation_payload(
            token_request_id,
            user_id,
            request,
            token_count,
            chosen_config,
            api_endpoint,
            expires_at,
        )
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._publisher.publish_allocation_request, payload
            )
        except aiobreaker.CircuitBreakerError:
            await self._rollback_redis_on_rmq_failure(
                request.llm_model_name, api_endpoint, token_count, token_request_id
            )
            return None

        logger.info(
            "[acquire] Fast-path ALLOCATED via RMQ",
            extra={
                "token_request_id": token_request_id,
                "model": request.llm_model_name,
                "token_count": token_count,
            },
        )
        return _build_fast_path_response(
            token_request_id,
            user_id,
            request,
            token_count,
            chosen_config,
            api_endpoint,
            now,
            expires_at,
        )

    async def _rollback_redis_on_rmq_failure(
        self,
        model_name: str,
        api_endpoint: str,
        token_count: int,
        token_request_id: str,
    ) -> None:
        """
        Release the Redis reservation when RMQ circuit breaker is open.

        Args:
            model_name: The logical model name whose counter to release.
            api_endpoint: The endpoint whose counter was incremented.
            token_count: Number of tokens to roll back.
            token_request_id: For structured log correlation.
        """
        rollback_result = await self._redis_counter.release_tokens(
            model_name, api_endpoint, token_count
        )
        if rollback_result is None:
            logger.warning(
                "[acquire] Redis rollback deferred (Redis CB open); "
                "reconciler will correct counter drift",
                extra={"token_request_id": token_request_id},
            )
        logger.warning(
            "[acquire] RMQ CB open — Redis reservation rolled back; "
            "falling through to DB path",
            extra={"token_request_id": token_request_id},
        )

    # ------------------------------------------------------------------
    # DB path
    # ------------------------------------------------------------------

    async def _create_db_allocation(
        self,
        user_id: UUID,
        request: TokenAllocationClientRequest,
        token_count: int,
        chosen_config: dict[str, Any],
    ) -> TokenAllocationResponse:
        """
        Create an allocation record synchronously via PostgreSQL.

        Decides WAITING vs ACQUIRED based on current capacity, then
        persists the record. Wrapped with the DB circuit breaker.

        Args:
            user_id: Authenticated user's UUID.
            request: The original allocation request.
            token_count: Estimated token count to reserve.
            chosen_config: Deployment config dict.

        Returns:
            TokenAllocationResponse with status ACQUIRED or WAITING.

        Raises:
            DeploymentConfigurationError: If max_tokens is not configured.
            TokenLimitExceededError: If token_count exceeds max_tokens.
            DatabaseUnavailableError: If the DB circuit breaker is open.
        """
        max_token_limit = _require_max_token_limit(
            chosen_config, request.llm_model_name
        )
        if token_count > max_token_limit:
            raise TokenLimitExceededError(
                token_count, max_token_limit, request.llm_model_name
            )

        record = await self._insert_allocation_record(
            user_id, request, token_count, chosen_config
        )
        logger.info(
            "[acquire] DB-path allocation created",
            extra={
                "token_request_id": record.get("token_request_id"),
                "status": record.get("allocation_status"),
                "model": request.llm_model_name,
            },
        )
        return TokenAllocationResponse(**record)

    async def _insert_allocation_record(
        self,
        user_id: UUID,
        request: TokenAllocationClientRequest,
        token_count: int,
        chosen_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create an allocation through the atomic DB capacity-check primitive.

        Args:
            user_id: Authenticated user's UUID.
            request: The original allocation request.
            token_count: Token count to persist.
            chosen_config: Deployment config dict.

        Returns:
            Raw allocation record dict as returned by the persistence layer.

        Raises:
            DatabaseUnavailableError: If the DB circuit breaker is open.
        """
        max_lock_secs: int = chosen_config.get("max_token_lock_time_secs", 70)
        expires_at = datetime.now() + timedelta(seconds=max_lock_secs)
        deployment = _extract_deployment_fields(chosen_config)
        params = _build_capacity_checked_create_params(
            f"req_{uuid.uuid4().hex}",
            user_id,
            request,
            token_count,
            chosen_config,
            deployment,
            expires_at,
        )
        try:
            return cast(
                "dict[str, Any]",
                await self._db_circuit_breaker.call_async(
                    self._allocation_persistence.create_token_allocation_with_capacity_check,
                    **params,
                ),
            )
        except aiobreaker.CircuitBreakerError as exc:
            raise DatabaseUnavailableError(
                "DB circuit breaker open during DB-path allocation"
            ) from exc


# ---------------------------------------------------------------------------
# Module-level pure helpers (no service state)
# ---------------------------------------------------------------------------


def _require_max_token_limit(chosen_config: dict[str, Any], model_name: str) -> int:
    """
    Extract max_tokens from deployment config; raise if absent.

    Args:
        chosen_config: Deployment configuration dict.
        model_name: Used in the error message when max_tokens is missing.

    Returns:
        Maximum token allocation as a positive integer.

    Raises:
        DeploymentConfigurationError: If max_tokens is None or absent.
    """
    max_token_limit: int | None = chosen_config.get("max_tokens")
    if max_token_limit is None:
        raise DeploymentConfigurationError(model_name, "max_tokens")
    return max_token_limit


def _extract_deployment_fields(chosen_config: dict[str, Any]) -> dict[str, Any]:
    """
    Return selected deployment fields for the atomic DB allocation attempt.

    Args:
        chosen_config: Deployment configuration dict from the persistence layer.

    Returns:
        Dict with keys: deployment_name, api_endpoint_url,
        deployment_region, temperature, seed.
    """
    return {
        "deployment_name": chosen_config.get("deployment_name", ""),
        "api_endpoint_url": chosen_config.get("api_endpoint_url", ""),
        "deployment_region": chosen_config.get("deployment_region", ""),
        "temperature": chosen_config.get("temperature", 0.0),
        "top_p": chosen_config.get("top_p"),
        "seed": chosen_config.get("seed", chosen_config.get("random_seed", 42)),
    }


def _build_capacity_checked_create_params(
    token_request_id: str,
    user_id: UUID,
    request: TokenAllocationClientRequest,
    token_count: int,
    chosen_config: dict[str, Any],
    deployment: dict[str, Any],
    expires_at: datetime,
) -> dict[str, Any]:
    """
    Build kwargs for create_token_allocation_with_capacity_check.

    Args:
        token_request_id: Generated unique allocation identifier.
        user_id: Requesting user's UUID.
        request: The original allocation request.
        token_count: Tokens to reserve.
        chosen_config: Deployment configuration dict.
        deployment: Extracted deployment fields (from _extract_deployment_fields).
        expires_at: Computed expiration timestamp.

    Returns:
        Dict ready to unpack into the persistence capacity-check primitive.
    """
    return {
        "token_request_identifier": token_request_id,
        "user_id": user_id,
        "llm_provider": request.llm_provider.value,
        "llm_model_name": request.llm_model_name,
        "token_count": token_count,
        "expiration_timestamp": expires_at,
        "deployment_name": deployment["deployment_name"] or None,
        "cloud_provider_name": chosen_config.get("cloud_provider"),
        "api_endpoint_url": deployment["api_endpoint_url"],
        "deployment_region": deployment["deployment_region"] or None,
        "request_metadata": request.request_context,
        "temperature": deployment["temperature"],
        "top_p": deployment["top_p"],
        "seed": deployment["seed"],
    }


def _build_allocation_payload(
    token_request_id: str,
    user_id: UUID,
    request: TokenAllocationClientRequest,
    token_count: int,
    chosen_config: dict[str, Any],
    api_endpoint: str,
    expires_at: datetime,
) -> dict[str, Any]:
    """
    Build the RabbitMQ message payload for async DB persistence.

    Args:
        token_request_id: Generated unique allocation identifier.
        user_id: Requesting user's UUID.
        request: The original allocation request.
        token_count: Tokens reserved in Redis.
        chosen_config: Deployment configuration dict.
        api_endpoint: Selected API endpoint URL.
        expires_at: Computed expiration timestamp.

    Returns:
        Dict ready to publish via TokenAllocationPublisher.
    """
    return {
        "token_request_id": token_request_id,
        "user_id": str(user_id),
        "llm_provider": request.llm_provider.value,
        "llm_model_name": request.llm_model_name,
        "token_count": token_count,
        "api_endpoint_url": api_endpoint,
        "allocation_status": "ACQUIRED",
        "deployment_name": chosen_config.get("deployment_name", ""),
        "cloud_provider": chosen_config.get("cloud_provider"),
        "deployment_region": chosen_config.get("deployment_region", ""),
        "request_context": request.request_context,
        "temperature": chosen_config.get("temperature", 0.0),
        "seed": chosen_config.get("seed", 42),
        "expires_at": expires_at.isoformat(),
    }


def _build_fast_path_response(
    token_request_id: str,
    user_id: UUID,
    request: TokenAllocationClientRequest,
    token_count: int,
    chosen_config: dict[str, Any],
    api_endpoint: str,
    now: datetime,
    expires_at: datetime,
) -> TokenAllocationResponse:
    """
    Construct a TokenAllocationResponse for the Redis + RMQ fast path.

    Args:
        token_request_id: Generated allocation identifier.
        user_id: Requesting user's UUID.
        request: The original allocation request.
        token_count: Tokens reserved in Redis.
        chosen_config: Deployment configuration dict.
        api_endpoint: Selected API endpoint URL.
        now: Allocation timestamp.
        expires_at: Expiration timestamp.

    Returns:
        Fully populated TokenAllocationResponse with ACQUIRED status.
    """
    return TokenAllocationResponse(
        token_request_id=token_request_id,
        user_id=user_id,
        allocation_status="ACQUIRED",
        llm_model_name=request.llm_model_name,
        llm_provider=request.llm_provider.value,
        token_count=token_count,
        api_endpoint_url=api_endpoint,
        deployment_name=chosen_config.get("deployment_name", "") or None,
        cloud_provider=chosen_config.get("cloud_provider"),
        deployment_region=chosen_config.get("deployment_region", "") or None,
        temperature=chosen_config.get("temperature", 0.0),
        seed=chosen_config.get("seed", 42),
        allocated_at=now,
        expires_at=expires_at,
        request_context=request.request_context,
    )


def _log_redis_miss_reason(
    reservation_result: TokenReservationResult, model_name: str
) -> None:
    """
    Log the reason the Redis fast path did not return ALLOCATED.

    Args:
        reservation_result: The result code from reserve_tokens.
        model_name: For structured log context.
    """
    if reservation_result == TokenReservationResult.COUNTER_MISS:
        logger.debug(
            "[acquire] Redis fast-path MISS (counter not seeded) — using DB path",
            extra={"model": model_name},
        )
    else:
        logger.debug(
            "[acquire] Redis fast-path EXHAUSTED"
            " — DB path will decide WAITING/ACQUIRED",
            extra={"model": model_name},
        )
