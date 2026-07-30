"""
TokenAcquisitionService — token allocation use-case orchestration.

Implements the resilience acquisition flow, coordinating across Redis, RabbitMQ,
and PostgreSQL without embedding that logic in the API or persistence layers.

Architecture:
-------------
    acquire_tokens API (app/api/)
            │  Depends()
            ▼
    TokenAcquisitionService ──▶ DeploymentLoadBalancer   (which endpoint?)
            │               ──▶ LLMTokenAllocationPersistence (reserve)
            │               ──▶ RedisTokenCounterService  (fast-path reserve)
            └──────────────────▶ TokenAllocationPublisher (async DB persist)

Layer execution order:
    0. Rate limit        (API layer via Depends)
    1. Back pressure     (API layer via Depends)
    2. Token estimation  (this service)
    3. Deployment pick   (this service → DeploymentLoadBalancer)
    4. Redis fast path   (this service → RedisTokenCounterService)
    4a.  ALLOCATED → RMQ publish; RMQ CB open → Redis rollback → DB path
    4b.  MISS / EXHAUSTED → DB path
    5. DB path           (this service → LLMTokenAllocationPersistence)
       Atomic capacity check decides WAITING vs ACQUIRED; creates the record.

Identity (user_id, tenant_id) comes from the verified JWT — this service never
queries the users table (that is llm_services' concern).
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
    TokenLimitExceededError,
)
from app.models.response_models import TokenAllocationResponse
from app.resilience.redis_token_counter import TokenReservationResult
from app.utils.token_count_estimation import estimate_tokens

if TYPE_CHECKING:
    from app.models.request_models import TokenAllocationClientRequest
    from app.persistence.allocations import LLMTokenAllocationPersistence
    from app.resilience.redis_token_counter import RedisTokenCounterService
    from app.resilience.token_queue import TokenAllocationPublisher
    from app.services.deployment_load_balancer import DeploymentLoadBalancer

logger = logging.getLogger(__name__)

_DEFAULT_LOCK_SECONDS = 70


class TokenAcquisitionService:
    """
    Orchestrates the complete token acquisition use case.

    Picks a deployment via the load balancer, then coordinates Redis fast-path
    reservation, RabbitMQ async persistence, and the PostgreSQL DB fallback.
    Does not perform DB queries directly — delegates to injected adapters.
    """

    def __init__(
        self,
        allocation_persistence: LLMTokenAllocationPersistence,
        load_balancer: DeploymentLoadBalancer,
        redis_counter: RedisTokenCounterService,
        publisher: TokenAllocationPublisher,
        db_circuit_breaker: aiobreaker.CircuitBreaker,
    ) -> None:
        """Initialise with injected dependencies."""
        self._allocation_persistence = allocation_persistence
        self._load_balancer = load_balancer
        self._redis_counter = redis_counter
        self._publisher = publisher
        self._db_circuit_breaker = db_circuit_breaker

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def acquire_tokens(
        self,
        user_id: UUID,
        tenant_id: UUID,
        request: TokenAllocationClientRequest,
    ) -> TokenAllocationResponse:
        """
        Acquire tokens following the resilience path.

        Args:
            user_id: Authenticated user's UUID (from JWT).
            tenant_id: Tenant the user is acting within (from JWT).
            request: Validated client allocation request.

        Returns:
            TokenAllocationResponse with status ACQUIRED or WAITING.

        Raises:
            DeploymentConfigurationError: No active deployment for provider/model.
            TokenLimitExceededError: Request exceeds the deployment's capacity.
            DatabaseUnavailableError: DB circuit breaker is open.
        """
        provider_name = request.llm_provider.value
        model_name = request.llm_model_name
        estimated_token_count = self._estimate_token_count(request)

        deployment = await self._choose_deployment(tenant_id, provider_name, model_name)
        api_endpoint = deployment["api_endpoint_url"]

        reservation_result = await self._redis_counter.reserve_tokens(
            model_name=model_name,
            api_endpoint_url=api_endpoint,
            token_count=estimated_token_count,
        )

        if reservation_result == TokenReservationResult.ALLOCATED:
            fast_path_response = await self._handle_fast_path(
                user_id, tenant_id, request, estimated_token_count, deployment
            )
            if fast_path_response is not None:
                return fast_path_response
            # RMQ CB open — Redis reservation rolled back; fall to DB path.

        _log_redis_miss_reason(reservation_result, model_name)
        return await self._create_db_allocation(
            user_id, tenant_id, request, estimated_token_count, deployment
        )

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _estimate_token_count(self, request: TokenAllocationClientRequest) -> int:
        """Return estimated token count for the request's input data."""
        return estimate_tokens(request.input_data, request.llm_model_name).total_tokens

    # ------------------------------------------------------------------
    # Deployment selection (load balancer + circuit breaker)
    # ------------------------------------------------------------------

    async def _choose_deployment(
        self, tenant_id: UUID, provider_name: str, model_name: str
    ) -> dict[str, Any]:
        """
        Ask the load balancer for the least-loaded active deployment.

        Raises:
            DatabaseUnavailableError: If the DB circuit breaker is open.
            DeploymentConfigurationError: If no active deployment exists.
        """
        try:
            return cast(
                "dict[str, Any]",
                await self._db_circuit_breaker.call_async(
                    self._load_balancer.choose_least_loaded,
                    tenant_id,
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
        tenant_id: UUID,
        request: TokenAllocationClientRequest,
        token_count: int,
        deployment: dict[str, Any],
    ) -> TokenAllocationResponse | None:
        """
        Attempt RMQ publish after a successful Redis reservation.

        Returns the allocation response on success, or None if the RabbitMQ
        circuit breaker is open (Redis reservation already rolled back).
        """
        token_request_id = f"req_{uuid.uuid4().hex}"
        lock_secs = (
            deployment.get("token_lock_duration_seconds") or _DEFAULT_LOCK_SECONDS
        )
        now = datetime.now()
        expires_at = now + timedelta(seconds=lock_secs)
        api_endpoint = deployment["api_endpoint_url"]

        payload = _build_persist_payload(
            token_request_id,
            user_id,
            tenant_id,
            request,
            token_count,
            deployment,
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
            },
        )
        return _build_allocation_response(
            token_request_id,
            user_id,
            tenant_id,
            request,
            token_count,
            deployment,
            "ACQUIRED",
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
        """Release the Redis reservation when the RMQ circuit breaker is open."""
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
            "[acquire] RMQ CB open — Redis reservation rolled back; using DB path",
            extra={"token_request_id": token_request_id},
        )

    # ------------------------------------------------------------------
    # DB path
    # ------------------------------------------------------------------

    async def _create_db_allocation(
        self,
        user_id: UUID,
        tenant_id: UUID,
        request: TokenAllocationClientRequest,
        token_count: int,
        deployment: dict[str, Any],
    ) -> TokenAllocationResponse:
        """
        Create an allocation record synchronously via PostgreSQL.

        The atomic capacity-check primitive decides WAITING vs ACQUIRED.

        Raises:
            TokenLimitExceededError: If token_count exceeds the deployment capacity.
            DatabaseUnavailableError: If the DB circuit breaker is open.
        """
        capacity_limit = deployment["token_capacity_limit"]
        if token_count > capacity_limit:
            raise TokenLimitExceededError(
                token_count, capacity_limit, request.llm_model_name
            )

        lock_secs = (
            deployment.get("token_lock_duration_seconds") or _DEFAULT_LOCK_SECONDS
        )
        expires_at = datetime.now() + timedelta(seconds=lock_secs)
        try:
            record = cast(
                "dict[str, Any]",
                await self._db_circuit_breaker.call_async(
                    self._allocation_persistence.create_allocation_with_capacity_check,
                    token_request_identifier=f"req_{uuid.uuid4().hex}",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    deployment_id=deployment["deployment_id"],
                    provider_name=request.llm_provider.value,
                    model_name=request.llm_model_name,
                    token_count=token_count,
                    expiration_timestamp=expires_at,
                    request_metadata=request.request_context,
                ),
            )
        except aiobreaker.CircuitBreakerError as exc:
            raise DatabaseUnavailableError(
                "DB circuit breaker open during DB-path allocation"
            ) from exc

        logger.info(
            "[acquire] DB-path allocation created",
            extra={
                "token_request_id": record.get("token_request_id"),
                "status": record.get("allocation_status"),
                "model": request.llm_model_name,
            },
        )
        return TokenAllocationResponse(**record)


# ---------------------------------------------------------------------------
# Module-level pure helpers (no service state)
# ---------------------------------------------------------------------------


def _build_persist_payload(
    token_request_id: str,
    user_id: UUID,
    tenant_id: UUID,
    request: TokenAllocationClientRequest,
    token_count: int,
    deployment: dict[str, Any],
    expires_at: datetime,
) -> dict[str, Any]:
    """Build the RabbitMQ message payload for async DB persistence."""
    return {
        "token_request_id": token_request_id,
        "tenant_id": str(tenant_id),
        "user_id": str(user_id),
        "deployment_id": str(deployment["deployment_id"]),
        "provider_name": request.llm_provider.value,
        "model_name": request.llm_model_name,
        "deployment_key": deployment["deployment_key"],
        "token_count": token_count,
        "api_endpoint_url": deployment["api_endpoint_url"],
        "allocation_status": "ACQUIRED",
        "deployment_name": deployment.get("deployment_name"),
        "provider_deployment_name": deployment.get("provider_deployment_name"),
        "cloud_provider": deployment.get("cloud_provider"),
        "cloud_region": deployment.get("cloud_region"),
        "temperature": deployment.get("default_temperature"),
        "top_p": deployment.get("default_top_p"),
        "seed": None,
        "request_context": request.request_context,
        "expires_at": expires_at.isoformat(),
    }


def _build_allocation_response(
    token_request_id: str,
    user_id: UUID,
    tenant_id: UUID,
    request: TokenAllocationClientRequest,
    token_count: int,
    deployment: dict[str, Any],
    allocation_status: str,
    now: datetime,
    expires_at: datetime,
) -> TokenAllocationResponse:
    """Construct a TokenAllocationResponse for the Redis + RMQ fast path."""
    return TokenAllocationResponse(
        token_request_id=token_request_id,
        user_id=user_id,
        tenant_id=tenant_id,
        allocation_status=allocation_status,
        provider_name=request.llm_provider.value,
        model_name=request.llm_model_name,
        token_count=token_count,
        api_endpoint_url=deployment["api_endpoint_url"],
        deployment_name=deployment.get("deployment_name"),
        cloud_provider=deployment.get("cloud_provider"),
        cloud_region=deployment.get("cloud_region"),
        temperature=deployment.get("default_temperature"),
        top_p=deployment.get("default_top_p"),
        seed=None,
        allocated_at=now,
        expires_at=expires_at,
        request_context=request.request_context,
    )


def _log_redis_miss_reason(
    reservation_result: TokenReservationResult, model_name: str
) -> None:
    """Log the reason the Redis fast path did not return ALLOCATED."""
    if reservation_result == TokenReservationResult.COUNTER_MISS:
        logger.debug(
            "[acquire] Redis fast-path MISS (counter not seeded) — using DB path",
            extra={"model": model_name},
        )
    else:
        logger.debug(
            "[acquire] Redis fast-path EXHAUSTED — DB path decides WAITING/ACQUIRED",
            extra={"model": model_name},
        )
