"""
Token Management Endpoints
--------------------------
Thin FastAPI routers for the four token management operations.

Responsibility of this layer:
- HTTP routing and request/response serialisation (Pydantic models).
- Authentication and object-level authorisation checks.
- Mapping domain exceptions from app/services/ to HTTP status codes.

What this layer does NOT do:
- Business logic or orchestration (→ app/services/).
- Database queries (→ app/persistence/).
- Redis or RabbitMQ operations (→ app/resilience/).

Architecture:
-------------
    ┌────────────────────────────────┐
    │  token_manager_endpoints.py    │  ← You are here
    │  (Interface Layer)             │
    │  auth + routing + HTTP mapping │
    └──────────────┬─────────────────┘
                   │  FastAPI Depends()
    ┌──────────────▼─────────────────┐
    │  app/services/                 │
    │  TokenAcquisitionService       │
    │  TokenReleaseService           │
    │  TokenRetryService             │
    └────────────────────────────────┘
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger

from app.auth import AuthTokenPayload, CurrentUser
from app.core.exceptions import (
    AllocationNotFoundError,
    AllocationStateError,
    DatabaseUnavailableError,
    DeploymentConfigurationError,
    TokenLimitExceededError,
)
from app.core.redis_rate_limiter import token_acquire_rate_limiter
from app.models.request_models import (
    PauseDeploymentRequest,
    TokenAllocationClientRequest,
    TokenRetryRequest,
)
from app.models.response_models import (
    PauseDeploymentAllocationResponse,
    TokenAllocationResponse,
)
from app.persistence.allocations import LLMTokenAllocationPersistence
from app.persistence.deployed_llm_endpoints import DeployedLLMReadPersistence
from app.resilience.backpressure import backpressure_dependency
from app.resilience.circuit_breaker import get_db_circuit_breaker
from app.resilience.redis_token_counter import get_shared_redis_token_counter_service
from app.resilience.token_queue import TokenAllocationPublisher
from app.services.deployment_load_balancer import DeploymentLoadBalancer
from app.services.token_acquisition_service import TokenAcquisitionService
from app.services.token_retry_service import TokenRetryService

# ---------------------------------------------------------------------------
# Module-level stateless infrastructure singletons (safe to share across
# requests — no mutable state between calls).
# ---------------------------------------------------------------------------
_shared_token_counter_service = get_shared_redis_token_counter_service()
_publisher = TokenAllocationPublisher()

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/api/v1/tokens", tags=["Token Management"])

# ============================================================================
# FastAPI dependency providers (composition root)
#
# Enterprise/testing note:
# - FastAPI's Depends() captures the callable at definition time.
# - Tests patch these get_*_service() functions to inject fakes without
#   rebuilding the router/app dependency graph.
# ============================================================================


def get_deployment_load_balancer() -> DeploymentLoadBalancer:
    """Factory for the deployment load balancer; overridable in tests."""
    return DeploymentLoadBalancer(endpoint_reads=DeployedLLMReadPersistence())


def get_token_acquisition_service() -> TokenAcquisitionService:
    """Factory for TokenAcquisitionService with all injected dependencies."""
    return TokenAcquisitionService(
        allocation_persistence=LLMTokenAllocationPersistence(),
        load_balancer=get_deployment_load_balancer(),
        redis_counter=_shared_token_counter_service,
        publisher=_publisher,
        db_circuit_breaker=get_db_circuit_breaker(),
    )


def get_token_retry_service() -> TokenRetryService:
    """Factory for TokenRetryService with all injected dependencies."""
    return TokenRetryService(
        allocation_persistence=LLMTokenAllocationPersistence(),
        load_balancer=get_deployment_load_balancer(),
    )


def get_allocation_persistence() -> LLMTokenAllocationPersistence:
    """Factory for LLMTokenAllocationPersistence; overridable in tests."""
    return LLMTokenAllocationPersistence()


# ============================================================================
# Object-level authorisation helper
# ============================================================================


def _is_authorized_for_token_request(
    allocation: dict,
    current_user: AuthTokenPayload,
) -> bool:
    """
    Check object-level authorisation for token request operations.

    Access policy:
    - Token owner can operate on their own request.
    - admin / owner roles can operate on any request.

    Args:
        allocation: The allocation record dict (must contain 'user_id').
        current_user: JWT payload of the authenticated requester.

    Returns:
        True if the requester is authorised, False otherwise.
    """
    if current_user.role in {"admin", "owner"}:
        return True
    owner_id = allocation.get("user_id")
    if owner_id is None:
        return False
    return str(owner_id) == str(current_user.user_id)


# ============================================================================
# TOKEN ALLOCATION ENDPOINTS
# ============================================================================


@router.post(
    "/acquire",
    response_model=TokenAllocationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Acquire tokens for LLM usage",
    description=(
        "Reserve token capacity for LLM calls. "
        "Fast path: Redis atomic reservation + async RabbitMQ persistence (~1ms). "
        "Fallback: synchronous PostgreSQL path wrapped with circuit breaker."
    ),
)
async def acquire_tokens(
    request: TokenAllocationClientRequest,
    current_user: CurrentUser,
    # Layer 0: per-service rate limit (X-Service-Id bucketing)
    _rate_limit: None = Depends(token_acquire_rate_limiter()),
    # Layer 1: back pressure — fail-fast 503 when system is saturated
    _backpressure: None = Depends(backpressure_dependency),
    service: TokenAcquisitionService = Depends(get_token_acquisition_service),
):
    """
    Acquire tokens for LLM usage — 5-layer resilience hot path.

    Execution path:
    1. Pydantic validation (automatic)
    2. Per-service rate limit check  (Layer 0 — X-Service-Id sliding window)
    3. Back pressure check           (Layer 1 — queue depth + DB pool + CB state)
    4–7. TokenAcquisitionService     (user guard → estimation → Redis → RMQ → DB)

    Returns:
        201 ACQUIRED — tokens reserved (fast-path or DB-path)
        201 WAITING  — capacity full; client should retry via /acquire/retry
        429          — rate limit exceeded
        503          — system saturated or circuit breaker open
    """
    try:
        return await service.acquire_tokens(
            current_user.user_id, current_user.tenant_id, request
        )

    except (DatabaseUnavailableError, DeploymentConfigurationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "30"},
        )
    except TokenLimitExceededError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ValueError as exc:
        logger.warning("[acquire] Validation error", extra={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("[acquire] Unexpected error acquiring tokens")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acquire tokens. Please try again later.",
        )


@router.post(
    "/acquire/retry",
    response_model=TokenAllocationResponse,
    status_code=status.HTTP_200_OK,
    summary="Retry acquiring tokens for a WAITING allocation",
    description="Check if capacity is now available for a WAITING allocation.",
)
async def retry_acquire_tokens(
    request: TokenRetryRequest,
    response: Response,
    current_user: CurrentUser,
    service: TokenRetryService = Depends(get_token_retry_service),
):
    """
    Retry acquiring tokens for a WAITING allocation.

    Returns 200 if successfully promoted to ACQUIRED, or 202 if still WAITING.

    Args:
        request: Contains the token_request_id to retry.

    Returns:
        TokenAllocationResponse with allocation_status ACQUIRED (200)
        or WAITING (202).
    """
    try:
        allocation_response = await service.retry_acquire(request.token_request_id)

    except AllocationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except AllocationStateError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DeploymentConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except Exception:
        logger.exception("[retry] Unexpected error retrying token acquisition")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry token acquisition. Please try again later.",
        )

    if not _is_authorized_for_token_request(
        allocation_response.model_dump(), current_user
    ):
        logger.warning(
            "[retry] Unauthorised retry attempt",
            extra={
                "requester": str(current_user.user_id),
                "token_request_id": request.token_request_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorised to operate on this token request",
        )

    # Enterprise FastAPI pattern: inject Response to vary success status code
    # rather than returning a (payload, status_code) tuple which is fragile.
    if allocation_response.allocation_status == "WAITING":
        response.status_code = status.HTTP_202_ACCEPTED

    return allocation_response


@router.put(
    "/release",
    status_code=status.HTTP_410_GONE,
    summary="[REMOVED] Manual token release is no longer available",
    description=(
        "Token release is now handled automatically by the LLM worker after "
        "each job completes (SUCCESS or FAILURE). Calling this endpoint is no "
        "longer necessary and will return 410 Gone. "
        "Submit a job via POST /api/v1/llm/jobs — the worker releases tokens for you."
    ),
    include_in_schema=True,
)
async def release_tokens_removed() -> dict:
    """
    Endpoint retired: manual token release is no longer available.

    Tokens are released automatically by the gateway worker when an LLM job
    reaches a terminal state (SUCCESS or FAILURE). The caller does not need to
    release manually — doing so would break the lifecycle guarantee.

    Returns:
        HTTP 410 Gone with an explanation.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=(
            "Manual token release has been removed. "
            "Tokens are released automatically by the LLM worker when the job finishes. "
            "Submit your job via POST /api/v1/llm/jobs."
        ),
    )


# ============================================================================
# DEPLOYMENT MANAGEMENT ENDPOINTS
# ============================================================================


@router.put(
    "/pause-deployment",
    response_model=PauseDeploymentAllocationResponse,
    status_code=status.HTTP_200_OK,
    summary="Pause a failing deployment",
    description=(
        "Pause a failing deployment for emergency failover. "
        "Creates a PAUSED capacity-blocker allocation so the load balancer "
        "routes all new traffic away from the problematic endpoint. "
        "Returns 409 if the deployment is already paused."
    ),
)
async def pause_deployment(
    request: PauseDeploymentRequest,
    current_user: CurrentUser,
    allocation_persistence: LLMTokenAllocationPersistence = Depends(
        get_allocation_persistence
    ),
):
    """
    Pause a failing deployment for emergency failover.

    Mechanism: Creates a PAUSED allocation consuming the full capacity of
    the target deployment.  The load balancer sees 100% utilisation and
    routes all new traffic to other deployments.  The underlying persistence
    call is atomic — the deployment row is locked before the duplicate check
    and the INSERT, so concurrent pause requests cannot produce duplicate
    capacity-blocker rows.

    Identity (user_id, tenant_id) comes from the verified JWT.

    Args:
        request: Pause parameters (model, endpoint, reason, duration).

    Returns:
        PauseDeploymentAllocationResponse with allocation_status = 'PAUSED'.
        409 if the deployment is already paused.
        404 if the deployment does not exist.
    """
    try:
        if request.api_endpoint_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="api_endpoint_url is required for pause deployment",
            )

        result = await allocation_persistence.pause_deployment(
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            provider_name=request.llm_provider.value,
            model_name=request.llm_model_name,
            api_endpoint=request.api_endpoint_url,
            pause_reason=request.pause_reason,
            pause_duration_minutes=request.pause_duration_minutes or 30,
        )

        alloc_status = result.get("alloc_status")
        if alloc_status == "NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Deployment '{request.llm_provider}/{request.llm_model_name}' "
                    f"at '{request.api_endpoint_url}' not found"
                ),
            )
        if alloc_status == "ALREADY_PAUSED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Deployment '{request.llm_provider}/{request.llm_model_name}' "
                    f"at '{request.api_endpoint_url}' is already paused"
                ),
            )

        logger.info(
            "[pause] Deployment paused",
            extra={
                "provider": str(request.llm_provider),
                "model": request.llm_model_name,
                "endpoint": request.api_endpoint_url,
            },
        )
        return PauseDeploymentAllocationResponse(**result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("[pause] Unexpected error pausing deployment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause deployment. Please try again later.",
        )
