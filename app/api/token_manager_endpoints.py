"""
Token Management Endpoints
--------------------------
Production-ready API endpoints for token allocation management.
Provides comprehensive token allocation operations with robust error handling.

Core Operations:
- Acquire tokens for LLM usage (immediate or waiting status)
- Retry acquiring tokens for waiting allocations
- Release allocated tokens back to pool
- Pause failing deployments for failover

Based on reference Flask service patterns from token_manager_service.py
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.psql_db_services.token_allocation_manager import TokenAllocationService
from app.models.request_models import (
    TokenAllocationRequest,
    TokenReleaseRequest,
    TokenRetryRequest,
    PauseDeploymentRequest,
)
from app.models.response_models import (
    TokenAllocationResponse,
    TokenReleaseResponse,
)

# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/api/v1/tokens", tags=["Token Management"])


# ============================================================================
# TOKEN ALLOCATION ENDPOINTS
# ============================================================================


@router.post(
    "/acquire",
    response_model=TokenAllocationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Acquire tokens for LLM usage",
    description="Reserve token capacity for LLM calls. Returns immediate allocation if capacity available, otherwise creates waiting allocation.",
)
async def acquire_tokens(request: TokenAllocationRequest):
    """
    Acquire tokens for LLM usage.

    Process:
    1. Validate input (handled by Pydantic)
    2. Find least loaded deployment for the model
    3. Check if capacity is available
    4. Create allocation (ACQUIRED if immediate, WAITING if capacity full)
    5. Return allocation details with deployment configuration

    Args:
        request: Token allocation parameters (user_id, model_name, token_count, etc.)

    Returns:
        TokenAllocationResponse: Allocation details with status and deployment info

    Raises:
        HTTPException 400: If validation fails or token count exceeds limit
        HTTPException 404: If no deployments found for model
        HTTPException 500: On internal server error
    """
    logger.info(
        f"Acquiring tokens: user={request.user_id}, model={request.llm_model_name}, tokens={request.token_count}"
    )

    try:
        # Create allocation service instance
        allocation_service = TokenAllocationService()

        # Acquire tokens
        allocation = await allocation_service.acquire_tokens(
            user_id=request.user_id,
            model_name=request.llm_model_name,
            token_count=request.token_count,
            request_context=request.request_context,
        )

        # Check for errors in response
        if "error" in allocation:
            error_msg = allocation["error"]
            logger.warning(f"Token allocation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
            )

        logger.info(
            f"Token allocation successful: {allocation['token_request_id']} - {allocation['allocation_status']}"
        )
        return TokenAllocationResponse(**allocation)

    except ValueError as e:
        # Handle validation errors
        error_msg = str(e)
        logger.warning(f"Validation error acquiring tokens: {error_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    except HTTPException:
        raise

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error acquiring tokens: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acquire tokens. Please try again later.",
        )


@router.post(
    "/acquire/retry",
    response_model=TokenAllocationResponse,
    summary="Retry acquiring tokens for waiting allocation",
    description="Retry acquiring tokens for a WAITING allocation. Checks if capacity is now available.",
)
async def retry_acquire_tokens(request: TokenRetryRequest):
    """
    Retry acquiring tokens for a WAITING allocation.

    Process:
    1. Validate token_request_id
    2. Find the waiting allocation
    3. Check if capacity is now available
    4. Update to ACQUIRED if possible, otherwise keep WAITING
    5. Return updated allocation status

    Args:
        request: Token retry parameters (token_request_id)

    Returns:
        TokenAllocationResponse: Updated allocation details
        Status 200: If successfully acquired
        Status 202: If still waiting for capacity

    Raises:
        HTTPException 404: If token_request_id not found
        HTTPException 400: If allocation is not in WAITING status
        HTTPException 500: On internal server error
    """
    logger.info(f"Retrying token acquisition: {request.token_request_id}")

    try:
        # Create allocation service instance
        allocation_service = TokenAllocationService()

        # Retry acquiring tokens
        allocation = await allocation_service.retry_acquire_tokens(
            request.token_request_id
        )

        # Check if allocation is None
        if allocation is None:
            logger.warning(f"Token request not found: {request.token_request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Token request '{request.token_request_id}' not found",
            )

        # Check for errors in response
        if "error" in allocation:
            error_msg = allocation["error"]
            logger.warning(f"Token retry failed: {error_msg}")

            # Determine appropriate status code based on error type
            if "not found" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=error_msg
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
                )

        # Check if still waiting
        if allocation.get("allocation_status") == "WAITING":
            logger.info(f"Token allocation still waiting: {request.token_request_id}")
            return TokenAllocationResponse(**allocation), status.HTTP_202_ACCEPTED

        # Successfully acquired
        logger.info(f"Token allocation acquired: {request.token_request_id}")
        return TokenAllocationResponse(**allocation)

    except HTTPException:
        raise

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error retrying token acquisition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry token acquisition. Please try again later.",
        )


@router.put(
    "/release",
    response_model=TokenReleaseResponse,
    summary="Release allocated tokens",
    description="Release allocated tokens back to the pool. Idempotent operation - safe to call multiple times.",
)
async def release_tokens(request: TokenReleaseRequest):
    """
    Release allocated tokens back to the pool.

    Process:
    1. Validate token_request_id
    2. Update allocation status to RELEASED
    3. Calculate latency metrics
    4. Return confirmation

    Args:
        request: Token release parameters (token_request_id)

    Returns:
        TokenReleaseResponse: Release confirmation with status

    Raises:
        HTTPException 500: On internal server error
    """
    logger.info(f"Releasing tokens: {request.token_request_id}")

    try:
        # Create allocation service instance
        allocation_service = TokenAllocationService()

        # Release tokens (update to RELEASED status)
        await allocation_service.update_allocation_completed(request.token_request_id)

        # Always return success for idempotency (like the reference Flask service)
        logger.info(f"Tokens released successfully: {request.token_request_id}")
        return TokenReleaseResponse(
            token_request_id=request.token_request_id,
            allocation_status="RELEASED",
            message="Tokens released successfully",
        )

    except Exception as e:
        # Log error but still return success for idempotency
        logger.warning(f"Error releasing tokens {request.token_request_id}: {e}")
        return TokenReleaseResponse(
            token_request_id=request.token_request_id,
            allocation_status="RELEASED",
            message="Tokens released successfully",
        )


# ============================================================================
# DEPLOYMENT MANAGEMENT ENDPOINTS
# ============================================================================


@router.put(
    "/pause-deployment",
    summary="Pause a failing deployment",
    description="Pause a failing deployment for emergency failover. Blocks all new allocations to the specified deployment.",
)
async def pause_deployment(request: PauseDeploymentRequest):
    """
    Pause a failing deployment for emergency failover.

    Process:
    1. Validate input parameters
    2. Find the deployment configuration
    3. Create a PAUSED allocation to block the deployment
    4. Return pause confirmation

    Args:
        request: Pause deployment parameters (model_name, api_base, pause_reason, etc.)

    Returns:
        Dictionary with pause status and details

    Raises:
        HTTPException 400: If validation fails
        HTTPException 404: If deployment not found
        HTTPException 500: On internal server error
    """
    logger.info(
        f"Pausing deployment: model={request.llm_model_name}, endpoint={request.api_base}, reason={request.pause_reason}"
    )

    try:
        # Validate api_base is not None
        if not request.api_base:
            logger.warning(
                f"Missing api_base for pause deployment: {request.llm_model_name}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="api_base parameter is required for pause deployment operation",
            )

        # Create allocation service instance
        allocation_service = TokenAllocationService()

        # Pause the deployment
        result = await allocation_service.pause_deployment(
            model_name=request.llm_model_name,
            api_endpoint=request.api_base,
            pause_reason=request.pause_reason,
            pause_duration_minutes=request.pause_duration_minutes or 30,
        )

        # Check for errors in response
        if "error" in result:
            error_msg = result["error"]
            logger.warning(f"Deployment pause failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
            )

        # Check if deployment not found
        if result.get("alloc_status") == "NOT_FOUND":
            logger.warning(
                f"Deployment not found: {request.llm_model_name} at {request.api_base}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment '{request.llm_model_name}' at '{request.api_base}' not found",
            )

        logger.info(
            f"Deployment paused successfully: {request.llm_model_name} at {request.api_base}"
        )
        return result

    except HTTPException:
        raise

    except ValueError as e:
        # Handle validation errors
        error_msg = str(e)
        logger.warning(f"Validation error pausing deployment: {error_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error pausing deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause deployment. Please try again later.",
        )
