"""
LLM Model Configuration Endpoints
---------------------------------
Production-ready API endpoints for LLM model configuration management.
Provides comprehensive CRUD operations with robust error handling and admin-only access control.

Admin-Only Operations:
- Create new LLM model configurations
- Update existing model configurations
- Deactivate/activate models

Public Operations:
- Retrieve model configurations
- List models by provider
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends
from loguru import logger

from app.psql_db_services.llm_models_service import LLMModelsService
from app.models.request_models import LLMModelCreateRequest, LLMModelUpdateRequest
from app.models.response_models import LLMModelResponse, LLMModelListResponse
from app.auth import require_developer, require_admin, TokenPayload

# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/api/v1/llm-models", tags=["LLM Model Configuration"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================


# Removed placeholder require_admin_user - now using real auth dependencies


# ============================================================================
# CREATE ENDPOINTS
# ============================================================================


# ============================================================================
#                           CREATE LLM MODEL ENDPOINT
# ============================================================================
@router.post(
    "/",
    response_model=LLMModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new LLM model configuration",
    description="Register a new LLM model configuration with rate limits, endpoints, and settings. Admin access required.",
)
async def create_llm_model(
    request: LLMModelCreateRequest, current_user: TokenPayload = Depends(require_admin)
):
    """
    Create a new LLM model configuration in the system.

    Process:
    1. Validate input (handled by Pydantic)
    2. Generate unique model ID
    3. Validate provider and model name constraints
    4. Set defaults (timestamps, active status)
    5. Insert into database
    6. Return model configuration (without sensitive data)

    Args:
        request: LLM model creation parameters (provider, model name, rate limits, etc.)

    Returns:
        LLMModelResponse: Created model configuration

    Raises:
        HTTPException 400: If validation fails or model already exists
        HTTPException 500: On internal server error
    """
    logger.info(
        f"Creating LLM model: provider={request.provider_name}, model={request.llm_model_name}"
    )

    try:
        # Create model in database
        llm_service = LLMModelsService()
        model = await llm_service.create_llm_model(
            provider_name=request.provider_name,
            llm_model_name=request.llm_model_name,
            api_key_variable_name=request.api_key_variable_name,
            llm_model_version=request.llm_model_version,
            max_tokens=request.max_tokens,
            tokens_per_minute_limit=request.tokens_per_minute_limit,
            requests_per_minute_limit=request.requests_per_minute_limit,
            deployment_name=request.deployment_name,
            api_endpoint_url=request.api_endpoint_url,
            is_active_status=request.is_active_status,
            temperature=request.temperature,
            random_seed=request.random_seed,
            deployment_region=request.deployment_region,
        )

        logger.info(
            f"LLM model created successfully: {model['provider_name']}/{model['llm_model_name']}/{model['llm_model_version']}"
        )
        return LLMModelResponse(**model)

    except ValueError as e:
        # Handle validation errors and constraint violations
        error_msg = str(e)
        logger.warning(f"Validation error creating LLM model: {error_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error creating LLM model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create LLM model configuration. Please try again later.",
        )


# ============================================================================
# READ ENDPOINTS
# ============================================================================

# FASTAPI ROUTE PRECEDENCE NOTE:
# ===============================
# IMPORTANT: Route order matters in FastAPI! More specific routes must come BEFORE less specific ones.
#
# Route precedence issue example:
# - "/provider/{provider}" (more specific - matches /provider/openai)
# - "/{provider}/{model_name}" (less specific - matches /provider/openai as provider="provider", model_name="openai")
#
# WRONG ORDER (causes conflicts):
#   @router.get("/{provider}/{model_name}")  # This matches /provider/openai first!
#   @router.get("/provider/{provider}")     # This never gets reached
#
# CORRECT ORDER (fixed):
#   @router.get("/provider/{provider}")     # More specific - matches first
#   @router.get("/{provider}/{model_name}") # Less specific - matches second
#
# This is why the routes are ordered with more specific paths first in this file.


@router.get(
    "/provider/{provider}",
    response_model=LLMModelListResponse,
    summary="List LLM models by provider",
    description="Retrieve all LLM model configurations for a specific provider with optional filtering and pagination.",
)
async def list_llm_models_by_provider(
    provider: str,
    active_only: Optional[bool] = Query(
        None, description="Filter for active models only"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: TokenPayload = Depends(require_developer),
):
    """
    List LLM model configurations for a specific provider.

    Args:
        provider: LLM provider name (e.g., 'openai', 'anthropic')
        active_only: Optional filter for active models only
        limit: Maximum results per page (1-1000)
        offset: Pagination offset

    Returns:
        LLMModelListResponse: Paginated list of model configurations

    Raises:
        HTTPException 400: On invalid parameters
        HTTPException 500: On internal server error
    """
    logger.debug(
        f"Listing LLM models: provider={provider}, active_only={active_only}, limit={limit}, offset={offset}"
    )

    try:
        llm_service = LLMModelsService()
        models = await llm_service.get_llm_models_by_provider(
            provider_name=provider, active_only=active_only, limit=limit, offset=offset
        )

        # Calculate pagination info
        total_count = len(
            models
        )  # For simplicity, in production you'd get this from a count query
        has_next = (offset + limit) < total_count
        has_previous = offset > 0

        return LLMModelListResponse(
            models=[LLMModelResponse(**model) for model in models],
            total_count=total_count,
            page=(offset // limit) + 1 if limit > 0 else 1,
            page_size=limit,
            has_next=has_next,
            has_previous=has_previous,
        )

    except ValueError as validation_error:
        logger.warning(f"Validation error listing LLM models: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
        )
    except Exception as error:
        logger.error(f"Error listing LLM models: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LLM model configurations",
        )


@router.get(
    "/{provider}/{model_name}",
    response_model=LLMModelResponse,
    summary="Get LLM model configuration",
    description="Retrieve a specific LLM model configuration by provider and model name.",
)
async def get_llm_model(
    provider: str,
    model_name: str,
    version: Optional[str] = Query(None, description="Optional model version"),
    current_user: TokenPayload = Depends(require_developer),
):
    """
    Retrieve an LLM model configuration by provider and model name.

    Args:
        provider: LLM provider name (e.g., 'openai', 'anthropic')
        model_name: Model name (e.g., 'gpt-4o', 'claude-3.5-sonnet')
        version: Optional model version (e.g., '2024-08')

    Returns:
        LLMModelResponse: Model configuration details

    Raises:
        HTTPException 404: If model configuration not found
        HTTPException 500: On internal server error
    """
    logger.debug(
        f"Fetching LLM model: provider={provider}, model={model_name}, version={version}"
    )

    try:
        llm_service = LLMModelsService()
        model = await llm_service.get_llm_model_by_provider_and_model(
            provider_name=provider, llm_model_name=model_name, llm_model_version=version
        )

        if not model:
            logger.warning(
                f"LLM model not found: provider={provider}, model={model_name}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM model '{model_name}' for provider '{provider}' not found",
            )

        return LLMModelResponse(**model)

    except HTTPException:
        raise
    except ValueError as validation_error:
        logger.warning(f"Invalid parameters: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
        )
    except Exception as error:
        logger.error(f"Error fetching LLM model: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LLM model configuration",
        )


# ============================================================================
# UPDATE ENDPOINTS
# ============================================================================


@router.patch(
    "/{provider}/{model_name}",
    response_model=LLMModelResponse,
    summary="Update LLM model configuration",
    description="Update LLM model configuration fields. Only provided fields will be updated. Admin access required.",
)
async def update_llm_model(
    provider: str,
    model_name: str,
    request: LLMModelUpdateRequest,
    version: Optional[str] = Query(None, description="Optional model version"),
    current_user: TokenPayload = Depends(require_admin),
):
    """
    Update an LLM model configuration.

    Args:
        provider: Current provider name identifying the model
        model_name: Current model name identifying the model
        version: Optional current model version
        request: Fields to update (all optional)

    Returns:
        LLMModelResponse: Updated model configuration

    Raises:
        HTTPException 404: If model configuration not found
        HTTPException 400: On invalid parameters or constraint violations
        HTTPException 500: On internal server error
    """
    logger.info(f"Updating LLM model: provider={provider}, model={model_name}")

    try:
        llm_service = LLMModelsService()
        updated_model = await llm_service.update_llm_model(
            provider_name=provider,
            llm_model_name=model_name,
            llm_model_version=version,
            new_provider_name=request.provider_name,
            new_llm_model_name=request.llm_model_name,
            deployment_name=request.deployment_name,
            api_key_variable_name=request.api_key_variable_name,
            api_endpoint_url=request.api_endpoint_url,
            new_llm_model_version=request.llm_model_version,
            max_tokens=request.max_tokens,
            tokens_per_minute_limit=request.tokens_per_minute_limit,
            requests_per_minute_limit=request.requests_per_minute_limit,
            is_active_status=request.is_active_status,
            temperature=request.temperature,
            random_seed=request.random_seed,
            deployment_region=request.deployment_region,
        )

        if not updated_model:
            logger.warning(
                f"LLM model not found for update: provider={provider}, model={model_name}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM model '{model_name}' for provider '{provider}' not found",
            )

        logger.info(
            f"LLM model updated successfully: provider={provider}, model={model_name}"
        )
        return LLMModelResponse(**updated_model)

    except HTTPException:
        raise
    except ValueError as validation_error:
        logger.warning(f"Validation error updating LLM model: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
        )
    except Exception as error:
        # Check if it's a constraint violation (duplicate key, etc.)
        error_str = str(error).lower()
        if "unique" in error_str or "duplicate" in error_str:
            logger.warning(f"Constraint violation in update: {request}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model configuration already exists with these parameters",
            )

        logger.error(f"Error updating LLM model: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update LLM model configuration",
        )


# ============================================================================
# ACTIVATION/DEACTIVATION ENDPOINTS
# ============================================================================


@router.patch(
    "/{provider}/{model_name}/activate",
    response_model=LLMModelResponse,
    summary="Activate LLM model",
    description="Activate an LLM model configuration for token allocation. Admin access required.",
)
async def activate_llm_model(
    provider: str,
    model_name: str,
    version: Optional[str] = Query(None, description="Optional model version"),
    current_user: TokenPayload = Depends(require_admin),
):
    """
    Activate an LLM model configuration.

    Args:
        provider: Provider name identifying the model
        model_name: Model name identifying the model
        version: Optional model version

    Returns:
        LLMModelResponse: Updated model configuration

    Raises:
        HTTPException 404: If model configuration not found
        HTTPException 500: On internal server error
    """
    logger.info(f"Activating LLM model: provider={provider}, model={model_name}")

    try:
        llm_service = LLMModelsService()
        activated_model = await llm_service.activate_llm_model(
            provider_name=provider,
            llm_model_name=model_name,
            llm_model_version=version,
        )

        if not activated_model:
            logger.warning(
                f"LLM model not found for activation: provider={provider}, model={model_name}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM model '{model_name}' for provider '{provider}' not found",
            )

        logger.info(
            f"LLM model activated successfully: provider={provider}, model={model_name}"
        )
        return LLMModelResponse(**activated_model)

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error activating LLM model: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate LLM model configuration",
        )


@router.patch(
    "/{provider}/{model_name}/deactivate",
    response_model=LLMModelResponse,
    summary="Deactivate LLM model",
    description="Deactivate an LLM model configuration to prevent token allocation. Admin access required.",
)
async def deactivate_llm_model(
    provider: str,
    model_name: str,
    version: Optional[str] = Query(None, description="Optional model version"),
    current_user: TokenPayload = Depends(require_admin),
):
    """
    Deactivate an LLM model configuration.

    Args:
        provider: Provider name identifying the model
        model_name: Model name identifying the model
        version: Optional model version

    Returns:
        LLMModelResponse: Updated model configuration

    Raises:
        HTTPException 404: If model configuration not found
        HTTPException 500: On internal server error
    """
    logger.info(f"Deactivating LLM model: provider={provider}, model={model_name}")

    try:
        llm_service = LLMModelsService()
        deactivated_model = await llm_service.deactivate_llm_model(
            provider_name=provider,
            llm_model_name=model_name,
            llm_model_version=version,
        )

        if not deactivated_model:
            logger.warning(
                f"LLM model not found for deactivation: provider={provider}, model={model_name}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM model '{model_name}' for provider '{provider}' not found",
            )

        logger.info(
            f"LLM model deactivated successfully: provider={provider}, model={model_name}"
        )
        return LLMModelResponse(**deactivated_model)

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error deactivating LLM model: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate LLM model configuration",
        )
