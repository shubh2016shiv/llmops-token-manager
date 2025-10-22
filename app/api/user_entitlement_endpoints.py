"""
User LLM Entitlements Endpoints
-------------------------------
Production-ready API endpoints for managing user entitlements to LLM models.
Provides comprehensive entitlement management with robust error handling and security.

Core Operations:
- Create entitlements (admin/owner only)
- List user entitlements
- Get specific entitlement
- Update entitlements (admin/owner only)
- Delete entitlements (admin/owner only)

Security Features:
- Role-based access control (admin/owner for modifications)
- API key encryption using bcrypt
- API keys never exposed in responses
- Comprehensive validation at all layers
"""

from uuid import UUID
from fastapi import APIRouter, HTTPException, status, Depends, Query
from loguru import logger

from app.psql_db_services.user_entitlements_service import UserEntitlementsService
from app.psql_db_services.users_service import UsersService
from app.utils.passwrd_hashing import PasswordHasher
from app.auth import require_developer, require_admin, AuthTokenPayload

from app.models.request_models import (
    UserEntitlementCreateRequest,
)
from app.models.response_models import (
    UserEntitlementResponse,
    UserEntitlementListResponse,
)

# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(
    prefix="/api/v1/users/{user_id}/entitlements", tags=["User LLM Entitlements"]
)

# ============================================================================
# CREATE ENDPOINTS
# ============================================================================


@router.post(
    "/",
    response_model=UserEntitlementResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user LLM entitlement",
    description="Create a new LLM entitlement for a user. Only admin and owner roles can create entitlements.",
)
async def create_user_entitlement(
    user_id: UUID,
    request: UserEntitlementCreateRequest,
    current_user: AuthTokenPayload = Depends(require_admin),
):
    """
    Create a new LLM entitlement for a user (admin/owner only).

    Comprehensive validation ensures data integrity:
    1. Current user must have admin or owner role
    2. Target user must exist in the system
    3. Provider/model must exist in llm_models table
    4. No duplicate entitlements allowed
    5. API key encrypted using bcrypt before storage

    Process:
    1. Validate request data (Pydantic)
    2. Check user role (admin/owner only)
    3. Use path parameter user_id as the target user
    4. Encrypt API key using PasswordHasher
    5. Call service to create entitlement with all validations
    6. Return created entitlement (API key excluded for security)

    Args:
        user_id: User ID from path parameter (the target user to receive the entitlement)
        request: Entitlement creation parameters
        current_user: Current authenticated user (from JWT)

    Returns:
        UserEntitlementResponse: Created entitlement details (API key excluded)

    Raises:
        HTTPException 400: If validation fails or duplicate exists
        HTTPException 403: If user is not admin/owner
        HTTPException 404: If user or provider/model not found
        HTTPException 500: On internal server error
    """
    logger.info(
        f"Creating entitlement: target_user={user_id}, provider={request.llm_provider.value}, model={request.llm_model_name}"
    )

    # Validate current user has admin or owner role
    if current_user.role not in ["admin", "owner"]:
        logger.warning(
            f"User {current_user.user_id} with role {current_user.role} attempted to create entitlement"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin and owner roles can create entitlements",
        )

    try:
        # Encrypt API key using bcrypt
        encrypted_api_key = PasswordHasher.hash_password(request.api_key)

        # Create entitlement service
        entitlements_service = UserEntitlementsService()

        # Create entitlement (service handles all validations)
        entitlement = await entitlements_service.create_entitlement(
            user_id=user_id,  # Use the path parameter as the target user ID
            llm_provider=request.llm_provider.value,
            llm_model_name=request.llm_model_name,
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=current_user.user_id,
            api_endpoint_url=request.api_endpoint_url,
            cloud_provider=request.cloud_provider,
            deployment_name=request.deployment_name,
            region=request.region,
        )

        logger.info(
            f"Entitlement created successfully: entitlement_id={entitlement['entitlement_id']}"
        )
        return UserEntitlementResponse(**entitlement)

    except ValueError as e:
        # Handle validation errors from service layer
        error_msg = str(e)
        logger.warning(f"Validation error creating entitlement: {error_msg}")

        # Enhance error message for missing provider/model with specific guidance
        if "does not exist in llm_models table" in error_msg:
            enhanced_error = (
                f"{error_msg} "
                f"Please create the LLM model configuration first using: "
                f"POST /api/v1/llm-models/ "
                f"Required fields: llm_provider, llm_model_name, api_key_variable_name, "
                f"max_tokens, tokens_per_minute_limit, requests_per_minute_limit. "
                f"Admin role required for model creation."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=enhanced_error
            )

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    except HTTPException:
        raise

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error creating entitlement: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create entitlement. Please try again later.",
        )


# ============================================================================
# READ ENDPOINTS
# ============================================================================


@router.get(
    "/",
    response_model=UserEntitlementListResponse,
    summary="List user entitlements",
    description="Retrieve all LLM entitlements for a specific user with pagination.",
)
async def list_user_entitlements(
    user_id: UUID,
    current_user: AuthTokenPayload = Depends(require_developer),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
):
    """
    List all LLM entitlements for a user with pagination.

    Args:
        user_id: User ID from path parameter (the target user whose entitlements to retrieve)
        current_user: Current authenticated user (from JWT)
        page: Page number (starts at 1)
        page_size: Number of items per page (1-100)

    Returns:
        UserEntitlementListResponse: Paginated list of entitlements

    Raises:
        HTTPException 404: If user not found
        HTTPException 500: On internal server error
    """
    logger.debug(
        f"Listing entitlements for user {user_id}: page={page}, size={page_size}"
    )

    try:
        # Verify user exists
        users_service = UsersService()
        user = await users_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found",
            )

        # Calculate offset
        offset = (page - 1) * page_size

        # Get entitlements
        entitlements_service = UserEntitlementsService()
        entitlements = await entitlements_service.get_user_entitlements(
            user_id=user_id, limit=page_size, offset=offset
        )

        # Get total count
        total_count = await entitlements_service.count_user_entitlements(user_id)

        return UserEntitlementListResponse(
            entitlements=[UserEntitlementResponse(**ent) for ent in entitlements],
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Error listing entitlements for user {user_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entitlements. Please try again later.",
        )


# @router.get(
#     "/{entitlement_id}",
#     response_model=UserEntitlementResponse,
#     summary="Get specific entitlement",
#     description="Retrieve details of a specific LLM entitlement.",
# )
# async def get_entitlement(
#     user_id: UUID,
#     entitlement_id: int,
#     current_user: AuthTokenPayload = Depends(require_developer),
# ):
#     """
#     Get details of a specific LLM entitlement.
#
#     Args:
#         user_id: User ID from path parameter (the target user whose entitlement to retrieve)
#         entitlement_id: Unique entitlement identifier
#         current_user: Current authenticated user (from JWT)
#
#     Returns:
#         UserEntitlementResponse: Entitlement details
#
#     Raises:
#         HTTPException 404: If entitlement not found
#         HTTPException 500: On internal server error
#     """
#     logger.debug(f"Fetching entitlement: user={user_id}, entitlement_id={entitlement_id}")
#
#     try:
#         entitlements_service = UserEntitlementsService()
#         entitlement = await entitlements_service.get_entitlement_by_id(entitlement_id)
#
#         if not entitlement:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Entitlement with ID '{entitlement_id}' not found",
#             )
#
#         # Verify entitlement belongs to specified user
#         if entitlement["user_id"] != user_id:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Entitlement '{entitlement_id}' not found for user '{user_id}'",
#             )
#
#         return UserEntitlementResponse(**entitlement)
#
#     except HTTPException:
#         raise
#
#     except Exception as e:
#         logger.error(f"Error fetching entitlement {entitlement_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to retrieve entitlement. Please try again later.",
#         )


# ============================================================================
# UPDATE ENDPOINTS
# ============================================================================


# @router.patch(
#     "/{entitlement_id}",
#     response_model=UserEntitlementResponse,
#     summary="Update entitlement",
#     description="Update an existing LLM entitlement. Only admin and owner roles can update.",
# )
# async def update_entitlement(
#     user_id: UUID,
#     entitlement_id: int,
#     request: UserEntitlementUpdateRequest,
#     current_user: AuthTokenPayload = Depends(require_admin),
# ):
#     """
#     Update an existing LLM entitlement (admin/owner only).
#
#     Only provided fields will be updated. API key is encrypted if provided.
#
#     Process:
#     1. Validate user role (admin/owner only)
#     2. Verify entitlement exists and belongs to user
#     3. Encrypt new API key if provided
#     4. Update entitlement with new values
#     5. Return updated entitlement (API key excluded)
#
#     Args:
#         user_id: User ID from path parameter (the target user whose entitlement to update)
#         entitlement_id: Unique entitlement identifier
#         request: Fields to update
#         current_user: Current authenticated user (from JWT)
#
#     Returns:
#         UserEntitlementResponse: Updated entitlement details
#
#     Raises:
#         HTTPException 403: If user is not admin/owner
#         HTTPException 404: If entitlement not found
#         HTTPException 500: On internal server error
#     """
#     logger.info(f"Updating entitlement: user={user_id}, entitlement_id={entitlement_id}")
#
#     # Validate current user has admin or owner role
#     if current_user.role not in ["admin", "owner"]:
#         logger.warning(
#             f"User {current_user.user_id} with role {current_user.role} attempted to update entitlement"
#         )
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Only admin and owner roles can update entitlements",
#         )
#
#     try:
#         entitlements_service = UserEntitlementsService()
#
#         # Verify entitlement exists and belongs to user
#         existing_entitlement = await entitlements_service.get_entitlement_by_id(
#             entitlement_id
#         )
#         if not existing_entitlement:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Entitlement with ID '{entitlement_id}' not found",
#             )
#
#         if existing_entitlement["user_id"] != user_id:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Entitlement '{entitlement_id}' not found for user '{user_id}'",
#             )
#
#         # Encrypt API key if provided
#         encrypted_api_key = None
#         if request.api_key:
#             encrypted_api_key = PasswordHasher.hash_password(request.api_key)
#
#         # Update entitlement
#         updated_entitlement = await entitlements_service.update_entitlement(
#             entitlement_id=entitlement_id,
#             encrypted_api_key=encrypted_api_key,
#             api_endpoint_url=request.api_endpoint_url,
#             cloud_provider=request.cloud_provider,
#             deployment_name=request.deployment_name,
#             region=request.region,
#         )
#
#         if not updated_entitlement:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Entitlement with ID '{entitlement_id}' not found",
#             )
#
#         logger.info(f"Entitlement updated successfully: entitlement_id={entitlement_id}")
#         return UserEntitlementResponse(**updated_entitlement)
#
#     except HTTPException:
#         raise
#
#     except Exception as e:
#         logger.error(f"Error updating entitlement {entitlement_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to update entitlement. Please try again later.",
#         )


# ============================================================================
# DELETE ENDPOINTS
# ============================================================================


@router.delete(
    "/{entitlement_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete entitlement",
    description="Delete an LLM entitlement. Only admin and owner roles can delete.",
)
async def delete_entitlement(
    user_id: UUID,
    entitlement_id: int,
    current_user: AuthTokenPayload = Depends(require_admin),
):
    """
    Delete an LLM entitlement (admin/owner only).

    Process:
    1. Validate user role (admin/owner only)
    2. Verify entitlement exists and belongs to user
    3. Get user details for response
    4. Delete entitlement
    5. Return detailed response with user info and deletion status

    Args:
        user_id: User ID from path parameter (the target user whose entitlement to delete)
        entitlement_id: Unique entitlement identifier
        current_user: Current authenticated user (from JWT)

    Returns:
        Dict containing user details, entitlement info, and deletion status

    Raises:
        HTTPException 403: If user is not admin/owner
        HTTPException 404: If entitlement not found
        HTTPException 500: On internal server error
    """
    logger.info(
        f"Deleting entitlement: user={user_id}, entitlement_id={entitlement_id}"
    )

    # Validate current user has admin or owner role
    if current_user.role not in ["admin", "owner"]:
        logger.warning(
            f"User {current_user.user_id} with role {current_user.role} attempted to delete entitlement"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin and owner roles can delete entitlements",
        )

    try:
        entitlements_service = UserEntitlementsService()
        users_service = UsersService()

        # Verify entitlement exists and belongs to user
        existing_entitlement = await entitlements_service.get_entitlement_by_id(
            entitlement_id
        )
        if not existing_entitlement:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entitlement with ID '{entitlement_id}' not found",
            )

        if existing_entitlement["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entitlement '{entitlement_id}' not found for user '{user_id}'",
            )

        # Get user details for response
        user_details = await users_service.get_user_by_id(user_id)
        if not user_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found",
            )

        # Delete entitlement
        was_deleted = await entitlements_service.delete_entitlement(entitlement_id)

        if not was_deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entitlement with ID '{entitlement_id}' not found",
            )

        # Prepare response with user details and deletion status
        response_data = {
            "deletion_status": "success",
            "entitlement_id": entitlement_id,
            "user_details": {
                "user_id": str(user_id),
                "username": user_details.username if user_details else "N/A",
                "email": user_details.email if user_details else "N/A",
            },
            "entitlement_details": {
                "llm_provider": existing_entitlement.get("llm_provider", "N/A"),
                "llm_model_name": existing_entitlement.get("llm_model_name", "N/A"),
                "cloud_provider": existing_entitlement.get("cloud_provider"),
                "api_endpoint_url": existing_entitlement.get("api_endpoint_url"),
            },
            "deleted_by": {
                "admin_user_id": str(current_user.user_id),
                "admin_username": current_user.username
                if hasattr(current_user, "username")
                else "N/A",
            },
            "message": f"Entitlement for {user_details.username if user_details else 'user'} ({user_details.email if user_details else 'N/A'}) has been successfully deleted",
        }

        logger.info(
            f"Entitlement deleted successfully: entitlement_id={entitlement_id}"
        )
        return response_data

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error deleting entitlement {entitlement_id}: {e}", exc_info=True)

        # Try to get user details for error response
        try:
            users_service = UsersService()
            user_details = await users_service.get_user_by_id(user_id)
            username = user_details.username if user_details else "N/A"
            email = user_details.email if user_details else "N/A"
        except:
            username = "N/A"
            email = "N/A"

        error_response = {
            "deletion_status": "failure",
            "entitlement_id": entitlement_id,
            "user_details": {
                "user_id": str(user_id),
                "username": username,
                "email": email,
            },
            "error": "Failed to delete entitlement due to internal server error",
            "message": f"Failed to delete entitlement for {username} ({email})",
        }

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response,
        )
