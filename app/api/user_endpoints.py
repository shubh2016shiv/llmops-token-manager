"""
User Management Endpoints
-------------------------
Production-ready API endpoints for user CRUD operations.
Provides essential user management functionality with robust error handling.
"""

from uuid import uuid4, UUID
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger

from app.utils.passwrd_hashing import PasswordHasher
from app.psql_db_services.users_service import UsersService
from app.auth import require_developer, require_admin, TokenPayload

from app.models.request_models import UserCreateRequest, UserUpdateRequest
from app.models.response_models import UserResponse


# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


# ============================================================================
# CREATE ENDPOINTS
# ============================================================================


# ============================================================================
#                           CREATE USER ENDPOINT
# ============================================================================
@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with username, email, and password.",
)
async def create_user(request: UserCreateRequest):
    """
    Create a new user in the system.

    Process:
    1. Validate input (handled by Pydantic)
    2. Generate unique user_id
    3. Check username and email uniqueness
    4. Hash password
    5. Set defaults (role=developer, status=active)
    6. Insert into database
    7. Return user data (without password)

    Args:
        request: User creation parameters (first_name, last_name, username, email, password)

    Returns:
        UserResponse: Created user information

    Raises:
        HTTPException 400: If email/username exists or validation fails
        HTTPException 500: On internal server error
    """
    logger.info(f"Creating user: username={request.username}, email={request.email}")

    try:
        # Generate unique user ID
        user_id = uuid4()

        # Hash password
        password_hash = PasswordHasher.hash_password(request.password)

        # Get current timestamp
        now = datetime.utcnow()

        # Create user in database
        users_service = UsersService()
        user = await users_service.create_user(
            user_id=user_id,
            username=request.username,
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name,
            password_hash=password_hash,
            user_role="developer",  # Default role
            user_status="active",  # Default status
            created_at=now,
            updated_at=now,
        )

        logger.info(f"User created successfully: user_id={user['user_id']}")
        return UserResponse(**user)

    except ValueError as e:
        # Handle uniqueness violations and validation errors
        error_msg = str(e)
        logger.warning(f"Validation error: {error_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error creating user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user. Please try again later.",
        )


# ============================================================================
# READ ENDPOINTS
# ============================================================================


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Retrieve a specific user by their unique identifier.",
)
async def get_user(
    user_id: UUID, current_user: TokenPayload = Depends(require_developer)
):
    """
    Retrieve a user by their ID.

    Args:
        user_id: User's unique identifier

    Returns:
        UserResponse: User information

    Raises:
        HTTPException 404: If user not found
        HTTPException 500: On internal server error
    """
    logger.debug(f"Fetching user: user_id={user_id}")

    try:
        users_service = UsersService()
        user = await users_service.get_user_by_id(user_id)

        if not user:
            logger.warning(f"User not found: user_id={user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found",
            )

        return user

    except HTTPException:
        raise
    except ValueError as validation_error:
        logger.warning(f"Invalid user ID: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
        )
    except Exception as error:
        logger.error(f"Error fetching user: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user",
        )


@router.get(
    "/email/{email}",
    response_model=UserResponse,
    summary="Get user by email",
    description="Retrieve a specific user by their email address.",
)
async def get_user_by_email(
    email: str, current_user: TokenPayload = Depends(require_developer)
):
    """
    Retrieve a user by their email address.

    Args:
        email: User's email address

    Returns:
        UserResponse: User information

    Raises:
        HTTPException 404: If user not found
        HTTPException 500: On internal server error
    """
    logger.debug(f"Fetching user by email: email={email}")

    try:
        users_service = UsersService()
        user = await users_service.get_user_by_email(email)

        if not user:
            logger.warning(f"User not found: email={email}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with email '{email}' not found",
            )

        return UserResponse(**user)

    except HTTPException:
        raise
    except ValueError as validation_error:
        logger.warning(f"Invalid email: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
        )
    except Exception as error:
        logger.error(f"Error fetching user by email: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user",
        )


# @router.get(
#     "/",
#     response_model=UserListResponse,
#     summary="List all users",
#     description="Retrieve a paginated list of users with optional filtering by role and status.",
# )
# async def list_users(
#     role: Optional[str] = Query(
#         None, description="Filter by role: owner, admin, developer, or viewer"
#     ),
#     status: Optional[str] = Query(
#         None, description="Filter by status: active, suspended, or inactive"
#     ),
#     limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
#     offset: int = Query(0, ge=0, description="Number of results to skip"),
# ):
#     """
#     List users with optional filtering and pagination.

#     Args:
#         role: Optional role filter
#         status: Optional status filter
#         limit: Maximum results per page (1-1000)
#         offset: Pagination offset

#     Returns:
#         UserListResponse: Paginated list of users

#     Raises:
#         HTTPException 400: On invalid parameters
#         HTTPException 500: On internal server error
#     """
#     logger.debug(
#         f"Listing users: role={role}, status={status}, limit={limit}, offset={offset}"
#     )

#     try:
#         users_service = UsersService()
#         users = await users_service.get_all_users(
#             role_filter=role, status_filter=status, limit=limit, offset=offset
#         )

#         return UserListResponse(
#             users=[UserResponse(**user) for user in users],
#             total=len(users),
#             limit=limit,
#             offset=offset,
#         )

#     except ValueError as validation_error:
#         logger.warning(f"Validation error listing users: {validation_error}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
#         )
#     except Exception as error:
#         logger.error(f"Error listing users: {error}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to retrieve users",
#         )


# ============================================================================
# UPDATE ENDPOINTS
# ============================================================================


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update user information. Only provided fields will be updated.",
)
async def update_user(
    user_id: UUID,
    request: UserUpdateRequest,
    current_user: TokenPayload = Depends(require_admin),
):
    """
    Update user information.

    Args:
        user_id: User's unique identifier
        request: Fields to update

    Returns:
        UserResponse: Updated user information

    Raises:
        HTTPException 404: If user not found
        HTTPException 400: On invalid parameters or duplicate email
        HTTPException 500: On internal server error
    """
    logger.info(f"Updating user: user_id={user_id}")

    try:
        users_service = UsersService()
        updated_user = await users_service.update_user(
            user_id=user_id,
            email_address=request.email,
            user_role=request.role,
            user_status=request.status,
        )

        if not updated_user:
            logger.warning(f"User not found for update: user_id={user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found",
            )

        logger.info(f"User updated successfully: user_id={user_id}")
        return updated_user

    except HTTPException:
        raise
    except ValueError as validation_error:
        logger.warning(f"Validation error updating user: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
        )
    except Exception as error:
        # Check if it's a duplicate email error
        error_str = str(error).lower()
        if "unique" in error_str or "duplicate" in error_str:
            logger.warning(f"Duplicate email in update: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{request.email}' is already in use",
            )

        logger.error(f"Error updating user: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user",
        )


@router.patch(
    "/{user_id}/suspend",
    response_model=UserResponse,
    summary="Suspend user",
    description="Suspend a user account by setting status to 'suspended'.",
)
async def suspend_user(
    user_id: UUID, current_user: TokenPayload = Depends(require_admin)
):
    """
    Suspend a user account.

    Args:
        user_id: User's unique identifier

    Returns:
        UserResponse: Updated user information

    Raises:
        HTTPException 404: If user not found
        HTTPException 500: On internal server error
    """
    logger.info(f"Suspending user: user_id={user_id}")

    try:
        users_service = UsersService()
        suspended_user = await users_service.suspend_user(user_id)

        if not suspended_user:
            logger.warning(f"User not found for suspension: user_id={user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found",
            )

        logger.info(f"User suspended successfully: user_id={user_id}")
        return suspended_user

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error suspending user: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to suspend user",
        )


@router.patch(
    "/{user_id}/activate",
    response_model=UserResponse,
    summary="Activate user",
    description="Activate a user account by setting status to 'active'.",
)
async def activate_user(
    user_id: UUID, current_user: TokenPayload = Depends(require_admin)
):
    """
    Activate a user account.

    Args:
        user_id: User's unique identifier

    Returns:
        UserResponse: Updated user information

    Raises:
        HTTPException 404: If user not found
        HTTPException 500: On internal server error
    """
    logger.info(f"Activating user: user_id={user_id}")

    try:
        users_service = UsersService()
        activated_user = await users_service.activate_user(user_id)

        if not activated_user:
            logger.warning(f"User not found for activation: user_id={user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found",
            )

        logger.info(f"User activated successfully: user_id={user_id}")
        return activated_user

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error activating user: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate user",
        )


# ============================================================================
# DELETE ENDPOINTS
# ============================================================================


# @router.delete(
#     "/{user_id}",
#     response_model=MessageResponse,
#     summary="Delete user",
#     description="Permanently delete a user account from the system.",
# )
# async def delete_user(user_id: uuid4):
#     """
#     Delete a user account.

#     Args:
#         user_id: User's unique identifier

#     Returns:
#         MessageResponse: Deletion status message

#     Raises:
#         HTTPException 404: If user not found
#         HTTPException 500: On internal server error
#     """
#     logger.info(f"Deleting user: user_id={user_id}")

#     try:
#         users_service = UsersService()
#         was_deleted = await users_service.delete_user(user_id)

#         if not was_deleted:
#             logger.warning(f"User not found for deletion: user_id={user_id}")
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"User with ID '{user_id}' not found",
#             )

#         logger.info(f"User deleted successfully: user_id={user_id}")
#         return MessageResponse(message="User deleted successfully", success=True)

#     except HTTPException:
#         raise
#     except ValueError as validation_error:
#         logger.warning(f"Invalid user ID for deletion: {validation_error}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, detail=str(validation_error)
#         )
#     except Exception as error:
#         logger.error(f"Error deleting user: {error}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to delete user",
#         )
