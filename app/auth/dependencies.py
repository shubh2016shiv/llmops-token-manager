"""
FastAPI Authentication Dependencies
-----------------------------------
FastAPI dependencies for JWT-based authorization and role-based access control.
Provides reusable dependencies for protecting endpoints with different permission levels.

Role Hierarchy (from request_models.py):
OWNER > ADMIN > OPERATOR > DEVELOPER

Security Best Practices:
- Stateless authorization using JWT tokens
- Hierarchical role checking (higher roles inherit lower permissions)
- Minimal database queries (only when user validation is required)
- Clear error messages for debugging without exposing sensitive information
"""

from typing import List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from loguru import logger

from app.auth.jwt_utils import decode_token, verify_token_type
from app.auth.models import AuthTokenPayload
from app.psql_db_services.users_service import UsersService
from app.models.response_models import UserResponse

# OAuth2 scheme for extracting Bearer tokens from Authorization header
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",  # Production login endpoint
    auto_error=False,  # Don't auto-raise 401, let us handle it
)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> AuthTokenPayload:
    """
    Extract and validate JWT token from Authorization header.

    Performs cryptographic validation of the JWT token without database queries.
    This is the core dependency for all authenticated endpoints.

    Args:
        token: JWT token from Authorization header (extracted by OAuth2PasswordBearer)

    Returns:
        AuthTokenPayload: Decoded and validated token payload with user_id and role

    Raises:
        HTTPException 401: If token is missing, invalid, or expired
    """
    if not token:
        logger.warning("Missing authorization token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Decode and validate token cryptographically
        payload = decode_token(token)

        # Verify this is an access token (not refresh token)
        verify_token_type(payload, "access")

        logger.debug(
            f"Token validated for user {payload.user_id} with role {payload.role}"
        )
        return payload

    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except ValueError as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_active_user(
    payload: AuthTokenPayload = Depends(get_current_user),
) -> AuthTokenPayload:
    """
    Verify user exists and is active in database.

    This dependency performs a database query to ensure the user still exists
    and is active. Only use this for endpoints that require user validation.
    For high-performance endpoints, use get_current_user instead.

    Args:
        payload: Token payload from get_current_user dependency

    Returns:
        AuthTokenPayload: Validated token payload if user is active

    Raises:
        HTTPException 403: If user not found or not active
        HTTPException 500: If database query fails
    """
    try:
        # Query database to verify user exists and is active
        users_service = UsersService()
        user: Optional[UserResponse] = await users_service.get_user_by_id(
            payload.user_id
        )

        if not user:
            logger.warning(f"User not found: {payload.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="User not found"
            )

        if user.status != "active":
            logger.warning(f"User not active: {payload.user_id}, status: {user.status}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is not active",
            )

        logger.debug(f"User {payload.user_id} verified as active")
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error during user validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User validation failed",
        )


class RoleChecker:
    """
    Dependency class for role-based authorization.

    Implements hierarchical role checking where higher roles inherit
    permissions from lower roles:
    OWNER > ADMIN > OPERATOR > DEVELOPER

    Usage:
        require_admin = RoleChecker(["admin", "owner"])
        @app.get("/admin-only", dependencies=[Depends(require_admin)])
    """

    def __init__(self, allowed_roles: List[str]):
        """
        Initialize role checker with allowed roles.

        Args:
            allowed_roles: List of roles that can access the endpoint
        """
        self.allowed_roles = allowed_roles

        # Validate roles
        valid_roles = ["developer", "operator", "admin", "owner"]
        for role in allowed_roles:
            if role not in valid_roles:
                raise ValueError(
                    f"Invalid role '{role}'. Must be one of: {', '.join(valid_roles)}"
                )

    def __call__(
        self, payload: AuthTokenPayload = Depends(get_current_user)
    ) -> AuthTokenPayload:
        """
        Check if user's role is authorized for the endpoint.

        Args:
            payload: Token payload from get_current_user dependency

        Returns:
            AuthTokenPayload: Authorized token payload

        Raises:
            HTTPException 403: If user's role is not authorized
        """
        if payload.role not in self.allowed_roles:
            logger.warning(
                f"Access denied for user {payload.user_id} with role {payload.role}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(self.allowed_roles)}",
            )

        logger.debug(
            f"Access granted for user {payload.user_id} with role {payload.role}"
        )
        return payload


# Convenience role checkers for common permission levels
# These implement the hierarchical role system

require_developer = RoleChecker(["developer", "operator", "admin", "owner"])
"""
Allow any authenticated user (developer, operator, admin, or owner).
Use for basic authenticated endpoints.
"""

require_operator = RoleChecker(["operator", "admin", "owner"])
"""
Allow operators, admins, and owners.
Use for operational endpoints like pausing deployments.
"""

require_admin = RoleChecker(["admin", "owner"])
"""
Allow admins and owners only.
Use for configuration management endpoints.
"""

require_owner = RoleChecker(["owner"])
"""
Allow owners only.
Use for system administration endpoints.
"""


# Specialized dependencies for specific use cases


async def require_active_developer(
    payload: AuthTokenPayload = Depends(require_developer),
) -> AuthTokenPayload:
    """
    Require developer role or higher AND verify user is active.

    Use for endpoints that need both role checking and user validation.
    """
    return await get_active_user(payload)


async def require_active_operator(
    payload: AuthTokenPayload = Depends(require_operator),
) -> AuthTokenPayload:
    """
    Require operator role or higher AND verify user is active.

    Use for operational endpoints that need user validation.
    """
    return await get_active_user(payload)


async def require_active_admin(
    payload: AuthTokenPayload = Depends(require_admin),
) -> AuthTokenPayload:
    """
    Require admin role or higher AND verify user is active.

    Use for admin endpoints that need user validation.
    """
    return await get_active_user(payload)


async def require_active_owner(
    payload: AuthTokenPayload = Depends(require_owner),
) -> AuthTokenPayload:
    """
    Require owner role AND verify user is active.

    Use for owner-only endpoints that need user validation.
    """
    return await get_active_user(payload)
