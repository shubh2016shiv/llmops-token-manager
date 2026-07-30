"""
JWT Authentication Endpoints
----------------------------
FastAPI endpoints for JWT token management and authentication.

Token manager does not own user identity (that's llm_services' concern) and
never issues tokens from credentials. It only validates JWTs issued elsewhere,
plus provides /token/generate (dev/testing only) and /token/refresh for
working with already-trusted identity (user_id, role, tenant_id).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from loguru import logger
from pydantic import BaseModel

from app.auth.auth_dependencies import CurrentUser
from app.auth.jwt_auth_token_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiration_seconds,
    is_refresh_enabled,
    verify_token_type,
)
from app.core.config import settings
from app.core.redis_rate_limiter import (
    auth_token_generate_rate_limiter,
    auth_token_refresh_rate_limiter,
)
from app.models.auth_models import (
    AuthTokenGenerateRequest,
    AuthTokenPayload,
    AuthTokenRefreshRequest,
    AuthTokenResponse,
)


class AuthConfigResponse(BaseModel):
    """Authentication configuration response (non-sensitive fields only)."""

    jwt_algorithm: str
    access_token_expire_hours: int
    refresh_enabled: bool
    refresh_token_expire_days: int | None
    token_type: str


# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================


# ============================================================================
# TOKEN GENERATION ENDPOINTS
# ============================================================================


@router.post(
    "/token/generate",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(auth_token_generate_rate_limiter())],
    summary="Generate JWT token (Development Only)",
    description="""
    Generate JWT access and refresh tokens for a user.

    ⚠️ DEVELOPMENT/TESTING ONLY ⚠️

    Requires a valid JWT access token (any authenticated caller — no
    specific role needed since authorization is handled by llm_services).

    This endpoint is for development and testing purposes only.
    In production, token generation should be handled by a separate
    authentication service.

    Use this endpoint to:
    - Generate tokens for API testing
    - Create tokens for development environments
    - Test authorization flows

    Security Note: This endpoint bypasses normal authentication flows.
    """,
)
async def generate_token(
    request: AuthTokenGenerateRequest,
    current_user: CurrentUser,
):
    """
    Generate JWT tokens for a user.

    Creates both access and refresh tokens (if refresh is enabled).
    Requires a valid JWT access token.  This is a development/testing
    endpoint — in production it returns 403.

    Args:
        request: Token generation parameters (user_id, role, tenant_id).
        current_user: Authenticated user (any role — no authorization check).

    Returns:
        AuthTokenResponse: Generated access token and optional refresh token.

    Raises:
        HTTPException 400: If role is invalid or refresh tokens disabled.
        HTTPException 403: If called in production environment.
        HTTPException 500: If token generation fails.
    """
    logger.info(
        f"Generating tokens for user {request.user_id} with role {request.role}"
    )
    logger.debug(
        f"Token generation requested by privileged user: {current_user.user_id}"
    )

    app_environment = getattr(settings, "app_environment", "development")
    if app_environment == "production":
        logger.warning(
            "Token generation endpoint is disabled in production environment"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token generation endpoint is disabled in production",
        )

    try:
        # Generate access token
        access_token = create_access_token(
            request.user_id, request.role, request.tenant_id
        )

        # Generate refresh token if enabled
        refresh_token = None
        if is_refresh_enabled():
            refresh_token = create_refresh_token(
                request.user_id, request.role, request.tenant_id
            )
            logger.debug("Refresh token generated")
        else:
            logger.debug("Refresh tokens disabled in configuration")

        # Calculate expiration time
        expires_in = get_token_expiration_seconds()

        response = AuthTokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            refresh_token=refresh_token,
        )

        logger.info(f"Tokens generated successfully for user {request.user_id}")
        return response

    except ValueError as e:
        logger.warning(f"Token generation failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Token generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate tokens",
        )


@router.post(
    "/token/refresh",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(auth_token_refresh_rate_limiter())],
    summary="Refresh access token",
    description="""
    Refresh an access token using a valid refresh token.

    Only available when refresh tokens are enabled in configuration.
    Exchanges a refresh token for a new access token.

    Use Cases:
    - Extend session without re-authentication
    - Get new access token when current one expires
    - Maintain user session across token expiration
    """,
)
async def refresh_access_token(request: AuthTokenRefreshRequest):
    """
    Refresh access token using refresh token.

    Validates the refresh token and generates a new access token.
    Only available when jwt_refresh_enabled=True in configuration.

    Args:
        request: Refresh token request

    Returns:
        AuthTokenResponse: New access token and optional new refresh token

    Raises:
        HTTPException 400: If refresh tokens disabled or invalid refresh token
        HTTPException 401: If refresh token is invalid or expired
        HTTPException 500: If token generation fails

    """
    if not is_refresh_enabled():
        logger.warning("Refresh token request received but refresh is disabled")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh tokens are disabled",
        )

    logger.info("Refreshing access token")

    try:
        # Decode and validate refresh token
        payload = decode_token(request.refresh_token)
        verify_token_type(payload, "refresh")

        # Generate new access token with same user_id, role, and tenant_id
        new_access_token = create_access_token(
            payload.user_id, payload.role, payload.tenant_id
        )

        # Optionally generate new refresh token (rotate refresh token)
        new_refresh_token = None
        if is_refresh_enabled():
            new_refresh_token = create_refresh_token(
                payload.user_id, payload.role, payload.tenant_id
            )
            logger.debug("New refresh token generated")

        # Calculate expiration time
        expires_in = get_token_expiration_seconds()

        response = AuthTokenResponse(
            access_token=new_access_token,
            token_type="bearer",
            expires_in=expires_in,
            refresh_token=new_refresh_token,
        )

        logger.info(f"Access token refreshed for user {payload.user_id}")
        return response

    except ValueError as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except JWTError as e:
        logger.warning(f"Refresh token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    except Exception as e:
        logger.error(f"Unexpected error refreshing token: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh access token",
        )


# ============================================================================
# TOKEN VALIDATION ENDPOINTS
# ============================================================================


@router.get(
    "/token/validate",
    response_model=AuthTokenPayload,
    summary="Validate current token",
    description="""
    Validate the current JWT token and return its payload.

    Use this endpoint to:
    - Check if a token is valid
    - Get current user information from token
    - Debug token issues
    - Verify token expiration
    """,
)
async def validate_token(
    current_user: CurrentUser,
) -> AuthTokenPayload:
    """
    Validate the current JWT token.

    Returns the token payload if the token is valid.
    This endpoint is protected and requires a valid JWT token.

    Args:
        current_user: Current user from JWT token (injected by dependency)

    Returns:
        AuthTokenPayload: Current token payload with user information

    Raises:
        HTTPException 401: If token is invalid or expired

    """
    logger.debug(f"Token validated for user {current_user.user_id}")
    return current_user


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================


@router.get(
    "/config",
    response_model=AuthConfigResponse,
    summary="Get authentication configuration",
    description="""
    Get current JWT authentication configuration.

    Returns configuration information about:
    - Token expiration times
    - Refresh token support
    - Available algorithms
    """,
)
async def get_auth_config():
    """
    Get authentication configuration.

    Returns current JWT configuration without sensitive information.

    Returns:
        Dictionary with authentication configuration

    """
    return AuthConfigResponse(
        jwt_algorithm=settings.jwt_algorithm,
        access_token_expire_hours=settings.jwt_access_token_expire_hours,
        refresh_enabled=settings.jwt_refresh_enabled,
        refresh_token_expire_days=settings.jwt_refresh_token_expire_days
        if settings.jwt_refresh_enabled
        else None,
        token_type="bearer",
    )
