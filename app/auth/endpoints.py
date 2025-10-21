"""
JWT Authentication Endpoints
----------------------------
FastAPI endpoints for JWT token management and testing.
Provides token generation and refresh capabilities for development and testing.

WARNING: These endpoints are for development/testing only.
In production, token generation should be handled by a separate authentication service.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger

from app.auth.models import (
    TokenGenerateRequest,
    TokenResponse,
    TokenRefreshRequest,
    TokenPayload,
)
from app.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token_type,
    get_token_expiration_seconds,
    is_refresh_enabled,
)
from app.auth.dependencies import get_current_user

# ============================================================================
# ROUTER INITIALIZATION
# ============================================================================

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ============================================================================
# TOKEN GENERATION ENDPOINTS
# ============================================================================


@router.post(
    "/token/generate",
    response_model=TokenResponse,
    summary="Generate JWT token (Development Only)",
    description="""
    Generate JWT access and refresh tokens for a user.

    ⚠️ DEVELOPMENT/TESTING ONLY ⚠️

    This endpoint is for development and testing purposes only.
    In production, token generation should be handled by a separate authentication service.

    Use this endpoint to:
    - Generate tokens for API testing
    - Create tokens for development environments
    - Test authorization flows

    Security Note: This endpoint bypasses normal authentication flows.
    """,
)
async def generate_token(request: TokenGenerateRequest):
    """
    Generate JWT tokens for a user.

    Creates both access and refresh tokens (if refresh is enabled).
    This is a development/testing endpoint and should not be used in production.

    Args:
        request: Token generation parameters (user_id, role)

    Returns:
        TokenResponse: Generated access token and optional refresh token

    Raises:
        HTTPException 400: If role is invalid or refresh tokens disabled
        HTTPException 500: If token generation fails
    """
    logger.info(
        f"Generating tokens for user {request.user_id} with role {request.role}"
    )

    try:
        # Generate access token
        access_token = create_access_token(request.user_id, request.role)

        # Generate refresh token if enabled
        refresh_token = None
        if is_refresh_enabled():
            refresh_token = create_refresh_token(request.user_id, request.role)
            logger.debug("Refresh token generated")
        else:
            logger.debug("Refresh tokens disabled in configuration")

        # Calculate expiration time
        expires_in = get_token_expiration_seconds()

        response = TokenResponse(
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
    response_model=TokenResponse,
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
async def refresh_access_token(request: TokenRefreshRequest):
    """
    Refresh access token using refresh token.

    Validates the refresh token and generates a new access token.
    Only available when jwt_refresh_enabled=True in configuration.

    Args:
        request: Refresh token request

    Returns:
        TokenResponse: New access token and optional new refresh token

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

        # Generate new access token with same user_id and role
        new_access_token = create_access_token(payload.user_id, payload.role)

        # Optionally generate new refresh token (rotate refresh token)
        new_refresh_token = None
        if is_refresh_enabled():
            new_refresh_token = create_refresh_token(payload.user_id, payload.role)
            logger.debug("New refresh token generated")

        # Calculate expiration time
        expires_in = get_token_expiration_seconds()

        response = TokenResponse(
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
    except Exception as e:
        logger.warning(f"Refresh token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


# ============================================================================
# TOKEN VALIDATION ENDPOINTS
# ============================================================================


@router.get(
    "/token/validate",
    response_model=TokenPayload,
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
    current_user: TokenPayload = Depends(get_current_user),
) -> TokenPayload:
    """
    Validate the current JWT token.

    Returns the token payload if the token is valid.
    This endpoint is protected and requires a valid JWT token.

    Args:
        current_user: Current user from JWT token (injected by dependency)

    Returns:
        TokenPayload: Current token payload with user information

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
    from app.core.config_manager import settings

    return {
        "jwt_algorithm": settings.jwt_algorithm,
        "access_token_expire_hours": settings.jwt_access_token_expire_hours,
        "refresh_enabled": settings.jwt_refresh_enabled,
        "refresh_token_expire_days": settings.jwt_refresh_token_expire_days
        if settings.jwt_refresh_enabled
        else None,
        "token_type": "bearer",
    }
