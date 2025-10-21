"""
JWT Utilities
-------------
Core JWT operations for token generation, validation, and decoding.
Implements secure JWT handling with proper error handling and validation.

Security Best Practices:
- Use python-jose[cryptography] for cryptographic operations
- Always validate token expiration and signature
- Include token type in payload to prevent token confusion attacks
- Use cryptographic operations for all validation (no DB queries)
- Follow RFC 8725 JWT Best Current Practices
"""

from datetime import datetime, timedelta
from uuid import UUID
from jose import JWTError, jwt
from loguru import logger

from app.core.config_manager import settings
from app.auth.models import TokenPayload


def create_access_token(user_id: UUID, role: str) -> str:
    """
    Create a JWT access token for a user.

    Args:
        user_id: User's unique identifier
        role: User's role (developer, operator, admin, owner)

    Returns:
        JWT access token string

    Raises:
        ValueError: If role is invalid
        JWTError: If token creation fails
    """
    # Validate role
    valid_roles = ["developer", "operator", "admin", "owner"]
    if role not in valid_roles:
        raise ValueError(
            f"Invalid role '{role}'. Must be one of: {', '.join(valid_roles)}"
        )

    # Calculate expiration time
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_access_token_expire_hours)

    # Create token payload
    payload = {
        "user_id": str(user_id),
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    try:
        # Generate JWT token
        token: str = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

        logger.debug(f"Access token created for user {user_id} with role {role}")
        return token

    except JWTError as e:
        logger.error(f"Failed to create access token: {e}")
        raise JWTError(f"Token creation failed: {str(e)}")


def create_refresh_token(user_id: UUID, role: str) -> str:
    """
    Create a JWT refresh token for a user.

    Only available when jwt_refresh_enabled=True in configuration.

    Args:
        user_id: User's unique identifier
        role: User's role (developer, operator, admin, owner)

    Returns:
        JWT refresh token string

    Raises:
        ValueError: If refresh tokens are disabled or role is invalid
        JWTError: If token creation fails
    """
    if not settings.jwt_refresh_enabled:
        raise ValueError("Refresh tokens are disabled in configuration")

    # Validate role
    valid_roles = ["developer", "operator", "admin", "owner"]
    if role not in valid_roles:
        raise ValueError(
            f"Invalid role '{role}'. Must be one of: {', '.join(valid_roles)}"
        )

    # Calculate expiration time (longer than access token)
    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)

    # Create token payload
    payload = {
        "user_id": str(user_id),
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }

    try:
        # Generate JWT token
        token: str = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )

        logger.debug(f"Refresh token created for user {user_id} with role {role}")
        return token

    except JWTError as e:
        logger.error(f"Failed to create refresh token: {e}")
        raise JWTError(f"Refresh token creation failed: {str(e)}")


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Performs cryptographic validation of the token signature and expiration.
    No database queries are performed - all validation is cryptographic.

    Args:
        token: JWT token string

    Returns:
        TokenPayload: Decoded and validated token payload

    Raises:
        JWTError: If token is invalid, expired, or malformed
        ValueError: If token payload is invalid
    """
    try:
        # Decode and verify token
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )

        # Validate required fields
        if "user_id" not in payload:
            raise ValueError("Token missing user_id")
        if "role" not in payload:
            raise ValueError("Token missing role")
        if "exp" not in payload:
            raise ValueError("Token missing expiration")
        if "iat" not in payload:
            raise ValueError("Token missing issued at time")
        if "type" not in payload:
            raise ValueError("Token missing type")

        # Convert Unix timestamps to datetime objects
        exp_datetime = datetime.utcfromtimestamp(payload["exp"])
        iat_datetime = datetime.utcfromtimestamp(payload["iat"])

        # Convert to TokenPayload
        token_payload = TokenPayload(
            user_id=UUID(payload["user_id"]),
            role=payload["role"],
            exp=exp_datetime,
            iat=iat_datetime,
            type=payload["type"],
        )

        logger.debug(f"Token decoded successfully for user {token_payload.user_id}")
        return token_payload

    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        raise JWTError(f"Invalid token: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.warning(f"Token payload validation failed: {e}")
        raise ValueError(f"Invalid token payload: {str(e)}")


def verify_token_type(payload: TokenPayload, expected_type: str) -> None:
    """
    Verify that a token is of the expected type.

    Prevents token confusion attacks by ensuring access tokens
    are not used as refresh tokens and vice versa.

    Args:
        payload: Token payload to verify
        expected_type: Expected token type ("access" or "refresh")

    Raises:
        ValueError: If token type doesn't match expected type
    """
    if payload.type != expected_type:
        raise ValueError(
            f"Token type mismatch. Expected '{expected_type}', got '{payload.type}'"
        )

    logger.debug(f"Token type verified: {payload.type}")


def get_token_expiration_seconds() -> int:
    """
    Get the access token expiration time in seconds.

    Returns:
        Number of seconds until access token expires
    """
    return settings.jwt_access_token_expire_hours * 3600


def is_refresh_enabled() -> bool:
    """
    Check if refresh tokens are enabled.

    Returns:
        True if refresh tokens are enabled, False otherwise
    """
    return settings.jwt_refresh_enabled
