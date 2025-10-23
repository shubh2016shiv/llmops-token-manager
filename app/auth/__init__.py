"""
JWT Authentication Module
-------------------------
Enterprise-grade JWT authorization layer with Role-Based Access Control (RBAC).

This module provides:
- JWT token generation and validation
- Role-based endpoint protection
- FastAPI dependencies for authorization
- Token refresh capabilities (configurable)

Core Components:
- models: Pydantic models for JWT operations
- jwt_utils: Core JWT operations (create, decode, validate)
- dependencies: FastAPI dependencies for endpoint protection
- endpoints: Token management endpoints (development/testing)

Security Features:
- Stateless authorization using JWT tokens
- Hierarchical role checking (owner > admin > operator > developer)
- Cryptographic token validation (no database queries per request)
- Configurable token refresh support
- Protection against token confusion attacks

Usage:
    from app.auth import require_developer, get_current_user

    @app.get("/protected")
    async def protected_endpoint(user: TokenPayload = Depends(require_developer)):
        return {"user_id": user.user_id, "role": user.role}
"""

# Core dependencies for endpoint protection
from app.auth.auth_dependencies import (
    get_current_user,
    get_active_user,
    require_developer,
    require_operator,
    require_admin,
    require_owner,
    require_active_developer,
    require_active_operator,
    require_active_admin,
    require_active_owner,
    RoleChecker,
)

# JWT utilities for token operations
from app.auth.jwt_auth_token_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token_type,
    get_token_expiration_seconds,
    is_refresh_enabled,
    authenticate_user,
)

# Pydantic models for JWT operations
from app.models.auth_models import (
    AuthTokenPayload,
    AuthTokenResponse,
    AuthTokenRefreshRequest,
    AuthTokenGenerateRequest,
    AuthLoginRequest,
)

# Router for auth endpoints
from app.api.auth_endpoints import router as auth_router

__all__ = [
    # Dependencies
    "get_current_user",
    "get_active_user",
    "require_developer",
    "require_operator",
    "require_admin",
    "require_owner",
    "require_active_developer",
    "require_active_operator",
    "require_active_admin",
    "require_active_owner",
    "RoleChecker",
    # JWT Utilities
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token_type",
    "get_token_expiration_seconds",
    "is_refresh_enabled",
    "authenticate_user",
    # Models
    "AuthTokenPayload",
    "AuthTokenResponse",
    "AuthTokenRefreshRequest",
    "AuthTokenGenerateRequest",
    "AuthLoginRequest",
    # Router
    "auth_router",
]
