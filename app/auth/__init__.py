"""
JWT Authentication Module — token validation, no authorization.

This module answers one question: "Is the Bearer token real?"

It validates JWT access tokens cryptographically (signature + expiration)
and injects the decoded payload into protected route handlers.  No role
checking — the upstream llm_services microservice owns authorization.

----
Reading order (each file builds on the previous one)
-----------------------------------------------------
    1. jwt_auth_token_service.py
       Core JWT operations: creating tokens (dev only), decoding them,
       and verifying token types to prevent confusion attacks.

    2. auth_dependencies.py
       The FastAPI dependencies that wire token validation into your
       routes — one import, one Depends(), zero boilerplate per endpoint.

----
How to protect an endpoint (one line)
--------------------------------------
    from app.auth import CurrentUser

    @router.get("/protected")
    async def my_endpoint(current_user: CurrentUser):
        return {"user_id": str(current_user.user_id)}

----
Enterprise architecture note
-----------------------------
This service is a RESOURCE SERVER, not an AUTHENTICATION SERVER.  It
validates JWTs issued by llm_services but never issues them from user
credentials.  The /token/generate endpoint exists for development only
and is disabled in production (returns 403).
"""

# ---- The one dependency every protected endpoint uses ----
from app.auth.auth_dependencies import (
    CurrentUser,
    get_current_user,
    oauth2_scheme,
)

# ---- JWT utilities (validation needed in production; creation dev-only) ----
from app.auth.jwt_auth_token_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiration_seconds,
    is_refresh_enabled,
    verify_token_type,
)

# ---- Pydantic models for JWT operations ----
from app.models.auth_models import (
    AuthTokenGenerateRequest,
    AuthTokenPayload,
    AuthTokenRefreshRequest,
    AuthTokenResponse,
)

__all__ = [
    # Authentication dependency
    "CurrentUser",
    "get_current_user",
    "oauth2_scheme",
    # JWT utilities
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token_type",
    "get_token_expiration_seconds",
    "is_refresh_enabled",
    # Models
    "AuthTokenPayload",
    "AuthTokenResponse",
    "AuthTokenRefreshRequest",
    "AuthTokenGenerateRequest",
]
