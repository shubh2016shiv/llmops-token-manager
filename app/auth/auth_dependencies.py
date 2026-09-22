"""
FastAPI boundary for bearer authentication.

This dependency authenticates identity and token type. Route handlers own
action-specific role, ownership, and tenant authorization.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from loguru import logger

from app.auth.jwt_auth_token_service import decode_token, verify_token_type
from app.models.auth_models import AuthTokenPayload

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token/generate",
    auto_error=False,
)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> AuthTokenPayload:
    """Return a verified access-token identity or raise HTTP 401."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(token)
        verify_token_type(payload, "access")
        return payload
    except JWTError as exc:
        logger.warning("JWT validation failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except ValueError as exc:
        logger.warning("Token payload validation failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentUser = Annotated[AuthTokenPayload, Depends(get_current_user)]
