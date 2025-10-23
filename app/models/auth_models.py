"""
JWT Authentication Models
-------------------------
Pydantic models for JWT token operations and responses.
Defines the structure for token payloads, responses, and requests.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class AuthTokenPayload(BaseModel):
    """
    JWT token payload structure.

    Contains the essential user information embedded in JWT tokens.
    This payload is cryptographically signed and cannot be tampered with.

    Security Note: Only include non-sensitive data in JWT payloads.
    Sensitive information should be retrieved from database when needed.
    """

    user_id: UUID = Field(..., description="User's unique identifier")
    role: str = Field(
        ..., description="User role: developer, operator, admin, or owner"
    )
    exp: datetime = Field(..., description="Token expiration timestamp")
    iat: datetime = Field(..., description="Token issued at timestamp")
    type: str = Field(..., description="Token type: 'access' or 'refresh'")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "role": "developer",
                "exp": "2025-10-21T10:30:00Z",
                "iat": "2025-10-20T10:30:00Z",
                "type": "access",
            }
        }


class AuthTokenResponse(BaseModel):
    """
    Token generation response.

    Returned when generating new access tokens.
    Includes both access and optional refresh tokens.
    """

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(
        default="bearer", description="Token type (always 'bearer')"
    )
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(
        default=None, description="Refresh token (only if refresh is enabled)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            }
        }


class AuthTokenRefreshRequest(BaseModel):
    """
    Request model for refreshing access tokens.

    Used when exchanging a refresh token for a new access token.
    Only available when jwt_refresh_enabled=True in configuration.
    """

    refresh_token: str = Field(..., description="Valid refresh token")

    class Config:
        json_schema_extra = {
            "example": {"refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}
        }


class AuthTokenGenerateRequest(BaseModel):
    """
    Request model for generating tokens (testing/development only).

    TEMPORARY ENDPOINT for testing - remove in production.
    In real system, this would be called by authentication service.
    """

    user_id: UUID = Field(..., description="User ID to generate token for")
    role: str = Field(..., description="User role for the token")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "role": "developer",
            }
        }


class AuthLoginRequest(BaseModel):
    """
    Request model for user authentication login.

    Used to authenticate users with username/password credentials
    and generate JWT auth tokens upon successful authentication.
    """

    username: str = Field(..., description="User's username")
    password: str = Field(..., description="User's password")

    class Config:
        json_schema_extra = {
            "example": {"username": "johndoe", "password": "SecurePass123"}
        }
