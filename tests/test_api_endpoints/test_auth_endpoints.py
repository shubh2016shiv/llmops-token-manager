"""Unit tests for auth endpoints."""

import asyncio
from datetime import datetime, timedelta, timezone
import os
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import JWTError
import pytest

os.environ.setdefault("RATE_LIMIT_STORAGE", "memory")

from app.api.auth_endpoints import router
from app.models.auth_models import AuthTokenPayload


@pytest.fixture
def app():
    """Create a FastAPI app with auth routes."""
    from app.core.redis_rate_limiter import rate_limiter_manager

    rate_limiter_manager.initialize()
    app = FastAPI()
    app.include_router(router)
    yield app
    asyncio.run(rate_limiter_manager.close())


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestGenerateToken:
    """A development helper must not mint arbitrary identities for developers."""

    @patch("app.api.auth_endpoints.settings.app_environment", "development")
    def test_developer_cannot_mint_admin_token(self, client, app, mock_developer_user):
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user
        response = client.post(
            "/api/v1/auth/token/generate",
            json={
                "user_id": str(mock_developer_user.user_id),
                "tenant_id": str(mock_developer_user.tenant_id),
                "role": "admin",
            },
        )

        assert response.status_code == 403


class TestRefreshAccessToken:
    """Refresh token endpoint tests."""

    @patch("app.api.auth_endpoints.is_refresh_enabled", return_value=True)
    @patch("app.api.auth_endpoints.decode_token")
    def test_refresh_access_token_invalid_jwt_returns_401(
        self,
        mock_decode_token,
        _mock_is_refresh_enabled,
        client,
    ):
        """JWT decoding failures should map to 401."""
        mock_decode_token.side_effect = JWTError("invalid token")

        response = client.post(
            "/api/v1/auth/token/refresh",
            json={"refresh_token": "bad_refresh_token"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired refresh token"

    @patch("app.api.auth_endpoints.is_refresh_enabled", return_value=True)
    @patch("app.api.auth_endpoints.decode_token")
    @patch("app.api.auth_endpoints.verify_token_type")
    def test_refresh_access_token_bad_token_type_returns_400(
        self,
        mock_verify_token_type,
        mock_decode_token,
        _mock_is_refresh_enabled,
        client,
    ):
        """Token type/payload validation errors should map to 400."""
        mock_decode_token.return_value = AuthTokenPayload(
            user_id=uuid4(),
            tenant_id=uuid4(),
            role="developer",
            expire_at_time=datetime.now(timezone.utc) + timedelta(hours=1),
            issued_at_time=datetime.now(timezone.utc),
            type="refresh",
        )
        mock_verify_token_type.side_effect = ValueError("Token type mismatch")

        response = client.post(
            "/api/v1/auth/token/refresh",
            json={"refresh_token": "refresh_token"},
        )

        assert response.status_code == 400
        assert "Token type mismatch" in response.json()["detail"]

    @patch("app.api.auth_endpoints.is_refresh_enabled", return_value=True)
    @patch("app.api.auth_endpoints.decode_token")
    @patch("app.api.auth_endpoints.verify_token_type")
    @patch("app.api.auth_endpoints.create_access_token")
    def test_refresh_access_token_unexpected_error_returns_500(
        self,
        mock_create_access_token,
        mock_verify_token_type,
        mock_decode_token,
        _mock_is_refresh_enabled,
        client,
    ):
        """Unexpected failures should map to 500."""
        mock_decode_token.return_value = AuthTokenPayload(
            user_id=uuid4(),
            tenant_id=uuid4(),
            role="developer",
            expire_at_time=datetime.now(timezone.utc) + timedelta(hours=1),
            issued_at_time=datetime.now(timezone.utc),
            type="refresh",
        )
        mock_verify_token_type.return_value = None
        mock_create_access_token.side_effect = RuntimeError("db transient failure")

        response = client.post(
            "/api/v1/auth/token/refresh",
            json={"refresh_token": "refresh_token"},
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to refresh access token"
