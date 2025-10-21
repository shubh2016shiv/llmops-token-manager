"""
Auth Dependencies Tests
----------------------
Test FastAPI dependencies for JWT authentication and role-based authorization.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from uuid import uuid4

from app.auth.dependencies import (
    get_current_user,
    get_active_user,
    RoleChecker,
    require_developer,
    require_operator,
    require_admin,
    require_owner,
)
from app.auth.models import AuthTokenPayload
from app.models.response_models import UserResponse


class TestAuthDependencies:
    """Test authentication dependencies."""

    def setup_method(self):
        """Set up test data."""
        self.test_user_id = uuid4()
        self.test_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="developer",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

    @pytest.mark.asyncio
    async def test_get_current_user_success(self):
        """Test successful user extraction from valid token."""
        with patch("app.auth.dependencies.decode_token") as mock_decode:
            mock_decode.return_value = self.test_payload

            result = await get_current_user("valid_token")

            assert result == self.test_payload
            mock_decode.assert_called_once_with("valid_token")

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self):
        """Test user extraction with no token."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)

        assert exc_info.value.status_code == 401
        assert "Authorization token required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test user extraction with invalid token."""
        with patch("app.auth.dependencies.decode_token") as mock_decode:
            from jose import JWTError

            mock_decode.side_effect = JWTError("Invalid token")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("invalid_token")

            assert exc_info.value.status_code == 401
            assert "Invalid or expired token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_wrong_token_type(self):
        """Test user extraction with refresh token instead of access token."""
        refresh_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="developer",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="refresh",
        )

        with patch("app.auth.dependencies.decode_token") as mock_decode:
            mock_decode.return_value = refresh_payload

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("refresh_token")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_active_user_success(self):
        """Test successful active user validation."""
        active_user = UserResponse(
            user_id=self.test_user_id,
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="active",
            created_at=None,
            updated_at=None,
        )

        with patch("app.auth.dependencies.UsersService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.get_user_by_id.return_value = active_user
            mock_service_class.return_value = mock_service

            result = await get_active_user(self.test_payload)

            assert result == self.test_payload
            mock_service.get_user_by_id.assert_called_once_with(self.test_user_id)

    @pytest.mark.asyncio
    async def test_get_active_user_not_found(self):
        """Test active user validation with user not found."""
        with patch("app.auth.dependencies.UsersService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.get_user_by_id.return_value = None
            mock_service_class.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await get_active_user(self.test_payload)

            assert exc_info.value.status_code == 403
            assert "User not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_active_user_inactive(self):
        """Test active user validation with inactive user."""
        inactive_user = UserResponse(
            user_id=self.test_user_id,
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="suspended",
            created_at=None,
            updated_at=None,
        )

        with patch("app.auth.dependencies.UsersService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.get_user_by_id.return_value = inactive_user
            mock_service_class.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await get_active_user(self.test_payload)

            assert exc_info.value.status_code == 403
            assert "User account is not active" in exc_info.value.detail

    def test_role_checker_success(self):
        """Test successful role checking."""
        checker = RoleChecker(["developer", "admin"])

        # Should not raise exception
        result = checker(self.test_payload)
        assert result == self.test_payload

    def test_role_checker_insufficient_permissions(self):
        """Test role checking with insufficient permissions."""
        checker = RoleChecker(["admin", "owner"])

        with pytest.raises(HTTPException) as exc_info:
            checker(self.test_payload)

        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in exc_info.value.detail

    def test_role_checker_invalid_roles(self):
        """Test role checker initialization with invalid roles."""
        with pytest.raises(ValueError, match="Invalid role"):
            RoleChecker(["invalid_role"])

    def test_require_developer_success(self):
        """Test require_developer with developer role."""
        result = require_developer(self.test_payload)
        assert result == self.test_payload

    def test_require_developer_admin_success(self):
        """Test require_developer with admin role (should pass)."""
        admin_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="admin",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        result = require_developer(admin_payload)
        assert result == admin_payload

    def test_require_operator_success(self):
        """Test require_operator with operator role."""
        operator_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="operator",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        result = require_operator(operator_payload)
        assert result == operator_payload

    def test_require_operator_developer_fails(self):
        """Test require_operator with developer role (should fail)."""
        with pytest.raises(HTTPException) as exc_info:
            require_operator(self.test_payload)

        assert exc_info.value.status_code == 403

    def test_require_admin_success(self):
        """Test require_admin with admin role."""
        admin_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="admin",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        result = require_admin(admin_payload)
        assert result == admin_payload

    def test_require_admin_developer_fails(self):
        """Test require_admin with developer role (should fail)."""
        with pytest.raises(HTTPException) as exc_info:
            require_admin(self.test_payload)

        assert exc_info.value.status_code == 403

    def test_require_owner_success(self):
        """Test require_owner with owner role."""
        owner_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="owner",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        result = require_owner(owner_payload)
        assert result == owner_payload

    def test_require_owner_admin_fails(self):
        """Test require_owner with admin role (should fail)."""
        admin_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="admin",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        with pytest.raises(HTTPException) as exc_info:
            require_owner(admin_payload)

        assert exc_info.value.status_code == 403

    def test_role_hierarchy(self):
        """Test that role hierarchy works correctly."""
        # Owner should have access to all roles
        owner_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="owner",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        assert require_developer(owner_payload) == owner_payload
        assert require_operator(owner_payload) == owner_payload
        assert require_admin(owner_payload) == owner_payload
        assert require_owner(owner_payload) == owner_payload

        # Admin should have access to admin and below
        admin_payload = AuthTokenPayload(
            user_id=self.test_user_id,
            role="admin",
            exp="2025-12-31T23:59:59Z",
            iat="2025-01-01T00:00:00Z",
            type="access",
        )

        assert require_developer(admin_payload) == admin_payload
        assert require_operator(admin_payload) == admin_payload
        assert require_admin(admin_payload) == admin_payload

        with pytest.raises(HTTPException):
            require_owner(admin_payload)
