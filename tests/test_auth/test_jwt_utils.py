"""
JWT Utilities Tests
------------------
Test JWT token generation, decoding, and validation functions.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from jose import JWTError

from app.auth.jwt_auth_token_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token_type,
    get_token_expiration_seconds,
    is_refresh_enabled,
)
from app.models.auth_models import AuthTokenPayload
from app.core.config_manager import settings


class TestJWTUtils:
    """Test JWT utility functions."""

    def setup_method(self):
        """Set up test data."""
        self.test_user_id = uuid4()
        self.test_role = "developer"
        self.test_admin_role = "admin"

    def test_create_access_token_success(self):
        """Test successful access token creation."""
        token = create_access_token(self.test_user_id, self.test_role)

        assert isinstance(token, str)
        assert len(token) > 0
        # Token should be a valid JWT format (3 parts separated by dots)
        assert len(token.split(".")) == 3

    def test_create_access_token_invalid_role(self):
        """Test access token creation with invalid role."""
        with pytest.raises(ValueError, match="Invalid role"):
            create_access_token(self.test_user_id, "invalid_role")

    def test_create_refresh_token_success(self):
        """Test successful refresh token creation when enabled."""
        # Temporarily enable refresh tokens for testing
        original_setting = settings.jwt_refresh_enabled
        settings.jwt_refresh_enabled = True

        try:
            token = create_refresh_token(self.test_user_id, self.test_role)

            assert isinstance(token, str)
            assert len(token) > 0
            assert len(token.split(".")) == 3
        finally:
            settings.jwt_refresh_enabled = original_setting

    def test_create_refresh_token_disabled(self):
        """Test refresh token creation when disabled."""
        original_setting = settings.jwt_refresh_enabled
        settings.jwt_refresh_enabled = False

        try:
            with pytest.raises(ValueError, match="Refresh tokens are disabled"):
                create_refresh_token(self.test_user_id, self.test_role)
        finally:
            settings.jwt_refresh_enabled = original_setting

    def test_decode_token_success(self):
        """Test successful token decoding."""
        token = create_access_token(self.test_user_id, self.test_role)
        payload = decode_token(token)

        assert isinstance(payload, AuthTokenPayload)
        assert payload.user_id == self.test_user_id
        assert payload.role == self.test_role
        assert payload.type == "access"
        assert isinstance(payload.expire_at_time, datetime)
        assert isinstance(payload.issued_at_time, datetime)

    def test_decode_token_invalid_format(self):
        """Test token decoding with invalid format."""
        with pytest.raises(JWTError):
            decode_token("invalid.token.format")

    def test_decode_token_expired(self):
        """Test token decoding with expired token."""
        # Create token with very short expiration (1 second)
        original_expire = settings.jwt_access_token_expire_hours
        settings.jwt_access_token_expire_hours = (
            0  # Set to 0 hours (will expire immediately)
        )

        try:
            token = create_access_token(self.test_user_id, self.test_role)

            # Manually modify the token to have an expired timestamp
            import base64
            import json

            # Decode the token header and payload
            header, payload, signature = token.split(".")
            decoded_payload = base64.urlsafe_b64decode(
                payload + "=" * (4 - len(payload) % 4)
            )
            payload_data = json.loads(decoded_payload)

            # Set expiration to past time (1 hour ago)
            payload_data["exp"] = int(datetime.utcnow().timestamp()) - 3600

            # Re-encode the payload
            new_payload = (
                base64.urlsafe_b64encode(json.dumps(payload_data).encode())
                .decode()
                .rstrip("=")
            )
            expired_token = f"{header}.{new_payload}.{signature}"

            with pytest.raises(JWTError):
                decode_token(expired_token)
        finally:
            settings.jwt_access_token_expire_hours = original_expire

    def test_verify_token_type_success(self):
        """Test successful token type verification."""
        token = create_access_token(self.test_user_id, self.test_role)
        payload = decode_token(token)

        # Should not raise exception
        verify_token_type(payload, "access")

    def test_verify_token_type_mismatch(self):
        """Test token type verification with type mismatch."""
        token = create_access_token(self.test_user_id, self.test_role)
        payload = decode_token(token)

        with pytest.raises(ValueError, match="Token type mismatch"):
            verify_token_type(payload, "refresh")

    def test_get_token_expiration_seconds(self):
        """Test token expiration calculation."""
        original_expire = settings.jwt_access_token_expire_hours
        settings.jwt_access_token_expire_hours = 2

        try:
            seconds = get_token_expiration_seconds()
            assert seconds == 2 * 3600  # 2 hours in seconds
        finally:
            settings.jwt_access_token_expire_hours = original_expire

    def test_is_refresh_enabled(self):
        """Test refresh token enabled check."""
        original_setting = settings.jwt_refresh_enabled

        settings.jwt_refresh_enabled = True
        assert is_refresh_enabled() is True

        settings.jwt_refresh_enabled = False
        assert is_refresh_enabled() is False

        # Restore original setting
        settings.jwt_refresh_enabled = original_setting

    def test_token_payload_structure(self):
        """Test that token payload contains all required fields."""
        token = create_access_token(self.test_user_id, self.test_role)
        payload = decode_token(token)

        # Check all required fields are present
        assert hasattr(payload, "user_id")
        assert hasattr(payload, "role")
        assert hasattr(payload, "expire_at_time")
        assert hasattr(payload, "issued_at_time")
        assert hasattr(payload, "type")

        # Check field types
        assert isinstance(payload.user_id, type(self.test_user_id))
        assert isinstance(payload.role, str)
        assert isinstance(payload.expire_at_time, datetime)
        assert isinstance(payload.issued_at_time, datetime)
        assert isinstance(payload.type, str)

    def test_token_expiration_time(self):
        """Test that token expiration is set correctly."""
        token = create_access_token(self.test_user_id, self.test_role)
        payload = decode_token(token)

        # Expiration should be in the future
        assert payload.expire_at_time > datetime.utcnow()

        # Should be approximately the configured hours from now
        expected_exp = datetime.utcnow() + timedelta(
            hours=settings.jwt_access_token_expire_hours
        )
        time_diff = abs((payload.expire_at_time - expected_exp).total_seconds())
        assert time_diff < 60  # Within 1 minute tolerance

    def test_token_issued_at_time(self):
        """Test that token issued at time is set correctly."""
        before_creation = datetime.utcnow()
        token = create_access_token(self.test_user_id, self.test_role)
        after_creation = datetime.utcnow()

        payload = decode_token(token)

        # Issued at should be between before and after creation (with some tolerance)
        # Allow for slight timing differences due to precision
        tolerance = timedelta(seconds=5)  # 5 second tolerance
        assert (
            (before_creation - tolerance)
            <= payload.issued_at_time
            <= (after_creation + tolerance)
        )
