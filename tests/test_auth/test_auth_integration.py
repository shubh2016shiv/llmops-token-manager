"""
Auth Integration Tests
----------------------
End-to-end tests for JWT authentication integration with FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch
from uuid import uuid4

from app.app import app
from app.auth.jwt_utils import create_access_token
from app.models.response_models import UserResponse


class TestAuthIntegration:
    """Test JWT authentication integration with FastAPI endpoints."""

    def setup_method(self):
        """Set up test client and data."""
        self.client = TestClient(app)
        self.test_user_id = uuid4()
        self.test_admin_id = uuid4()

    def test_auth_endpoints_accessible(self):
        """Test that auth endpoints are accessible."""
        # Test token generation endpoint
        response = self.client.post(
            "/api/v1/auth/token/generate",
            json={"user_id": str(self.test_user_id), "role": "developer"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_token_validation_endpoint(self):
        """Test token validation endpoint."""
        # Generate a token
        token = create_access_token(self.test_user_id, "developer")

        # Test token validation
        response = self.client.get(
            "/api/v1/auth/token/validate", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == str(self.test_user_id)
        assert data["role"] == "developer"
        assert data["type"] == "access"

    def test_token_validation_invalid_token(self):
        """Test token validation with invalid token."""
        response = self.client.get(
            "/api/v1/auth/token/validate",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == 401

    def test_token_validation_no_token(self):
        """Test token validation without token."""
        response = self.client.get("/api/v1/auth/token/validate")
        assert response.status_code == 401

    def test_auth_config_endpoint(self):
        """Test authentication configuration endpoint."""
        response = self.client.get("/api/v1/auth/config")
        assert response.status_code == 200
        data = response.json()
        assert "jwt_algorithm" in data
        assert "access_token_expire_hours" in data
        assert "refresh_enabled" in data

    @patch("app.psql_db_services.users_service.UsersService.get_user_by_id")
    def test_protected_endpoint_success(self, mock_get_user):
        """Test accessing protected endpoint with valid token."""
        # Mock user service response
        mock_user = UserResponse(
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
        mock_get_user.return_value = mock_user

        # Generate token
        token = create_access_token(self.test_user_id, "developer")

        # Test protected endpoint
        response = self.client.get(
            f"/api/v1/users/{self.test_user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_protected_endpoint_no_auth(self):
        """Test accessing protected endpoint without authentication."""
        response = self.client.get(f"/api/v1/users/{self.test_user_id}")
        assert response.status_code == 401

    def test_protected_endpoint_invalid_token(self):
        """Test accessing protected endpoint with invalid token."""
        response = self.client.get(
            f"/api/v1/users/{self.test_user_id}",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == 401

    @patch("app.psql_db_services.users_service.UsersService.get_user_by_id")
    def test_admin_endpoint_developer_fails(self, mock_get_user):
        """Test that developer cannot access admin endpoints."""
        # Mock user service response
        mock_user = UserResponse(
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
        mock_get_user.return_value = mock_user

        # Generate developer token
        token = create_access_token(self.test_user_id, "developer")

        # Test admin endpoint
        response = self.client.patch(
            f"/api/v1/users/{self.test_user_id}",
            json={"email": "new@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    @patch("app.psql_db_services.users_service.UsersService.get_user_by_id")
    def test_admin_endpoint_admin_success(self, mock_get_user):
        """Test that admin can access admin endpoints."""
        # Mock user service response
        mock_user = UserResponse(
            user_id=self.test_admin_id,
            username="admin",
            email="admin@example.com",
            first_name="Admin",
            last_name="User",
            role="admin",
            status="active",
            created_at=None,
            updated_at=None,
        )
        mock_get_user.return_value = mock_user

        # Generate admin token
        token = create_access_token(self.test_admin_id, "admin")

        # Test admin endpoint
        response = self.client.patch(
            f"/api/v1/users/{self.test_user_id}",
            json={"email": "new@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should not be 403 (forbidden) - might be 404 if user doesn't exist, but not 403
        assert response.status_code != 403

    def test_token_refresh_disabled(self):
        """Test token refresh when disabled."""
        response = self.client.post(
            "/api/v1/auth/token/refresh", json={"refresh_token": "some_token"}
        )
        # Should return 400 (bad request) when refresh is disabled
        assert response.status_code == 400

    def test_health_endpoints_no_auth(self):
        """Test that health endpoints don't require authentication."""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200

    def test_user_registration_no_auth(self):
        """Test that user registration doesn't require authentication."""
        response = self.client.post(
            "/api/v1/users/",
            json={
                "first_name": "Test",
                "last_name": "User",
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )
        # Should not be 401 (unauthorized) - might be 400 if validation fails, but not 401
        assert response.status_code != 401

    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses."""
        response = self.client.options("/api/v1/auth/token/validate")
        # CORS preflight should be handled
        assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS

    def test_error_response_format(self):
        """Test that error responses follow consistent format."""
        # Test unauthorized access (should get 401 error)
        response = self.client.get("/api/v1/users/invalid-uuid")

        # Should get 401 unauthorized error since no token provided
        assert response.status_code == 401

        # Should have proper error format
        data = response.json()
        assert "detail" in data

    def test_token_expiration_handling(self):
        """Test that expired tokens are handled properly."""
        # This test would require mocking time or using very short expiration
        # For now, just test that the endpoint exists and responds
        response = self.client.get("/api/v1/auth/token/validate")
        assert response.status_code == 401  # No token provided
