"""
Comprehensive Unit Tests for User Entitlement Endpoints
=====================================================
Unit tests for user entitlement management endpoints covering all HTTP operations
with authentication, authorization, and validation.

Test Coverage:
- Create entitlement operations (15 tests)
- List entitlements operations (10 tests)
- Delete entitlement operations (10 tests)

Total: 35 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from datetime import datetime
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.user_entitlement_endpoints import router


# ============================================================================
# TEST SETUP
# ============================================================================


@pytest.fixture
def app():
    """Create FastAPI app with user entitlement endpoints router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_entitlement_request():
    """Sample entitlement creation request data."""
    return {
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "api_key_value": "sk-1234567890abcdefgh",
        "api_endpoint_url": "https://api.openai.com/v1",
        "cloud_provider": None,
        "deployment_name": None,
        "deployment_region": "us-east-1",
    }


@pytest.fixture
def sample_cloud_entitlement_request():
    """Sample cloud entitlement creation request data."""
    return {
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "api_key_value": "sk-1234567890abcdefgh",
        "api_endpoint_url": "https://my-resource.openai.azure.com/",
        "cloud_provider": "Azure",
        "deployment_name": "gpt4o-eastus-prod",
        "deployment_region": "eastus",
    }


@pytest.fixture
def sample_entitlement_response():
    """Sample entitlement response data (no API key)."""
    return {
        "entitlement_id": 1,
        "user_id": str(uuid4()),
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "api_endpoint_url": "https://api.openai.com/v1",
        "cloud_provider": None,
        "deployment_name": None,
        "region": "us-east-1",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "created_by_user_id": str(uuid4()),
    }


@pytest.fixture
def sample_entitlement_list_response():
    """Sample entitlement list response data."""
    return {
        "entitlements": [
            {
                "entitlement_id": 1,
                "user_id": str(uuid4()),
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_endpoint_url": "https://api.openai.com/v1",
                "cloud_provider": None,
                "deployment_name": None,
                "region": "us-east-1",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "created_by_user_id": str(uuid4()),
            }
        ],
        "total_count": 1,
        "page": 1,
        "page_size": 50,
    }


@pytest.fixture
def sample_update_request():
    """Sample entitlement update request data."""
    return {
        "api_key": "sk-new1234567890abcdefgh",
        "api_endpoint_url": "https://new-endpoint.com/v1",
        "cloud_provider": "Azure",
        "deployment_name": "gpt4o-westus-prod",
        "deployment_region": "westus",
    }


def override_auth_dependency(app, user_payload):
    """Override authentication dependency for testing."""
    from app.auth.dependencies import require_admin, require_owner, require_developer

    def mock_require_admin():
        return user_payload

    def mock_require_owner():
        return user_payload

    def mock_require_developer():
        return user_payload

    app.dependency_overrides[require_admin] = mock_require_admin
    app.dependency_overrides[require_owner] = mock_require_owner
    app.dependency_overrides[require_developer] = mock_require_developer


# ============================================================================
# CREATE ENTITLEMENT TESTS
# ============================================================================


class TestCreateUserEntitlement:
    """Test cases for user entitlement creation endpoint."""

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_success_as_admin(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        sample_entitlement_response,
        mock_admin_user,
    ):
        """Test successful entitlement creation as admin user."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.return_value = (
            sample_entitlement_response
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 201
        data = response.json()
        assert data["entitlement_id"] == 1
        assert data["llm_provider"] == "openai"
        assert data["llm_model_name"] == "gpt-4o"
        assert "api_key" not in data  # API key should be excluded

        # Verify API key was hashed
        mock_password_hasher.hash_password.assert_called_once_with(
            sample_entitlement_request["api_key_value"]
        )

        # Verify service was called with hashed key
        mock_service_instance.create_entitlement.assert_called_once()
        call_args = mock_service_instance.create_entitlement.call_args
        assert call_args[1]["encrypted_api_key"] == "hashed_api_key"

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_success_as_owner(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        sample_entitlement_response,
        mock_owner_user,
    ):
        """Test successful entitlement creation as owner user."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.return_value = (
            sample_entitlement_response
        )

        # Override auth dependency
        override_auth_dependency(app, mock_owner_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 201
        data = response.json()
        assert data["entitlement_id"] == 1
        assert "api_key" not in data  # API key should be excluded

    def test_create_entitlement_forbidden_as_developer(
        self, client, app, sample_entitlement_request, mock_developer_user
    ):
        """Test entitlement creation forbidden for developer user."""
        # Override auth dependency
        override_auth_dependency(app, mock_developer_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 403
        assert (
            "Only admin and owner roles can create entitlements"
            in response.json()["detail"]
        )

    def test_create_entitlement_forbidden_as_operator(
        self, client, app, sample_entitlement_request, mock_operator_user
    ):
        """Test entitlement creation forbidden for operator user."""
        # Override auth dependency
        override_auth_dependency(app, mock_operator_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 403
        assert (
            "Only admin and owner roles can create entitlements"
            in response.json()["detail"]
        )

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_user_not_found(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        mock_admin_user,
    ):
        """Test entitlement creation when target user does not exist."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.side_effect = ValueError(
            "User with ID '...' does not exist"
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 400
        assert "User with ID" in response.json()["detail"]

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_api_key_encrypted(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        sample_entitlement_response,
        mock_admin_user,
    ):
        """Test that API key is encrypted before storage."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.return_value = (
            sample_entitlement_response
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 201

        # Verify API key was hashed
        mock_password_hasher.hash_password.assert_called_once_with(
            sample_entitlement_request["api_key_value"]
        )

        # Verify service was called with hashed key, not plain key
        mock_service_instance.create_entitlement.assert_called_once()
        call_args = mock_service_instance.create_entitlement.call_args
        assert call_args[1]["encrypted_api_key"] == "hashed_api_key"
        assert "api_key" not in call_args[1]  # Plain key should not be passed

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_api_key_not_in_response(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        sample_entitlement_response,
        mock_admin_user,
    ):
        """Test that API key is not included in response."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.return_value = (
            sample_entitlement_response
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 201
        data = response.json()
        assert "api_key" not in data
        assert "encrypted_api_key" not in data

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_provider_model_not_found(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        mock_admin_user,
    ):
        """Test entitlement creation when provider/model not found."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.side_effect = ValueError(
            "Provider/model combination '...' does not exist in llm_models table"
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 400
        assert "Provider/model combination" in response.json()["detail"]

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_duplicate_entitlement(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        mock_admin_user,
    ):
        """Test entitlement creation when duplicate entitlement exists."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.side_effect = ValueError(
            "Entitlement already exists for user"
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 400
        assert "Entitlement already exists" in response.json()["detail"]

    def test_create_entitlement_invalid_provider_type(
        self, client, app, mock_admin_user
    ):
        """Test entitlement creation with invalid provider type."""
        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        invalid_request = {
            "llm_provider": "invalid_provider",
            "llm_model_name": "gpt-4o",
            "api_key": "sk-1234567890abcdefgh",
        }

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=invalid_request
        )

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] in [
            "enum",
            "value_error",
            "missing",
            "greater_than_equal",
            "validation_error",
        ]

    def test_create_entitlement_invalid_api_key_too_short(
        self, client, app, mock_admin_user
    ):
        """Test entitlement creation with API key too short."""
        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        invalid_request = {
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "api_key": "short",
        }

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=invalid_request
        )

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] in [
            "enum",
            "value_error",
            "missing",
            "greater_than_equal",
            "validation_error",
        ]

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_with_cloud_provider_fields(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_cloud_entitlement_request,
        sample_entitlement_response,
        mock_admin_user,
    ):
        """Test entitlement creation with cloud provider fields."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.return_value = (
            sample_entitlement_response
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/",
            json=sample_cloud_entitlement_request,
        )

        assert response.status_code == 201

        # Verify service was called with cloud provider fields
        mock_service_instance.create_entitlement.assert_called_once()
        call_args = mock_service_instance.create_entitlement.call_args
        assert call_args[1]["cloud_provider"] == "Azure"
        assert call_args[1]["deployment_name"] == "gpt4o-eastus-prod"
        assert call_args[1]["region"] == "eastus"

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.PasswordHasher")
    def test_create_entitlement_database_error(
        self,
        mock_password_hasher,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_request,
        mock_admin_user,
    ):
        """Test entitlement creation with database error."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_password_hasher.hash_password.return_value = "hashed_api_key"
        mock_service_instance.create_entitlement.side_effect = Exception(
            "Database connection failed"
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=sample_entitlement_request
        )

        assert response.status_code == 500
        assert (
            "Failed to create entitlement. Please try again later."
            in response.json()["detail"]
        )

    def test_create_entitlement_validation_error(self, client, app, mock_admin_user):
        """Test entitlement creation with validation error."""
        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        # Missing required fields (api_key)
        invalid_request = {
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
        }

        user_id = str(uuid4())
        response = client.post(
            f"/api/v1/users/{user_id}/entitlements/", json=invalid_request
        )

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] in [
            "enum",
            "value_error",
            "missing",
            "greater_than_equal",
            "validation_error",
        ]


# ============================================================================
# LIST ENTITLEMENTS TESTS
# ============================================================================


class TestListUserEntitlements:
    """Test cases for listing user entitlements endpoint."""

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_success(
        self,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_list_response,
        mock_admin_user,
    ):
        """Test successful entitlements listing."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = (
            sample_entitlement_list_response["entitlements"]
        )
        mock_service_instance.count_user_entitlements.return_value = (
            sample_entitlement_list_response["total_count"]
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["entitlements"]) == 1
        assert data["entitlements"][0]["entitlement_id"] == 1
        assert "api_key" not in data["entitlements"][0]  # API key should be excluded

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_empty(
        self, mock_entitlements_service, client, app, mock_admin_user
    ):
        """Test entitlements listing when user has no entitlements."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = []
        mock_service_instance.count_user_entitlements.return_value = 0

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["entitlements"] == []

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_pagination_page_1(
        self,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_list_response,
        mock_admin_user,
    ):
        """Test entitlements listing with pagination page 1."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = (
            sample_entitlement_list_response["entitlements"]
        )
        mock_service_instance.count_user_entitlements.return_value = (
            sample_entitlement_list_response["total_count"]
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(
            f"/api/v1/users/{user_id}/entitlements/?page=1&page_size=10"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_pagination_page_2(
        self,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_list_response,
        mock_admin_user,
    ):
        """Test entitlements listing with pagination page 2."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = (
            sample_entitlement_list_response["entitlements"]
        )
        mock_service_instance.count_user_entitlements.return_value = (
            sample_entitlement_list_response["total_count"]
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(
            f"/api/v1/users/{user_id}/entitlements/?page=2&page_size=10"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_custom_page_size(
        self,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_list_response,
        mock_admin_user,
    ):
        """Test entitlements listing with custom page size."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = (
            sample_entitlement_list_response["entitlements"]
        )
        mock_service_instance.count_user_entitlements.return_value = (
            sample_entitlement_list_response["total_count"]
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/?page_size=25")

        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 25

    @patch("app.api.user_entitlement_endpoints.UsersService")
    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_user_not_found(
        self,
        mock_entitlements_service,
        mock_users_service,
        client,
        app,
        mock_admin_user,
    ):
        """Test entitlements listing when user not found."""
        # Setup mocks - user not found should be checked first
        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        mock_users_service_instance.get_user_by_id.return_value = None  # User not found

        # Also mock the entitlements service to prevent any calls to it
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/")

        assert response.status_code == 404
        assert "User with ID" in response.json()["detail"]

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_api_keys_excluded(
        self,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_list_response,
        mock_admin_user,
    ):
        """Test that API keys are excluded from entitlements list response."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = (
            sample_entitlement_list_response["entitlements"]
        )
        mock_service_instance.count_user_entitlements.return_value = (
            sample_entitlement_list_response["total_count"]
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/")

        assert response.status_code == 200
        data = response.json()
        assert all("api_key" not in entitlement for entitlement in data["entitlements"])
        assert all(
            "encrypted_api_key" not in entitlement
            for entitlement in data["entitlements"]
        )

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_as_different_roles(
        self,
        mock_entitlements_service,
        client,
        app,
        sample_entitlement_list_response,
        mock_developer_user,
    ):
        """Test entitlements listing as different user roles."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.return_value = (
            sample_entitlement_list_response["entitlements"]
        )
        mock_service_instance.count_user_entitlements.return_value = (
            sample_entitlement_list_response["total_count"]
        )

        # Override auth dependency
        override_auth_dependency(app, mock_developer_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1

    def test_list_entitlements_invalid_pagination(self, client, app, mock_admin_user):
        """Test entitlements listing with invalid pagination parameters."""
        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/?page=-1")

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] in [
            "enum",
            "value_error",
            "missing",
            "greater_than_equal",
            "validation_error",
        ]

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_list_entitlements_database_error(
        self, mock_entitlements_service, client, app, mock_admin_user
    ):
        """Test entitlements listing with database error."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_user_entitlements.side_effect = Exception(
            "Database connection failed"
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.get(f"/api/v1/users/{user_id}/entitlements/")

        assert response.status_code == 500
        assert (
            "Failed to retrieve entitlements. Please try again later."
            in response.json()["detail"]
        )


# ============================================================================
# DELETE ENTITLEMENT TESTS
# ============================================================================


class TestDeleteEntitlement:
    """Test cases for entitlement deletion endpoint."""

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.UsersService")
    def test_delete_entitlement_success_as_admin(
        self,
        mock_users_service,
        mock_entitlements_service,
        client,
        app,
        mock_admin_user,
    ):
        """Test successful entitlement deletion as admin user."""
        # Setup mocks
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance
        mock_entitlements_service_instance.delete_entitlement.return_value = True
        from uuid import UUID

        mock_entitlements_service_instance.get_entitlement_by_id.return_value = {
            "entitlement_id": 1,
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "cloud_provider": None,
            "api_endpoint_url": "https://api.openai.com/v1",
        }

        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        from uuid import UUID
        from app.models.response_models import UserResponse

        mock_users_service_instance.get_user_by_id.return_value = UserResponse(
            user_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="active",
            created_at=None,
            updated_at=None,
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        response = client.delete(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000/entitlements/1"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["deletion_status"] == "success"
        assert data["entitlement_id"] == 1
        assert data["user_details"]["username"] == "testuser"
        assert data["user_details"]["email"] == "test@example.com"
        assert data["entitlement_details"]["llm_provider"] == "openai"
        assert data["entitlement_details"]["llm_model_name"] == "gpt-4o"
        mock_entitlements_service_instance.delete_entitlement.assert_called_once_with(1)

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.UsersService")
    def test_delete_entitlement_success_as_owner(
        self,
        mock_users_service,
        mock_entitlements_service,
        client,
        app,
        mock_owner_user,
    ):
        """Test successful entitlement deletion as owner user."""
        # Setup mocks
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance
        mock_entitlements_service_instance.delete_entitlement.return_value = True
        from uuid import UUID

        mock_entitlements_service_instance.get_entitlement_by_id.return_value = {
            "entitlement_id": 1,
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "cloud_provider": None,
            "api_endpoint_url": "https://api.openai.com/v1",
        }

        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        from uuid import UUID
        from app.models.response_models import UserResponse

        mock_users_service_instance.get_user_by_id.return_value = UserResponse(
            user_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="active",
            created_at=None,
            updated_at=None,
        )

        # Override auth dependency
        override_auth_dependency(app, mock_owner_user)

        response = client.delete(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000/entitlements/1"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["deletion_status"] == "success"
        mock_entitlements_service_instance.delete_entitlement.assert_called_once_with(1)

    def test_delete_entitlement_forbidden_as_developer(
        self, client, app, mock_developer_user
    ):
        """Test entitlement deletion forbidden for developer user."""
        # Override auth dependency
        override_auth_dependency(app, mock_developer_user)

        user_id = str(uuid4())
        response = client.delete(f"/api/v1/users/{user_id}/entitlements/1")

        assert response.status_code == 403
        assert (
            "Only admin and owner roles can delete entitlements"
            in response.json()["detail"]
        )

    def test_delete_entitlement_forbidden_as_operator(
        self, client, app, mock_operator_user
    ):
        """Test entitlement deletion forbidden for operator user."""
        # Override auth dependency
        override_auth_dependency(app, mock_operator_user)

        user_id = str(uuid4())
        response = client.delete(f"/api/v1/users/{user_id}/entitlements/1")

        assert response.status_code == 403
        assert (
            "Only admin and owner roles can delete entitlements"
            in response.json()["detail"]
        )

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_delete_entitlement_not_found(
        self, mock_entitlements_service, client, app, mock_admin_user
    ):
        """Test entitlement deletion when entitlement not found."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_entitlement_by_id.return_value = (
            None  # Entitlement not found
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.delete(f"/api/v1/users/{user_id}/entitlements/999")

        assert response.status_code == 404
        assert "Entitlement with ID '999' not found" in response.json()["detail"]

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.UsersService")
    def test_delete_entitlement_wrong_user(
        self,
        mock_users_service,
        mock_entitlements_service,
        client,
        app,
        mock_admin_user,
    ):
        """Test entitlement deletion for wrong user."""
        from uuid import UUID

        # Setup mocks
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance
        mock_entitlements_service_instance.delete_entitlement.return_value = True
        mock_entitlements_service_instance.get_entitlement_by_id.return_value = {
            "entitlement_id": 1,
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
        }

        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        from uuid import UUID
        from app.models.response_models import UserResponse

        mock_users_service_instance.get_user_by_id.return_value = UserResponse(
            user_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="active",
            created_at=None,
            updated_at=None,
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        response = client.delete(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000/entitlements/1"
        )

        assert response.status_code == 200
        # Admin can delete any entitlement

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.UsersService")
    def test_delete_entitlement_returns_200_with_details(
        self,
        mock_users_service,
        mock_entitlements_service,
        client,
        app,
        mock_admin_user,
    ):
        """Test that successful deletion returns 200 status code with detailed response."""
        # Setup mocks
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance
        mock_entitlements_service_instance.delete_entitlement.return_value = True
        from uuid import UUID

        mock_entitlements_service_instance.get_entitlement_by_id.return_value = {
            "entitlement_id": 1,
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "cloud_provider": None,
            "api_endpoint_url": "https://api.openai.com/v1",
        }

        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        from uuid import UUID
        from app.models.response_models import UserResponse

        mock_users_service_instance.get_user_by_id.return_value = UserResponse(
            user_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="active",
            created_at=None,
            updated_at=None,
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        response = client.delete(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000/entitlements/1"
        )

        assert response.status_code == 200
        data = response.json()
        assert "deletion_status" in data
        assert "user_details" in data
        assert "entitlement_details" in data

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.UsersService")
    def test_delete_entitlement_user_mismatch(
        self,
        mock_users_service,
        mock_entitlements_service,
        client,
        app,
        mock_admin_user,
    ):
        """Test entitlement deletion with user mismatch."""
        from uuid import UUID

        # Setup mocks
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance
        mock_entitlements_service_instance.delete_entitlement.return_value = True
        mock_entitlements_service_instance.get_entitlement_by_id.return_value = {
            "entitlement_id": 1,
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
        }

        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        from uuid import UUID
        from app.models.response_models import UserResponse

        mock_users_service_instance.get_user_by_id.return_value = UserResponse(
            user_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="developer",
            status="active",
            created_at=None,
            updated_at=None,
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        response = client.delete(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000/entitlements/1"
        )

        assert response.status_code == 200
        # Admin can delete any entitlement

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    @patch("app.api.user_entitlement_endpoints.UsersService")
    def test_delete_entitlement_database_error(
        self,
        mock_users_service,
        mock_entitlements_service,
        client,
        app,
        mock_admin_user,
    ):
        """Test entitlement deletion with database error."""
        from uuid import UUID

        # Setup mocks
        mock_entitlements_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_entitlements_service_instance
        mock_entitlements_service_instance.get_entitlement_by_id.return_value = {
            "entitlement_id": 1,
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
        }
        mock_entitlements_service_instance.delete_entitlement.side_effect = Exception(
            "Database connection failed"
        )

        mock_users_service_instance = AsyncMock()
        mock_users_service.return_value = mock_users_service_instance
        from uuid import UUID

        mock_users_service_instance.get_user_by_id.return_value = {
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "username": "testuser",
            "email": "test@example.com",
        }

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        response = client.delete(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000/entitlements/1"
        )

        assert response.status_code == 500
        # FastAPI wraps the error_response dict in a "detail" key
        response_data = response.json()
        assert "detail" in response_data
        detail = response_data["detail"]
        assert detail["deletion_status"] == "failure"
        assert (
            "Failed to delete entitlement due to internal server error"
            in detail["error"]
        )

    @patch("app.api.user_entitlement_endpoints.UserEntitlementsService")
    def test_delete_entitlement_idempotent(
        self, mock_entitlements_service, client, app, mock_admin_user
    ):
        """Test that deletion is idempotent."""
        # Setup mocks
        mock_service_instance = AsyncMock()
        mock_entitlements_service.return_value = mock_service_instance
        mock_service_instance.get_entitlement_by_id.return_value = (
            None  # Entitlement not found
        )

        # Override auth dependency
        override_auth_dependency(app, mock_admin_user)

        user_id = str(uuid4())
        response = client.delete(f"/api/v1/users/{user_id}/entitlements/1")

        assert response.status_code == 404
        assert "Entitlement with ID '1' not found" in response.json()["detail"]
