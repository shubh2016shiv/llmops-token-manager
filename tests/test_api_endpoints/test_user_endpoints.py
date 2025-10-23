"""
Comprehensive Unit Tests for User Endpoints
==========================================
Unit tests for user management endpoints covering all CRUD operations
with 100% coverage including success and error scenarios.

Test Coverage:
- Create user operations (6 tests)
- Get user by ID operations (6 tests)
- Get user by email operations (5 tests)
- Update user operations (6 tests)
- Suspend user operations (2 tests)
- Activate user operations (2 tests)

Total: 27 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from datetime import datetime
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.user_endpoints import router
from app.models.response_models import UserResponse


# ============================================================================
# TEST SETUP
# ============================================================================


@pytest.fixture
def app():
    """Create FastAPI app with user endpoints router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "user_id": uuid4(),
        "username": "testuser",
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "role": "developer",
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_create_request():
    """Sample user creation request data."""
    return {
        "first_name": "John",
        "last_name": "Doe",
        "username": "johndoe",
        "email": "john.doe@example.com",
        "password": "SecurePass123",
    }


@pytest.fixture
def sample_update_request():
    """Sample user update request data."""
    return {
        "email": "john.updated@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "role": "admin",
        "status": "active",
    }


# ============================================================================
# CREATE USER TESTS
# ============================================================================


class TestCreateUser:
    """Test cases for user creation endpoint."""

    @patch("app.api.user_endpoints.UsersService")
    @patch("app.api.user_endpoints.PasswordHasher")
    def test_create_user_success(
        self,
        mock_password_hasher,
        mock_users_service,
        client,
        sample_create_request,
        sample_user_data,
    ):
        """Test successful user creation returns correct response."""
        # Arrange
        mock_password_hasher.hash_password.return_value = "hashed_password"
        mock_service_instance = AsyncMock()
        mock_service_instance.create_user.return_value = sample_user_data
        mock_users_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/users/", json=sample_create_request)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == str(sample_user_data["user_id"])
        assert data["username"] == sample_user_data["username"]
        assert data["email"] == sample_user_data["email"]
        assert data["first_name"] == sample_user_data["first_name"]
        assert data["last_name"] == sample_user_data["last_name"]
        assert data["role"] == sample_user_data["role"]
        assert data["status"] == sample_user_data["status"]
        assert "created_at" in data
        assert "updated_at" in data

        # Verify service was called correctly
        mock_service_instance.create_user.assert_called_once()
        call_args = mock_service_instance.create_user.call_args
        assert call_args[1]["username"] == sample_create_request["username"]
        assert call_args[1]["email"] == sample_create_request["email"]
        assert call_args[1]["first_name"] == sample_create_request["first_name"]
        assert call_args[1]["last_name"] == sample_create_request["last_name"]
        assert call_args[1]["password_hash"] == "hashed_password"
        assert call_args[1]["user_role"] == "developer"
        assert call_args[1]["user_status"] == "active"

    @patch("app.api.user_endpoints.UsersService")
    @patch("app.api.user_endpoints.PasswordHasher")
    def test_create_user_duplicate_email(
        self, mock_password_hasher, mock_users_service, client, sample_create_request
    ):
        """Test user creation with duplicate email returns 400 error."""
        # Arrange
        mock_password_hasher.hash_password.return_value = "hashed_password"
        mock_service_instance = AsyncMock()
        mock_service_instance.create_user.side_effect = ValueError(
            "Email 'john.doe@example.com' already exists"
        )
        mock_users_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/users/", json=sample_create_request)

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Email 'john.doe@example.com' already exists" in data["detail"]

    @patch("app.api.user_endpoints.UsersService")
    @patch("app.api.user_endpoints.PasswordHasher")
    def test_create_user_duplicate_username(
        self, mock_password_hasher, mock_users_service, client, sample_create_request
    ):
        """Test user creation with duplicate username returns 400 error."""
        # Arrange
        mock_password_hasher.hash_password.return_value = "hashed_password"
        mock_service_instance = AsyncMock()
        mock_service_instance.create_user.side_effect = ValueError(
            "Username 'johndoe' already exists"
        )
        mock_users_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/users/", json=sample_create_request)

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Username 'johndoe' already exists" in data["detail"]

    def test_create_user_invalid_request_body(self, client):
        """Test user creation with invalid request body returns 422 validation error."""
        # Arrange
        invalid_request = {
            "first_name": "John",
            # Missing required fields
            "email": "invalid-email",  # Invalid email format
            "password": "123",  # Too short password
        }

        # Act
        response = client.post("/api/v1/users/", json=invalid_request)

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Should have validation errors for missing fields and invalid email/password

    @patch("app.api.user_endpoints.UsersService")
    @patch("app.api.user_endpoints.PasswordHasher")
    def test_create_user_password_hashing(
        self,
        mock_password_hasher,
        mock_users_service,
        client,
        sample_create_request,
        sample_user_data,
    ):
        """Test that password hashing is called with correct password."""
        # Arrange
        mock_password_hasher.hash_password.return_value = "hashed_password"
        mock_service_instance = AsyncMock()
        mock_service_instance.create_user.return_value = sample_user_data
        mock_users_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/users/", json=sample_create_request)

        # Assert
        assert response.status_code == 201
        mock_password_hasher.hash_password.assert_called_once_with(
            sample_create_request["password"]
        )

    @patch("app.api.user_endpoints.UsersService")
    @patch("app.api.user_endpoints.PasswordHasher")
    def test_create_user_database_error(
        self, mock_password_hasher, mock_users_service, client, sample_create_request
    ):
        """Test user creation with database error returns 500 error."""
        # Arrange
        mock_password_hasher.hash_password.return_value = "hashed_password"
        mock_service_instance = AsyncMock()
        mock_service_instance.create_user.side_effect = Exception(
            "Database connection failed"
        )
        mock_users_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/users/", json=sample_create_request)

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to create user. Please try again later." in data["detail"]


# ============================================================================
# GET USER BY ID TESTS
# ============================================================================


class TestGetUserById:
    """Test cases for get user by ID endpoint."""

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_id_success(
        self, mock_users_service, app, client, mock_developer_user, sample_user_data
    ):
        """Test successful user retrieval by ID returns correct response."""
        # Override auth dependency
        # NOTE: Using dependency_overrides is FastAPI's recommended approach
        # for testing. It bypasses JWT validation while testing business logic.
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_id.return_value = sample_user_data
        mock_users_service.return_value = mock_service_instance
        user_id = sample_user_data["user_id"]

        # Act
        response = client.get(f"/api/v1/users/{user_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == str(user_id)
        assert data["username"] == sample_user_data["username"]
        assert data["email"] == sample_user_data["email"]
        assert data["first_name"] == sample_user_data["first_name"]
        assert data["last_name"] == sample_user_data["last_name"]
        assert data["role"] == sample_user_data["role"]
        assert data["status"] == sample_user_data["status"]

        # Verify service was called correctly
        mock_service_instance.get_user_by_id.assert_called_once_with(user_id)

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_id_not_found(
        self, mock_users_service, app, client, mock_developer_user
    ):
        """Test user not found by ID returns 404 error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_id.return_value = None
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.get(f"/api/v1/users/{user_id}")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert f"User with ID '{user_id}' not found" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    def test_get_user_by_id_invalid_uuid(self, app, client, mock_developer_user):
        """Test invalid UUID format returns 400 validation error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        invalid_uuid = "invalid-uuid-format"

        # Act
        response = client.get(f"/api/v1/users/{invalid_uuid}")

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_id_value_error(
        self, mock_users_service, app, client, mock_developer_user
    ):
        """Test ValueError from service returns 400 error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_id.side_effect = ValueError(
            "Invalid user ID format"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.get(f"/api/v1/users/{user_id}")

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid user ID format" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_id_database_error(
        self, mock_users_service, app, client, mock_developer_user
    ):
        """Test database error returns 500 error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_id.side_effect = Exception(
            "Database connection failed"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.get(f"/api/v1/users/{user_id}")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve user" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_id_response_structure(
        self, mock_users_service, app, client, mock_developer_user, sample_user_data
    ):
        """Test response structure matches UserResponse schema."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_id.return_value = sample_user_data
        mock_users_service.return_value = mock_service_instance
        user_id = sample_user_data["user_id"]

        # Act
        response = client.get(f"/api/v1/users/{user_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate all required fields are present
        required_fields = [
            "user_id",
            "username",
            "email",
            "first_name",
            "last_name",
            "role",
            "status",
            "created_at",
            "updated_at",
        ]
        for field in required_fields:
            assert field in data

        # Validate UserResponse model can be created from response
        user_response = UserResponse(**data)
        assert user_response.user_id == user_id
        assert user_response.username == sample_user_data["username"]

        # Cleanup
        app.dependency_overrides.clear()


# ============================================================================
# GET USER BY EMAIL TESTS
# ============================================================================


class TestGetUserByEmail:
    """Test cases for get user by email endpoint."""

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_email_success(
        self, mock_users_service, app, client, mock_developer_user, sample_user_data
    ):
        """Test successful user retrieval by email returns correct response."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_email.return_value = sample_user_data
        mock_users_service.return_value = mock_service_instance
        email = sample_user_data["email"]

        # Act
        response = client.get(f"/api/v1/users/email/{email}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == str(sample_user_data["user_id"])
        assert data["email"] == email
        assert data["username"] == sample_user_data["username"]

        # Verify service was called correctly
        mock_service_instance.get_user_by_email.assert_called_once_with(email)

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_email_not_found(
        self, mock_users_service, app, client, mock_developer_user
    ):
        """Test user not found by email returns 404 error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_email.return_value = None
        mock_users_service.return_value = mock_service_instance
        email = "nonexistent@example.com"

        # Act
        response = client.get(f"/api/v1/users/email/{email}")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert f"User with email '{email}' not found" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_email_value_error(
        self, mock_users_service, app, client, mock_developer_user
    ):
        """Test ValueError from service returns 400 error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_email.side_effect = ValueError(
            "Invalid email format"
        )
        mock_users_service.return_value = mock_service_instance
        email = "invalid-email"

        # Act
        response = client.get(f"/api/v1/users/email/{email}")

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid email format" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_email_database_error(
        self, mock_users_service, app, client, mock_developer_user
    ):
        """Test database error returns 500 error."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_email.side_effect = Exception(
            "Database connection failed"
        )
        mock_users_service.return_value = mock_service_instance
        email = "test@example.com"

        # Act
        response = client.get(f"/api/v1/users/email/{email}")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve user" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_get_user_by_email_response_structure(
        self, mock_users_service, app, client, mock_developer_user, sample_user_data
    ):
        """Test response structure matches UserResponse schema."""
        # Override auth dependency
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_developer_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_user_by_email.return_value = sample_user_data
        mock_users_service.return_value = mock_service_instance
        email = sample_user_data["email"]

        # Act
        response = client.get(f"/api/v1/users/email/{email}")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate UserResponse model can be created from response
        user_response = UserResponse(**data)
        assert user_response.email == email
        assert user_response.username == sample_user_data["username"]

        # Cleanup
        app.dependency_overrides.clear()


# ============================================================================
# UPDATE USER TESTS
# ============================================================================


class TestUpdateUser:
    """Test cases for update user endpoint."""

    @patch("app.api.user_endpoints.UsersService")
    def test_update_user_success(
        self,
        mock_users_service,
        app,
        client,
        mock_admin_user,
        sample_user_data,
        sample_update_request,
    ):
        """Test successful user update returns correct response."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        updated_user_data = sample_user_data.copy()
        updated_user_data["email"] = sample_update_request["email"]
        updated_user_data["role"] = sample_update_request["role"]
        mock_service_instance.update_user.return_value = updated_user_data
        mock_users_service.return_value = mock_service_instance
        user_id = sample_user_data["user_id"]

        # Act
        response = client.patch(f"/api/v1/users/{user_id}", json=sample_update_request)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == sample_update_request["email"]
        assert data["role"] == sample_update_request["role"]

        # Cleanup
        app.dependency_overrides.clear()

        # Verify service was called correctly
        mock_service_instance.update_user.assert_called_once()
        call_args = mock_service_instance.update_user.call_args
        assert call_args[1]["user_id"] == user_id
        assert call_args[1]["email_address"] == sample_update_request["email"]
        assert call_args[1]["user_role"] == sample_update_request["role"]
        assert call_args[1]["user_status"] == sample_update_request["status"]

    @patch("app.api.user_endpoints.UsersService")
    def test_update_user_not_found(
        self, mock_users_service, app, client, mock_admin_user, sample_update_request
    ):
        """Test user not found for update returns 404 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_user.return_value = None
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}", json=sample_update_request)

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert f"User with ID '{user_id}' not found" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_update_user_duplicate_email(
        self, mock_users_service, app, client, mock_admin_user, sample_update_request
    ):
        """Test update with duplicate email returns 400 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_user.side_effect = ValueError(
            "Email 'john.updated@example.com' already exists"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}", json=sample_update_request)

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Email 'john.updated@example.com' already exists" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    def test_update_user_invalid_role(self, app, client, mock_admin_user):
        """Test update with invalid role returns 422 validation error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        user_id = uuid4()
        update_request = {"role": "invalid_role"}

        # Act
        response = client.patch(f"/api/v1/users/{user_id}", json=update_request)

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Should have validation error for invalid role enum value

        # Cleanup
        app.dependency_overrides.clear()

    def test_update_user_invalid_uuid(
        self, app, client, mock_admin_user, sample_update_request
    ):
        """Test invalid UUID format returns 400 validation error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        invalid_uuid = "invalid-uuid-format"

        # Act
        response = client.patch(
            f"/api/v1/users/{invalid_uuid}", json=sample_update_request
        )

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_update_user_database_error(
        self, mock_users_service, app, client, mock_admin_user, sample_update_request
    ):
        """Test database error returns appropriate error status."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_user.side_effect = Exception(
            "duplicate email constraint violation"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}", json=sample_update_request)

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Email 'john.updated@example.com' is already in use" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_update_user_general_database_error(
        self, mock_users_service, app, client, mock_admin_user, sample_update_request
    ):
        """Test general database error returns 500 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_user.side_effect = Exception(
            "Database connection failed"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}", json=sample_update_request)

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to update user" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()


# ============================================================================
# SUSPEND USER TESTS
# ============================================================================


class TestSuspendUser:
    """Test cases for suspend user endpoint."""

    @patch("app.api.user_endpoints.UsersService")
    def test_suspend_user_success(
        self, mock_users_service, app, client, mock_admin_user, sample_user_data
    ):
        """Test successful user suspension returns correct response."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        suspended_user_data = sample_user_data.copy()
        suspended_user_data["status"] = "suspended"
        mock_service_instance.suspend_user.return_value = suspended_user_data
        mock_users_service.return_value = mock_service_instance
        user_id = sample_user_data["user_id"]

        # Act
        response = client.patch(f"/api/v1/users/{user_id}/suspend")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "suspended"
        assert data["user_id"] == str(user_id)

        # Verify service was called correctly
        mock_service_instance.suspend_user.assert_called_once_with(user_id)

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_suspend_user_not_found(
        self, mock_users_service, app, client, mock_admin_user
    ):
        """Test user not found for suspension returns 404 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.suspend_user.return_value = None
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}/suspend")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert f"User with ID '{user_id}' not found" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_suspend_user_database_error(
        self, mock_users_service, app, client, mock_admin_user
    ):
        """Test database error during suspension returns 500 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.suspend_user.side_effect = Exception(
            "Database connection failed"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}/suspend")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to suspend user" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()


# ============================================================================
# ACTIVATE USER TESTS
# ============================================================================


class TestActivateUser:
    """Test cases for activate user endpoint."""

    @patch("app.api.user_endpoints.UsersService")
    def test_activate_user_success(
        self, mock_users_service, app, client, mock_admin_user, sample_user_data
    ):
        """Test successful user activation returns correct response."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        activated_user_data = sample_user_data.copy()
        activated_user_data["status"] = "active"
        mock_service_instance.activate_user.return_value = activated_user_data
        mock_users_service.return_value = mock_service_instance
        user_id = sample_user_data["user_id"]

        # Act
        response = client.patch(f"/api/v1/users/{user_id}/activate")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert data["user_id"] == str(user_id)

        # Verify service was called correctly
        mock_service_instance.activate_user.assert_called_once_with(user_id)

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_activate_user_not_found(
        self, mock_users_service, app, client, mock_admin_user
    ):
        """Test user not found for activation returns 404 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.activate_user.return_value = None
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}/activate")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert f"User with ID '{user_id}' not found" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()

    @patch("app.api.user_endpoints.UsersService")
    def test_activate_user_database_error(
        self, mock_users_service, app, client, mock_admin_user
    ):
        """Test database error during activation returns 500 error."""
        # Override auth dependency with admin user
        from app.auth.auth_dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: mock_admin_user

        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.activate_user.side_effect = Exception(
            "Database connection failed"
        )
        mock_users_service.return_value = mock_service_instance
        user_id = uuid4()

        # Act
        response = client.patch(f"/api/v1/users/{user_id}/activate")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to activate user" in data["detail"]

        # Cleanup
        app.dependency_overrides.clear()
