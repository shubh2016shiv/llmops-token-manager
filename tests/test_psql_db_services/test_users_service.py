"""
Comprehensive Unit Tests for UsersService
========================================
Async unit tests for the UsersService class covering all CRUD operations
with both positive and negative test cases.

Test Coverage:
- Validation methods (9 tests)
- Create operations (6 tests)
- Read single user operations (6 tests)
- Read multiple users operations (13 tests)
- Update operations (15 tests)
- Delete operations (6 tests)

Total: 55 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime

import psycopg
from psycopg import IntegrityError

from app.psql_db_services.users_service import UsersService
from app.core.database_connection import DatabaseManager


def setup_mock_database_connection(mock_db_manager, mock_cursor_data=None):
    """
    Helper function to set up mock database connection and cursor for psycopg3 async context managers.

    Args:
        mock_db_manager: The mock database manager
        mock_cursor_data: Optional data to return from cursor operations
    """
    # Create mock cursor with async context manager support
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    # Set up cursor methods
    if mock_cursor_data is not None:
        if isinstance(mock_cursor_data, list):
            mock_cursor.fetchall = AsyncMock(return_value=mock_cursor_data)
        else:
            mock_cursor.fetchone = AsyncMock(return_value=mock_cursor_data)
    else:
        # When mock_cursor_data is None, set up fetchone to return None (for "not found" cases)
        mock_cursor.fetchone = AsyncMock(return_value=None)

    mock_cursor.execute = AsyncMock()
    mock_cursor.rowcount = (
        getattr(mock_cursor_data, "rowcount", 1)
        if hasattr(mock_cursor_data, "rowcount")
        else 1
    )

    # Create mock connection
    mock_connection = AsyncMock()

    # KEY FIX: cursor() should be a regular MagicMock (not AsyncMock) that returns the async context manager
    # In psycopg3, cursor() is NOT an async method - it's a synchronous method that returns an async CM
    mock_connection.cursor = MagicMock(return_value=mock_cursor)

    mock_connection.commit = AsyncMock()
    mock_connection.rollback = AsyncMock()

    # Set up database manager to return the connection
    mock_db_manager.get_connection = AsyncMock(return_value=mock_connection)

    return mock_connection, mock_cursor


class TestUsersServiceValidation:
    """Test validation methods for UsersService."""

    @pytest.fixture
    def users_service(self):
        """Create UsersService instance for testing."""
        return UsersService()

    # Positive validation tests
    @pytest.mark.parametrize("valid_role", ["owner", "admin", "developer", "viewer"])
    def test_validate_user_role_valid(self, users_service, valid_role):
        """Test that valid user roles pass validation."""
        # Should not raise any exception
        users_service.validate_user_role(valid_role)

    @pytest.mark.parametrize("valid_status", ["active", "suspended", "inactive"])
    def test_validate_user_status_valid(self, users_service, valid_status):
        """Test that valid user statuses pass validation."""
        # Should not raise any exception
        users_service.validate_user_status(valid_status)

    @pytest.mark.parametrize(
        "valid_email",
        [
            "test@example.com",
            "user.name@domain.co.uk",
            "test+tag@example.org",
            "user123@test-domain.com",
        ],
    )
    def test_validate_email_address_valid(self, users_service, valid_email):
        """Test that valid email addresses pass validation."""
        # Should not raise any exception
        users_service.validate_email_address(valid_email)

    # Negative validation tests
    @pytest.mark.parametrize("invalid_role", ["invalid", "superuser", "", None, 123])
    def test_validate_user_role_invalid(self, users_service, invalid_role):
        """Test that invalid user roles raise ValueError."""
        with pytest.raises(ValueError, match="Invalid user role"):
            users_service.validate_user_role(invalid_role)

    @pytest.mark.parametrize("invalid_status", ["invalid", "pending", "", None, 123])
    def test_validate_user_status_invalid(self, users_service, invalid_status):
        """Test that invalid user statuses raise ValueError."""
        with pytest.raises(ValueError, match="Invalid user status"):
            users_service.validate_user_status(invalid_status)

    @pytest.mark.parametrize(
        "invalid_email,expected_error",
        [
            ("", "email address must be a non-empty string"),
            (None, "email address must be a non-empty string"),
            ("   ", "email address cannot be only whitespace"),
            ("invalid-email", "Invalid email address format"),
            ("@domain.com", "Invalid email address format"),
            ("user@", "Invalid email address format"),
        ],
    )
    def test_validate_email_address_invalid(
        self, users_service, invalid_email, expected_error
    ):
        """Test that invalid email addresses raise ValueError."""
        with pytest.raises(ValueError, match=expected_error):
            users_service.validate_email_address(invalid_email)

    def test_validate_uuid_invalid(self, users_service):
        """Test that invalid UUID raises ValueError."""
        with pytest.raises(ValueError, match="must be a valid UUID instance"):
            users_service.validate_uuid("invalid-uuid", "user_id")


class TestUsersServiceCreate:
    """Test create operations for UsersService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def users_service(self, mock_db_manager):
        """UsersService instance with mocked database manager."""
        return UsersService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for tests."""
        return {
            "user_id": uuid4(),
            "email": "test@example.com",
            "role": "developer",
            "status": "active",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

    @pytest.mark.asyncio
    async def test_create_user_success(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test successfully creating a user with all fields."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_user_data
        )

        # Act
        result = await users_service.create_user(
            email_address="test@example.com",
            user_role="developer",
            user_status="active",
        )

        # Assert
        assert result == sample_user_data
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_defaults(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test creating a user with default role and status."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_user_data
        )

        # Act
        result = await users_service.create_user(email_address="test@example.com")

        # Assert
        assert result == sample_user_data
        # Verify default values were used
        call_args = mock_cursor.execute.call_args[0]
        assert call_args[1][1] == "developer"  # default role
        assert call_args[1][2] == "active"  # default status

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, users_service, mock_db_manager):
        """Test that duplicate email raises IntegrityError."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(
            side_effect=IntegrityError("duplicate key value")
        )

        # Act & Assert
        with pytest.raises(IntegrityError):
            await users_service.create_user(email_address="duplicate@example.com")

        mock_connection.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_invalid_role(self, users_service):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid user role"):
            await users_service.create_user(
                email_address="test@example.com", user_role="invalid_role"
            )

    @pytest.mark.asyncio
    async def test_create_user_invalid_status(self, users_service):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid user status"):
            await users_service.create_user(
                email_address="test@example.com", user_status="invalid_status"
            )

    @pytest.mark.asyncio
    async def test_create_user_database_error(self, users_service, mock_db_manager):
        """Test that database errors are properly handled."""
        # Arrange
        mock_db_manager.get_connection.side_effect = psycopg.Error("Connection failed")

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await users_service.create_user(email_address="test@example.com")


class TestUsersServiceReadSingle:
    """Test single user read operations for UsersService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def users_service(self, mock_db_manager):
        """UsersService instance with mocked database manager."""
        return UsersService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for tests."""
        return {
            "user_id": uuid4(),
            "email": "test@example.com",
            "role": "developer",
            "status": "active",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

    @pytest.mark.asyncio
    async def test_get_user_by_id_found(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test getting user by ID when user exists."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_user_data
        )
        user_id = sample_user_data["user_id"]

        # Act
        result = await users_service.get_user_by_id(user_id)

        # Assert
        assert result == sample_user_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_by_email_found(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test getting user by email when user exists."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_user_data
        )

        # Act
        result = await users_service.get_user_by_email("test@example.com")

        # Assert
        assert result == sample_user_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self, users_service, mock_db_manager):
        """Test getting user by ID when user doesn't exist."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, None
        )

        # Act
        result = await users_service.get_user_by_id(uuid4())

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self, users_service, mock_db_manager):
        """Test getting user by email when user doesn't exist."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, None
        )

        # Act
        result = await users_service.get_user_by_email("nonexistent@example.com")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_by_id_invalid_uuid(self, users_service):
        """Test that invalid UUID raises ValueError."""
        with pytest.raises(ValueError, match="must be a valid UUID instance"):
            await users_service.get_user_by_id("invalid-uuid")

    @pytest.mark.asyncio
    async def test_get_user_by_id_database_error(self, users_service, mock_db_manager):
        """Test that database errors are properly handled."""
        # Arrange
        mock_db_manager.get_connection.side_effect = psycopg.Error("Connection failed")

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await users_service.get_user_by_id(uuid4())


class TestUsersServiceReadMultiple:
    """Test multiple users read operations for UsersService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def users_service(self, mock_db_manager):
        """UsersService instance with mocked database manager."""
        return UsersService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_users_data(self):
        """Sample users data for tests."""
        return [
            {
                "user_id": uuid4(),
                "email": "user1@example.com",
                "role": "admin",
                "status": "active",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            },
            {
                "user_id": uuid4(),
                "email": "user2@example.com",
                "role": "developer",
                "status": "active",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            },
        ]

    @pytest.mark.asyncio
    async def test_get_all_users_no_filters(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting all users without filters."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_all_users()

        # Assert
        assert result == sample_users_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_users_role_filter(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting users filtered by role."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_all_users(role_filter="admin")

        # Assert
        assert result == sample_users_data
        # Verify role filter was applied
        call_args = mock_cursor.execute.call_args[0]
        assert "role = %s" in call_args[0]
        assert call_args[1][0] == "admin"

    @pytest.mark.asyncio
    async def test_get_all_users_status_filter(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting users filtered by status."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_all_users(status_filter="active")

        # Assert
        assert result == sample_users_data
        # Verify status filter was applied
        call_args = mock_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]
        assert call_args[1][0] == "active"

    @pytest.mark.asyncio
    async def test_get_all_users_both_filters(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting users filtered by both role and status."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_all_users(
            role_filter="admin", status_filter="active"
        )

        # Assert
        assert result == sample_users_data
        # Verify both filters were applied
        call_args = mock_cursor.execute.call_args[0]
        assert "role = %s" in call_args[0]
        assert "status = %s" in call_args[0]
        assert call_args[1][0] == "admin"
        assert call_args[1][1] == "active"

    @pytest.mark.asyncio
    async def test_get_all_users_pagination(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting users with pagination."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_all_users(limit=10, offset=5)

        # Assert
        assert result == sample_users_data
        # Verify pagination was applied
        call_args = mock_cursor.execute.call_args[0]
        assert "LIMIT %s OFFSET %s" in call_args[0]
        assert call_args[1][-2] == 10  # limit
        assert call_args[1][-1] == 5  # offset

    @pytest.mark.asyncio
    async def test_get_users_by_role(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting users by specific role."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_users_by_role("admin")

        # Assert
        assert result == sample_users_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_users(
        self, users_service, mock_db_manager, sample_users_data
    ):
        """Test getting only active users."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_users_data
        )

        # Act
        result = await users_service.get_active_users()

        # Assert
        assert result == sample_users_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_users_by_status(self, users_service, mock_db_manager):
        """Test counting users by status."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, (5,)
        )

        # Act
        result = await users_service.count_users_by_status("active")

        # Assert
        assert result == 5
        mock_cursor.execute.assert_called_once()

    # Negative test cases
    @pytest.mark.asyncio
    async def test_get_all_users_invalid_limit(self, users_service):
        """Test that negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be positive"):
            await users_service.get_all_users(limit=-1)

    @pytest.mark.asyncio
    async def test_get_all_users_invalid_offset(self, users_service):
        """Test that negative offset raises ValueError."""
        with pytest.raises(ValueError, match="offset must be non-negative"):
            await users_service.get_all_users(offset=-1)

    @pytest.mark.asyncio
    async def test_get_all_users_limit_exceeds_max(self, users_service):
        """Test that limit > 1000 raises ValueError."""
        with pytest.raises(ValueError, match="limit cannot exceed 1000"):
            await users_service.get_all_users(limit=1001)

    @pytest.mark.asyncio
    async def test_get_all_users_invalid_role_filter(self, users_service):
        """Test that invalid role filter raises ValueError."""
        with pytest.raises(ValueError, match="Invalid user role"):
            await users_service.get_all_users(role_filter="invalid")

    @pytest.mark.asyncio
    async def test_get_all_users_database_error(self, users_service, mock_db_manager):
        """Test that database errors are properly handled."""
        # Arrange
        mock_db_manager.get_connection.side_effect = psycopg.Error("Connection failed")

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await users_service.get_all_users()


class TestUsersServiceUpdate:
    """Test update operations for UsersService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def users_service(self, mock_db_manager):
        """UsersService instance with mocked database manager."""
        return UsersService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for tests."""
        return {
            "user_id": uuid4(),
            "email": "test@example.com",
            "role": "developer",
            "status": "active",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

    @pytest.mark.asyncio
    async def test_update_user_email_only(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test updating only user email."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["email"] = "updated@example.com"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.update_user(
            user_id=sample_user_data["user_id"], email_address="updated@example.com"
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_role_only(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test updating only user role."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["role"] = "admin"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.update_user(
            user_id=sample_user_data["user_id"], user_role="admin"
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_status_only(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test updating only user status."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["status"] = "suspended"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.update_user(
            user_id=sample_user_data["user_id"], user_status="suspended"
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_all_fields(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test updating all user fields."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["email"] = "updated@example.com"
        updated_data["role"] = "admin"
        updated_data["status"] = "suspended"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.update_user(
            user_id=sample_user_data["user_id"],
            email_address="updated@example.com",
            user_role="admin",
            user_status="suspended",
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_no_changes(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test updating user with no fields provided returns current user."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_user_data
        )

        # Act
        result = await users_service.update_user(user_id=sample_user_data["user_id"])

        # Assert
        assert result == sample_user_data

    @pytest.mark.asyncio
    async def test_update_user_role_method(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test using update_user_role helper method."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["role"] = "admin"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.update_user_role(
            user_id=sample_user_data["user_id"], new_user_role="admin"
        )

        # Assert
        assert result == updated_data

    @pytest.mark.asyncio
    async def test_update_user_status_method(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test using update_user_status helper method."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["status"] = "suspended"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.update_user_status(
            user_id=sample_user_data["user_id"], new_user_status="suspended"
        )

        # Assert
        assert result == updated_data

    @pytest.mark.asyncio
    async def test_suspend_user(self, users_service, mock_db_manager, sample_user_data):
        """Test suspending user sets status to suspended."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["status"] = "suspended"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.suspend_user(user_id=sample_user_data["user_id"])

        # Assert
        assert result == updated_data

    @pytest.mark.asyncio
    async def test_activate_user(
        self, users_service, mock_db_manager, sample_user_data
    ):
        """Test activating user sets status to active."""
        # Arrange
        updated_data = sample_user_data.copy()
        updated_data["status"] = "active"

        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )

        # Act
        result = await users_service.activate_user(user_id=sample_user_data["user_id"])

        # Assert
        assert result == updated_data

    # Negative test cases
    @pytest.mark.asyncio
    async def test_update_user_not_found(self, users_service, mock_db_manager):
        """Test updating user that doesn't exist returns None."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, None
        )

        # Act
        result = await users_service.update_user(
            user_id=uuid4(), email_address="updated@example.com"
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_update_user_invalid_uuid(self, users_service):
        """Test that invalid UUID raises ValueError."""
        with pytest.raises(ValueError, match="must be a valid UUID instance"):
            await users_service.update_user(
                user_id="invalid-uuid", email_address="updated@example.com"
            )

    @pytest.mark.asyncio
    async def test_update_user_duplicate_email(self, users_service, mock_db_manager):
        """Test that duplicate email raises IntegrityError."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(
            side_effect=IntegrityError("duplicate key value")
        )

        # Act & Assert
        with pytest.raises(IntegrityError):
            await users_service.update_user(
                user_id=uuid4(), email_address="duplicate@example.com"
            )

        mock_connection.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_invalid_role(self, users_service):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid user role"):
            await users_service.update_user(user_id=uuid4(), user_role="invalid_role")

    @pytest.mark.asyncio
    async def test_update_user_invalid_status(self, users_service):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid user status"):
            await users_service.update_user(
                user_id=uuid4(), user_status="invalid_status"
            )

    @pytest.mark.asyncio
    async def test_update_user_database_error(self, users_service, mock_db_manager):
        """Test that database errors are properly handled."""
        # Arrange
        mock_db_manager.get_connection.side_effect = psycopg.Error("Connection failed")

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await users_service.update_user(
                user_id=uuid4(), email_address="updated@example.com"
            )


class TestUsersServiceDelete:
    """Test delete operations for UsersService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def users_service(self, mock_db_manager):
        """UsersService instance with mocked database manager."""
        return UsersService(database_manager=mock_db_manager)

    @pytest.mark.asyncio
    async def test_delete_user_by_id_success(self, users_service, mock_db_manager):
        """Test successfully deleting user by ID."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 1  # Simulate successful deletion
        user_id = uuid4()

        # Act
        result = await users_service.delete_user(user_id)

        # Assert
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_user_by_email_success(self, users_service, mock_db_manager):
        """Test successfully deleting user by email."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 1  # Simulate successful deletion

        # Act
        result = await users_service.delete_user_by_email("test@example.com")

        # Assert
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_user_by_id_not_found(self, users_service, mock_db_manager):
        """Test deleting user by ID when user doesn't exist."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 0  # Simulate no rows affected

        # Act
        result = await users_service.delete_user(uuid4())

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_user_by_email_not_found(self, users_service, mock_db_manager):
        """Test deleting user by email when user doesn't exist."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 0  # Simulate no rows affected

        # Act
        result = await users_service.delete_user_by_email("nonexistent@example.com")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_user_invalid_uuid(self, users_service):
        """Test that invalid UUID raises ValueError."""
        with pytest.raises(ValueError, match="must be a valid UUID instance"):
            await users_service.delete_user("invalid-uuid")

    @pytest.mark.asyncio
    async def test_delete_user_database_error(self, users_service, mock_db_manager):
        """Test that database errors are properly handled."""
        # Arrange
        mock_db_manager.get_connection.side_effect = psycopg.Error("Connection failed")

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await users_service.delete_user(uuid4())


# Run with: pytest tests/test_psql_db_services/test_users_service.py -v
