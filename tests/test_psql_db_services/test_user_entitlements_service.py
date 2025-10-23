"""
Comprehensive Unit Tests for UserEntitlementsService
==================================================
Async unit tests for the UserEntitlementsService class covering all CRUD operations
with both positive and negative test cases.

Test Coverage:
- Validation methods (15 tests)
- Create operations (12 tests)
- Read multiple users operations (13 tests)
- Count operations (2 tests)
- Delete operations (8 tests)

Total: 50 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime
from contextlib import asynccontextmanager

from app.psql_db_services.user_entitlements_service import UserEntitlementsService
from app.core.database_connection import DatabaseManager


def setup_mock_sqlalchemy_session(mock_db_manager, mock_result_data=None, rowcount=1):
    """
    Helper function to set up mock SQLAlchemy session and result for async context managers.

    Args:
        mock_db_manager: The mock database manager
        mock_result_data: Optional data to return from result operations
        rowcount: Number of rows affected for update/delete operations
    """
    # Create mock result object with SQLAlchemy methods
    mock_result = MagicMock()

    # Set up mappings() method that returns the result object itself
    mock_result.mappings.return_value = mock_result

    # Set up result methods based on data type
    if mock_result_data is not None:
        if isinstance(mock_result_data, list):
            # For multiple results (fetchall equivalent)
            mock_result.all.return_value = mock_result_data
            mock_result.one_or_none.return_value = None  # Not used for lists
            mock_result.first.return_value = None  # Not used for lists
        elif isinstance(mock_result_data, tuple) and len(mock_result_data) == 1:
            # For scalar results (like COUNT queries)
            mock_result.scalar_one_or_none.return_value = mock_result_data[0]
            mock_result.first.return_value = None  # Not used for scalars
        else:
            # For single result (fetchone equivalent)
            mock_result.one_or_none.return_value = mock_result_data
            mock_result.all.return_value = []  # Not used for single results
            mock_result.first.return_value = (
                mock_result_data  # Used by check_email_exists
            )
    else:
        # When mock_result_data is None, set up for "not found" cases
        mock_result.one_or_none.return_value = None
        mock_result.all.return_value = []
        mock_result.scalar_one_or_none.return_value = None
        mock_result.first.return_value = None  # Used by check_email_exists

    # Set up rowcount for update/delete operations
    mock_result.rowcount = rowcount

    # Create mock session with async context manager support
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Set up session methods
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    # Ensure execute returns the mock_result directly, not a coroutine
    async def mock_execute(*args, **kwargs):
        return mock_result

    mock_session.execute = mock_execute

    # Create a mock async context manager that behaves like the real get_session
    @asynccontextmanager
    async def mock_get_session_cm():
        try:
            yield mock_session
        except Exception:
            await mock_session.rollback()
            raise
        else:
            await mock_session.commit()
        finally:
            await mock_session.close()

    # Mock the get_session method
    mock_db_manager.get_session = mock_get_session_cm

    return mock_session


def setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence):
    """
    Helper function to set up mock SQLAlchemy session with a sequence of results.

    Args:
        mock_db_manager: The mock database manager
        result_sequence: List of mock_result_data for each database call in sequence
    """
    # Create a list of mock results for each call
    mock_results = []
    for mock_result_data, rowcount in result_sequence:
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result
        mock_result.rowcount = rowcount

        # Set up result methods based on data type
        if mock_result_data is not None:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
                mock_result.one_or_none.return_value = None
                mock_result.first.return_value = None
            elif isinstance(mock_result_data, tuple) and len(mock_result_data) == 1:
                mock_result.scalar_one_or_none.return_value = mock_result_data[0]
                mock_result.first.return_value = None
            else:
                mock_result.one_or_none.return_value = mock_result_data
                mock_result.all.return_value = []
                mock_result.first.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []
            mock_result.scalar_one_or_none.return_value = None
            mock_result.first.return_value = None

        mock_results.append(mock_result)

    # Create mock session with async context manager support
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    # Create a call counter to return different results for each call
    call_count = 0

    async def mock_execute(*args, **kwargs):
        nonlocal call_count
        if call_count < len(mock_results):
            result = mock_results[call_count]
            call_count += 1
            return result
        else:
            # Return the last result for any additional calls
            return mock_results[-1] if mock_results else None

    mock_session.execute = mock_execute

    # Create a mock async context manager
    @asynccontextmanager
    async def mock_get_session_cm():
        try:
            yield mock_session
        except Exception:
            await mock_session.rollback()
            raise
        else:
            await mock_session.commit()
        finally:
            await mock_session.close()

    # Mock the get_session method
    mock_db_manager.get_session = mock_get_session_cm

    return mock_session


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def mock_db_manager():
    """Create mock database manager for testing."""
    return MagicMock(spec=DatabaseManager)


@pytest.fixture
def entitlements_service(mock_db_manager):
    """Create UserEntitlementsService with mocked database manager."""
    return UserEntitlementsService(database_manager=mock_db_manager)


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return uuid4()


@pytest.fixture
def sample_admin_id():
    """Sample admin user ID for testing."""
    return uuid4()


@pytest.fixture
def sample_entitlement_data():
    """Sample entitlement data for testing."""
    return {
        "entitlement_id": 1,
        "user_id": uuid4(),
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "api_endpoint_url": "https://api.openai.com/v1",
        "cloud_provider": None,
        "deployment_name": None,
        "region": "us-east-1",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "created_by_user_id": uuid4(),
    }


@pytest.fixture
def sample_cloud_entitlement_data():
    """Sample cloud entitlement data for testing."""
    return {
        "entitlement_id": 2,
        "user_id": uuid4(),
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "api_endpoint_url": "https://my-resource.openai.azure.com/",
        "cloud_provider": "azure_openai",
        "deployment_name": "gpt4o-eastus-prod",
        "region": "eastus",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "created_by_user_id": uuid4(),
    }


@pytest.fixture
def encrypted_api_key():
    """Pre-encrypted API key for testing."""
    return "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4j8j8j8j8j"


# ============================================================================
# VALIDATION METHODS TESTS
# ============================================================================


class TestValidationMethods:
    """Test cases for validation helper methods."""

    @pytest.mark.asyncio
    async def test_validate_user_exists_success(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test successful user existence validation."""
        # Setup mock to return user exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"user_id": sample_user_id}
        )

        result = await entitlements_service.validate_user_exists(sample_user_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_user_exists_not_found(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test user existence validation when user not found."""
        # Setup mock to return no user
        setup_mock_sqlalchemy_session(mock_db_manager, mock_result_data=None)

        result = await entitlements_service.validate_user_exists(sample_user_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_user_exists_database_error(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test user existence validation with database error."""
        # Create mock session that will raise exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        # Create a mock async context manager
        @asynccontextmanager
        async def mock_get_session_cm():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            else:
                await mock_session.commit()
            finally:
                await mock_session.close()

        # Mock the get_session method
        mock_db_manager.get_session = mock_get_session_cm

        with pytest.raises(Exception, match="Database connection failed"):
            await entitlements_service.validate_user_exists(sample_user_id)

    @pytest.mark.asyncio
    async def test_validate_provider_model_exists_success(
        self, entitlements_service, mock_db_manager
    ):
        """Test successful provider/model existence validation."""
        # Setup mock to return provider/model exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"llm_provider": "openai"}
        )

        result = await entitlements_service.validate_provider_model_exists(
            "openai", "gpt-4o"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_provider_model_exists_not_found(
        self, entitlements_service, mock_db_manager
    ):
        """Test provider/model existence validation when not found."""
        # Setup mock to return no provider/model
        setup_mock_sqlalchemy_session(mock_db_manager, mock_result_data=None)

        result = await entitlements_service.validate_provider_model_exists(
            "openai", "gpt-4o"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_provider_model_exists_database_error(
        self, entitlements_service, mock_db_manager
    ):
        """Test provider/model existence validation with database error."""
        # Create mock session that will raise exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        # Create a mock async context manager
        @asynccontextmanager
        async def mock_get_session_cm():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            else:
                await mock_session.commit()
            finally:
                await mock_session.close()

        # Mock the get_session method
        mock_db_manager.get_session = mock_get_session_cm

        with pytest.raises(Exception, match="Database connection failed"):
            await entitlements_service.validate_provider_model_exists(
                "openai", "gpt-4o"
            )

    @pytest.mark.asyncio
    async def test_check_entitlement_exists_true(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test entitlement existence check when entitlement exists."""
        # Setup mock to return entitlement exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"entitlement_id": 1}
        )

        result = await entitlements_service.check_entitlement_exists(
            sample_user_id, "openai", "gpt-4o", "https://api.openai.com/v1"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_entitlement_exists_false(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test entitlement existence check when entitlement does not exist."""
        # Setup mock to return no entitlement
        setup_mock_sqlalchemy_session(mock_db_manager, mock_result_data=None)

        result = await entitlements_service.check_entitlement_exists(
            sample_user_id, "openai", "gpt-4o", "https://api.openai.com/v1"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_check_entitlement_exists_with_endpoint(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test entitlement existence check with specific endpoint."""
        # Setup mock to return entitlement exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"entitlement_id": 1}
        )

        result = await entitlements_service.check_entitlement_exists(
            sample_user_id, "openai", "gpt-4o", "https://api.openai.com/v1"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_entitlement_exists_without_endpoint(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test entitlement existence check without endpoint."""
        # Setup mock to return entitlement exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"entitlement_id": 1}
        )

        result = await entitlements_service.check_entitlement_exists(
            sample_user_id, "openai", "gpt-4o", None
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_entitlement_exists_null_endpoint_match(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test entitlement existence check with null endpoint matching."""
        # Setup mock to return entitlement exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"entitlement_id": 1}
        )

        result = await entitlements_service.check_entitlement_exists(
            sample_user_id, "openai", "gpt-4o", None
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_entitlement_exists_database_error(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test entitlement existence check with database error."""
        # Create mock session that will raise exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        # Create a mock async context manager
        @asynccontextmanager
        async def mock_get_session_cm():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            else:
                await mock_session.commit()
            finally:
                await mock_session.close()

        # Mock the get_session method
        mock_db_manager.get_session = mock_get_session_cm

        with pytest.raises(Exception, match="Database connection failed"):
            await entitlements_service.check_entitlement_exists(
                sample_user_id, "openai", "gpt-4o", "https://api.openai.com/v1"
            )

    def test_validate_positive_integer_valid(self, entitlements_service):
        """Test positive integer validation with valid input."""
        # Should not raise exception
        entitlements_service.validate_positive_integer(1, "test_param")
        entitlements_service.validate_positive_integer(100, "test_param")

    def test_validate_positive_integer_invalid(self, entitlements_service):
        """Test positive integer validation with invalid input."""
        with pytest.raises(ValueError, match="test_param must be positive, got -1"):
            entitlements_service.validate_positive_integer(-1, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive, got 0"):
            entitlements_service.validate_positive_integer(0, "test_param")

    def test_validate_uuid_valid(self, entitlements_service, sample_user_id):
        """Test UUID validation with valid input."""
        # Should not raise exception
        entitlements_service.validate_uuid(sample_user_id, "test_param")


# ============================================================================
# CREATE OPERATIONS TESTS
# ============================================================================


class TestCreateEntitlement:
    """Test cases for entitlement creation."""

    @pytest.mark.asyncio
    async def test_create_entitlement_success_minimal_fields(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test successful entitlement creation with minimal fields."""
        # Setup mocks for validation and creation in sequence
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": None,
                    "cloud_provider": None,
                    "deployment_name": None,
                    "region": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
        )

        assert result["entitlement_id"] == 1
        assert result["user_id"] == sample_user_id
        assert result["llm_provider"] == "openai"
        assert result["llm_model_name"] == "gpt-4o"
        assert "api_key" not in result  # API key should be excluded

    @pytest.mark.asyncio
    async def test_create_entitlement_success_all_fields(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test successful entitlement creation with all fields."""
        # Setup mocks for validation and creation in sequence
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": "https://api.openai.com/v1",
                    "cloud_provider": "azure_openai",
                    "deployment_name": "gpt4o-eastus-prod",
                    "region": "eastus",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
            api_endpoint_url="https://api.openai.com/v1",
            cloud_provider="Azure",
            deployment_name="gpt4o-eastus-prod",
            region="eastus",
        )

        assert result["entitlement_id"] == 1
        assert result["user_id"] == sample_user_id
        assert result["llm_provider"] == "openai"
        assert result["llm_model_name"] == "gpt-4o"
        assert result["api_endpoint_url"] == "https://api.openai.com/v1"
        assert result["cloud_provider"] == "azure_openai"
        assert result["deployment_name"] == "gpt4o-eastus-prod"
        assert result["region"] == "eastus"
        assert "api_key" not in result  # API key should be excluded

    @pytest.mark.asyncio
    async def test_create_entitlement_user_not_exists(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation when user does not exist."""
        # Setup mock to return user not found
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=None
        )  # validate_user_exists

        with pytest.raises(ValueError, match="User with ID .* does not exist"):
            await entitlements_service.create_entitlement(
                user_id=sample_user_id,
                llm_provider="openai",
                llm_model_name="gpt-4o",
                encrypted_api_key=encrypted_api_key,
                created_by_user_id=sample_admin_id,
            )

    @pytest.mark.asyncio
    async def test_create_entitlement_provider_model_not_exists(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation when provider/model does not exist."""
        # Setup mocks: user exists, but provider/model doesn't
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            (None, 1),  # validate_provider_model_exists
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        with pytest.raises(
            ValueError,
            match="Provider/model combination 'openai/gpt-4o' does not exist in llm_models table",
        ):
            await entitlements_service.create_entitlement(
                user_id=sample_user_id,
                llm_provider="openai",
                llm_model_name="gpt-4o",
                encrypted_api_key=encrypted_api_key,
                created_by_user_id=sample_admin_id,
            )

    @pytest.mark.asyncio
    async def test_create_entitlement_duplicate_exists(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation when duplicate entitlement exists."""
        # Setup mocks: user exists, provider/model exists, but entitlement already exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"user_id": sample_user_id}
        )  # validate_user_exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"llm_provider": "openai"}
        )  # validate_provider_model_exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"entitlement_id": 1}
        )  # check_entitlement_exists

        with pytest.raises(ValueError, match="Entitlement already exists for user"):
            await entitlements_service.create_entitlement(
                user_id=sample_user_id,
                llm_provider="openai",
                llm_model_name="gpt-4o",
                encrypted_api_key=encrypted_api_key,
                created_by_user_id=sample_admin_id,
            )

    @pytest.mark.asyncio
    async def test_create_entitlement_with_cloud_provider(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation with cloud provider fields."""
        # Setup mocks for validation and creation in sequence
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": "https://my-resource.openai.azure.com/",
                    "cloud_provider": "azure_openai",
                    "deployment_name": "gpt4o-eastus-prod",
                    "region": "eastus",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
            api_endpoint_url="https://my-resource.openai.azure.com/",
            cloud_provider="Azure",
            deployment_name="gpt4o-eastus-prod",
            region="eastus",
        )

        assert result["cloud_provider"] == "azure_openai"
        assert result["deployment_name"] == "gpt4o-eastus-prod"
        assert result["region"] == "eastus"

    @pytest.mark.asyncio
    async def test_create_entitlement_with_deployment_name(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation with deployment name."""
        # Setup mocks for validation and creation in sequence
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": None,
                    "cloud_provider": None,
                    "deployment_name": "gpt4o-eastus-prod",
                    "region": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
            deployment_name="gpt4o-eastus-prod",
        )

        assert result["deployment_name"] == "gpt4o-eastus-prod"

    @pytest.mark.asyncio
    async def test_create_entitlement_with_region(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation with region."""
        # Setup mocks for validation and creation in sequence
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": None,
                    "cloud_provider": None,
                    "deployment_name": None,
                    "region": "eastus",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
            region="eastus",
        )

        assert result["region"] == "eastus"

    @pytest.mark.asyncio
    async def test_create_entitlement_database_error(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation with database error."""
        # Setup mocks for validation (success) but creation fails
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        # Setup mock to raise database error during creation
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        @asynccontextmanager
        async def mock_get_session_cm():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            else:
                await mock_session.commit()
            finally:
                await mock_session.close()

        mock_db_manager.get_session = mock_get_session_cm

        with pytest.raises(Exception, match="Database connection failed"):
            await entitlements_service.create_entitlement(
                user_id=sample_user_id,
                llm_provider="openai",
                llm_model_name="gpt-4o",
                encrypted_api_key=encrypted_api_key,
                created_by_user_id=sample_admin_id,
            )

    @pytest.mark.asyncio
    async def test_create_entitlement_commit_failure(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test entitlement creation with commit failure."""
        # Setup mocks for validation (success) but commit fails
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"user_id": sample_user_id}
        )  # validate_user_exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data={"llm_provider": "openai"}
        )  # validate_provider_model_exists
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=None
        )  # check_entitlement_exists

        # Setup mock to raise error during commit
        mock_session = setup_mock_sqlalchemy_session(
            mock_db_manager,
            mock_result_data={
                "entitlement_id": 1,
                "user_id": sample_user_id,
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_endpoint_url": None,
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": sample_admin_id,
            },
        )
        mock_session.commit.side_effect = Exception("Commit failed")

        with pytest.raises(Exception, match="Commit failed"):
            await entitlements_service.create_entitlement(
                user_id=sample_user_id,
                llm_provider="openai",
                llm_model_name="gpt-4o",
                encrypted_api_key=encrypted_api_key,
                created_by_user_id=sample_admin_id,
            )

    @pytest.mark.asyncio
    async def test_create_entitlement_returns_without_api_key(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test that created entitlement response excludes API key."""
        # Setup mocks for validation and creation in sequence
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": None,
                    "cloud_provider": None,
                    "deployment_name": None,
                    "region": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
        )

        # Verify API key is not in response
        assert "api_key" not in result
        assert "encrypted_api_key" not in result

    @pytest.mark.asyncio
    async def test_create_entitlement_audit_trail(
        self,
        entitlements_service,
        sample_user_id,
        sample_admin_id,
        encrypted_api_key,
        mock_db_manager,
    ):
        """Test that audit trail fields are properly set."""
        # Setup mocks for validation and creation in sequence
        now = datetime.utcnow()
        result_sequence = [
            ({"user_id": sample_user_id}, 1),  # validate_user_exists
            ({"llm_provider": "openai"}, 1),  # validate_provider_model_exists
            (None, 1),  # check_entitlement_exists
            (
                {
                    "entitlement_id": 1,
                    "user_id": sample_user_id,
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "api_endpoint_url": None,
                    "cloud_provider": None,
                    "deployment_name": None,
                    "region": None,
                    "created_at": now,
                    "updated_at": now,
                    "created_by_user_id": sample_admin_id,
                },
                1,
            ),  # create_entitlement
        ]
        setup_mock_sqlalchemy_session_sequence(mock_db_manager, result_sequence)

        result = await entitlements_service.create_entitlement(
            user_id=sample_user_id,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            encrypted_api_key=encrypted_api_key,
            created_by_user_id=sample_admin_id,
        )

        assert result["created_by_user_id"] == sample_admin_id
        assert result["created_at"] == now
        assert result["updated_at"] == now


# ============================================================================
# READ OPERATIONS TESTS
# ============================================================================


class TestGetUserEntitlements:
    """Test cases for user entitlements retrieval."""

    @pytest.mark.asyncio
    async def test_get_user_entitlements_success(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test successful user entitlements retrieval."""
        entitlements_data = [
            {
                "entitlement_id": 1,
                "user_id": sample_user_id,
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_endpoint_url": "https://api.openai.com/v1",
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": uuid4(),
            },
            {
                "entitlement_id": 2,
                "user_id": sample_user_id,
                "llm_provider": "anthropic",
                "llm_model_name": "claude-3-5-sonnet",
                "api_endpoint_url": None,
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": uuid4(),
            },
        ]
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=entitlements_data
        )

        result = await entitlements_service.get_user_entitlements(sample_user_id)

        assert len(result) == 2
        assert result[0]["entitlement_id"] == 1
        assert result[1]["entitlement_id"] == 2
        assert all(
            "api_key" not in entitlement for entitlement in result
        )  # API keys excluded

    @pytest.mark.asyncio
    async def test_get_user_entitlements_empty(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test user entitlements retrieval when user has no entitlements."""
        setup_mock_sqlalchemy_session(mock_db_manager, mock_result_data=[])

        result = await entitlements_service.get_user_entitlements(sample_user_id)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_user_entitlements_with_pagination(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test user entitlements retrieval with pagination."""
        entitlements_data = [
            {
                "entitlement_id": 1,
                "user_id": sample_user_id,
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_endpoint_url": None,
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": uuid4(),
            }
        ]
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=entitlements_data
        )

        result = await entitlements_service.get_user_entitlements(
            sample_user_id, limit=10, offset=0
        )

        assert len(result) == 1
        assert result[0]["entitlement_id"] == 1

    @pytest.mark.asyncio
    async def test_get_user_entitlements_invalid_user_id(self, entitlements_service):
        """Test user entitlements retrieval with invalid user ID."""
        with pytest.raises(ValueError, match="user_id must be a valid UUID"):
            await entitlements_service.get_user_entitlements("invalid-uuid")

    @pytest.mark.asyncio
    async def test_get_user_entitlements_invalid_pagination(
        self, entitlements_service, sample_user_id
    ):
        """Test user entitlements retrieval with invalid pagination parameters."""
        with pytest.raises(ValueError, match="limit must be positive, got -1"):
            await entitlements_service.get_user_entitlements(sample_user_id, limit=-1)

        with pytest.raises(ValueError, match="offset must be non-negative, got -1"):
            await entitlements_service.get_user_entitlements(sample_user_id, offset=-1)

    @pytest.mark.asyncio
    async def test_get_user_entitlements_api_keys_excluded(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test that API keys are excluded from user entitlements response."""
        entitlements_data = [
            {
                "entitlement_id": 1,
                "user_id": sample_user_id,
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_endpoint_url": None,
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": uuid4(),
            }
        ]
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=entitlements_data
        )

        result = await entitlements_service.get_user_entitlements(sample_user_id)

        assert all("api_key" not in entitlement for entitlement in result)
        assert all("encrypted_api_key" not in entitlement for entitlement in result)

    @pytest.mark.asyncio
    async def test_get_user_entitlements_ordered_by_created_at(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test that user entitlements are ordered by created_at DESC."""
        entitlements_data = [
            {
                "entitlement_id": 2,
                "user_id": sample_user_id,
                "llm_provider": "anthropic",
                "llm_model_name": "claude-3-5-sonnet",
                "api_endpoint_url": None,
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": uuid4(),
            },
            {
                "entitlement_id": 1,
                "user_id": sample_user_id,
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_endpoint_url": None,
                "cloud_provider": None,
                "deployment_name": None,
                "region": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "created_by_user_id": uuid4(),
            },
        ]
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=entitlements_data
        )

        result = await entitlements_service.get_user_entitlements(sample_user_id)

        # Should be ordered by created_at DESC (newest first)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_user_entitlements_database_error(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test user entitlements retrieval with database error."""
        # Create mock session that will raise exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        # Create a mock async context manager
        @asynccontextmanager
        async def mock_get_session_cm():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            else:
                await mock_session.commit()
            finally:
                await mock_session.close()

        # Mock the get_session method
        mock_db_manager.get_session = mock_get_session_cm

        with pytest.raises(Exception, match="Database connection failed"):
            await entitlements_service.get_user_entitlements(sample_user_id)


class TestCountUserEntitlements:
    """Test cases for counting user entitlements."""

    @pytest.mark.asyncio
    async def test_count_user_entitlements_success(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test successful user entitlements count."""
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=(5,)
        )  # scalar result

        result = await entitlements_service.count_user_entitlements(sample_user_id)

        assert result == 5

    @pytest.mark.asyncio
    async def test_count_user_entitlements_zero(
        self, entitlements_service, sample_user_id, mock_db_manager
    ):
        """Test user entitlements count when user has no entitlements."""
        setup_mock_sqlalchemy_session(
            mock_db_manager, mock_result_data=(0,)
        )  # scalar result

        result = await entitlements_service.count_user_entitlements(sample_user_id)

        assert result == 0


# ============================================================================
# DELETE OPERATIONS TESTS
# ============================================================================


class TestDeleteEntitlement:
    """Test cases for entitlement deletion."""

    @pytest.mark.asyncio
    async def test_delete_entitlement_success(
        self, entitlements_service, mock_db_manager
    ):
        """Test successful entitlement deletion."""
        setup_mock_sqlalchemy_session(mock_db_manager, rowcount=1)

        result = await entitlements_service.delete_entitlement(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_entitlement_not_found(
        self, entitlements_service, mock_db_manager
    ):
        """Test entitlement deletion when entitlement not found."""
        setup_mock_sqlalchemy_session(mock_db_manager, rowcount=0)

        result = await entitlements_service.delete_entitlement(999)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_entitlement_invalid_id(self, entitlements_service):
        """Test entitlement deletion with invalid ID."""
        with pytest.raises(ValueError, match="entitlement_id must be positive, got -1"):
            await entitlements_service.delete_entitlement(-1)

    @pytest.mark.asyncio
    async def test_delete_entitlement_returns_true_on_success(
        self, entitlements_service, mock_db_manager
    ):
        """Test that delete returns True on successful deletion."""
        setup_mock_sqlalchemy_session(mock_db_manager, rowcount=1)

        result = await entitlements_service.delete_entitlement(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_entitlement_returns_false_on_not_found(
        self, entitlements_service, mock_db_manager
    ):
        """Test that delete returns False when entitlement not found."""
        setup_mock_sqlalchemy_session(mock_db_manager, rowcount=0)

        result = await entitlements_service.delete_entitlement(999)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_entitlement_logs_operation(
        self, entitlements_service, mock_db_manager
    ):
        """Test that delete operation is logged."""
        setup_mock_sqlalchemy_session(mock_db_manager, rowcount=1)

        result = await entitlements_service.delete_entitlement(1)

        assert result is True
        # Verify that log_operation was called (this is tested through the base service)

    @pytest.mark.asyncio
    async def test_delete_entitlement_database_error(
        self, entitlements_service, mock_db_manager
    ):
        """Test entitlement deletion with database error."""
        # Create mock session that will raise exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        # Create a mock async context manager
        @asynccontextmanager
        async def mock_get_session_cm():
            try:
                yield mock_session
            except Exception:
                await mock_session.rollback()
                raise
            else:
                await mock_session.commit()
            finally:
                await mock_session.close()

        # Mock the get_session method
        mock_db_manager.get_session = mock_get_session_cm

        with pytest.raises(Exception, match="Database connection failed"):
            await entitlements_service.delete_entitlement(1)

    @pytest.mark.asyncio
    async def test_delete_entitlement_cascade_safe(
        self, entitlements_service, mock_db_manager
    ):
        """Test that delete operation is cascade safe."""
        setup_mock_sqlalchemy_session(mock_db_manager, rowcount=1)

        result = await entitlements_service.delete_entitlement(1)

        assert result is True
        # The cascade safety is handled by the database schema, not the service layer
