"""
Comprehensive Unit Tests for LLMModelsService
============================================
Async unit tests for the LLMModelsService class covering all CRUD operations
with both positive and negative test cases.

Test Coverage:
- Validation methods (12 tests)
- Create operations (6 tests)
- Read single model operations (6 tests)
- Read multiple models operations (15 tests)
- Update operations (12 tests)
- Delete operations (5 tests)

Total: 56 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime

import psycopg
from psycopg import IntegrityError

from app.psql_db_services.llm_models_service import LLMModelsService
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


class TestLLMModelsServiceValidation:
    """Test validation methods for LLMModelsService."""

    @pytest.fixture
    def llm_models_service(self):
        """Create LLMModelsService instance for testing."""
        return LLMModelsService()

    # Positive validation tests
    @pytest.mark.parametrize("valid_provider", ["openai", "gemini", "anthropic"])
    def test_validate_llm_provider_valid(self, llm_models_service, valid_provider):
        """Test that valid LLM providers pass validation."""
        # Should not raise any exception
        llm_models_service.validate_llm_provider(valid_provider)

    def test_validate_model_numerical_parameters_valid(self, llm_models_service):
        """Test that valid numerical parameters pass validation."""
        # Should not raise any exception
        llm_models_service.validate_model_numerical_parameters(
            maximum_tokens=8192,
            tokens_per_minute_limit=90000,
            requests_per_minute_limit=3500,
            temperature_value=1.0,
            random_seed=42,
        )

    def test_validate_model_numerical_parameters_edge_cases(self, llm_models_service):
        """Test edge cases for numerical parameters."""
        # Temperature at boundaries
        llm_models_service.validate_model_numerical_parameters(temperature_value=0.0)
        llm_models_service.validate_model_numerical_parameters(temperature_value=2.0)

    # Negative validation tests
    @pytest.mark.parametrize(
        "invalid_provider", ["invalid", "azure", "claude", "", None, 123]
    )
    def test_validate_llm_provider_invalid(self, llm_models_service, invalid_provider):
        """Test that invalid LLM providers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            llm_models_service.validate_llm_provider(invalid_provider)

    @pytest.mark.parametrize("invalid_max_tokens", [-1, 0, "invalid"])
    def test_validate_maximum_tokens_invalid(
        self, llm_models_service, invalid_max_tokens
    ):
        """Test that invalid maximum_tokens raise ValueError."""
        with pytest.raises(ValueError, match="must be (an integer|positive)"):
            llm_models_service.validate_model_numerical_parameters(
                maximum_tokens=invalid_max_tokens
            )

    @pytest.mark.parametrize("invalid_tokens_limit", [-1, "invalid"])
    def test_validate_tokens_per_minute_limit_invalid(
        self, llm_models_service, invalid_tokens_limit
    ):
        """Test that invalid tokens_per_minute_limit raise ValueError."""
        with pytest.raises(ValueError, match="must be (an integer|positive)"):
            llm_models_service.validate_model_numerical_parameters(
                tokens_per_minute_limit=invalid_tokens_limit
            )

    @pytest.mark.parametrize("invalid_requests_limit", [-1, "invalid"])
    def test_validate_requests_per_minute_limit_invalid(
        self, llm_models_service, invalid_requests_limit
    ):
        """Test that invalid requests_per_minute_limit raise ValueError."""
        with pytest.raises(ValueError, match="must be (an integer|positive)"):
            llm_models_service.validate_model_numerical_parameters(
                requests_per_minute_limit=invalid_requests_limit
            )

    @pytest.mark.parametrize("invalid_temperature", [-0.1, 2.1, "invalid"])
    def test_validate_temperature_value_invalid(
        self, llm_models_service, invalid_temperature
    ):
        """Test that invalid temperature values raise ValueError."""
        with pytest.raises(
            ValueError, match="temperature_value must be (a number|between)"
        ):
            llm_models_service.validate_model_numerical_parameters(
                temperature_value=invalid_temperature
            )

    @pytest.mark.parametrize("invalid_seed", ["invalid", 1.5])
    def test_validate_random_seed_invalid(self, llm_models_service, invalid_seed):
        """Test that invalid random_seed values raise ValueError."""
        with pytest.raises(ValueError, match="random_seed must be an integer"):
            llm_models_service.validate_model_numerical_parameters(
                random_seed=invalid_seed
            )

    def test_validate_uuid_invalid(self, llm_models_service):
        """Test that invalid UUID raises ValueError."""
        with pytest.raises(ValueError, match="must be a valid UUID"):
            llm_models_service.validate_uuid("invalid-uuid", "model_id")


class TestLLMModelsServiceCreate:
    """Test create operations for LLMModelsService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def llm_models_service(self, mock_db_manager):
        """LLMModelsService instance with mocked database manager."""
        return LLMModelsService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for tests."""
        return {
            "model_id": uuid4(),
            "provider": "openai",
            "model_name": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "api_key_vault_id": "vault-key-123",
            "api_endpoint": "https://api.openai.com/v1",
            "model_version": "0613",
            "max_tokens": 8192,
            "tokens_per_minute_limit": 90000,
            "requests_per_minute_limit": 3500,
            "is_active": True,
            "temperature": 1.0,
            "seed": 42,
            "region": "us-east-1",
            "total_requests": 0,
            "total_tokens_processed": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_used_at": None,
        }

    @pytest.mark.asyncio
    async def test_create_llm_model_success(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test successfully creating a model with all fields."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_model_data
        )

        # Act
        result = await llm_models_service.create_llm_model(
            provider_name="openai",
            model_name="gpt-4",
            deployment_name="gpt-4-deployment",
            api_key_vault_identifier="vault-key-123",
            api_endpoint_url="https://api.openai.com/v1",
            model_version="0613",
            maximum_tokens=8192,
            tokens_per_minute_limit=90000,
            requests_per_minute_limit=3500,
            is_active_status=True,
            temperature_value=1.0,
            random_seed=42,
            geographic_region="us-east-1",
        )

        # Assert
        assert result == sample_model_data
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_llm_model_minimal(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test creating a model with only required fields."""
        # Arrange
        minimal_data = {
            "model_id": uuid4(),
            "provider": "openai",
            "model_name": "gpt-4",
            "deployment_name": None,
            "api_key_vault_id": None,
            "api_endpoint": None,
            "model_version": None,
            "max_tokens": None,
            "tokens_per_minute_limit": None,
            "requests_per_minute_limit": None,
            "is_active": True,
            "temperature": None,
            "seed": None,
            "region": None,
            "total_requests": 0,
            "total_tokens_processed": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_used_at": None,
        }
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, minimal_data
        )

        # Act
        result = await llm_models_service.create_llm_model(
            provider_name="openai",
            model_name="gpt-4",
        )

        # Assert
        assert result == minimal_data
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_llm_model_duplicate(
        self, llm_models_service, mock_db_manager
    ):
        """Test that creating a duplicate model raises IntegrityError."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(side_effect=IntegrityError("Duplicate key"))

        # Act & Assert
        with pytest.raises(IntegrityError):
            await llm_models_service.create_llm_model(
                provider_name="openai",
                model_name="gpt-4",
            )

    @pytest.mark.asyncio
    async def test_create_llm_model_invalid_provider(
        self, llm_models_service, mock_db_manager
    ):
        """Test that invalid provider raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            await llm_models_service.create_llm_model(
                provider_name="invalid",
                model_name="gpt-4",
            )

    @pytest.mark.asyncio
    async def test_create_llm_model_invalid_temperature(
        self, llm_models_service, mock_db_manager
    ):
        """Test that invalid temperature raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="temperature_value must be between"):
            await llm_models_service.create_llm_model(
                provider_name="openai",
                model_name="gpt-4",
                temperature_value=3.0,
            )

    @pytest.mark.asyncio
    async def test_create_llm_model_database_error(
        self, llm_models_service, mock_db_manager
    ):
        """Test that database errors are properly handled."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Database error"))

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await llm_models_service.create_llm_model(
                provider_name="openai",
                model_name="gpt-4",
            )


class TestLLMModelsServiceReadSingle:
    """Test single model read operations for LLMModelsService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def llm_models_service(self, mock_db_manager):
        """LLMModelsService instance with mocked database manager."""
        return LLMModelsService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for tests."""
        return {
            "model_id": uuid4(),
            "provider": "openai",
            "model_name": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "api_key_vault_id": "vault-key-123",
            "api_endpoint": "https://api.openai.com/v1",
            "model_version": "0613",
            "max_tokens": 8192,
            "tokens_per_minute_limit": 90000,
            "requests_per_minute_limit": 3500,
            "is_active": True,
            "temperature": 1.0,
            "seed": 42,
            "region": "us-east-1",
            "total_requests": 0,
            "total_tokens_processed": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_used_at": None,
        }

    @pytest.mark.asyncio
    async def test_get_llm_model_by_id_found(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test getting model by ID when model exists."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_model_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.get_llm_model_by_id(model_id)

        # Assert
        assert result == sample_model_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_model_by_id_not_found(
        self, llm_models_service, mock_db_manager
    ):
        """Test getting model by ID when model doesn't exist."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, None
        )

        # Act
        result = await llm_models_service.get_llm_model_by_id(uuid4())

        # Assert
        assert result is None
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_model_by_id_invalid_uuid(
        self, llm_models_service, mock_db_manager
    ):
        """Test getting model by ID with invalid UUID."""
        # Act & Assert
        with pytest.raises(ValueError, match="must be a valid UUID"):
            await llm_models_service.get_llm_model_by_id("invalid-uuid")

    @pytest.mark.asyncio
    async def test_get_llm_model_by_name_and_endpoint_found(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test getting model by name and endpoint when model exists."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_model_data
        )

        # Act
        result = await llm_models_service.get_llm_model_by_name_and_endpoint(
            "gpt-4", "https://api.openai.com/v1"
        )

        # Assert
        assert result == sample_model_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_model_by_name_and_endpoint_not_found(
        self, llm_models_service, mock_db_manager
    ):
        """Test getting model by name and endpoint when model doesn't exist."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, None
        )

        # Act
        result = await llm_models_service.get_llm_model_by_name_and_endpoint(
            "gpt-4", "https://api.openai.com/v1"
        )

        # Assert
        assert result is None
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_model_by_id_database_error(
        self, llm_models_service, mock_db_manager
    ):
        """Test that database errors are properly handled."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Database error"))

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await llm_models_service.get_llm_model_by_id(uuid4())


class TestLLMModelsServiceReadMultiple:
    """Test multiple models read operations for LLMModelsService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def llm_models_service(self, mock_db_manager):
        """LLMModelsService instance with mocked database manager."""
        return LLMModelsService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_models_data(self):
        """Sample models data for tests."""
        return [
            {
                "model_id": uuid4(),
                "provider": "openai",
                "model_name": "gpt-4",
                "deployment_name": "gpt-4-deployment",
                "api_key_vault_id": "vault-key-123",
                "api_endpoint": "https://api.openai.com/v1",
                "model_version": "0613",
                "max_tokens": 8192,
                "tokens_per_minute_limit": 90000,
                "requests_per_minute_limit": 3500,
                "is_active": True,
                "temperature": 1.0,
                "seed": 42,
                "region": "us-east-1",
                "total_requests": 0,
                "total_tokens_processed": 0,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "last_used_at": None,
            },
            {
                "model_id": uuid4(),
                "provider": "gemini",
                "model_name": "gemini-pro",
                "deployment_name": "gemini-pro-deployment",
                "api_key_vault_id": "vault-key-456",
                "api_endpoint": "https://generativelanguage.googleapis.com/v1",
                "model_version": "001",
                "max_tokens": 32768,
                "tokens_per_minute_limit": 150000,
                "requests_per_minute_limit": 60,
                "is_active": True,
                "temperature": 0.7,
                "seed": 123,
                "region": "us-central1",
                "total_requests": 0,
                "total_tokens_processed": 0,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "last_used_at": None,
            },
        ]

    @pytest.mark.asyncio
    async def test_get_all_llm_models_no_filters(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting all models without filters."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_models_data
        )

        # Act
        result = await llm_models_service.get_all_llm_models()

        # Assert
        assert result == sample_models_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_llm_models_provider_filter(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting models with provider filter."""
        # Arrange
        openai_models = [
            model for model in sample_models_data if model["provider"] == "openai"
        ]
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, openai_models
        )

        # Act
        result = await llm_models_service.get_all_llm_models(provider_filter="openai")

        # Assert
        assert result == openai_models
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_llm_models_active_status_filter(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting models with active status filter."""
        # Arrange
        active_models = [model for model in sample_models_data if model["is_active"]]
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, active_models
        )

        # Act
        result = await llm_models_service.get_all_llm_models(active_status_filter=True)

        # Assert
        assert result == active_models
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_llm_models_both_filters(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting models with both provider and active status filters."""
        # Arrange
        filtered_models = [
            model
            for model in sample_models_data
            if model["provider"] == "openai" and model["is_active"]
        ]
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, filtered_models
        )

        # Act
        result = await llm_models_service.get_all_llm_models(
            provider_filter="openai", active_status_filter=True
        )

        # Assert
        assert result == filtered_models
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_llm_models_pagination(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting models with pagination."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_models_data
        )

        # Act
        result = await llm_models_service.get_all_llm_models(limit=10, offset=5)

        # Assert
        assert result == sample_models_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_llm_models(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting active models convenience method."""
        # Arrange
        active_models = [model for model in sample_models_data if model["is_active"]]
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, active_models
        )

        # Act
        result = await llm_models_service.get_active_llm_models()

        # Assert
        assert result == active_models
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_models_by_provider(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting models by provider."""
        # Arrange
        openai_models = [
            model for model in sample_models_data if model["provider"] == "openai"
        ]
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, openai_models
        )

        # Act
        result = await llm_models_service.get_llm_models_by_provider("openai")

        # Assert
        assert result == openai_models
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_models_by_name(
        self, llm_models_service, mock_db_manager, sample_models_data
    ):
        """Test getting all deployments for a model name."""
        # Arrange
        gpt4_models = [
            model for model in sample_models_data if model["model_name"] == "gpt-4"
        ]
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, gpt4_models
        )

        # Act
        result = await llm_models_service.get_llm_models_by_name("gpt-4")

        # Assert
        assert result == gpt4_models
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_llm_models_by_provider(
        self, llm_models_service, mock_db_manager
    ):
        """Test counting models by provider."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager,
            (5,),  # Return count as first element of tuple
        )

        # Act
        result = await llm_models_service.count_llm_models_by_provider("openai")

        # Assert
        assert result == 5
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_llm_models_invalid_limit(
        self, llm_models_service, mock_db_manager
    ):
        """Test that invalid limit raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="limit must be positive"):
            await llm_models_service.get_all_llm_models(limit=-1)

    @pytest.mark.asyncio
    async def test_get_all_llm_models_invalid_offset(
        self, llm_models_service, mock_db_manager
    ):
        """Test that invalid offset raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="offset must be non-negative"):
            await llm_models_service.get_all_llm_models(offset=-1)

    @pytest.mark.asyncio
    async def test_get_all_llm_models_limit_exceeds_max(
        self, llm_models_service, mock_db_manager
    ):
        """Test that limit exceeding maximum raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="limit cannot exceed"):
            await llm_models_service.get_all_llm_models(limit=1001)

    @pytest.mark.asyncio
    async def test_get_all_llm_models_invalid_provider_filter(
        self, llm_models_service, mock_db_manager
    ):
        """Test that invalid provider filter raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            await llm_models_service.get_all_llm_models(provider_filter="invalid")

    @pytest.mark.asyncio
    async def test_get_all_llm_models_database_error(
        self, llm_models_service, mock_db_manager
    ):
        """Test that database errors are properly handled."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Database error"))

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await llm_models_service.get_all_llm_models()


class TestLLMModelsServiceUpdate:
    """Test update operations for LLMModelsService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def llm_models_service(self, mock_db_manager):
        """LLMModelsService instance with mocked database manager."""
        return LLMModelsService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for tests."""
        return {
            "model_id": uuid4(),
            "provider": "openai",
            "model_name": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "api_key_vault_id": "vault-key-123",
            "api_endpoint": "https://api.openai.com/v1",
            "model_version": "0613",
            "max_tokens": 8192,
            "tokens_per_minute_limit": 90000,
            "requests_per_minute_limit": 3500,
            "is_active": True,
            "temperature": 1.0,
            "seed": 42,
            "region": "us-east-1",
            "total_requests": 0,
            "total_tokens_processed": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_used_at": None,
        }

    @pytest.mark.asyncio
    async def test_update_llm_model_single_field(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test updating a single field."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["temperature"] = 0.5
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.update_llm_model(
            model_id=model_id, temperature_value=0.5
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_llm_model_multiple_fields(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test updating multiple fields at once."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["temperature"] = 0.7
        updated_data["max_tokens"] = 4096
        updated_data["is_active"] = False
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.update_llm_model(
            model_id=model_id,
            temperature_value=0.7,
            maximum_tokens=4096,
            is_active_status=False,
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_llm_model_all_fields(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test updating all updatable fields."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["provider"] = "gemini"
        updated_data["model_name"] = "gemini-pro"
        updated_data["deployment_name"] = "gemini-pro-deployment"
        updated_data["api_key_vault_id"] = "vault-key-456"
        updated_data["api_endpoint"] = "https://generativelanguage.googleapis.com/v1"
        updated_data["model_version"] = "001"
        updated_data["max_tokens"] = 32768
        updated_data["tokens_per_minute_limit"] = 150000
        updated_data["requests_per_minute_limit"] = 60
        updated_data["is_active"] = False
        updated_data["temperature"] = 0.7
        updated_data["seed"] = 123
        updated_data["region"] = "us-central1"
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.update_llm_model(
            model_id=model_id,
            provider_name="gemini",
            model_name="gemini-pro",
            deployment_name="gemini-pro-deployment",
            api_key_vault_identifier="vault-key-456",
            api_endpoint_url="https://generativelanguage.googleapis.com/v1",
            model_version="001",
            maximum_tokens=32768,
            tokens_per_minute_limit=150000,
            requests_per_minute_limit=60,
            is_active_status=False,
            temperature_value=0.7,
            random_seed=123,
            geographic_region="us-central1",
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_llm_model_no_changes(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test updating with no fields provided (should return current)."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, sample_model_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.update_llm_model(model_id=model_id)

        # Assert
        assert result == sample_model_data
        # Should call get_llm_model_by_id instead of update query
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_llm_model_status(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test updating model status."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["is_active"] = False
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.update_llm_model_status(
            model_id=model_id, is_active_status=False
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_llm_model(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test activate model convenience method."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["is_active"] = True
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.activate_llm_model(model_id=model_id)

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_llm_model(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test deactivate model convenience method."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["is_active"] = False
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.deactivate_llm_model(model_id=model_id)

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_model_usage_statistics(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test atomic increment of usage statistics."""
        # Arrange
        updated_data = sample_model_data.copy()
        updated_data["total_requests"] = 5
        updated_data["total_tokens_processed"] = 1000
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, updated_data
        )
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.update_model_usage_statistics(
            model_id=model_id,
            request_count_increment=5,
            token_count_increment=1000,
        )

        # Assert
        assert result == updated_data
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_llm_model_not_found(
        self, llm_models_service, mock_db_manager
    ):
        """Test updating non-existent model returns None."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(
            mock_db_manager, None
        )

        # Act
        result = await llm_models_service.update_llm_model(
            model_id=uuid4(), temperature_value=0.5
        )

        # Assert
        assert result is None
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_llm_model_invalid_uuid(
        self, llm_models_service, mock_db_manager
    ):
        """Test updating with invalid UUID raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="must be a valid UUID"):
            await llm_models_service.update_llm_model(
                model_id="invalid-uuid", temperature_value=0.5
            )

    @pytest.mark.asyncio
    async def test_update_llm_model_invalid_temperature(
        self, llm_models_service, mock_db_manager
    ):
        """Test updating with invalid temperature raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="temperature_value must be between"):
            await llm_models_service.update_llm_model(
                model_id=uuid4(), temperature_value=3.0
            )

    @pytest.mark.asyncio
    async def test_update_llm_model_database_error(
        self, llm_models_service, mock_db_manager
    ):
        """Test that database errors are properly handled."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Database error"))

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await llm_models_service.update_llm_model(
                model_id=uuid4(), temperature_value=0.5
            )


class TestLLMModelsServiceDelete:
    """Test delete operations for LLMModelsService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_manager = AsyncMock(spec=DatabaseManager)
        return mock_manager

    @pytest.fixture
    def llm_models_service(self, mock_db_manager):
        """LLMModelsService instance with mocked database manager."""
        return LLMModelsService(database_manager=mock_db_manager)

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for tests."""
        return {
            "model_id": uuid4(),
            "provider": "openai",
            "model_name": "gpt-4",
            "deployment_name": "gpt-4-deployment",
            "api_key_vault_id": "vault-key-123",
            "api_endpoint": "https://api.openai.com/v1",
            "model_version": "0613",
            "max_tokens": 8192,
            "tokens_per_minute_limit": 90000,
            "requests_per_minute_limit": 3500,
            "is_active": True,
            "temperature": 1.0,
            "seed": 42,
            "region": "us-east-1",
            "total_requests": 0,
            "total_tokens_processed": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_used_at": None,
        }

    @pytest.mark.asyncio
    async def test_delete_llm_model_success(
        self, llm_models_service, mock_db_manager, sample_model_data
    ):
        """Test successfully deleting an existing model."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 1  # Simulate successful deletion
        model_id = sample_model_data["model_id"]

        # Act
        result = await llm_models_service.delete_llm_model(model_id)

        # Assert
        assert result is True
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_llm_model_not_found(
        self, llm_models_service, mock_db_manager
    ):
        """Test deleting non-existent model returns False."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 0  # Simulate no rows affected

        # Act
        result = await llm_models_service.delete_llm_model(uuid4())

        # Assert
        assert result is False
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_llm_model_invalid_uuid(
        self, llm_models_service, mock_db_manager
    ):
        """Test deleting with invalid UUID raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="must be a valid UUID"):
            await llm_models_service.delete_llm_model("invalid-uuid")

    @pytest.mark.asyncio
    async def test_delete_llm_models_by_provider_success(
        self, llm_models_service, mock_db_manager
    ):
        """Test successfully deleting models by provider."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.rowcount = 3  # Simulate 3 models deleted

        # Act
        result = await llm_models_service.delete_llm_models_by_provider("openai")

        # Assert
        assert result == 3
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_llm_models_by_provider_database_error(
        self, llm_models_service, mock_db_manager
    ):
        """Test that database errors are properly handled."""
        # Arrange
        mock_connection, mock_cursor = setup_mock_database_connection(mock_db_manager)
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Database error"))

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await llm_models_service.delete_llm_models_by_provider("openai")


# Run with: pytest tests/test_psql_db_services/test_llm_models_service.py -v
