"""
Unit tests for TokenAllocationService

Tests cover:
- CRUD operations for token allocations
- Load balancing and least-loaded endpoint selection
- Allocation lifecycle management
- Usage analytics and reporting
- Business logic for token acquisition and retry
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.persistence.llm_token_allocations import (
    LLMTokenAllocationPersistence,
    get_token_allocation_repository,
)


class TestTokenAllocationServiceValidation:
    """Test validation methods and constructor"""

    def test_constructor(self):
        """Test service initialization"""
        service = LLMTokenAllocationPersistence()
        assert service is not None
        assert hasattr(service, "VALID_ALLOCATION_STATUSES")
        assert hasattr(service, "DEFAULT_ALLOCATION_STATUS")

    def test_constructor_with_db_manager(self):
        """Test service initialization with database manager"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)
        assert service is not None
        assert service.database_manager == mock_db_manager

    @pytest.mark.parametrize(
        "valid_status",
        ["ACQUIRED", "WAITING", "PAUSED", "RELEASED", "EXPIRED", "FAILED"],
    )
    def test_validate_allocation_status_valid(self, valid_status):
        """Test valid allocation statuses"""
        service = LLMTokenAllocationPersistence()
        # Should not raise any exception
        service.validate_allocation_status(valid_status)

    def test_validate_allocation_status_invalid(self):
        """Test invalid allocation status raises ValueError"""
        service = LLMTokenAllocationPersistence()
        with pytest.raises(ValueError, match="Invalid allocation status"):
            service.validate_allocation_status("INVALID_STATUS")


class TestTokenAllocationServiceCreate:
    """Test CREATE operations"""

    @pytest.fixture
    def sample_allocation_data(self):
        return {
            "token_request_id": "req_123",
            "user_id": uuid4(),
            "llm_model_name": "gpt-4",
            "token_count": 1000,
            "allocation_status": "ACQUIRED",
            "allocated_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=1),
        }

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.asyncio
    async def test_create_token_allocation_success_minimal(
        self, sample_allocation_data
    ):
        """Test creating allocation with minimal required parameters"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, sample_allocation_data
        )

        # Call method
        result = await service.create_token_allocation(
            token_request_identifier="req_123",
            user_id=sample_allocation_data["user_id"],
            llm_provider="openai",
            llm_model_name="gpt-4",
            token_count=1000,
            api_endpoint_url="https://api.test.com/v1",  # Add required field
        )

        # Assertions
        assert result == sample_allocation_data
        mock_session.execute.assert_called_once()
        # Check that SQL query was called (TextClause object)
        call_args = mock_session.execute.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_create_token_allocation_success_full(self, sample_allocation_data):
        """Test creating allocation with all optional parameters"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, sample_allocation_data
        )

        # Call method with all parameters
        result = await service.create_token_allocation(
            token_request_identifier="req_123",
            user_id=sample_allocation_data["user_id"],
            llm_provider="openai",
            llm_model_name="gpt-4",
            token_count=1000,
            allocation_status="ACQUIRED",
            allocation_timestamp=datetime.now(),
            expiration_timestamp=datetime.now() + timedelta(hours=1),
            deployment_name="deployment-1",
            cloud_provider_name="azure",
            api_endpoint_url="https://api.openai.com",
            deployment_region="us-east-1",
            request_metadata={"key": "value"},
            temperature=0.7,
            top_p=0.9,
            seed=42,
        )

        # Assertions
        assert result == sample_allocation_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_token_allocation_validation_errors(self):
        """Test validation errors for create allocation"""
        service = LLMTokenAllocationPersistence()

        # Test empty token_request_identifier
        with pytest.raises(ValueError, match="must be a non-empty string"):
            await service.create_token_allocation(
                token_request_identifier="",
                user_id=uuid4(),
                llm_provider="openai",
                llm_model_name="gpt-4",
                token_count=1000,
                api_endpoint_url="https://api.test.com/v1",  # Add required field
            )

        # Test negative token_count
        with pytest.raises(ValueError, match="must be positive"):
            await service.create_token_allocation(
                token_request_identifier="req_123",
                user_id=uuid4(),
                llm_provider="openai",
                llm_model_name="gpt-4",
                token_count=-100,
                api_endpoint_url="https://api.test.com/v1",  # Add required field
            )

        # Test invalid allocation_status
        with pytest.raises(ValueError, match="Invalid allocation status"):
            await service.create_token_allocation(
                token_request_identifier="req_123",
                user_id=uuid4(),
                llm_provider="openai",
                llm_model_name="gpt-4",
                token_count=1000,
                api_endpoint_url="https://api.test.com/v1",  # Add required field
                allocation_status="INVALID",
            )

    @pytest.mark.asyncio
    async def test_create_token_allocation_creation_failure(self):
        """Test RuntimeError when creation fails"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return None (creation failure)
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, None
        )

        # Call method
        with pytest.raises(RuntimeError, match="Failed to create allocation record"):
            await service.create_token_allocation(
                token_request_identifier="req_123",
                user_id=uuid4(),
                llm_provider="openai",
                llm_model_name="gpt-4",
                token_count=1000,
                api_endpoint_url="https://api.test.com/v1",  # Add required field
            )

    @pytest.mark.asyncio
    async def test_create_token_allocation_database_error(self):
        """Test database error handling"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to raise exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=Exception("Database error"))

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())

        # Call method
        with pytest.raises(
            Exception,
            match="(Database error|TokenAllocationService.*missing.*api_endpoint_url)",
        ):
            await service.create_token_allocation(
                token_request_identifier="req_123",
                user_id=uuid4(),
                llm_provider="openai",
                llm_model_name="gpt-4",
                token_count=1000,
                api_endpoint_url="https://api.test.com/v1",  # Add required field
            )

    @pytest.mark.asyncio
    async def test_create_token_allocation_with_capacity_check_uses_locked_insert(
        self, sample_allocation_data
    ):
        """DB fallback allocation should check capacity and insert atomically."""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)
        mock_session, _ = self.setup_mock_session_for_allocation(
            mock_db_manager, sample_allocation_data
        )

        result = await service.create_token_allocation_with_capacity_check(
            token_request_identifier="req_123",
            user_id=sample_allocation_data["user_id"],
            llm_provider="openai",
            llm_model_name="gpt-4",
            token_count=1000,
            api_endpoint_url="https://api.test.com/v1",
            expiration_timestamp=datetime.now() + timedelta(hours=1),
        )

        executed_query = str(mock_session.execute.call_args.args[0])
        assert result == sample_allocation_data
        assert "chosen_deployment" in executed_query
        assert "current_load" in executed_query
        assert "FOR UPDATE" in executed_query
        assert "ELSE ''" not in executed_query
        assert "COALESCE(:deployment_name, deployment_name)" in executed_query
        assert "COALESCE(:cloud_provider, cloud_provider)" in executed_query
        assert "COALESCE(:temperature, temperature)" in executed_query

    @pytest.mark.asyncio
    async def test_create_token_allocation_with_metadata(self, sample_allocation_data):
        """Test creating allocation with request metadata"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, sample_allocation_data
        )

        # Call method with metadata
        metadata = {"user_agent": "test", "ip": "127.0.0.1"}
        result = await service.create_token_allocation(
            token_request_identifier="req_123",
            user_id=sample_allocation_data["user_id"],
            llm_provider="openai",
            llm_model_name="gpt-4",
            token_count=1000,
            api_endpoint_url="https://api.test.com/v1",  # Add required field
            request_metadata=metadata,
        )

        # Assertions
        assert result == sample_allocation_data
        # Verify the method was called (metadata testing is covered by the method working)
        mock_session.execute.assert_called_once()


class TestTokenAllocationServiceRead:
    """Test READ operations"""

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.asyncio
    async def test_get_allocation_by_request_id_found(self):
        """Test getting allocation by request ID when found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        allocation_data = {
            "token_request_id": "req_123",
            "user_id": uuid4(),
            "model_name": "gpt-4",
            "token_count": 1000,
            "allocation_status": "ACQUIRED",
        }

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, allocation_data
        )

        # Call method
        result = await service.get_allocation_by_request_id("req_123")

        # Assertions
        assert result == allocation_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_allocation_by_request_id_not_found(self):
        """Test getting allocation by request ID when not found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return None
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, None
        )

        # Call method
        result = await service.get_allocation_by_request_id("req_123")

        # Assertions
        assert result is None

    @pytest.mark.asyncio
    async def test_get_allocation_by_request_id_validation_error(self):
        """Test validation error for empty request ID"""
        service = LLMTokenAllocationPersistence()

        with pytest.raises(ValueError, match="must be a non-empty string"):
            await service.get_allocation_by_request_id("")

    @pytest.mark.asyncio
    async def test_get_total_allocated_tokens_by_model_default_statuses(self):
        """Test getting total allocated tokens with default statuses"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_data = [
            {
                "model_name": "gpt-4",
                "api_endpoint": "https://api.openai.com",
                "total_tokens": 5000,
                "allocation_count": 5,
            }
        ]

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, mock_data
        )

        # Call method
        result = await service.get_total_allocated_tokens_by_model("gpt-4")

        # Assertions
        assert result == mock_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_total_allocated_tokens_by_model_custom_statuses(self):
        """Test getting total allocated tokens with custom statuses"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_data = [
            {
                "model_name": "gpt-4",
                "api_endpoint": "https://api.openai.com",
                "total_tokens": 3000,
                "allocation_count": 3,
            }
        ]

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, mock_data
        )

        # Call method with custom statuses
        result = await service.get_total_allocated_tokens_by_model(
            "gpt-4", included_statuses=["ACQUIRED", "PAUSED", "WAITING"]
        )

        # Assertions
        assert result == mock_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_total_allocated_tokens_for_endpoint_success(self):
        """Test getting total tokens for specific endpoint"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, 5000
        )

        # Call method
        result = await service.get_total_allocated_tokens_for_endpoint(
            "gpt-4", "https://api.openai.com"
        )

        # Assertions
        assert result == 5000
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_total_allocated_tokens_for_endpoint_zero(self):
        """Test getting total tokens when none found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return None (no data)
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, None
        )

        # Call method
        result = await service.get_total_allocated_tokens_for_endpoint(
            "gpt-4", "https://api.openai.com"
        )

        # Assertions
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_user_allocations_with_status_filter(self):
        """Test getting user allocations with status filter"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_data = [
            {
                "token_request_id": "req_1",
                "user_id": uuid4(),
                "allocation_status": "ACQUIRED",
            }
        ]

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, mock_data
        )

        # Call method with status filter
        result = await service.get_user_allocations(
            uuid4(), status_filter=["ACQUIRED"], limit=50
        )

        # Assertions
        assert result == mock_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_allocations_without_status_filter(self):
        """Test getting user allocations without status filter"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_data = [
            {
                "token_request_id": "req_1",
                "user_id": uuid4(),
                "allocation_status": "ACQUIRED",
            }
        ]

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, mock_data
        )

        # Call method without status filter
        result = await service.get_user_allocations(uuid4(), limit=50)

        # Assertions
        assert result == mock_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_allocations_count_by_model(self):
        """Test getting active allocations count for model"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, 5
        )

        # Call method
        result = await service.get_active_allocations_count_by_model("gpt-4")

        # Assertions
        assert result == 5
        mock_session.execute.assert_called_once()


class TestTokenAllocationServiceUpdate:
    """Test UPDATE operations"""

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.asyncio
    async def test_update_allocation_status_basic(self):
        """Test updating allocation status with basic parameters"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        updated_data = {"token_request_id": "req_123", "allocation_status": "RELEASED"}

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, updated_data
        )

        # Call method
        result = await service.update_allocation_status("req_123", "RELEASED")

        # Assertions
        assert result == updated_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_allocation_status_with_all_fields(self):
        """Test updating allocation status with all optional fields"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        updated_data = {
            "token_request_id": "req_123",
            "allocation_status": "RELEASED",
            "api_endpoint": "https://api.openai.com",
            "region": "us-east-1",
            "expires_at": datetime.now() + timedelta(hours=1),
            "completed_at": datetime.now(),
            "latency_ms": 150,
        }

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, updated_data
        )

        # Call method with all fields
        result = await service.update_allocation_status(
            "req_123",
            "RELEASED",
            api_endpoint="https://api.openai.com",
            deployment_region="us-east-1",
            expires_at=datetime.now() + timedelta(hours=1),
            completed_at=datetime.now(),
            latency_ms=150,
        )

        # Assertions
        assert result == updated_data
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_allocation_status_not_found(self):
        """Test updating allocation status when not found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return None
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, None
        )

        # Call method
        result = await service.update_allocation_status("req_123", "RELEASED")

        # Assertions
        assert result is None

    @pytest.mark.asyncio
    async def test_transition_waiting_to_acquired_success(self):
        """Test successful transition from WAITING to ACQUIRED"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        updated_data = {
            "token_request_id": "req_123",
            "allocation_status": "ACQUIRED",
            "api_endpoint": "https://api.openai.com",
            "region": "us-east-1",
        }

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, updated_data
        )

        # Call method
        result = await service.transition_waiting_to_acquired(
            "req_123",
            "https://api.openai.com",
            "us-east-1",
            datetime.now() + timedelta(hours=1),
        )

        # Assertions
        assert result == updated_data
        mock_session.execute.assert_called_once()
        executed_query = str(mock_session.execute.call_args.args[0])
        assert "current_load" in executed_query
        assert "FOR UPDATE" in executed_query
        assert "current_load.total_tokens + waiting.token_count" in executed_query
        assert "deployment_name = eligible.deployment_name" in executed_query
        assert "cloud_provider = eligible.cloud_provider" in executed_query
        assert "temperature = eligible.temperature" in executed_query
        assert "top_p = eligible.top_p" in executed_query
        assert "seed = eligible.random_seed" in executed_query

    @pytest.mark.asyncio
    async def test_transition_waiting_to_acquired_not_waiting(self):
        """Test transition fails when not in WAITING status"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return None (no rows updated)
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, None
        )

        # Call method
        result = await service.transition_waiting_to_acquired(
            "req_123",
            "https://api.openai.com",
            "us-east-1",
            datetime.now() + timedelta(hours=1),
        )

        # Assertions
        assert result is None


class TestTokenAllocationServiceDelete:
    """Test DELETE operations"""

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.asyncio
    async def test_delete_allocation_success(self):
        """Test successful deletion of allocation"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock with rowcount > 0
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, rowcount=1
        )

        # Call method
        result = await service.delete_allocation("req_123")

        # Assertions
        assert result is True
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_allocation_not_found(self):
        """Test deletion when allocation not found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock with rowcount = 0
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, rowcount=0
        )

        # Call method
        result = await service.delete_allocation("req_123")

        # Assertions
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_expired_allocations_found(self):
        """Test deleting expired allocations when found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock with rowcount > 0
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, rowcount=5
        )

        # Call method
        result = await service.delete_expired_allocations()

        # Assertions
        assert result == 5
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_expired_allocations_none(self):
        """Test deleting expired allocations when none found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock with rowcount = 0
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, rowcount=0
        )

        # Call method
        result = await service.delete_expired_allocations()

        # Assertions
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_allocations_by_user_with_status(self):
        """Test deleting user allocations with status filter"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock with rowcount > 0
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, rowcount=3
        )

        # Call method with status filter
        result = await service.delete_allocations_by_user(uuid4(), "ACQUIRED")

        # Assertions
        assert result == 3
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_allocations_by_user_without_status(self):
        """Test deleting user allocations without status filter"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock with rowcount > 0
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, rowcount=2
        )

        # Call method without status filter
        result = await service.delete_allocations_by_user(uuid4())

        # Assertions
        assert result == 2
        mock_session.execute.assert_called_once()


class TestTokenAllocationServiceBusinessLogic:
    """Test business logic operations"""

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.skip(
        reason="acquire_tokens moved to TokenAcquisitionService; see app/services/token_acquisition_service.py"
    )
    @pytest.mark.asyncio
    async def test_acquire_tokens_immediate_allocation(self):
        """Test immediate token allocation when under limit"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Mock get_least_loaded_deployment to return low token count
        with patch.object(
            service, "get_least_loaded_deployment"
        ) as mock_get_deployment:
            mock_get_deployment.return_value = (
                1000,
                {
                    "model_id": uuid4(),
                    "max_tokens": 100000,
                    "max_token_lock_time_secs": 70,
                    "api_version": "v1",
                    "deployment_name": "deployment-1",
                    "api_base": "https://api.openai.com",
                    "region": "us-east-1",
                    "api_keyv_id": "keyv-123",
                    "temperature": 0.7,
                    "seed": 42,
                },
            )

            # Mock create_token_allocation
            with patch.object(service, "create_token_allocation") as mock_create:
                mock_create.return_value = {
                    "token_request_id": "req_123",
                    "allocation_status": "ACQUIRED",
                }

                # Call method
                result = await service.acquire_tokens(uuid4(), "openai", "gpt-4", 5000)

                # Assertions
                assert result["allocation_status"] == "ACQUIRED"
                mock_create.assert_called_once()

    @pytest.mark.skip(
        reason="acquire_tokens moved to TokenAcquisitionService; see app/services/token_acquisition_service.py"
    )
    @pytest.mark.asyncio
    async def test_acquire_tokens_waiting_allocation(self):
        """Test waiting allocation when at limit"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Mock get_least_loaded_deployment to return high token count
        with patch.object(
            service, "get_least_loaded_deployment"
        ) as mock_get_deployment:
            mock_get_deployment.return_value = (
                95000,
                {
                    "model_id": uuid4(),
                    "max_tokens": 100000,
                    "max_token_lock_time_secs": 70,
                },
            )

            # Mock create_token_allocation
            with patch.object(service, "create_token_allocation") as mock_create:
                mock_create.return_value = {
                    "token_request_id": "req_123",
                    "allocation_status": "WAITING",
                }

                # Call method
                result = await service.acquire_tokens(uuid4(), "openai", "gpt-4", 5000)

                # Assertions
                assert result["allocation_status"] == "WAITING"
                mock_create.assert_called_once()

    @pytest.mark.skip(
        reason="acquire_tokens moved to TokenAcquisitionService; see app/services/token_acquisition_service.py"
    )
    @pytest.mark.asyncio
    async def test_acquire_tokens_invalid_token_count(self):
        """Test validation error for invalid token count"""
        service = LLMTokenAllocationPersistence()

        with pytest.raises(ValueError, match="must be positive"):
            await service.acquire_tokens(uuid4(), "openai", "gpt-4", 0)

        with pytest.raises(ValueError, match="must be positive"):
            await service.acquire_tokens(uuid4(), "openai", "gpt-4", -100)

    @pytest.mark.skip(
        reason="acquire_tokens moved to TokenAcquisitionService; see app/services/token_acquisition_service.py"
    )
    @pytest.mark.asyncio
    async def test_acquire_tokens_exceeds_limit(self):
        """Test error when token count exceeds limit"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Mock get_least_loaded_deployment
        with patch.object(
            service, "get_least_loaded_deployment"
        ) as mock_get_deployment:
            mock_get_deployment.return_value = (
                50000,
                {"max_tokens": 100000, "region": "us-east-1"},
            )

            # Call method with token count exceeding single request limit (150000 > 100000)
            result = await service.acquire_tokens(uuid4(), "openai", "gpt-4", 150000)

            # Assertions - should return error when single request exceeds limit
            assert "error" in result
            assert "max limit exceeded" in result["error"]

    @pytest.mark.skip(
        reason="acquire_tokens moved to TokenAcquisitionService; see app/services/token_acquisition_service.py"
    )
    @pytest.mark.asyncio
    async def test_acquire_tokens_no_deployments(self):
        """Test error when no deployments found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Mock get_least_loaded_deployment to raise ValueError
        with patch.object(
            service, "get_least_loaded_deployment"
        ) as mock_get_deployment:
            mock_get_deployment.side_effect = ValueError("No deployments found")

            # Call method
            with pytest.raises(ValueError, match="No deployments found"):
                await service.acquire_tokens(uuid4(), "openai", "gpt-4", 1000)

    @pytest.mark.skip(
        reason="retry_acquire_tokens moved to TokenRetryService; see app/services/token_retry_service.py"
    )
    @pytest.mark.asyncio
    async def test_retry_acquire_tokens_success(self):
        """Test successful retry of token acquisition"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        allocation_data = {
            "token_request_id": "req_123",
            "allocation_status": "WAITING",
            "llm_model_name": "gpt-4",
            "token_count": 1000,
        }

        # Mock get_allocation_by_request_id
        with patch.object(
            service, "get_allocation_by_request_id"
        ) as mock_get_allocation:
            mock_get_allocation.return_value = allocation_data

            # Mock get_least_loaded_deployment
            with patch.object(
                service, "get_least_loaded_deployment"
            ) as mock_get_deployment:
                mock_get_deployment.return_value = (
                    5000,
                    {
                        "max_tokens": 100000,
                        "max_token_lock_time_secs": 70,
                        "api_base": "https://api.openai.com",
                        "region": "us-east-1",
                        "api_version": "v1",
                        "api_keyv_id": "keyv-123",
                        "temperature": 0.7,
                        "seed": 42,
                    },
                )

                # Mock transition_waiting_to_acquired
                with patch.object(
                    service, "transition_waiting_to_acquired"
                ) as mock_transition:
                    mock_transition.return_value = {
                        "token_request_id": "req_123",
                        "allocation_status": "ACQUIRED",
                    }

                    # Call method
                    result = await service.retry_acquire_tokens("req_123")

                    # Assertions
                    assert result["allocation_status"] == "ACQUIRED"
                    mock_transition.assert_called_once()

    @pytest.mark.skip(
        reason="retry_acquire_tokens moved to TokenRetryService; see app/services/token_retry_service.py"
    )
    @pytest.mark.asyncio
    async def test_retry_acquire_tokens_not_found(self):
        """Test retry when allocation not found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Mock get_allocation_by_request_id to return None
        with patch.object(
            service, "get_allocation_by_request_id"
        ) as mock_get_allocation:
            mock_get_allocation.return_value = None

            # Call method
            result = await service.retry_acquire_tokens("req_123")

            # Assertions
            assert "error" in result
            assert result["error"] == "Invalid token_request_id = req_123"

    @pytest.mark.skip(
        reason="retry_acquire_tokens moved to TokenRetryService; see app/services/token_retry_service.py"
    )
    @pytest.mark.asyncio
    async def test_retry_acquire_tokens_not_waiting(self):
        """Test retry when allocation not in WAITING status"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        allocation_data = {
            "token_request_id": "req_123",
            "allocation_status": "ACQUIRED",  # Not WAITING
            "llm_model_name": "gpt-4",
            "token_count": 1000,
        }

        # Mock get_allocation_by_request_id
        with patch.object(
            service, "get_allocation_by_request_id"
        ) as mock_get_allocation:
            mock_get_allocation.return_value = allocation_data

            # Call method
            result = await service.retry_acquire_tokens("req_123")

            # Assertions
            assert "error" in result
            assert "not in WAITING status" in result["error"]

    @pytest.mark.skip(
        reason="retry_acquire_tokens moved to TokenRetryService; see app/services/token_retry_service.py"
    )
    @pytest.mark.asyncio
    async def test_retry_acquire_tokens_still_waiting(self):
        """Test retry when still over limit"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        allocation_data = {
            "token_request_id": "req_123",
            "allocation_status": "WAITING",
            "llm_model_name": "gpt-4",
            "token_count": 1000,
        }

        # Mock get_allocation_by_request_id
        with patch.object(
            service, "get_allocation_by_request_id"
        ) as mock_get_allocation:
            mock_get_allocation.return_value = allocation_data

            # Mock get_least_loaded_deployment to return high token count
            with patch.object(
                service, "get_least_loaded_deployment"
            ) as mock_get_deployment:
                mock_get_deployment.return_value = (95000, {"max_tokens": 100000})

                # Mock transition_waiting_to_acquired to return None (still waiting)
                with patch.object(
                    service, "transition_waiting_to_acquired"
                ) as mock_transition:
                    mock_transition.return_value = None

                    # Call method
                    result = await service.retry_acquire_tokens("req_123")

                    # Assertions - when transition fails, returns error
                    assert "error" in result
                    assert "Failed to acquire tokens" in result["error"]

    def _make_execute_result(self, data=None, scalar=None):
        """Build a minimal mock SQLAlchemy result for a single execute() call."""
        r = MagicMock()
        r.mappings.return_value = r
        r.one_or_none.return_value = data
        r.scalar_one_or_none.return_value = scalar
        return r

    def _make_mock_session(self, mock_db_manager, side_effects):
        """
        Wire mock_db_manager.get_session to yield a session whose execute()
        returns each item in side_effects on successive calls.
        """
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=side_effects)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session

    @pytest.mark.asyncio
    async def test_pause_deployment_success(self):
        """
        pause_deployment() makes exactly three execute() calls within one session:
          1. SELECT … FOR UPDATE  (locked deployment lookup)
          2. SELECT 1 … PAUSED   (existing-pause check → None means no existing pause)
          3. INSERT … RETURNING * (create the PAUSED capacity-blocker)
        All three run in the same transaction; there is no cross-session TOCTOU gap.
        """
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        model_config = {
            "max_tokens": 100000,
            "cloud_provider": "Azure",
            "deployment_region": "us-east-1",
            "deployment_name": "deployment-1",
        }
        created_allocation = {
            "token_request_id": "pause_abc123",
            "user_id": str(uuid4()),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4",
            "deployment_name": "deployment-1",
            "cloud_provider": "Azure",
            "api_endpoint_url": "https://api.openai.com",
            "deployment_region": "us-east-1",
            "token_count": 100000,
            "allocation_status": "PAUSED",
            "allocated_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=30),
            "request_context": None,
            "temperature": None,
            "top_p": None,
            "seed": None,
        }

        mock_session = self._make_mock_session(
            mock_db_manager,
            side_effects=[
                self._make_execute_result(data=model_config),  # FOR UPDATE lock
                self._make_execute_result(data=None, scalar=None),  # no existing pause
                self._make_execute_result(data=created_allocation),  # INSERT
            ],
        )

        result = await service.pause_deployment(
            uuid4(), "openai", "gpt-4", "https://api.openai.com"
        )

        assert result["allocation_status"] == "PAUSED"
        assert mock_session.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_pause_deployment_already_paused(self):
        """
        When an active PAUSED allocation is found (step 2), pause_deployment()
        returns the ALREADY_PAUSED sentinel without inserting a duplicate row.
        """
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        model_config = {
            "max_tokens": 100000,
            "cloud_provider": None,
            "deployment_region": "us-east-1",
            "deployment_name": "deployment-1",
        }

        mock_session = self._make_mock_session(
            mock_db_manager,
            side_effects=[
                self._make_execute_result(data=model_config),  # FOR UPDATE lock
                self._make_execute_result(data=None, scalar=1),  # existing pause found
            ],
        )

        result = await service.pause_deployment(
            uuid4(), "openai", "gpt-4", "https://api.openai.com"
        )

        assert result["alloc_status"] == "ALREADY_PAUSED"
        assert mock_session.execute.call_count == 2  # INSERT must NOT be reached

    @pytest.mark.asyncio
    async def test_pause_deployment_not_found(self):
        """
        When the deployment row does not exist the method returns the NOT_FOUND
        sentinel immediately after the first execute() (the FOR UPDATE lock).
        """
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_session = self._make_mock_session(
            mock_db_manager,
            side_effects=[
                self._make_execute_result(data=None),  # deployment not found
            ],
        )

        result = await service.pause_deployment(
            uuid4(), "openai", "gpt-4", "https://api.openai.com"
        )

        assert result["alloc_status"] == "NOT_FOUND"
        assert "Deployment not found" in result["reason"]
        assert mock_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_pause_deployment_invalid_duration_raises(self):
        """pause_deployment() rejects pause_duration_minutes <= 0 before any DB call."""
        service = LLMTokenAllocationPersistence()

        with pytest.raises(ValueError, match="must be positive"):
            await service.pause_deployment(
                uuid4(),
                "openai",
                "gpt-4",
                "https://api.openai.com",
                pause_duration_minutes=0,
            )


class TestTokenAllocationServiceLoadBalancing:
    """Test load balancing operations"""

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.asyncio
    async def test_get_least_loaded_deployment_no_allocations(self):
        """Test getting least loaded deployment when no allocations exist"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        model_deployments = [
            {
                "model_id": uuid4(),
                "model_name": "gpt-4",
                "api_base": "https://api.openai.com",
                "region": "us-east-1",
                "max_tokens": 100000,
                "is_active": True,
            }
        ]

        # Setup mock for deployments query
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, model_deployments
        )

        # Mock second query (allocations) to return empty list
        def mock_execute_side_effect(query, params=None):
            # Convert TextClause to string for comparison
            query_str = str(query)
            if "llm_models" in query_str:
                return mock_result
            else:  # allocations query
                empty_result = MagicMock()
                empty_result.mappings.return_value = empty_result
                empty_result.all.return_value = []
                return empty_result

        mock_session.execute = AsyncMock(side_effect=mock_execute_side_effect)

        # Call method
        total_tokens, chosen_config = await service.get_least_loaded_deployment(
            "openai", "gpt-4"
        )

        # Assertions
        assert total_tokens == 0
        assert chosen_config["model_name"] == "gpt-4"
        for execute_call in mock_session.execute.await_args_list:
            assert "llm_provider = :llm_provider" in str(execute_call.args[0])
            assert execute_call.args[1]["llm_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_get_least_loaded_deployment_unused_deployment(self):
        """Test choosing unused deployment"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        model_deployments = [
            {
                "model_id": uuid4(),
                "model_name": "gpt-4",
                "api_endpoint_url": "https://api.openai.com",
                "region": "us-east-1",
                "max_tokens": 100000,
                "is_active": True,
            },
            {
                "model_id": uuid4(),
                "model_name": "gpt-4",
                "api_endpoint_url": "https://api.azure.com",
                "region": "us-west-1",
                "max_tokens": 100000,
                "is_active": True,
            },
        ]

        allocation_results = [
            {"api_endpoint_url": "https://api.openai.com", "total_tokens": 5000}
        ]

        # Setup mock for both queries
        def mock_execute_side_effect(query, params=None):
            # Convert TextClause to string for comparison
            query_str = str(query)
            if "llm_models" in query_str:
                result = MagicMock()
                result.mappings.return_value = result
                result.all.return_value = model_deployments
                return result
            else:  # allocations query
                result = MagicMock()
                result.mappings.return_value = result
                result.all.return_value = allocation_results
                return result

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=mock_execute_side_effect)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())

        # Call method
        total_tokens, chosen_config = await service.get_least_loaded_deployment(
            "openai", "gpt-4"
        )

        # Assertions
        assert total_tokens == 0  # Unused deployment
        assert chosen_config["api_endpoint_url"] == "https://api.azure.com"

    @pytest.mark.asyncio
    async def test_get_least_loaded_deployment_least_loaded(self):
        """Test choosing least loaded deployment"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        model_deployments = [
            {
                "model_id": uuid4(),
                "model_name": "gpt-4",
                "api_endpoint_url": "https://api.openai.com",
                "region": "us-east-1",
                "max_tokens": 100000,
                "is_active": True,
            },
            {
                "model_id": uuid4(),
                "model_name": "gpt-4",
                "api_endpoint_url": "https://api.azure.com",
                "region": "us-west-1",
                "max_tokens": 100000,
                "is_active": True,
            },
        ]

        allocation_results = [
            {"api_endpoint_url": "https://api.azure.com", "total_tokens": 5000},
            {"api_endpoint_url": "https://api.openai.com", "total_tokens": 10000},
        ]

        # Setup mock for both queries
        def mock_execute_side_effect(query, params=None):
            # Convert TextClause to string for comparison
            query_str = str(query)
            if "llm_models" in query_str:
                result = MagicMock()
                result.mappings.return_value = result
                result.all.return_value = model_deployments
                return result
            else:  # allocations query
                result = MagicMock()
                result.mappings.return_value = result
                result.all.return_value = allocation_results
                return result

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=mock_execute_side_effect)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())

        # Call method
        total_tokens, chosen_config = await service.get_least_loaded_deployment(
            "openai", "gpt-4"
        )

        # Assertions
        assert total_tokens == 5000  # Least loaded
        assert chosen_config["api_endpoint_url"] == "https://api.azure.com"

    @pytest.mark.asyncio
    async def test_get_least_loaded_deployment_no_deployments(self):
        """Test error when no deployments found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return empty deployments
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, []
        )

        # Call method
        with pytest.raises(ValueError, match="No model deployments found"):
            await service.get_least_loaded_deployment("openai", "gpt-4")

    @pytest.mark.asyncio
    async def test_get_least_loaded_deployment_no_match(self):
        """Test fallback when no matching deployment found"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        model_deployments = [
            {
                "model_id": uuid4(),
                "model_name": "gpt-4",
                "api_endpoint_url": "https://api.openai.com",
                "region": "us-east-1",
                "max_tokens": 100000,
                "is_active": True,
            }
        ]

        allocation_results = [
            {
                "api_endpoint_url": "https://api.unknown.com",  # No matching deployment
                "total_tokens": 5000,
            }
        ]

        # Setup mock for both queries
        def mock_execute_side_effect(query, params=None):
            # Convert TextClause to string for comparison
            query_str = str(query)
            if "llm_models" in query_str:
                result = MagicMock()
                result.mappings.return_value = result
                result.all.return_value = model_deployments
                return result
            else:  # allocations query
                result = MagicMock()
                result.mappings.return_value = result
                result.all.return_value = allocation_results
                return result

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=mock_execute_side_effect)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())

        # Call method
        total_tokens, chosen_config = await service.get_least_loaded_deployment(
            "openai", "gpt-4"
        )

        # Assertions - when no match found, uses first deployment with 0 tokens
        assert total_tokens == 0
        assert (
            chosen_config["api_endpoint_url"] == "https://api.openai.com"
        )  # Fallback to first


class TestTokenAllocationServiceAnalytics:
    """Test analytics operations"""

    def setup_mock_session_for_allocation(
        self, mock_db_manager, mock_result_data=None, rowcount=1
    ):
        """Helper to set up mock session for token allocation tests"""
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result

        if mock_result_data:
            if isinstance(mock_result_data, list):
                mock_result.all.return_value = mock_result_data
            else:
                mock_result.one_or_none.return_value = mock_result_data
        else:
            mock_result.one_or_none.return_value = None
            mock_result.all.return_value = []

        mock_result.scalar_one_or_none.return_value = (
            mock_result_data if isinstance(mock_result_data, int) else 0
        )
        mock_result.rowcount = rowcount

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session_cm():
            yield mock_session

        mock_db_manager.get_session = MagicMock(return_value=mock_get_session_cm())
        return mock_session, mock_result

    @pytest.mark.asyncio
    async def test_get_allocation_summary_by_model(self):
        """Test getting allocation summary by model"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_data = [
            {
                "allocation_status": "ACQUIRED",
                "count": 5,
                "total_tokens": 5000,
                "avg_tokens": 1000,
            },
            {
                "allocation_status": "WAITING",
                "count": 2,
                "total_tokens": 2000,
                "avg_tokens": 1000,
            },
        ]

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, mock_data
        )

        # Call method
        result = await service.get_allocation_summary_by_model("gpt-4")

        # Assertions
        assert result["llm_model_name"] == "gpt-4"
        assert len(result["by_status"]) == 2
        assert result["by_status"][0]["allocation_status"] == "ACQUIRED"

    @pytest.mark.asyncio
    async def test_get_allocation_summary_by_model_empty(self):
        """Test getting allocation summary when no data"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return empty list
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, []
        )

        # Call method
        result = await service.get_allocation_summary_by_model("gpt-4")

        # Assertions
        assert result["llm_model_name"] == "gpt-4"
        assert result["by_status"] == []

    @pytest.mark.asyncio
    async def test_get_user_token_usage_stats_with_data(self):
        """Test getting user usage stats with data"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        mock_data = {
            "total_requests": 10,
            "total_tokens": 50000,
            "avg_tokens_per_request": 5000,
            "avg_latency_ms": 150.5,
            "completed_requests": 8,
            "failed_requests": 2,
        }

        # Setup mock
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, mock_data
        )

        # Call method
        result = await service.get_user_token_usage_stats(uuid4())

        # Assertions
        assert result == mock_data

    @pytest.mark.asyncio
    async def test_get_user_token_usage_stats_no_data(self):
        """Test getting user usage stats when no data"""
        mock_db_manager = MagicMock()
        service = LLMTokenAllocationPersistence(mock_db_manager)

        # Setup mock to return None
        mock_session, mock_result = self.setup_mock_session_for_allocation(
            mock_db_manager, None
        )

        # Call method
        result = await service.get_user_token_usage_stats(uuid4())

        # Assertions
        assert result == {}


class TestConvenienceFunction:
    """Test convenience factory function"""

    def test_get_token_allocation_repository(self):
        """Test factory function returns instance"""
        # Test with no parameters
        repo = get_token_allocation_repository()
        assert isinstance(repo, LLMTokenAllocationPersistence)

        # Test with database manager
        mock_db_manager = MagicMock()
        repo = get_token_allocation_repository(mock_db_manager)
        assert isinstance(repo, LLMTokenAllocationPersistence)
        assert repo.database_manager == mock_db_manager
