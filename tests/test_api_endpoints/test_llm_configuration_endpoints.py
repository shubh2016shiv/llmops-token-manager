"""
Comprehensive Unit Tests for LLM Configuration Endpoints
=======================================================
Unit tests for LLM model configuration management endpoints covering all CRUD operations
with 100% coverage including success and error scenarios.

Test Coverage:
- Create LLM model operations (6 tests)
- List models by provider operations (6 tests)
- Get LLM model operations (7 tests)
- Update LLM model operations (8 tests)
- Activate LLM model operations (5 tests)
- Deactivate LLM model operations (5 tests)

Total: 37 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.llm_configuration_endpoints import router
from app.models.response_models import LLMModelResponse, LLMModelListResponse


# ============================================================================
# TEST SETUP
# ============================================================================


@pytest.fixture
def app():
    """Create FastAPI app with LLM configuration endpoints router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_llm_model_data():
    """Sample LLM model data for testing."""
    return {
        "provider_name": "openai",
        "llm_model_name": "gpt-4o",
        "deployment_name": "gpt-4o-eastus",
        "api_key_variable_name": "OPENAI_API_KEY_GPT4O",
        "api_endpoint_url": "https://api.openai.com/v1",
        "llm_model_version": "2024-08",
        "max_tokens": 8192,
        "tokens_per_minute_limit": 100000,
        "requests_per_minute_limit": 1000,
        "is_active_status": True,
        "temperature": 0.7,
        "random_seed": 42,
        "deployment_region": "eastus2",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_create_request():
    """Sample LLM model creation request data."""
    return {
        "provider_name": "openai",
        "llm_model_name": "gpt-4o",
        "api_key_variable_name": "OPENAI_API_KEY_GPT4O",
        "llm_model_version": "2024-08",
        "max_tokens": 8192,
        "tokens_per_minute_limit": 100000,
        "requests_per_minute_limit": 1000,
        "deployment_name": "gpt-4o-eastus",
        "api_endpoint_url": "https://api.openai.com/v1",
        "is_active_status": True,
        "temperature": 0.7,
        "random_seed": 42,
        "deployment_region": "eastus2",
    }


@pytest.fixture
def sample_update_request():
    """Sample LLM model update request data."""
    return {
        "max_tokens": 16384,
        "tokens_per_minute_limit": 200000,
        "is_active_status": True,
        "temperature": 0.5,
        "deployment_region": "westus2",
    }


# ============================================================================
# CREATE LLM MODEL TESTS
# ============================================================================


class TestCreateLLMModel:
    """Test cases for create LLM model endpoint."""

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_create_llm_model_success(
        self,
        mock_llm_service,
        client,
        sample_create_request,
        sample_llm_model_data,
    ):
        """Test successful LLM model creation returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.create_llm_model.return_value = sample_llm_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/llm-models/", json=sample_create_request)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["provider_name"] == sample_llm_model_data["provider_name"]
        assert data["llm_model_name"] == sample_llm_model_data["llm_model_name"]
        assert data["llm_model_version"] == sample_llm_model_data["llm_model_version"]
        assert data["max_tokens"] == sample_llm_model_data["max_tokens"]
        assert (
            data["tokens_per_minute_limit"]
            == sample_llm_model_data["tokens_per_minute_limit"]
        )
        assert (
            data["requests_per_minute_limit"]
            == sample_llm_model_data["requests_per_minute_limit"]
        )
        assert data["is_active_status"] == sample_llm_model_data["is_active_status"]

        # Verify service was called correctly
        mock_service_instance.create_llm_model.assert_called_once()
        call_args = mock_service_instance.create_llm_model.call_args
        assert call_args[1]["provider_name"] == sample_create_request["provider_name"]
        assert call_args[1]["llm_model_name"] == sample_create_request["llm_model_name"]
        assert (
            call_args[1]["api_key_variable_name"]
            == sample_create_request["api_key_variable_name"]
        )
        assert (
            call_args[1]["llm_model_version"]
            == sample_create_request["llm_model_version"]
        )
        assert call_args[1]["max_tokens"] == sample_create_request["max_tokens"]
        assert (
            call_args[1]["tokens_per_minute_limit"]
            == sample_create_request["tokens_per_minute_limit"]
        )
        assert (
            call_args[1]["requests_per_minute_limit"]
            == sample_create_request["requests_per_minute_limit"]
        )
        assert (
            call_args[1]["deployment_name"] == sample_create_request["deployment_name"]
        )
        assert (
            call_args[1]["api_endpoint_url"]
            == sample_create_request["api_endpoint_url"]
        )
        assert (
            call_args[1]["is_active_status"]
            == sample_create_request["is_active_status"]
        )
        assert call_args[1]["temperature"] == sample_create_request["temperature"]
        assert call_args[1]["random_seed"] == sample_create_request["random_seed"]
        assert (
            call_args[1]["deployment_region"]
            == sample_create_request["deployment_region"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_create_llm_model_duplicate_model(
        self, mock_llm_service, client, sample_create_request
    ):
        """Test LLM model creation with duplicate model returns 400 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.create_llm_model.side_effect = ValueError(
            "Model 'gpt-4o' for provider 'openai' already exists"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/llm-models/", json=sample_create_request)

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Model 'gpt-4o' for provider 'openai' already exists" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_create_llm_model_invalid_provider(
        self, mock_llm_service, client, sample_create_request
    ):
        """Test LLM model creation with invalid provider returns 400 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.create_llm_model.side_effect = ValueError(
            "Invalid provider name 'invalid_provider'"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/llm-models/", json=sample_create_request)

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid provider name 'invalid_provider'" in data["detail"]

    def test_create_llm_model_invalid_request_body(self, client):
        """Test LLM model creation with invalid request body returns 422 validation error."""
        # Arrange
        invalid_request = {
            "provider_name": "openai",
            # Missing required fields
            "llm_model_name": "",  # Empty model name
            "api_key_variable_name": "OPENAI_API_KEY",
            "max_tokens": -1,  # Invalid max_tokens
            "tokens_per_minute_limit": 100000,
            "requests_per_minute_limit": 1000,
        }

        # Act
        response = client.post("/api/v1/llm-models/", json=invalid_request)

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Should have validation errors for empty model name and negative max_tokens

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_create_llm_model_database_error(
        self, mock_llm_service, client, sample_create_request
    ):
        """Test LLM model creation with database error returns 500 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.create_llm_model.side_effect = Exception(
            "Database connection failed"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/llm-models/", json=sample_create_request)

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert (
            "Failed to create LLM model configuration. Please try again later."
            in data["detail"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_create_llm_model_response_structure(
        self, mock_llm_service, client, sample_create_request, sample_llm_model_data
    ):
        """Test response structure matches LLMModelResponse schema."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.create_llm_model.return_value = sample_llm_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.post("/api/v1/llm-models/", json=sample_create_request)

        # Assert
        assert response.status_code == 201
        data = response.json()

        # Validate all required fields are present
        required_fields = [
            "provider_name",
            "llm_model_name",
            "deployment_name",
            "api_key_variable_name",
            "api_endpoint_url",
            "llm_model_version",
            "max_tokens",
            "tokens_per_minute_limit",
            "requests_per_minute_limit",
            "is_active_status",
            "temperature",
            "random_seed",
            "deployment_region",
            "created_at",
            "updated_at",
        ]
        for field in required_fields:
            assert field in data

        # Validate LLMModelResponse model can be created from response
        llm_model_response = LLMModelResponse(**data)
        assert (
            llm_model_response.provider_name == sample_llm_model_data["provider_name"]
        )
        assert (
            llm_model_response.llm_model_name == sample_llm_model_data["llm_model_name"]
        )


# ============================================================================
# LIST MODELS BY PROVIDER TESTS
# ============================================================================


class TestListLLMModelsByProvider:
    """Test cases for list LLM models by provider endpoint."""

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_list_llm_models_by_provider_success(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful listing of LLM models by provider returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_models_by_provider.return_value = [
            sample_llm_model_data
        ]
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/provider/openai")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1
        assert (
            data["models"][0]["provider_name"] == sample_llm_model_data["provider_name"]
        )
        assert (
            data["models"][0]["llm_model_name"]
            == sample_llm_model_data["llm_model_name"]
        )
        assert data["total_count"] == 1
        assert data["page"] == 1
        assert data["page_size"] == 100
        assert data["has_next"] is False
        assert data["has_previous"] is False

        # Verify service was called correctly
        mock_service_instance.get_llm_models_by_provider.assert_called_once_with(
            provider_name="openai", active_only=None, limit=100, offset=0
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_list_llm_models_by_provider_with_active_filter(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test listing with active_only filter returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_models_by_provider.return_value = [
            sample_llm_model_data
        ]
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/provider/openai?active_only=true")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1

        # Verify service was called with active_only filter
        mock_service_instance.get_llm_models_by_provider.assert_called_once_with(
            provider_name="openai", active_only=True, limit=100, offset=0
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_list_llm_models_by_provider_with_pagination(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test listing with pagination parameters returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_models_by_provider.return_value = [
            sample_llm_model_data
        ]
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/provider/openai?limit=50&offset=10")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 50
        assert data["page"] == 1  # (10 // 50) + 1

        # Verify service was called with pagination parameters
        mock_service_instance.get_llm_models_by_provider.assert_called_once_with(
            provider_name="openai", active_only=None, limit=50, offset=10
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_list_llm_models_by_provider_value_error(self, mock_llm_service, client):
        """Test ValueError from service returns 400 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_models_by_provider.side_effect = ValueError(
            "Invalid provider name"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/provider/invalid")

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid provider name" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_list_llm_models_by_provider_database_error(self, mock_llm_service, client):
        """Test database error returns 500 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_models_by_provider.side_effect = Exception(
            "Database connection failed"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/provider/openai")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve LLM model configurations" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_list_llm_models_by_provider_response_structure(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test response structure matches LLMModelListResponse schema."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_models_by_provider.return_value = [
            sample_llm_model_data
        ]
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/provider/openai")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate all required fields are present
        required_fields = [
            "models",
            "total_count",
            "page",
            "page_size",
            "has_next",
            "has_previous",
        ]
        for field in required_fields:
            assert field in data

        # Validate LLMModelListResponse model can be created from response
        llm_model_list_response = LLMModelListResponse(**data)
        assert len(llm_model_list_response.models) == 1
        assert llm_model_list_response.total_count == 1


# ============================================================================
# GET LLM MODEL TESTS
# ============================================================================


class TestGetLLMModel:
    """Test cases for get LLM model endpoint."""

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_success(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful LLM model retrieval returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.return_value = (
            sample_llm_model_data
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/openai/gpt-4o")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["provider_name"] == sample_llm_model_data["provider_name"]
        assert data["llm_model_name"] == sample_llm_model_data["llm_model_name"]
        assert data["llm_model_version"] == sample_llm_model_data["llm_model_version"]

        # Verify service was called correctly
        mock_service_instance.get_llm_model_by_provider_and_model.assert_called_once_with(
            provider_name="openai", llm_model_name="gpt-4o", llm_model_version=None
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_with_version(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful LLM model retrieval with version parameter."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.return_value = (
            sample_llm_model_data
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/openai/gpt-4o?version=2024-08")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["llm_model_version"] == "2024-08"

        # Verify service was called with version
        mock_service_instance.get_llm_model_by_provider_and_model.assert_called_once_with(
            provider_name="openai", llm_model_name="gpt-4o", llm_model_version="2024-08"
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_not_found(self, mock_llm_service, client):
        """Test LLM model not found returns 404 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.return_value = None
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/openai/nonexistent-model")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert (
            "LLM model 'nonexistent-model' for provider 'openai' not found"
            in data["detail"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_value_error(self, mock_llm_service, client):
        """Test ValueError from service returns 400 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.side_effect = (
            ValueError("Invalid parameters")
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/invalid-provider/model")

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid parameters" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_database_error(self, mock_llm_service, client):
        """Test database error returns 500 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.side_effect = (
            Exception("Database connection failed")
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/openai/gpt-4o")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve LLM model configuration" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_response_structure(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test response structure matches LLMModelResponse schema."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.return_value = (
            sample_llm_model_data
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/openai/gpt-4o")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate LLMModelResponse model can be created from response
        llm_model_response = LLMModelResponse(**data)
        assert (
            llm_model_response.provider_name == sample_llm_model_data["provider_name"]
        )
        assert (
            llm_model_response.llm_model_name == sample_llm_model_data["llm_model_name"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_get_llm_model_http_exception_re_raise(self, mock_llm_service, client):
        """Test that HTTPException is re-raised and not caught by general handler."""
        # Arrange
        from fastapi import HTTPException

        mock_service_instance = AsyncMock()
        mock_service_instance.get_llm_model_by_provider_and_model.side_effect = (
            HTTPException(status_code=403, detail="Access denied")
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.get("/api/v1/llm-models/openai/gpt-4o")

        # Assert
        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]


# ============================================================================
# UPDATE LLM MODEL TESTS
# ============================================================================


class TestUpdateLLMModel:
    """Test cases for update LLM model endpoint."""

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_success(
        self, mock_llm_service, client, sample_llm_model_data, sample_update_request
    ):
        """Test successful LLM model update returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        updated_model_data = sample_llm_model_data.copy()
        updated_model_data["max_tokens"] = sample_update_request["max_tokens"]
        updated_model_data["tokens_per_minute_limit"] = sample_update_request[
            "tokens_per_minute_limit"
        ]
        mock_service_instance.update_llm_model.return_value = updated_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o", json=sample_update_request
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["max_tokens"] == sample_update_request["max_tokens"]
        assert (
            data["tokens_per_minute_limit"]
            == sample_update_request["tokens_per_minute_limit"]
        )

        # Verify service was called correctly
        mock_service_instance.update_llm_model.assert_called_once()
        call_args = mock_service_instance.update_llm_model.call_args
        assert call_args[1]["provider_name"] == "openai"
        assert call_args[1]["llm_model_name"] == "gpt-4o"
        assert call_args[1]["llm_model_version"] is None
        assert call_args[1]["max_tokens"] == sample_update_request["max_tokens"]
        assert (
            call_args[1]["tokens_per_minute_limit"]
            == sample_update_request["tokens_per_minute_limit"]
        )
        assert (
            call_args[1]["is_active_status"]
            == sample_update_request["is_active_status"]
        )
        assert call_args[1]["temperature"] == sample_update_request["temperature"]
        assert (
            call_args[1]["deployment_region"]
            == sample_update_request["deployment_region"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_partial_update(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test partial update with only some fields returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        updated_model_data = sample_llm_model_data.copy()
        updated_model_data["max_tokens"] = 16384
        mock_service_instance.update_llm_model.return_value = updated_model_data
        mock_llm_service.return_value = mock_service_instance

        partial_update = {"max_tokens": 16384}

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o", json=partial_update)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["max_tokens"] == 16384

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_not_found(
        self, mock_llm_service, client, sample_update_request
    ):
        """Test LLM model not found for update returns 404 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_llm_model.return_value = None
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/nonexistent-model", json=sample_update_request
        )

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert (
            "LLM model 'nonexistent-model' for provider 'openai' not found"
            in data["detail"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_value_error(
        self, mock_llm_service, client, sample_update_request
    ):
        """Test ValueError from service returns 400 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_llm_model.side_effect = ValueError(
            "Invalid update parameters"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o", json=sample_update_request
        )

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid update parameters" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_constraint_violation(
        self, mock_llm_service, client, sample_update_request
    ):
        """Test constraint violation (duplicate key) returns 400 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_llm_model.side_effect = Exception(
            "unique constraint violation on model configuration"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o", json=sample_update_request
        )

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert (
            "Model configuration already exists with these parameters" in data["detail"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_database_error(
        self, mock_llm_service, client, sample_update_request
    ):
        """Test database error returns 500 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_llm_model.side_effect = Exception(
            "Database connection failed"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o", json=sample_update_request
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to update LLM model configuration" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_service_call_verification(
        self, mock_llm_service, client, sample_update_request, sample_llm_model_data
    ):
        """Test that all parameters are correctly mapped to service call."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.update_llm_model.return_value = sample_llm_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o?version=2024-08",
            json=sample_update_request,
        )

        # Assert
        assert response.status_code == 200
        call_args = mock_service_instance.update_llm_model.call_args
        # Verify all parameters are passed correctly
        assert call_args[1]["provider_name"] == "openai"
        assert call_args[1]["llm_model_name"] == "gpt-4o"
        assert call_args[1]["llm_model_version"] == "2024-08"
        # Check only the fields that are present in sample_update_request
        assert call_args[1]["max_tokens"] == sample_update_request["max_tokens"]
        assert (
            call_args[1]["tokens_per_minute_limit"]
            == sample_update_request["tokens_per_minute_limit"]
        )
        assert (
            call_args[1]["is_active_status"]
            == sample_update_request["is_active_status"]
        )
        assert call_args[1]["temperature"] == sample_update_request["temperature"]
        assert (
            call_args[1]["deployment_region"]
            == sample_update_request["deployment_region"]
        )
        # Check that None values are passed for fields not in the update request
        assert call_args[1]["new_provider_name"] is None
        assert call_args[1]["new_llm_model_name"] is None
        assert call_args[1]["deployment_name"] is None
        assert call_args[1]["api_key_variable_name"] is None
        assert call_args[1]["api_endpoint_url"] is None
        assert call_args[1]["new_llm_model_version"] is None
        assert call_args[1]["requests_per_minute_limit"] is None
        assert call_args[1]["random_seed"] is None

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_update_llm_model_http_exception_re_raise(
        self, mock_llm_service, client, sample_update_request
    ):
        """Test that HTTPException is re-raised and not caught by general handler."""
        # Arrange
        from fastapi import HTTPException

        mock_service_instance = AsyncMock()
        mock_service_instance.update_llm_model.side_effect = HTTPException(
            status_code=403, detail="Access denied"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o", json=sample_update_request
        )

        # Assert
        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]


# ============================================================================
# ACTIVATE LLM MODEL TESTS
# ============================================================================


class TestActivateLLMModel:
    """Test cases for activate LLM model endpoint."""

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_activate_llm_model_success(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful LLM model activation returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        activated_model_data = sample_llm_model_data.copy()
        activated_model_data["is_active_status"] = True
        mock_service_instance.activate_llm_model.return_value = activated_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o/activate")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["is_active_status"] is True
        assert data["provider_name"] == sample_llm_model_data["provider_name"]
        assert data["llm_model_name"] == sample_llm_model_data["llm_model_name"]

        # Verify service was called correctly
        mock_service_instance.activate_llm_model.assert_called_once_with(
            provider_name="openai", llm_model_name="gpt-4o", llm_model_version=None
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_activate_llm_model_with_version(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful LLM model activation with version parameter."""
        # Arrange
        mock_service_instance = AsyncMock()
        activated_model_data = sample_llm_model_data.copy()
        activated_model_data["is_active_status"] = True
        mock_service_instance.activate_llm_model.return_value = activated_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o/activate?version=2024-08"
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["is_active_status"] is True

        # Verify service was called with version
        mock_service_instance.activate_llm_model.assert_called_once_with(
            provider_name="openai", llm_model_name="gpt-4o", llm_model_version="2024-08"
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_activate_llm_model_not_found(self, mock_llm_service, client):
        """Test LLM model not found for activation returns 404 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.activate_llm_model.return_value = None
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/nonexistent-model/activate")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert (
            "LLM model 'nonexistent-model' for provider 'openai' not found"
            in data["detail"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_activate_llm_model_database_error(self, mock_llm_service, client):
        """Test database error during activation returns 500 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.activate_llm_model.side_effect = Exception(
            "Database connection failed"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o/activate")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to activate LLM model configuration" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_activate_llm_model_http_exception_re_raise(self, mock_llm_service, client):
        """Test that HTTPException is re-raised and not caught by general handler."""
        # Arrange
        from fastapi import HTTPException

        mock_service_instance = AsyncMock()
        mock_service_instance.activate_llm_model.side_effect = HTTPException(
            status_code=403, detail="Access denied"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o/activate")

        # Assert
        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]


# ============================================================================
# DEACTIVATE LLM MODEL TESTS
# ============================================================================


class TestDeactivateLLMModel:
    """Test cases for deactivate LLM model endpoint."""

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_deactivate_llm_model_success(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful LLM model deactivation returns correct response."""
        # Arrange
        mock_service_instance = AsyncMock()
        deactivated_model_data = sample_llm_model_data.copy()
        deactivated_model_data["is_active_status"] = False
        mock_service_instance.deactivate_llm_model.return_value = deactivated_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o/deactivate")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["is_active_status"] is False
        assert data["provider_name"] == sample_llm_model_data["provider_name"]
        assert data["llm_model_name"] == sample_llm_model_data["llm_model_name"]

        # Verify service was called correctly
        mock_service_instance.deactivate_llm_model.assert_called_once_with(
            provider_name="openai", llm_model_name="gpt-4o", llm_model_version=None
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_deactivate_llm_model_with_version(
        self, mock_llm_service, client, sample_llm_model_data
    ):
        """Test successful LLM model deactivation with version parameter."""
        # Arrange
        mock_service_instance = AsyncMock()
        deactivated_model_data = sample_llm_model_data.copy()
        deactivated_model_data["is_active_status"] = False
        mock_service_instance.deactivate_llm_model.return_value = deactivated_model_data
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/gpt-4o/deactivate?version=2024-08"
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["is_active_status"] is False

        # Verify service was called with version
        mock_service_instance.deactivate_llm_model.assert_called_once_with(
            provider_name="openai", llm_model_name="gpt-4o", llm_model_version="2024-08"
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_deactivate_llm_model_not_found(self, mock_llm_service, client):
        """Test LLM model not found for deactivation returns 404 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.deactivate_llm_model.return_value = None
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch(
            "/api/v1/llm-models/openai/nonexistent-model/deactivate"
        )

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert (
            "LLM model 'nonexistent-model' for provider 'openai' not found"
            in data["detail"]
        )

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_deactivate_llm_model_database_error(self, mock_llm_service, client):
        """Test database error during deactivation returns 500 error."""
        # Arrange
        mock_service_instance = AsyncMock()
        mock_service_instance.deactivate_llm_model.side_effect = Exception(
            "Database connection failed"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o/deactivate")

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Failed to deactivate LLM model configuration" in data["detail"]

    @patch("app.api.llm_configuration_endpoints.LLMModelsService")
    def test_deactivate_llm_model_http_exception_re_raise(
        self, mock_llm_service, client
    ):
        """Test that HTTPException is re-raised and not caught by general handler."""
        # Arrange
        from fastapi import HTTPException

        mock_service_instance = AsyncMock()
        mock_service_instance.deactivate_llm_model.side_effect = HTTPException(
            status_code=403, detail="Access denied"
        )
        mock_llm_service.return_value = mock_service_instance

        # Act
        response = client.patch("/api/v1/llm-models/openai/gpt-4o/deactivate")

        # Assert
        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
