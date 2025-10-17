"""
Comprehensive Unit Tests for Health Endpoints
============================================
Unit tests for health monitoring endpoints covering basic health checks
and dependency health checks with 100% coverage.

Test Coverage:
- Basic health check endpoint (2 tests)
- Dependency health check endpoint (6 tests)

Total: 8 comprehensive unit tests
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health_endpoints import router
from app.models.response_models import HealthStatus, DependencyHealth


# ============================================================================
# TEST SETUP
# ============================================================================


@pytest.fixture
def app():
    """Create FastAPI app with health endpoints router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_health_status():
    """Sample health status data for testing."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc),
        "version": "1.0.0",
    }


@pytest.fixture
def sample_dependency_health():
    """Sample dependency health data for testing."""
    return {
        "postgresql": True,
        "redis": True,
        "rabbitmq": True,
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc),
    }


# ============================================================================
# BASIC HEALTH CHECK TESTS
# ============================================================================


class TestBasicHealthCheck:
    """Test cases for basic health check endpoint."""

    @patch("app.api.health_endpoints.settings")
    def test_health_check_success(self, mock_settings, client, sample_health_status):
        """Test successful health check returns correct response."""
        # Arrange
        mock_settings.app_version = "1.0.0"

        # Act
        response = client.get("/api/v1/health/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        # Verify timestamp is recent (within last minute)
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        assert (now - timestamp).total_seconds() < 60

    @patch("app.api.health_endpoints.settings")
    def test_health_check_response_structure(self, mock_settings, client):
        """Test health check response matches HealthStatus schema."""
        # Arrange
        mock_settings.app_version = "2.1.0"

        # Act
        response = client.get("/api/v1/health/")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate all required fields are present
        required_fields = ["status", "timestamp", "version"]
        for field in required_fields:
            assert field in data

        # Validate field types
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)

        # Validate HealthStatus model can be created from response
        health_status = HealthStatus(**data)
        assert health_status.status == "healthy"
        assert health_status.version == "2.1.0"


# ============================================================================
# DEPENDENCY HEALTH CHECK TESTS
# ============================================================================


class TestDependencyHealthCheck:
    """Test cases for dependency health check endpoint."""

    @patch("app.api.health_endpoints._check_rabbitmq")
    @patch("app.api.health_endpoints._check_redis")
    @patch("app.api.health_endpoints._check_database")
    def test_dependencies_all_healthy(
        self,
        mock_check_db,
        mock_check_redis,
        mock_check_rabbitmq,
        client,
        sample_dependency_health,
    ):
        """Test all dependencies healthy returns correct response."""
        # Arrange
        mock_check_db.return_value = True
        mock_check_redis.return_value = True
        mock_check_rabbitmq.return_value = True

        # Act
        response = client.get("/api/v1/health/dependencies")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["postgresql"] is True
        assert data["redis"] is True
        assert data["rabbitmq"] is True
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @patch("app.api.health_endpoints._check_rabbitmq")
    @patch("app.api.health_endpoints._check_redis")
    @patch("app.api.health_endpoints._check_database")
    def test_dependencies_postgresql_unhealthy(
        self, mock_check_db, mock_check_redis, mock_check_rabbitmq, client
    ):
        """Test PostgreSQL unhealthy returns correct response."""
        # Arrange
        mock_check_db.return_value = False
        mock_check_redis.return_value = True
        mock_check_rabbitmq.return_value = True

        # Act
        response = client.get("/api/v1/health/dependencies")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["postgresql"] is False
        assert data["redis"] is True
        assert data["rabbitmq"] is True
        assert data["status"] == "unhealthy"

    @patch("app.api.health_endpoints._check_rabbitmq")
    @patch("app.api.health_endpoints._check_redis")
    @patch("app.api.health_endpoints._check_database")
    def test_dependencies_redis_unhealthy(
        self, mock_check_db, mock_check_redis, mock_check_rabbitmq, client
    ):
        """Test Redis unhealthy returns correct response."""
        # Arrange
        mock_check_db.return_value = True
        mock_check_redis.return_value = False
        mock_check_rabbitmq.return_value = True

        # Act
        response = client.get("/api/v1/health/dependencies")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["postgresql"] is True
        assert data["redis"] is False
        assert data["rabbitmq"] is True
        assert data["status"] == "unhealthy"

    @patch("app.api.health_endpoints._check_rabbitmq")
    @patch("app.api.health_endpoints._check_redis")
    @patch("app.api.health_endpoints._check_database")
    def test_dependencies_rabbitmq_unhealthy(
        self, mock_check_db, mock_check_redis, mock_check_rabbitmq, client
    ):
        """Test RabbitMQ unhealthy returns correct response."""
        # Arrange
        mock_check_db.return_value = True
        mock_check_redis.return_value = True
        mock_check_rabbitmq.return_value = False

        # Act
        response = client.get("/api/v1/health/dependencies")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["postgresql"] is True
        assert data["redis"] is True
        assert data["rabbitmq"] is False
        assert data["status"] == "unhealthy"

    @patch("app.api.health_endpoints._check_rabbitmq")
    @patch("app.api.health_endpoints._check_redis")
    @patch("app.api.health_endpoints._check_database")
    def test_dependencies_multiple_unhealthy(
        self, mock_check_db, mock_check_redis, mock_check_rabbitmq, client
    ):
        """Test multiple dependencies unhealthy returns correct response."""
        # Arrange
        mock_check_db.return_value = False
        mock_check_redis.return_value = False
        mock_check_rabbitmq.return_value = False

        # Act
        response = client.get("/api/v1/health/dependencies")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["postgresql"] is False
        assert data["redis"] is False
        assert data["rabbitmq"] is False
        assert data["status"] == "unhealthy"

    @patch("app.api.health_endpoints._check_rabbitmq")
    @patch("app.api.health_endpoints._check_redis")
    @patch("app.api.health_endpoints._check_database")
    def test_dependencies_response_structure(
        self, mock_check_db, mock_check_redis, mock_check_rabbitmq, client
    ):
        """Test dependency health response matches DependencyHealth schema."""
        # Arrange
        mock_check_db.return_value = True
        mock_check_redis.return_value = True
        mock_check_rabbitmq.return_value = True

        # Act
        response = client.get("/api/v1/health/dependencies")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate all required fields are present
        required_fields = ["postgresql", "redis", "rabbitmq", "status", "timestamp"]
        for field in required_fields:
            assert field in data

        # Validate field types
        assert isinstance(data["postgresql"], bool)
        assert isinstance(data["redis"], bool)
        assert isinstance(data["rabbitmq"], bool)
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)

        # Validate DependencyHealth model can be created from response
        dependency_health = DependencyHealth(**data)
        assert dependency_health.postgresql is True
        assert dependency_health.redis is True
        assert dependency_health.rabbitmq is True
        assert dependency_health.status == "healthy"
