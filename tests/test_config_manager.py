"""
Comprehensive Unit Tests for Config Manager
==========================================
Unit tests for the ApplicationSettings configuration management system.

Test Coverage:
- Default configuration values (1 test)
- Field validators (4 tests)
- Computed properties (6 tests)
- Environment variable loading (2 tests)

Total: 13 comprehensive unit tests
"""

import pytest
from pydantic import ValidationError

from app.core.config_manager import ApplicationSettings


class TestApplicationSettingsDefaults:
    """Test default configuration values."""

    def test_default_settings(self):
        """Test that all default values are set correctly."""
        # Act
        settings = ApplicationSettings()

        # Assert - Application metadata
        assert settings.app_name == "LLM Token Manager"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"

        # Assert - FastAPI server configuration
        assert settings.fastapi_host == "localhost"
        assert settings.fastapi_port == 8000

        # Assert - PostgreSQL database configuration
        assert settings.database_host == "localhost"
        assert settings.database_port == 5432
        assert settings.database_user == "myuser"
        assert settings.database_password == "mypassword"
        assert settings.database_name == "mydb"
        assert settings.database_pool_size == 20
        assert settings.database_max_overflow == 10

        # Assert - Redis configuration
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.redis_db == 0
        assert settings.redis_password is None
        assert settings.redis_max_connections == 50

        # Assert - RabbitMQ configuration
        assert settings.rabbitmq_host == "localhost"
        assert settings.rabbitmq_port == 5672
        assert settings.rabbitmq_user == "rmq_user"
        assert settings.rabbitmq_password == "rmq_password"
        assert settings.rabbitmq_vhost == "/"

        # Assert - Celery configuration
        assert settings.celery_broker_url is None
        assert settings.celery_result_backend == "rpc://"
        assert settings.celery_worker_concurrency == 10
        assert settings.celery_task_soft_time_limit == 300
        assert settings.celery_task_time_limit == 600

        # Assert - Rate limiting configuration
        assert settings.rate_limit_requests_per_minute == 100
        assert settings.rate_limit_window_seconds == 60

        # Assert - Caching configuration
        assert settings.cache_enabled is True
        assert settings.cache_ttl_seconds == 300

        # Assert - LLM Provider API Keys
        assert settings.openai_api_key is None
        assert settings.azure_openai_api_key is None
        assert settings.azure_openai_endpoint is None
        assert settings.azure_openai_api_version == "2024-02-15-preview"
        assert settings.anthropic_api_key is None
        assert settings.google_api_key is None

        # Assert - Default LLM settings
        assert settings.default_max_tokens == 1000
        assert settings.default_temperature == 0.7


class TestApplicationSettingsValidators:
    """Test field validators."""

    @pytest.mark.parametrize(
        "valid_level",
        [
            "TRACE",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "trace",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        ],
    )
    def test_validate_log_level_valid(self, valid_level):
        """Test that valid log levels pass validation and return uppercase."""
        # Act
        settings = ApplicationSettings(log_level=valid_level)

        # Assert
        assert settings.log_level == valid_level.upper()

    def test_validate_log_level_invalid(self):
        """Test that invalid log level raises ValidationError (Line 115 coverage)."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(log_level="INVALID")

        # Assert error message contains expected text
        error_str = str(exc_info.value)
        assert "Log level must be one of" in error_str
        assert "TRACE" in error_str
        assert "DEBUG" in error_str
        assert "INFO" in error_str
        assert "WARNING" in error_str
        assert "ERROR" in error_str
        assert "CRITICAL" in error_str

    @pytest.mark.parametrize("valid_temperature", [0.0, 0.7, 1.0, 2.0])
    def test_validate_temperature_valid(self, valid_temperature):
        """Test that valid temperature values pass validation."""
        # Act
        settings = ApplicationSettings(default_temperature=valid_temperature)

        # Assert
        assert settings.default_temperature == valid_temperature

    @pytest.mark.parametrize(
        "invalid_temperature,expected_message",
        [
            (-0.1, "Temperature must be between 0.0 and 2.0"),
            (2.1, "Temperature must be between 0.0 and 2.0"),
            (-1.0, "Temperature must be between 0.0 and 2.0"),
            (3.0, "Temperature must be between 0.0 and 2.0"),
        ],
    )
    def test_validate_temperature_invalid(self, invalid_temperature, expected_message):
        """Test that invalid temperature values raise ValidationError (Line 123 coverage)."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ApplicationSettings(default_temperature=invalid_temperature)

        # Assert error message contains expected text
        error_str = str(exc_info.value)
        assert expected_message in error_str


class TestApplicationSettingsProperties:
    """Test computed properties."""

    def test_database_url_property(self):
        """Test async database URL construction (Line 129 coverage)."""
        # Arrange
        settings = ApplicationSettings(
            database_user="testuser",
            database_password="testpass",
            database_host="testhost",
            database_port=5432,
            database_name="testdb",
        )

        # Act
        url = settings.database_url

        # Assert
        expected = "postgresql+asyncpg://testuser:testpass@testhost:5432/testdb"
        assert url == expected

    def test_database_url_sync_property(self):
        """Test sync database URL construction (Line 137 coverage)."""
        # Arrange
        settings = ApplicationSettings(
            database_user="testuser",
            database_password="testpass",
            database_host="testhost",
            database_port=5432,
            database_name="testdb",
        )

        # Act
        url = settings.database_url_sync

        # Assert
        expected = "postgresql://testuser:testpass@testhost:5432/testdb"
        assert url == expected

    def test_redis_url_without_password(self):
        """Test Redis URL without password."""
        # Arrange
        settings = ApplicationSettings(
            redis_host="testhost", redis_port=6379, redis_db=0, redis_password=None
        )

        # Act
        url = settings.redis_url

        # Assert
        expected = "redis://testhost:6379/0"
        assert url == expected

    def test_redis_url_with_password(self):
        """Test Redis URL with password (Lines 145-147 coverage)."""
        # Arrange
        settings = ApplicationSettings(
            redis_host="testhost",
            redis_port=6379,
            redis_db=0,
            redis_password="testpass",
        )

        # Act
        url = settings.redis_url

        # Assert
        expected = "redis://:testpass@testhost:6379/0"
        assert url == expected

    def test_broker_url_default(self):
        """Test default RabbitMQ broker URL."""
        # Arrange
        settings = ApplicationSettings(
            celery_broker_url=None,
            rabbitmq_user="testuser",
            rabbitmq_password="testpass",
            rabbitmq_host="testhost",
            rabbitmq_port=5672,
            rabbitmq_vhost="/",
        )

        # Act
        url = settings.broker_url

        # Assert
        expected = "amqp://testuser:testpass@testhost:5672/"
        assert url == expected

    def test_broker_url_custom(self):
        """Test custom broker URL (Lines 152-154 coverage)."""
        # Arrange
        custom_url = "redis://localhost:6379/1"
        settings = ApplicationSettings(
            celery_broker_url=custom_url,
            rabbitmq_user="testuser",
            rabbitmq_password="testpass",
            rabbitmq_host="testhost",
            rabbitmq_port=5672,
            rabbitmq_vhost="/",
        )

        # Act
        url = settings.broker_url

        # Assert
        assert url == custom_url


class TestApplicationSettingsFromEnvironment:
    """Test environment variable loading."""

    def test_load_from_environment_variables(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Arrange
        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FASTAPI_PORT", "9000")
        monkeypatch.setenv("DATABASE_HOST", "testdb")
        monkeypatch.setenv("DATABASE_USER", "testuser")
        monkeypatch.setenv("DATABASE_PASSWORD", "testpass")
        monkeypatch.setenv("REDIS_HOST", "testredis")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "200")
        monkeypatch.setenv("DEFAULT_TEMPERATURE", "0.5")

        # Act
        settings = ApplicationSettings()

        # Assert
        assert settings.app_name == "Test App"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.fastapi_port == 9000
        assert settings.database_host == "testdb"
        assert settings.database_user == "testuser"
        assert settings.database_password == "testpass"
        assert settings.redis_host == "testredis"
        assert settings.rate_limit_requests_per_minute == 200
        assert settings.default_temperature == 0.5

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test case-insensitive environment variable loading."""
        # Arrange
        monkeypatch.setenv("app_name", "Test App Case")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("log_level", "WARNING")
        monkeypatch.setenv("fastapi_port", "8080")

        # Act
        settings = ApplicationSettings()

        # Assert
        assert settings.app_name == "Test App Case"
        assert settings.debug is False
        assert settings.log_level == "WARNING"
        assert settings.fastapi_port == 8080


# Run with: pytest tests/test_config_manager.py -v
