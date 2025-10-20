"""
Comprehensive Unit Tests for Startup Diagnostics
===============================================
Unit tests for service connectivity verification and error reporting.

Test Coverage:
- ServiceStatus dataclass (3 tests)
- Display functions (7 tests)
- Database verification (5 tests)
- Redis verification (5 tests)
- RabbitMQ verification (4 tests)

Total: 24 comprehensive unit tests
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import StringIO
import sys

from app.core.startup_diagnostics import (
    ServiceStatus,
    display_startup_failure,
    display_service_info,
    verify_database_connectivity,
    verify_redis_connectivity,
    verify_rabbitmq_connectivity,
)


class TestServiceStatus:
    """Test cases for ServiceStatus dataclass."""

    # Group 1: DataClass Initialization (3 tests)

    def test_service_status_basic_creation(self):
        """Test ServiceStatus creation with minimal fields."""
        # Act
        service = ServiceStatus(name="TestService", status="connected")

        # Assert
        assert service.name == "TestService"
        assert service.status == "connected"
        assert service.error_message is None
        assert service.suggestion is None
        assert service.connection_details is None

    def test_service_status_with_all_fields(self):
        """Test ServiceStatus creation with all optional fields."""
        # Act
        service = ServiceStatus(
            name="TestService",
            status="failed",
            error_message="Connection timeout",
            suggestion="Check network connectivity",
            connection_details={"host": "localhost", "port": "5432"},
        )

        # Assert
        assert service.name == "TestService"
        assert service.status == "failed"
        assert service.error_message == "Connection timeout"
        assert service.suggestion == "Check network connectivity"
        assert service.connection_details == {"host": "localhost", "port": "5432"}

    def test_service_status_connection_details_dict(self):
        """Test that connection_details accepts and stores dict properly."""
        # Setup
        connection_info = {"host": "localhost", "port": "6379", "database": "0"}

        # Act
        service = ServiceStatus(
            name="Redis", status="connected", connection_details=connection_info
        )

        # Assert
        assert service.connection_details == connection_info
        assert service.connection_details["host"] == "localhost"
        assert service.connection_details["port"] == "6379"
        assert service.connection_details["database"] == "0"


class TestDisplayFunctions:
    """Test cases for display functions."""

    # Group 2: Display Startup Failure (4 tests)

    def test_display_startup_failure_single_service(self):
        """Test display of single failed service."""
        # Setup
        failed_service = ServiceStatus(
            name="PostgreSQL",
            status="failed",
            error_message="Connection refused",
            suggestion="Start PostgreSQL server",
        )

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_startup_failure([failed_service])

            # Assert
            output = captured_output.getvalue()
            assert "APPLICATION STARTUP FAILED" in output
            assert "PostgreSQL" in output
            assert "FAILED" in output
            assert "Connection refused" in output
            assert "Start PostgreSQL server" in output
        finally:
            sys.stdout = original_stdout

    def test_display_startup_failure_multiple_services(self):
        """Test display of multiple failed services."""
        # Setup
        failed_services = [
            ServiceStatus(
                name="PostgreSQL",
                status="failed",
                error_message="Connection refused",
                suggestion="Start PostgreSQL server",
            ),
            ServiceStatus(
                name="Redis",
                status="failed",
                error_message="Timeout",
                suggestion="Check Redis configuration",
            ),
        ]

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_startup_failure(failed_services)

            # Assert
            output = captured_output.getvalue()
            assert "APPLICATION STARTUP FAILED" in output
            assert "PostgreSQL" in output
            assert "Redis" in output
            assert "Connection refused" in output
            assert "Timeout" in output
        finally:
            sys.stdout = original_stdout

    def test_display_startup_failure_with_connection_details(self):
        """Test display with connection details."""
        # Setup
        failed_service = ServiceStatus(
            name="PostgreSQL",
            status="failed",
            error_message="Connection refused",
            suggestion="Check host and port",
            connection_details={
                "host": "localhost",
                "port": "5432",
                "database": "mydb",
            },
        )

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_startup_failure([failed_service])

            # Assert
            output = captured_output.getvalue()
            assert "Connection Details:" in output
            assert "host: localhost" in output
            assert "port: 5432" in output
            assert "database: mydb" in output
        finally:
            sys.stdout = original_stdout

    def test_display_startup_failure_without_suggestion(self):
        """Test display without suggestion field."""
        # Setup
        failed_service = ServiceStatus(
            name="PostgreSQL",
            status="failed",
            error_message="Connection refused",
            # No suggestion provided
        )

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_startup_failure([failed_service])

            # Assert
            output = captured_output.getvalue()
            assert "PostgreSQL" in output
            assert "Connection refused" in output
            assert "Suggestion:" not in output
        finally:
            sys.stdout = original_stdout

    # Group 3: Display Service Info (3 tests)

    @patch("app.core.startup_diagnostics.settings")
    def test_display_service_info_fastapi_section(self, mock_settings):
        """Test FastAPI section display."""
        # Setup
        mock_settings.fastapi_port = 8000
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_settings.database_pool_size = 20
        mock_settings.database_max_overflow = 10
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_settings.redis_max_connections = 50

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_service_info()

            # Assert
            output = captured_output.getvalue()
            assert "SERVICE ENDPOINTS & CONNECTION INFORMATION" in output
            assert "FASTAPI SERVICE" in output
            assert "http://localhost:8000/" in output
            assert "http://localhost:8000/api/docs" in output
            assert "http://localhost:8000/api/redoc" in output
            assert "http://localhost:8000/api/v1/health" in output
        finally:
            sys.stdout = original_stdout

    @patch("app.core.startup_diagnostics.settings")
    def test_display_service_info_database_section(self, mock_settings):
        """Test database section display."""
        # Setup
        mock_settings.fastapi_port = 8000
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_settings.database_pool_size = 20
        mock_settings.database_max_overflow = 10
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_settings.redis_max_connections = 50

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_service_info()

            # Assert
            output = captured_output.getvalue()
            assert "POSTGRESQL DATABASE" in output
            assert "Host" in output
            assert "localhost" in output
            assert "Port" in output
            assert "5432" in output
            assert "Database" in output
            assert "mydb" in output
            assert "20 connections (+ 10 overflow)" in output
        finally:
            sys.stdout = original_stdout

    @patch("app.core.startup_diagnostics.settings")
    def test_display_service_info_redis_section(self, mock_settings):
        """Test Redis section display."""
        # Setup
        mock_settings.fastapi_port = 8000
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_settings.database_pool_size = 20
        mock_settings.database_max_overflow = 10
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_settings.redis_max_connections = 50

        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Act
            display_service_info()

            # Assert
            output = captured_output.getvalue()
            assert "REDIS CACHE" in output
            assert "Host" in output
            assert "localhost" in output
            assert "Port" in output
            assert "6379" in output
            assert "Database" in output
            assert "0" in output
            assert "Max Connections" in output
            assert "50" in output
        finally:
            sys.stdout = original_stdout


class TestVerifyDatabaseConnectivity:
    """Test cases for database connectivity verification."""

    # Group 4: Database Verification (5 tests)

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.db_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_database_connectivity_success(
        self, mock_settings, mock_db_manager
    ):
        """Test successful database connectivity verification."""
        # Setup
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        # Act
        result = await verify_database_connectivity()

        # Assert
        assert result.name == "PostgreSQL"
        assert result.status == "connected"
        assert result.error_message is None
        assert result.connection_details == {
            "host": "localhost",
            "port": "5432",
            "database": "mydb",
        }

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.db_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_database_connectivity_query_failed(
        self, mock_settings, mock_db_manager
    ):
        """Test database connectivity when query fails."""
        # Setup
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0  # Query failed
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        # Act
        result = await verify_database_connectivity()

        # Assert
        assert result.name == "PostgreSQL"
        assert result.status == "failed"
        assert "Connection test query failed" in result.error_message
        assert "Check database permissions" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.db_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_database_connectivity_connection_refused(
        self, mock_settings, mock_db_manager
    ):
        """Test database connectivity when connection is refused."""
        # Setup
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"

        mock_db_manager.get_session.side_effect = ConnectionRefusedError(
            "Connection refused"
        )

        # Act
        result = await verify_database_connectivity()

        # Assert
        assert result.name == "PostgreSQL"
        assert result.status == "failed"
        assert "Connection refused" in result.error_message
        assert "Start PostgreSQL server" in result.suggestion
        assert result.connection_details == {
            "host": "localhost",
            "port": "5432",
            "database": "mydb",
        }

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.db_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_database_connectivity_generic_exception(
        self, mock_settings, mock_db_manager
    ):
        """Test database connectivity with generic exception."""
        # Setup
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"

        mock_db_manager.get_session.side_effect = Exception("Database error")

        # Act
        result = await verify_database_connectivity()

        # Assert
        assert result.name == "PostgreSQL"
        assert result.status == "failed"
        assert result.error_message == "Database error"
        assert "Check database configuration" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.db_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_database_connectivity_connection_details_correct(
        self, mock_settings, mock_db_manager
    ):
        """Test that connection details match settings values."""
        # Setup
        mock_settings.database_host = "testhost"
        mock_settings.database_port = 9999
        mock_settings.database_name = "testdb"

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        # Act
        result = await verify_database_connectivity()

        # Assert
        assert result.connection_details == {
            "host": "testhost",
            "port": "9999",
            "database": "testdb",
        }


class TestVerifyRedisConnectivity:
    """Test cases for Redis connectivity verification."""

    # Group 5: Redis Verification (5 tests)

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.redis_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_redis_connectivity_success(
        self, mock_settings, mock_redis_manager
    ):
        """Test successful Redis connectivity verification."""
        # Setup
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0

        mock_redis_manager.ping = AsyncMock(return_value=True)

        # Act
        result = await verify_redis_connectivity()

        # Assert
        assert result.name == "Redis"
        assert result.status == "connected"
        assert result.error_message is None
        assert result.connection_details == {
            "host": "localhost",
            "port": "6379",
            "database": "0",
        }

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.redis_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_redis_connectivity_ping_failed(
        self, mock_settings, mock_redis_manager
    ):
        """Test Redis connectivity when ping fails."""
        # Setup
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0

        mock_redis_manager.ping = AsyncMock(return_value=False)

        # Act
        result = await verify_redis_connectivity()

        # Assert
        assert result.name == "Redis"
        assert result.status == "failed"
        assert "Redis server did not respond to ping" in result.error_message
        assert "Check if Redis server is running" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.redis_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_redis_connectivity_connection_refused(
        self, mock_settings, mock_redis_manager
    ):
        """Test Redis connectivity when connection is refused."""
        # Setup
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0

        mock_redis_manager.ping = AsyncMock(
            side_effect=ConnectionRefusedError("Connection refused")
        )

        # Act
        result = await verify_redis_connectivity()

        # Assert
        assert result.name == "Redis"
        assert result.status == "failed"
        assert "Connection refused" in result.error_message
        assert "redis-server" in result.suggestion
        assert result.connection_details == {
            "host": "localhost",
            "port": "6379",
            "database": "0",
        }

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.redis_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_redis_connectivity_generic_exception(
        self, mock_settings, mock_redis_manager
    ):
        """Test Redis connectivity with generic exception."""
        # Setup
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0

        mock_redis_manager.ping = AsyncMock(side_effect=Exception("Redis error"))

        # Act
        result = await verify_redis_connectivity()

        # Assert
        assert result.name == "Redis"
        assert result.status == "failed"
        assert result.error_message == "Redis error"
        assert "Check Redis configuration" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.startup_diagnostics.redis_manager")
    @patch("app.core.startup_diagnostics.settings")
    async def test_verify_redis_connectivity_connection_details_correct(
        self, mock_settings, mock_redis_manager
    ):
        """Test that Redis connection details match settings values."""
        # Setup
        mock_settings.redis_host = "redishost"
        mock_settings.redis_port = 9999
        mock_settings.redis_db = 5

        mock_redis_manager.ping = AsyncMock(return_value=True)

        # Act
        result = await verify_redis_connectivity()

        # Assert
        assert result.connection_details == {
            "host": "redishost",
            "port": "9999",
            "database": "5",
        }


class TestVerifyRabbitMQConnectivity:
    """Test cases for RabbitMQ connectivity verification."""

    # Group 6: RabbitMQ Verification (4 tests)

    @pytest.mark.asyncio
    @patch("app.llm_client_provisioning.celery_app.celery_app")
    async def test_verify_rabbitmq_connectivity_success(self, mock_celery_app):
        """Test successful RabbitMQ connectivity verification."""
        # Setup
        mock_connection = MagicMock()
        mock_connection.ensure_connection = MagicMock()
        mock_celery_app.connection.return_value.__enter__.return_value = mock_connection

        # Act
        result = await verify_rabbitmq_connectivity()

        # Assert
        assert result.name == "RabbitMQ"
        assert result.status == "connected"
        assert result.error_message is None
        assert result.connection_details == {
            "host": "localhost",
            "port": "5672",
            "broker": "Celery broker",
        }

    @pytest.mark.asyncio
    @patch("app.llm_client_provisioning.celery_app.celery_app")
    async def test_verify_rabbitmq_connectivity_connection_refused(
        self, mock_celery_app
    ):
        """Test RabbitMQ connectivity when connection is refused."""
        # Setup
        mock_celery_app.connection.side_effect = ConnectionRefusedError(
            "Connection refused"
        )

        # Act
        result = await verify_rabbitmq_connectivity()

        # Assert
        assert result.name == "RabbitMQ"
        assert result.status == "failed"
        assert "Connection refused" in result.error_message
        assert "Start RabbitMQ server" in result.suggestion
        assert result.connection_details == {
            "host": "localhost",
            "port": "5672",
            "broker": "Celery broker",
        }

    @pytest.mark.asyncio
    @patch("app.llm_client_provisioning.celery_app.celery_app")
    async def test_verify_rabbitmq_connectivity_generic_exception(
        self, mock_celery_app
    ):
        """Test RabbitMQ connectivity with generic exception."""
        # Setup
        mock_celery_app.connection.side_effect = Exception("RabbitMQ error")

        # Act
        result = await verify_rabbitmq_connectivity()

        # Assert
        assert result.name == "RabbitMQ"
        assert result.status == "failed"
        assert result.error_message == "RabbitMQ error"
        assert "Check RabbitMQ configuration" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.llm_client_provisioning.celery_app.celery_app")
    async def test_verify_rabbitmq_connectivity_connection_details_default(
        self, mock_celery_app
    ):
        """Test that RabbitMQ uses default connection details."""
        # Setup
        mock_connection = MagicMock()
        mock_connection.ensure_connection = MagicMock()
        mock_celery_app.connection.return_value.__enter__.return_value = mock_connection

        # Act
        result = await verify_rabbitmq_connectivity()

        # Assert
        assert result.connection_details == {
            "host": "localhost",
            "port": "5672",
            "broker": "Celery broker",
        }
