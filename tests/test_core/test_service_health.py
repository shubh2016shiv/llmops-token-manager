"""
Unit tests for app.core.service_health.

Scope note: this file previously imported `app.llm_client_provisioning.service_health`
(`display_provisioning_service_info`, `verify_celery_worker_readiness`) and tested
`verify_rabbitmq_connectivity` as if it were built on `celery_app.connection()`.
That module was renamed to `app.llm_client_provisioning_OBSELETE` — it is dead
code, not the current implementation — so the whole file failed to collect
(ModuleNotFoundError), giving zero real coverage of app.core.service_health
despite its "24 comprehensive unit tests" docstring. This rewrite tests the
actual current probes: database_connectivity_probe.py (SQLAlchemy),
redis_connectivity_probe.py (redis-py via RedisManager.client), and
rabbitmq_connectivity_probe.py (kombu.Connection directly — no Celery
involved). TestVerifyTokenMaintenanceReadiness is kept as-is: it tests
app.resilience.token_maintenance.health, a real, current module, and belongs
under the same "service health" umbrella even though it lives elsewhere.

Patch targets: each probe module does its own `from app.core.config import
settings` / `from app.core.database import db_manager` / etc. rather than
these being re-exported from `app.core.service_health.__init__`, so patches
target the probe submodule (e.g. `app.core.service_health.
redis_connectivity_probe.redis_manager`), not the package.
"""

from io import StringIO
import sys
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from kombu.exceptions import KombuError
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.service_health import (
    ServiceStatus,
    display_service_info,
    display_startup_failure,
    verify_database_connectivity,
    verify_rabbitmq_connectivity,
    verify_redis_connectivity,
)
from app.resilience.token_maintenance.health import (
    verify_token_maintenance_readiness,
)


class TestServiceStatus:
    """ServiceStatus is the shared Pydantic contract every probe returns."""

    def test_service_status_basic_creation(self):
        service = ServiceStatus(name="TestService", status="connected")

        assert service.name == "TestService"
        assert service.status == "connected"
        assert service.error_message is None
        assert service.suggestion is None
        assert service.connection_details is None

    def test_service_status_with_all_fields(self):
        service = ServiceStatus(
            name="TestService",
            status="failed",
            error_message="Connection timeout",
            suggestion="Check network connectivity",
            connection_details={"host": "localhost", "port": "5432"},
        )

        assert service.name == "TestService"
        assert service.status == "failed"
        assert service.error_message == "Connection timeout"
        assert service.suggestion == "Check network connectivity"
        assert service.connection_details == {"host": "localhost", "port": "5432"}

    def test_service_status_rejects_unknown_status_literal(self):
        # status is a Literal["connected", "failed", "skipped"] allow-list,
        # not an open string — a typo here must fail loudly at construction.
        with pytest.raises(ValueError):
            ServiceStatus(name="TestService", status="degraded")


class TestDisplayFunctions:
    """Console-only reporting: startup failure summary and service info."""

    def test_display_startup_failure_single_service(self):
        failed_service = ServiceStatus(
            name="PostgreSQL",
            status="failed",
            error_message="Connection refused",
            suggestion="Start PostgreSQL server",
        )
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_startup_failure([failed_service])
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "APPLICATION STARTUP FAILED" in output
        assert "PostgreSQL" in output
        assert "FAILED" in output
        assert "Connection refused" in output
        assert "Start PostgreSQL server" in output

    def test_display_startup_failure_multiple_services(self):
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
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_startup_failure(failed_services)
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "PostgreSQL" in output
        assert "Redis" in output
        assert "Connection refused" in output
        assert "Timeout" in output

    def test_display_startup_failure_with_connection_details(self):
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
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_startup_failure([failed_service])
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "Connection Details:" in output
        assert "host: localhost" in output
        assert "port: 5432" in output
        assert "database: mydb" in output

    def test_display_startup_failure_without_suggestion(self):
        failed_service = ServiceStatus(
            name="PostgreSQL", status="failed", error_message="Connection refused"
        )
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_startup_failure([failed_service])
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "PostgreSQL" in output
        assert "Connection refused" in output
        assert "Suggestion:" not in output

    @patch("app.core.service_health.service_status_console_report.settings")
    def test_display_service_info_fastapi_section(self, mock_settings):
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
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_service_info()
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "SERVICE ENDPOINTS & CONNECTION INFORMATION" in output
        assert "FASTAPI SERVICE" in output
        assert "http://localhost:8000/" in output
        assert "http://localhost:8000/api/docs" in output
        assert "http://localhost:8000/api/redoc" in output
        assert "http://localhost:8000/api/v1/health" in output

    @patch("app.core.service_health.service_status_console_report.settings")
    def test_display_service_info_database_section(self, mock_settings):
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
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_service_info()
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "POSTGRESQL DATABASE" in output
        assert "mydb" in output
        assert "20 connections (+ 10 overflow)" in output

    @patch("app.core.service_health.service_status_console_report.settings")
    def test_display_service_info_redis_section(self, mock_settings):
        mock_settings.fastapi_port = 8000
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_settings.database_pool_size = 20
        mock_settings.database_max_overflow = 10
        mock_settings.redis_host = "redishost"
        mock_settings.redis_port = 9999
        mock_settings.redis_db = 5
        mock_settings.redis_max_connections = 50
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            display_service_info()
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

        assert "REDIS CACHE" in output
        assert "redishost" in output
        assert "9999" in output


class TestVerifyDatabaseConnectivity:
    """database_connectivity_probe.py — SQLAlchemy async session, SELECT 1."""

    @pytest.mark.asyncio
    @patch("app.core.service_health.database_connectivity_probe.db_manager")
    @patch("app.core.service_health.database_connectivity_probe.settings")
    async def test_success(self, mock_settings, mock_db_manager):
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        result = await verify_database_connectivity()

        assert result.name == "PostgreSQL"
        assert result.status == "connected"
        assert result.error_message is None
        assert result.connection_details == {
            "host": "localhost",
            "port": "5432",
            "database": "mydb",
        }

    @pytest.mark.asyncio
    @patch("app.core.service_health.database_connectivity_probe.db_manager")
    @patch("app.core.service_health.database_connectivity_probe.settings")
    async def test_query_returns_wrong_value(self, mock_settings, mock_db_manager):
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0  # SELECT 1 didn't return 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

        result = await verify_database_connectivity()

        assert result.status == "failed"
        assert "Connection test query failed" in result.error_message
        assert "Check database permissions" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.service_health.database_connectivity_probe.db_manager")
    @patch("app.core.service_health.database_connectivity_probe.settings")
    async def test_connection_refused(self, mock_settings, mock_db_manager):
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_db_manager.get_session.side_effect = ConnectionRefusedError(
            "Connection refused"
        )

        result = await verify_database_connectivity()

        assert result.status == "failed"
        assert "Connection refused" in result.error_message
        assert "Start PostgreSQL server" in result.suggestion
        assert result.connection_details == {
            "host": "localhost",
            "port": "5432",
            "database": "mydb",
        }

    @pytest.mark.asyncio
    @patch("app.core.service_health.database_connectivity_probe.db_manager")
    @patch("app.core.service_health.database_connectivity_probe.settings")
    async def test_generic_exception(self, mock_settings, mock_db_manager):
        mock_settings.database_host = "localhost"
        mock_settings.database_port = 5432
        mock_settings.database_name = "mydb"
        mock_db_manager.get_session.side_effect = Exception("Database error")

        result = await verify_database_connectivity()

        assert result.status == "failed"
        assert result.error_message == "Database error"
        assert "Check database configuration" in result.suggestion


class TestVerifyRedisConnectivity:
    """
    redis_connectivity_probe.py — pings via `redis_manager.client` directly.

    Regression coverage for the fix in this pass: the probe used to call
    `redis_manager.ping()`, which swallows every exception and returns a
    bare bool, making the except branches below permanently unreachable for
    any real outage (verified empirically against a real refused
    connection). Going through `.client.ping()` restores the branches; these
    tests pin that behavior so it can't silently regress back to the
    swallowing call.
    """

    @pytest.mark.asyncio
    @patch("app.core.service_health.redis_connectivity_probe.redis_manager")
    @patch("app.core.service_health.redis_connectivity_probe.settings")
    async def test_success(self, mock_settings, mock_redis_manager):
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_redis_manager.client.ping = AsyncMock(return_value=True)

        result = await verify_redis_connectivity()

        assert result.name == "Redis"
        assert result.status == "connected"
        assert result.error_message is None
        assert result.connection_details == {
            "host": "localhost",
            "port": "6379",
            "database": "0",
        }

    @pytest.mark.asyncio
    @patch("app.core.service_health.redis_connectivity_probe.redis_manager")
    @patch("app.core.service_health.redis_connectivity_probe.settings")
    async def test_ping_returns_false(self, mock_settings, mock_redis_manager):
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_redis_manager.client.ping = AsyncMock(return_value=False)

        result = await verify_redis_connectivity()

        assert result.status == "failed"
        assert "did not respond to ping" in result.error_message
        assert "Check if Redis server is running" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.service_health.redis_connectivity_probe.redis_manager")
    @patch("app.core.service_health.redis_connectivity_probe.settings")
    async def test_redis_connection_error(self, mock_settings, mock_redis_manager):
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_redis_manager.client.ping = AsyncMock(
            side_effect=RedisConnectionError("Error 111 connecting to localhost:6379")
        )

        result = await verify_redis_connectivity()

        assert result.status == "failed"
        assert "Connection refused" in result.error_message
        assert "redis-server" in result.suggestion
        assert result.connection_details == {
            "host": "localhost",
            "port": "6379",
            "database": "0",
        }

    @pytest.mark.asyncio
    @patch("app.core.service_health.redis_connectivity_probe.redis_manager")
    @patch("app.core.service_health.redis_connectivity_probe.settings")
    async def test_client_not_initialized_reports_specific_message(
        self, mock_settings, mock_redis_manager
    ):
        # redis_manager.client raises RuntimeError (not a connectivity error)
        # when RedisManager.initialize() hasn't run — a startup-ordering bug.
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        type(mock_redis_manager).client = PropertyMock(
            side_effect=RuntimeError("Redis not initialized. Call initialize() first.")
        )

        result = await verify_redis_connectivity()

        assert result.status == "failed"
        assert result.error_message == "Redis not initialized. Call initialize() first."
        assert "initialize()" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.core.service_health.redis_connectivity_probe.redis_manager")
    @patch("app.core.service_health.redis_connectivity_probe.settings")
    async def test_generic_exception(self, mock_settings, mock_redis_manager):
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_redis_manager.client.ping = AsyncMock(
            side_effect=ValueError("unexpected protocol error")
        )

        result = await verify_redis_connectivity()

        assert result.status == "failed"
        assert result.error_message == "unexpected protocol error"
        assert "Check Redis configuration" in result.suggestion


class TestVerifyRabbitMQConnectivity:
    """
    rabbitmq_connectivity_probe.py — kombu.Connection directly, no Celery.

    ensure_connection/release run in a worker thread via asyncio.to_thread,
    so these tests patch the module's `Connection` constructor to return a
    plain (sync) MagicMock — no AsyncMock needed for its methods.
    """

    @pytest.mark.asyncio
    @patch("app.core.service_health.rabbitmq_connectivity_probe.Connection")
    @patch("app.core.service_health.rabbitmq_connectivity_probe.settings")
    async def test_success(self, mock_settings, mock_connection_cls):
        mock_settings.rabbitmq_host = "broker"
        mock_settings.rabbitmq_port = 5672
        mock_settings.rabbitmq_vhost = "/test"
        mock_settings.broker_url = "amqp://user:pass@broker:5672/test"
        mock_settings.rabbitmq_token_heartbeat_seconds = 60
        mock_connection = MagicMock()
        mock_connection_cls.return_value = mock_connection

        result = await verify_rabbitmq_connectivity()

        assert result.name == "RabbitMQ"
        assert result.status == "connected"
        assert result.error_message is None
        mock_connection.ensure_connection.assert_called_once_with(max_retries=1)
        mock_connection.release.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.service_health.rabbitmq_connectivity_probe.Connection")
    @patch("app.core.service_health.rabbitmq_connectivity_probe.settings")
    async def test_connection_refused_via_chained_cause(
        self, mock_settings, mock_connection_cls
    ):
        mock_settings.rabbitmq_host = "localhost"
        mock_settings.rabbitmq_port = 5672
        mock_settings.rabbitmq_vhost = "/"
        mock_settings.broker_url = "amqp://guest:guest@localhost:5672//"
        mock_settings.rabbitmq_token_heartbeat_seconds = 60
        mock_connection = MagicMock()
        # Kombu wraps socket errors in OperationalError, chaining the
        # original ConnectionRefusedError as __cause__ — reproduce that shape.
        wrapped_error = KombuError("op failed")
        wrapped_error.__cause__ = ConnectionRefusedError("refused")
        mock_connection.ensure_connection.side_effect = wrapped_error
        mock_connection_cls.return_value = mock_connection

        result = await verify_rabbitmq_connectivity()

        assert result.status == "failed"
        assert "Connection refused" in result.error_message
        assert "Start RabbitMQ server" in result.suggestion
        # Cleanup must still run even though the connect attempt raised.
        mock_connection.release.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.service_health.rabbitmq_connectivity_probe.Connection")
    @patch("app.core.service_health.rabbitmq_connectivity_probe.settings")
    async def test_other_kombu_error_reports_generic_message(
        self, mock_settings, mock_connection_cls
    ):
        mock_settings.rabbitmq_host = "localhost"
        mock_settings.rabbitmq_port = 5672
        mock_settings.rabbitmq_vhost = "/"
        mock_settings.broker_url = "amqp://guest:guest@localhost:5672//"
        mock_settings.rabbitmq_token_heartbeat_seconds = 60
        mock_connection = MagicMock()
        mock_connection.ensure_connection.side_effect = KombuError(
            "AMQP handshake failed"
        )
        mock_connection_cls.return_value = mock_connection

        result = await verify_rabbitmq_connectivity()

        assert result.status == "failed"
        assert result.error_message == "AMQP handshake failed"
        assert "Check RabbitMQ configuration" in result.suggestion
        mock_connection.release.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.service_health.rabbitmq_connectivity_probe.Connection")
    @patch("app.core.service_health.rabbitmq_connectivity_probe.settings")
    async def test_release_runs_even_when_ensure_connection_raises_unexpectedly(
        self, mock_settings, mock_connection_cls
    ):
        """Cleanup (release) must not be skipped when the primary path fails."""
        mock_settings.rabbitmq_host = "localhost"
        mock_settings.rabbitmq_port = 5672
        mock_settings.rabbitmq_vhost = "/"
        mock_settings.broker_url = "amqp://guest:guest@localhost:5672//"
        mock_settings.rabbitmq_token_heartbeat_seconds = 60
        mock_connection = MagicMock()
        mock_connection.ensure_connection.side_effect = OSError("network unreachable")
        mock_connection_cls.return_value = mock_connection

        result = await verify_rabbitmq_connectivity()

        assert result.status == "failed"
        mock_connection.release.assert_called_once()


class TestVerifyTokenMaintenanceReadiness:
    """
    app.resilience.token_maintenance.health — a different package, kept here
    because it answers the same "is this dependency ready" question as the
    probes above and app/api/health_endpoints.py treats it identically.
    """

    @pytest.mark.asyncio
    @patch("app.resilience.token_maintenance.health.inspect_token_maintenance_runtime")
    async def test_ready_schedule_reports_connected(self, mock_inspect_runtime):
        mock_inspect_runtime.return_value = (True, None)

        result = await verify_token_maintenance_readiness()

        assert result.name == "Token maintenance"
        assert result.status == "connected"
        assert result.connection_details["mode"] == "in-process scheduler"
        scheduled_jobs = result.connection_details["scheduled_jobs"]
        assert "reconciliation" in scheduled_jobs
        assert "cleanup" in scheduled_jobs
        assert "queue_depth_publish" in scheduled_jobs

    @pytest.mark.asyncio
    @patch("app.resilience.token_maintenance.health.inspect_token_maintenance_runtime")
    async def test_not_ready_schedule_reports_failed(self, mock_inspect_runtime):
        mock_inspect_runtime.return_value = (False, "runtime not ready")

        result = await verify_token_maintenance_readiness()

        assert result.name == "Token maintenance"
        assert result.status == "failed"
        assert result.error_message == "runtime not ready"
        assert "intervals are configured" in result.suggestion

    @pytest.mark.asyncio
    @patch("app.resilience.token_maintenance.health.inspect_token_maintenance_runtime")
    async def test_probe_raising_reports_failed(self, mock_inspect_runtime):
        mock_inspect_runtime.side_effect = RuntimeError("probe failed")

        result = await verify_token_maintenance_readiness()

        assert result.name == "Token maintenance"
        assert result.status == "failed"
        assert result.error_message == "probe failed"
