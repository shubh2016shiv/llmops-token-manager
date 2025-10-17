import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from loguru import logger
from app.core.redis_connection import RedisManager
from app.core.config_manager import settings


class TestRedisConnection:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down for each test."""
        # Setup
        self.redis_manager = RedisManager()
        # Mock logger to capture log messages
        self.logger_info_mock = MagicMock()
        self.logger_warning_mock = MagicMock()
        self.logger_error_mock = MagicMock()
        logger.info = self.logger_info_mock
        logger.warning = self.logger_warning_mock
        logger.error = self.logger_error_mock

        yield  # This is where the test runs

        # Teardown
        self.redis_manager._client = None
        self.redis_manager._pool = None

    @patch("app.core.redis_connection.aioredis.Redis")
    @patch("app.core.redis_connection.ConnectionPool")
    def test_initialize_success(self, mock_connection_pool, mock_redis):
        """Test successful initialization of RedisManager."""
        # Arrange
        mock_pool_instance = MagicMock()
        mock_connection_pool.return_value = mock_pool_instance
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance

        # Act
        self.redis_manager.initialize()

        # Assert
        mock_connection_pool.assert_called_once_with(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
        )
        mock_redis.assert_called_once_with(connection_pool=mock_pool_instance)
        assert self.redis_manager._pool == mock_pool_instance
        assert self.redis_manager._client == mock_redis_instance
        self.logger_info_mock.assert_any_call(
            f"Initializing Redis connection to {settings.redis_host}:{settings.redis_port}"
        )
        self.logger_info_mock.assert_any_call(
            "Redis connection initialized successfully"
        )

    @patch("app.core.redis_connection.aioredis.Redis")
    @patch("app.core.redis_connection.ConnectionPool")
    def test_initialize_already_initialized(self, mock_connection_pool, mock_redis):
        """Test initialization when already initialized."""
        # Arrange
        self.redis_manager._pool = MagicMock()
        self.redis_manager._client = AsyncMock()

        # Act
        self.redis_manager.initialize()

        # Assert
        mock_connection_pool.assert_not_called()
        mock_redis.assert_not_called()
        self.logger_warning_mock.assert_called_once_with(
            "Redis connection pool already initialized"
        )

    def test_client_property_not_initialized(self):
        """Test accessing client property before initialization raises RuntimeError."""
        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            _ = self.redis_manager.client
        assert str(exc_info.value) == "Redis not initialized. Call initialize() first."

    @patch("app.core.redis_connection.aioredis.Redis")
    def test_client_property_initialized(self, mock_redis):
        """Test accessing client property after initialization."""
        # Arrange
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance
        self.redis_manager._client = mock_redis_instance
        self.redis_manager._pool = MagicMock()

        # Act
        client = self.redis_manager.client

        # Assert
        assert client == mock_redis_instance

    @pytest.mark.asyncio
    @patch("app.core.redis_connection.aioredis.Redis")
    async def test_ping_success(self, mock_redis):
        """Test ping method when Redis responds successfully."""
        # Arrange
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True
        self.redis_manager._client = mock_redis_instance
        self.redis_manager._pool = MagicMock()

        # Act
        result = await self.redis_manager.ping()

        # Assert
        mock_redis_instance.ping.assert_awaited_once()
        assert result is True
        self.logger_error_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_ping_not_initialized(self):
        """Test ping method when client is not initialized."""
        # Act
        result = await self.redis_manager.ping()

        # Assert
        assert result is False
        self.logger_error_mock.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.core.redis_connection.aioredis.Redis")
    async def test_ping_failure(self, mock_redis):
        """Test ping method when Redis raises an exception."""
        # Arrange
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.side_effect = Exception("Connection error")
        self.redis_manager._client = mock_redis_instance
        self.redis_manager._pool = MagicMock()

        # Act
        result = await self.redis_manager.ping()

        # Assert
        mock_redis_instance.ping.assert_awaited_once()
        assert result is False
        self.logger_error_mock.assert_called_once_with(
            "Redis ping failed: Connection error"
        )

    @pytest.mark.asyncio
    @patch("app.core.redis_connection.aioredis.Redis")
    async def test_close_success(self, mock_redis):
        """Test close method when client and pool are initialized."""
        # Arrange
        mock_redis_instance = AsyncMock()
        mock_pool_instance = AsyncMock()
        self.redis_manager._client = mock_redis_instance
        self.redis_manager._pool = mock_pool_instance

        # Act
        await self.redis_manager.close()

        # Assert
        mock_redis_instance.close.assert_awaited_once()
        mock_pool_instance.disconnect.assert_awaited_once()
        assert self.redis_manager._client is None
        assert self.redis_manager._pool is None
        self.logger_info_mock.assert_any_call("Closing Redis connections")
        self.logger_info_mock.assert_any_call("Redis connections closed")

    @pytest.mark.asyncio
    async def test_close_not_initialized(self):
        """Test close method when client is not initialized."""
        # Act
        await self.redis_manager.close()

        # Assert
        self.logger_info_mock.assert_not_called()
        self.logger_warning_mock.assert_not_called()
        self.logger_error_mock.assert_not_called()
        assert self.redis_manager._client is None
        assert self.redis_manager._pool is None
