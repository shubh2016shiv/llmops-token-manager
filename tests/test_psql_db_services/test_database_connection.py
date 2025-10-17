"""
Unit tests for DatabaseManager using SQLAlchemy async engine
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.core.database_connection import DatabaseManager


class TestDatabaseManager:
    """Test cases for the DatabaseManager class."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up and clean up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None
        yield
        # Clean up after each test
        if DatabaseManager._engine is not None:
            try:
                await DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    def test_singleton_pattern(self):
        """Test that DatabaseManager implements singleton pattern."""
        # Act
        instance1 = DatabaseManager()
        instance2 = DatabaseManager()

        # Assert
        assert instance1 is instance2

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_initialize(self, mock_sessionmaker, mock_create_engine):
        """Test that initialize creates a SQLAlchemy async engine."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()

        # Act
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
                "min_connections": 5,
                "max_connections": 15,
            }
        )

        # Assert
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args
        assert (
            "postgresql+asyncpg://test_user:test_password@localhost:5432/test_db"
            in call_args[0]
        )
        assert call_args[1]["pool_size"] == 5
        assert call_args[1]["max_overflow"] == 10
        assert call_args[1]["pool_pre_ping"] is True

        mock_sessionmaker.assert_called_once()
        assert db_manager._engine is not None
        assert db_manager._sessionmaker is not None

    @patch("app.core.database_connection.create_async_engine")
    @pytest.mark.asyncio
    async def test_initialize_with_error(self, mock_create_engine):
        """Test that initialize handles errors."""
        # Arrange
        mock_create_engine.side_effect = SQLAlchemyError("Connection error")
        db_manager = DatabaseManager()

        # Act & Assert
        with pytest.raises(Exception):
            await db_manager.initialize(
                config={
                    "host": "localhost",
                    "port": 5432,
                    "dbname": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                }
            )

    @patch("app.core.database_connection.create_async_engine")
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_create_engine):
        """Test that initialize warns when already initialized."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine
        db_manager = DatabaseManager()
        db_manager._engine = mock_engine  # Already initialized

        # Act
        await db_manager.initialize()

        # Assert - should not create a new engine
        mock_create_engine.assert_not_called()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_close(self, mock_sessionmaker, mock_create_engine):
        """Test that close disposes the engine."""
        # Arrange
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = MagicMock()

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        await db_manager.close()

        # Assert
        mock_engine.dispose.assert_called_once()
        assert DatabaseManager._engine is None
        assert DatabaseManager._sessionmaker is None

    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self):
        """Test that get_session raises error when not initialized."""
        # Arrange
        db_manager = DatabaseManager()

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            async with db_manager.get_session():
                pass

        assert "not initialized" in str(exc_info.value).lower()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_get_session_success(self, mock_sessionmaker, mock_create_engine):
        """Test that get_session returns a session and commits on success."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        async with db_manager.get_session() as session:
            assert session == mock_session

        # Assert
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test that get_session rolls back on exception."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act & Assert
        with pytest.raises(ValueError):
            async with db_manager.get_session() as _:
                raise ValueError("Test error")

        # Assert
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()


class TestDatabaseManagerQueries:
    """Test cases for query execution in DatabaseManager."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up and clean up test fixtures."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None
        yield
        if DatabaseManager._engine is not None:
            try:
                await DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_all(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test executing a query and fetching all results."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        # Mock result object
        mock_row1 = {"id": 1, "name": "Test1"}
        mock_row2 = {"id": 2, "name": "Test2"}
        mock_mappings = MagicMock()
        mock_mappings.all.return_value = [mock_row1, mock_row2]

        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_mappings

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = await db_manager.execute_raw_query(
            "SELECT * FROM test_table", fetch_mode="all"
        )

        # Assert
        assert result == [mock_row1, mock_row2]
        mock_session.execute.assert_called_once()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_one(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test executing a query and fetching one result."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_row = {"id": 1, "name": "Test"}
        mock_mappings = MagicMock()
        mock_mappings.one_or_none.return_value = mock_row

        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_mappings

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = await db_manager.execute_raw_query(
            "SELECT * FROM test_table WHERE id = :id",
            params={"id": 1},
            fetch_mode="one",
        )

        # Assert
        assert result == mock_row
        mock_session.execute.assert_called_once()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_scalar(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test executing a query and fetching a scalar value."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 42

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = await db_manager.execute_raw_query(
            "SELECT COUNT(*) FROM test_table", fetch_mode="scalar"
        )

        # Assert
        assert result == 42
        mock_session.execute.assert_called_once()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_count(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test executing a query and getting row count."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.rowcount = 5

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = await db_manager.execute_raw_query(
            "UPDATE test_table SET name = :name",
            params={"name": "NewName"},
            fetch_mode="count",
        )

        # Assert
        assert result == 5
        mock_session.execute.assert_called_once()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_raw_query_no_fetch(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test executing a query without fetching results."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_result = MagicMock()

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = await db_manager.execute_raw_query(
            "INSERT INTO test_table (name) VALUES (:name)",
            params={"name": "Test"},
            fetch_mode=None,
        )

        # Assert
        assert result is None
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()


class TestDatabaseManagerTransactions:
    """Test cases for transaction management in DatabaseManager."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up and clean up test fixtures."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None
        yield
        if DatabaseManager._engine is not None:
            try:
                await DatabaseManager._engine.dispose()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_transaction(self, mock_sessionmaker, mock_create_engine):
        """Test successful transaction execution."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_result = MagicMock()

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        queries = [
            {
                "query": "INSERT INTO test_table (name) VALUES (:name)",
                "params": {"name": "Test"},
            },
            {
                "query": "UPDATE test_table SET name = :name WHERE id = :id",
                "params": {"name": "Updated", "id": 1},
            },
        ]

        # Act
        result = await db_manager.execute_transaction(queries)

        # Assert
        assert result is True
        assert mock_session.execute.call_count == 2
        mock_session.commit.assert_called_once()

    @patch("app.core.database_connection.create_async_engine")
    @patch("app.core.database_connection.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_execute_transaction_with_error(
        self, mock_sessionmaker, mock_create_engine
    ):
        """Test transaction rollback on error."""
        # Arrange
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(
            side_effect=SQLAlchemyError("Transaction error")
        )
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_session_factory = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        db_manager = DatabaseManager()
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        queries = [
            {
                "query": "INSERT INTO test_table (name) VALUES (:name)",
                "params": {"name": "Test"},
            },
        ]

        # Act
        result = await db_manager.execute_transaction(queries)

        # Assert
        assert result is False
        mock_session.rollback.assert_called_once()


# Run with: pytest tests/test_psql_db_services/test_database_connection.py -v
