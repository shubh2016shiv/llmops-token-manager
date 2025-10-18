"""
Comprehensive Unit Tests for Database Connection Manager
======================================================
Unit tests for the DatabaseManager class and convenience functions.

Test Coverage:
- Singleton pattern (2 tests)
- Database initialization (5 tests)
- Connection cleanup (2 tests)
- Session context manager (5 tests)
- Raw query execution (6 tests)
- Transaction execution (3 tests)
- Convenience functions (3 tests)

Total: 26 comprehensive unit tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.database_connection import (
    DatabaseManager,
    initialize_db,
    close_db,
    get_db_manager,
    db_manager,
)


class TestDatabaseManagerSingleton:
    """Test singleton pattern implementation."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    def test_singleton_pattern(self):
        """Test that multiple instances return the same object."""
        # Act
        db1 = DatabaseManager()
        db2 = DatabaseManager()

        # Assert
        assert db1 is db2
        assert id(db1) == id(db2)

    def test_singleton_persistence(self):
        """Test that state persists across different instance references."""
        # Arrange
        db1 = DatabaseManager()
        db1._engine = MagicMock()
        db1._sessionmaker = MagicMock()

        # Act
        db2 = DatabaseManager()

        # Assert
        assert db2._engine is db1._engine
        assert db2._sessionmaker is db1._sessionmaker


class TestDatabaseManagerInitialization:
    """Test database initialization."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @pytest.mark.asyncio
    async def test_initialize_with_default_config(self):
        """Test initialization with default settings from config_manager."""
        # Arrange
        with patch("app.core.database_connection.create_async_engine") as mock_engine:
            with patch("app.core.database_connection.async_sessionmaker"):
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                # Act
                db = DatabaseManager()
                await db.initialize()

                # Assert
                mock_engine.assert_called_once()
                assert db._engine == mock_engine_instance

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(self):
        """Test initialization with custom configuration."""
        # Arrange
        custom_config = {
            "host": "customhost",
            "port": 5433,
            "dbname": "customdb",
            "user": "customuser",
            "password": "custompass",
            "min_connections": 5,
            "max_connections": 15,
        }

        with patch("app.core.database_connection.create_async_engine") as mock_engine:
            with patch("app.core.database_connection.async_sessionmaker"):
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                # Act
                db = DatabaseManager()
                await db.initialize(config=custom_config)

                # Assert
                mock_engine.assert_called_once()
                call_args = mock_engine.call_args
                assert (
                    "postgresql+asyncpg://customuser:custompass@customhost:5433/customdb"
                    in call_args[0][0]
                )

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, caplog):
        """Test warning when database is already initialized (Line 71 coverage)."""
        # Arrange
        with patch("app.core.database_connection.create_async_engine") as mock_engine:
            with patch("app.core.database_connection.async_sessionmaker"):
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                db = DatabaseManager()
                await db.initialize()

                # Act
                await db.initialize()

                # Assert
                assert "Database engine already initialized" in caplog.text

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, caplog):
        """Test exception handling during initialization."""
        # Arrange
        with patch("app.core.database_connection.create_async_engine") as mock_engine:
            mock_engine.side_effect = Exception("Connection failed")

            # Act & Assert
            db = DatabaseManager()
            with pytest.raises(Exception, match="Connection failed"):
                await db.initialize()

            assert (
                "Error initializing SQLAlchemy engine: Connection failed" in caplog.text
            )

    @pytest.mark.asyncio
    async def test_initialize_creates_sessionmaker(self):
        """Test that sessionmaker is created with correct parameters."""
        # Arrange
        with patch("app.core.database_connection.create_async_engine") as mock_engine:
            with patch(
                "app.core.database_connection.async_sessionmaker"
            ) as mock_sessionmaker:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                # Act
                db = DatabaseManager()
                await db.initialize()

                # Assert
                mock_sessionmaker.assert_called_once()
                call_kwargs = mock_sessionmaker.call_args[1]
                assert call_kwargs["bind"] == mock_engine_instance
                assert call_kwargs["expire_on_commit"] is False


class TestDatabaseManagerClose:
    """Test connection cleanup."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @pytest.mark.asyncio
    async def test_close_disposes_engine(self):
        """Test that close disposes the engine and resets state."""
        # Arrange
        with patch("app.core.database_connection.create_async_engine") as mock_engine:
            with patch("app.core.database_connection.async_sessionmaker"):
                mock_engine_instance = MagicMock()
                mock_engine_instance.dispose = AsyncMock()
                mock_engine.return_value = mock_engine_instance

                db = DatabaseManager()
                await db.initialize()

                # Act
                await db.close()

                # Assert
                mock_engine_instance.dispose.assert_called_once()
                assert db._engine is None
                assert db._sessionmaker is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self):
        """Test closing when database is not initialized."""
        # Arrange
        db = DatabaseManager()
        assert db._engine is None

        # Act & Assert
        # Should not raise any exception
        await db.close()


class TestDatabaseManagerGetSession:
    """Test session context manager."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @pytest.mark.asyncio
    async def test_get_session_success(self):
        """Test successful session retrieval and commit."""
        # Arrange
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_sessionmaker = MagicMock(return_value=mock_session)

        db = DatabaseManager()
        db._sessionmaker = mock_sessionmaker

        # Act
        async with db.get_session() as session:
            assert session == mock_session

        # Assert
        mock_sessionmaker.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self):
        """Test error when database is not initialized."""
        # Arrange
        db = DatabaseManager()
        db._sessionmaker = None

        # Act & Assert
        with pytest.raises(
            RuntimeError, match="Database not initialized. Call initialize\\(\\) first."
        ):
            async with db.get_session():
                pass

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_exception(self):
        """Test rollback when exception occurs in context."""
        # Arrange
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_sessionmaker = MagicMock(return_value=mock_session)

        db = DatabaseManager()
        db._sessionmaker = mock_sessionmaker

        # Act & Assert
        with pytest.raises(ValueError, match="Test error"):
            async with db.get_session():
                raise ValueError("Test error")

        # Assert
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_always_closes(self):
        """Test that session always closes even on exception."""
        # Arrange
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_sessionmaker = MagicMock(return_value=mock_session)

        db = DatabaseManager()
        db._sessionmaker = mock_sessionmaker

        # Act
        try:
            async with db.get_session():
                raise ValueError("Test error")
        except ValueError:
            pass

        # Assert
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_context_manager_flow(self):
        """Test complete session context manager flow."""
        # Arrange
        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_sessionmaker = MagicMock(return_value=mock_session)

        db = DatabaseManager()
        db._sessionmaker = mock_sessionmaker

        # Act
        async with db.get_session():
            # Simulate some work
            pass

        # Assert
        mock_sessionmaker.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()


class TestDatabaseManagerExecuteRawQuery:
    """Test raw query execution."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_all(self):
        """Test execute_raw_query with fetch_mode='all'."""
        # Arrange
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result
        mock_result.all.return_value = [{"id": 1, "name": "test"}]

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_raw_query("SELECT * FROM test", fetch_mode="all")

            # Assert
            assert result == [{"id": 1, "name": "test"}]
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_one(self):
        """Test execute_raw_query with fetch_mode='one'."""
        # Arrange
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result
        mock_result.one_or_none.return_value = {"id": 1, "name": "test"}

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_raw_query(
                "SELECT * FROM test WHERE id = 1", fetch_mode="one"
            )

            # Assert
            assert result == {"id": 1, "name": "test"}
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_scalar(self):
        """Test execute_raw_query with fetch_mode='scalar'."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 42

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_raw_query(
                "SELECT COUNT(*) FROM test", fetch_mode="scalar"
            )

            # Assert
            assert result == 42
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_count(self):
        """Test execute_raw_query with fetch_mode='count'."""
        # Arrange
        mock_result = MagicMock()
        mock_result.rowcount = 5

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_raw_query(
                "UPDATE test SET name = 'updated'", fetch_mode="count"
            )

            # Assert
            assert result == 5
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_raw_query_fetch_none(self):
        """Test execute_raw_query with fetch_mode=None."""
        # Arrange
        mock_result = MagicMock()

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_raw_query(
                "CREATE TABLE test (id INT)", fetch_mode=None
            )

            # Assert
            assert result is None
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_raw_query_with_parameters(self):
        """Test execute_raw_query with parameters."""
        # Arrange
        mock_result = MagicMock()
        mock_result.mappings.return_value = mock_result
        mock_result.all.return_value = [{"id": 1, "name": "test"}]

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_raw_query(
                "SELECT * FROM test WHERE name = :name",
                params={"name": "test"},
                fetch_mode="all",
            )

            # Assert
            assert result == [{"id": 1, "name": "test"}]
            mock_session.execute.assert_called_once()
            call_args = mock_session.execute.call_args
            assert call_args[0][1] == {"name": "test"}


class TestDatabaseManagerExecuteTransaction:
    """Test transaction execution."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @pytest.mark.asyncio
    async def test_execute_transaction_success(self):
        """Test successful transaction execution."""
        # Arrange
        mock_session = MagicMock()
        mock_session.execute = AsyncMock()

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            queries = [
                {
                    "query": "INSERT INTO test (name) VALUES (:name)",
                    "params": {"name": "test1"},
                },
                {
                    "query": "UPDATE test SET status = :status",
                    "params": {"status": "active"},
                },
            ]

            # Act
            result = await db.execute_transaction(queries)

            # Assert
            assert result is True
            assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_transaction_failure(self, caplog):
        """Test transaction rollback on error."""
        # Arrange
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("Database error"))

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            queries = [
                {
                    "query": "INSERT INTO test (name) VALUES (:name)",
                    "params": {"name": "test1"},
                },
            ]

            # Act
            result = await db.execute_transaction(queries)

            # Assert
            assert result is False
            assert "Transaction error: Database error" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_transaction_empty_queries(self):
        """Test transaction with empty query list."""
        # Arrange
        mock_session = MagicMock()
        mock_session.execute = AsyncMock()

        with patch.object(DatabaseManager, "get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            db = DatabaseManager()

            # Act
            result = await db.execute_transaction([])

            # Assert
            assert result is True
            mock_session.execute.assert_not_called()


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def setup_method(self):
        """Reset singleton state before each test."""
        DatabaseManager._instance = None
        DatabaseManager._engine = None
        DatabaseManager._sessionmaker = None

    @pytest.mark.asyncio
    async def test_initialize_db_function(self):
        """Test initialize_db convenience function (Line 253 coverage)."""
        # Arrange
        with patch.object(db_manager, "initialize") as mock_initialize:
            config = {"host": "testhost", "port": 5432}

            # Act
            await initialize_db(config)

            # Assert
            mock_initialize.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_close_db_function(self):
        """Test close_db convenience function (Line 258 coverage)."""
        # Arrange
        with patch.object(db_manager, "close") as mock_close:
            # Act
            await close_db()

            # Assert
            mock_close.assert_called_once()

    def test_get_db_manager_function(self):
        """Test get_db_manager convenience function (Line 263 coverage)."""
        # Act
        result = get_db_manager()

        # Assert
        assert result is db_manager


# Run with: pytest tests/test_database_connection.py -v
