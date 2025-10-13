import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg

from app.core.database_connection import DatabaseManager


class TestDatabaseManager:
    """Test cases for the DatabaseManager class."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up and clean up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None
        yield
        # Clean up after each test
        if DatabaseManager._pool is not None:
            try:
                await DatabaseManager._pool.close()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def test_singleton_pattern(self):
        """Test that DatabaseManager implements singleton pattern."""
        # Act
        instance1 = DatabaseManager()
        instance2 = DatabaseManager()

        # Assert
        assert instance1 is instance2

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_initialize(self, mock_pool_class):
        """Test that initialize creates a connection pool."""
        # Arrange
        mock_pool_instance = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Act
        await db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
                "min_connections": 1,
                "max_connections": 10,
            }
        )

        # Assert
        mock_pool_class.assert_called_once_with(
            min_size=1,
            max_size=10,
            kwargs={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            },
        )
        mock_pool_instance.open.assert_called_once()
        assert db_manager._pool is not None
        assert db_manager._pool == mock_pool_instance

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_initialize_with_error(self, mock_pool_class):
        """Test that initialize handles errors."""
        # Arrange
        mock_pool_class.side_effect = psycopg.Error("Connection error")
        db_manager = DatabaseManager()

        # Act & Assert
        with pytest.raises(psycopg.Error):
            await db_manager.initialize(
                config={
                    "host": "localhost",
                    "port": 5432,
                    "dbname": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                }
            )

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_get_connection(self, mock_pool_class):
        """Test that get_connection returns a connection from the pool."""
        # Arrange
        mock_connection = AsyncMock()
        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Initialize the connection pool
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
        connection = await db_manager.get_connection()

        # Assert
        assert connection == mock_connection
        mock_pool_instance.getconn.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_connection_not_initialized(self):
        """Test that get_connection raises error when not initialized."""
        # Arrange
        db_manager = DatabaseManager()

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            await db_manager.get_connection()

        assert "not initialized" in str(exc_info.value)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_release_connection(self, mock_pool_class):
        """Test that release_connection returns a connection to the pool."""
        # Arrange
        mock_connection = AsyncMock()
        mock_pool_instance = AsyncMock()
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Initialize the connection pool
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
        await db_manager.release_connection(mock_connection)

        # Assert
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_close(self, mock_pool_class):
        """Test that close closes all connections in the pool."""
        # Arrange
        mock_pool_instance = AsyncMock()
        mock_pool_instance.close = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Initialize the connection pool
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
        mock_pool_instance.close.assert_called_once()
        assert DatabaseManager._pool is None


class TestDatabaseManagerQueries:
    """Test cases for query execution in DatabaseManager."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up and clean up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None
        yield
        # Clean up after each test
        if DatabaseManager._pool is not None:
            try:
                await DatabaseManager._pool.close()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_query_fetch_all(self, mock_pool_class):
        """Test executing a query and fetching all results."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}]
        )
        mock_cursor.execute = AsyncMock()

        # Mock the cursor context manager properly
        cursor_context = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=False)
        mock_connection.cursor = MagicMock(return_value=cursor_context)
        mock_connection.commit = AsyncMock()

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
        result = await db_manager.execute_query("SELECT * FROM test_table", fetch="all")

        # Assert
        assert result == [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}]
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table", None)
        mock_cursor.fetchall.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_query_fetch_one(self, mock_pool_class):
        """Test executing a query and fetching one result."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"id": 1, "name": "Test"})
        mock_cursor.execute = AsyncMock()
        cursor_context = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=False)
        mock_connection.cursor = MagicMock(return_value=cursor_context)
        mock_connection.commit = AsyncMock()

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
        result = await db_manager.execute_query(
            "SELECT * FROM test_table WHERE id = %s", params=(1,), fetch="one"
        )

        # Assert
        assert result == {"id": 1, "name": "Test"}
        mock_cursor.execute.assert_called_once_with(
            "SELECT * FROM test_table WHERE id = %s", (1,)
        )
        mock_cursor.fetchone.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_query_fetch_count(self, mock_pool_class):
        """Test executing a query and getting row count."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 5
        mock_cursor.execute = AsyncMock()
        cursor_context = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=False)
        mock_connection.cursor = MagicMock(return_value=cursor_context)
        mock_connection.commit = AsyncMock()

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
        result = await db_manager.execute_query(
            "UPDATE test_table SET name = %s", params=("NewName",), fetch="count"
        )

        # Assert
        assert result == 5
        mock_cursor.execute.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_query_no_fetch(self, mock_pool_class):
        """Test executing a query without fetching results."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        cursor_context = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=False)
        mock_connection.cursor = MagicMock(return_value=cursor_context)
        mock_connection.commit = AsyncMock()

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
        result = await db_manager.execute_query(
            "INSERT INTO test_table (name) VALUES (%s)", params=("Test",), fetch=None
        )

        # Assert
        assert result is None
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_query_with_error(self, mock_pool_class):
        """Test that execute_query handles errors."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Query error"))
        cursor_context = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=False)
        mock_connection.cursor = MagicMock(return_value=cursor_context)
        mock_connection.rollback = AsyncMock()

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
        with pytest.raises(psycopg.Error):
            await db_manager.execute_query("INVALID SQL")

        mock_connection.rollback.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)


class TestDatabaseManagerTransactions:
    """Test cases for transaction management in DatabaseManager."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Set up and clean up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None
        yield
        # Clean up after each test
        if DatabaseManager._pool is not None:
            try:
                await DatabaseManager._pool.close()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_transaction(self, mock_pool_class):
        """Test successful transaction execution."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        cursor_context = AsyncMock()
        cursor_context.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_context.__aexit__ = AsyncMock(return_value=False)
        mock_connection.cursor = MagicMock(return_value=cursor_context)
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=False)

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
            ("INSERT INTO test_table (name) VALUES (%s)", ("Test",)),
            ("UPDATE test_table SET name = %s WHERE id = %s", ("Updated", 1)),
        ]

        # Act
        result = await db_manager.execute_transaction(queries)

        # Assert
        assert result is True
        assert mock_cursor.execute.call_count == 2
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.AsyncConnectionPool")
    @pytest.mark.asyncio
    async def test_execute_transaction_with_error(self, mock_pool_class):
        """Test transaction rollback on error."""
        # Arrange
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock(side_effect=psycopg.Error("Transaction error"))
        mock_connection.cursor.return_value.__aenter__ = AsyncMock(
            return_value=mock_cursor
        )
        mock_connection.cursor.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_connection.__aenter__ = AsyncMock(
            side_effect=psycopg.Error("Transaction error")
        )
        mock_connection.__aexit__ = AsyncMock(return_value=False)

        mock_pool_instance = AsyncMock()
        mock_pool_instance.getconn = AsyncMock(return_value=mock_connection)
        mock_pool_instance.putconn = AsyncMock()
        mock_pool_instance.open = AsyncMock()
        mock_pool_class.return_value = mock_pool_instance

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
            ("INSERT INTO test_table (name) VALUES (%s)", ("Test",)),
        ]

        # Act
        result = await db_manager.execute_transaction(queries)

        # Assert
        assert result is False
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)


# Run with: pytest tests/test_psql_db_services/test_database_connection.py -v
