import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import psycopg2

from app.core.database_connection import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for the DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def tearDown(self):
        """Clean up after each test."""
        # Close any open connections
        if DatabaseManager._pool is not None:
            try:
                DatabaseManager._pool.closeall()
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
        self.assertIs(instance1, instance2)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_initialize(self, mock_pool):
        """Test that initialize creates a connection pool."""
        # Arrange
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Act
        db_manager.initialize(
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
        mock_pool.assert_called_once_with(
            minconn=1,
            maxconn=10,
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )
        self.assertIsNotNone(db_manager._pool)
        self.assertEqual(db_manager._pool, mock_pool_instance)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_initialize_with_error(self, mock_pool):
        """Test that initialize handles errors."""
        # Arrange
        mock_pool.side_effect = psycopg2.Error("Connection error")
        db_manager = DatabaseManager()

        # Act & Assert
        with self.assertRaises(psycopg2.Error):
            db_manager.initialize(
                config={
                    "host": "localhost",
                    "port": 5432,
                    "dbname": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                }
            )

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_get_connection(self, mock_pool):
        """Test that get_connection returns a connection from the pool."""
        # Arrange
        mock_connection = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Initialize the connection pool
        db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        connection = db_manager.get_connection()

        # Assert
        self.assertEqual(connection, mock_connection)
        mock_pool_instance.getconn.assert_called_once()

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_get_connection_not_initialized(self, mock_pool):
        """Test that get_connection raises error when not initialized."""
        # Arrange
        db_manager = DatabaseManager()

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            db_manager.get_connection()

        self.assertIn("not initialized", str(context.exception))

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_release_connection(self, mock_pool):
        """Test that release_connection returns a connection to the pool."""
        # Arrange
        mock_connection = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Initialize the connection pool
        db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        db_manager.release_connection(mock_connection)

        # Assert
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_close(self, mock_pool):
        """Test that close closes all connections in the pool."""
        # Arrange
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        db_manager = DatabaseManager()

        # Initialize the connection pool
        db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        db_manager.close()

        # Assert
        mock_pool_instance.closeall.assert_called_once()
        self.assertIsNone(DatabaseManager._pool)


class TestDatabaseManagerQueries(unittest.TestCase):
    """Test cases for query execution in DatabaseManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def tearDown(self):
        """Clean up after each test."""
        if DatabaseManager._pool is not None:
            try:
                DatabaseManager._pool.closeall()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_execute_query_fetch_all(self, mock_pool):
        """Test executing a query and fetching all results."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}]
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        db_manager = DatabaseManager()
        db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = db_manager.execute_query("SELECT * FROM test_table", fetch="all")

        # Assert
        self.assertEqual(result, [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}])
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table", None)
        mock_cursor.fetchall.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_execute_query_fetch_one(self, mock_pool):
        """Test executing a query and fetching one result."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "Test"}
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        db_manager = DatabaseManager()
        db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act
        result = db_manager.execute_query("SELECT * FROM test_table WHERE id = %s", params=(1,), fetch="one")

        # Assert
        self.assertEqual(result, {"id": 1, "name": "Test"})
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table WHERE id = %s", (1,))
        mock_cursor.fetchone.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_execute_query_with_error(self, mock_pool):
        """Test that execute_query handles errors."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = psycopg2.Error("Query error")
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        db_manager = DatabaseManager()
        db_manager.initialize(
            config={
                "host": "localhost",
                "port": 5432,
                "dbname": "test_db",
                "user": "test_user",
                "password": "test_password",
            }
        )

        # Act & Assert
        with self.assertRaises(psycopg2.Error):
            db_manager.execute_query("INVALID SQL")

        mock_connection.rollback.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)


class TestDatabaseManagerTransactions(unittest.TestCase):
    """Test cases for transaction management in DatabaseManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def tearDown(self):
        """Clean up after each test."""
        if DatabaseManager._pool is not None:
            try:
                DatabaseManager._pool.closeall()
            except Exception:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_execute_transaction(self, mock_pool):
        """Test successful transaction execution."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor_context.__exit__ = MagicMock(return_value=False)
        mock_connection.cursor.return_value = mock_cursor_context
        mock_connection.__enter__ = MagicMock(return_value=mock_connection)
        mock_connection.__exit__ = MagicMock(return_value=False)
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        db_manager = DatabaseManager()
        db_manager.initialize(
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
        result = db_manager.execute_transaction(queries)

        # Assert
        self.assertTrue(result)
        self.assertEqual(mock_cursor.execute.call_count, 2)
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_execute_transaction_with_error(self, mock_pool):
        """Test transaction rollback on error."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = psycopg2.Error("Transaction error")
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.__enter__.side_effect = psycopg2.Error("Transaction error")
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        db_manager = DatabaseManager()
        db_manager.initialize(
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
        result = db_manager.execute_transaction(queries)

        # Assert
        self.assertFalse(result)
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)


if __name__ == "__main__":
    unittest.main()
