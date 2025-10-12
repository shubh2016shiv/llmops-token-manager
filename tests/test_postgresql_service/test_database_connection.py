import unittest
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import psycopg2

from app.core.database_connection import DatabaseError
from app.core.database_connection import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for the DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_get_instance(self, mock_pool):
        """Test that get_instance returns a singleton instance."""
        # Arrange
        mock_pool.return_value = MagicMock()

        # Act
        instance1 = DatabaseManager.get_instance()
        instance2 = DatabaseManager.get_instance()

        # Assert
        self.assertIs(instance1, instance2)
        mock_pool.assert_called_once()

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

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_init_connection_pool(self, mock_pool):
        """Test that init_connection_pool creates a connection pool."""
        # Arrange
        mock_pool.return_value = MagicMock()

        # Act
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
            min_connections=1,
            max_connections=10,
        )

        # Assert
        mock_pool.assert_called_once_with(
            1,
            10,
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )
        self.assertIsNotNone(DatabaseManager._pool)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_init_connection_pool_with_error(self, mock_pool):
        """Test that init_connection_pool handles errors."""
        # Arrange
        mock_pool.side_effect = psycopg2.Error("Connection error")

        # Act & Assert
        with self.assertRaises(DatabaseError):
            DatabaseManager.init_connection_pool(
                host="localhost",
                port=5432,
                dbname="test_db",
                user="test_user",
                password="test_password",
            )

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_get_connection(self, mock_pool):
        """Test that get_connection returns a connection from the pool."""
        # Arrange
        mock_connection = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act
        connection = DatabaseManager.get_instance().get_connection()

        # Assert
        self.assertEqual(connection, mock_connection)
        mock_pool_instance.getconn.assert_called_once()

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_get_connection_with_error(self, mock_pool):
        """Test that get_connection handles errors."""
        # Arrange
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.side_effect = psycopg2.Error("Connection error")
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act & Assert
        with self.assertRaises(DatabaseError):
            DatabaseManager.get_instance().get_connection()

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_release_connection(self, mock_pool):
        """Test that release_connection returns a connection to the pool."""
        # Arrange
        mock_connection = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act
        DatabaseManager.get_instance().release_connection(mock_connection)

        # Assert
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_release_connection_with_error(self, mock_pool):
        """Test that release_connection handles errors."""
        # Arrange
        mock_connection = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.putconn.side_effect = psycopg2.Error("Connection error")
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act & Assert
        with self.assertRaises(DatabaseError):
            DatabaseManager.get_instance().release_connection(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_close_all_connections(self, mock_pool):
        """Test that close_all_connections closes all connections in the pool."""
        # Arrange
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act
        DatabaseManager.get_instance().close_all_connections()

        # Assert
        mock_pool_instance.closeall.assert_called_once()

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_close_all_connections_with_error(self, mock_pool):
        """Test that close_all_connections handles errors."""
        # Arrange
        mock_pool_instance = MagicMock()
        mock_pool_instance.closeall.side_effect = psycopg2.Error("Connection error")
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act & Assert
        with self.assertRaises(DatabaseError):
            DatabaseManager.get_instance().close_all_connections()


class TestDatabaseManagerIntegration(unittest.TestCase):
    """Integration tests for the DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance before each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None

        # Initialize the connection pool with test database credentials
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
            min_connections=1,
            max_connections=5,
        )

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
    def test_execute_query(self, mock_pool):
        """Test executing a query."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        # Act
        instance = DatabaseManager.get_instance()
        instance.execute_query("SELECT * FROM test_table WHERE id = %s", (1,))

        # Assert
        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table WHERE id = %s", (1,))
        mock_cursor.close.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_fetch_one(self, mock_pool):
        """Test fetching a single row."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "Test"}
        mock_connection.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        # Act
        instance = DatabaseManager.get_instance()
        result = instance.fetch_one("SELECT * FROM test_table WHERE id = %s", (1,))

        # Assert
        self.assertEqual(result, {"id": 1, "name": "Test"})
        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table WHERE id = %s", (1,))
        mock_cursor.fetchone.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_fetch_all(self, mock_pool):
        """Test fetching all rows."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}]
        mock_connection.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        # Act
        instance = DatabaseManager.get_instance()
        result = instance.fetch_all("SELECT * FROM test_table")

        # Assert
        self.assertEqual(result, [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}])
        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table", None)
        mock_cursor.fetchall.assert_called_once()
        mock_cursor.close.assert_called_once()
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
    def test_transaction_commit(self, mock_pool):
        """Test successful transaction commit."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act
        instance = DatabaseManager.get_instance()
        with instance.transaction() as cursor:
            cursor.execute("INSERT INTO test_table (name) VALUES (%s)", ("Test",))
            cursor.execute("UPDATE test_table SET name = %s WHERE id = %s", ("Updated", 1))

        # Assert
        mock_connection.cursor.assert_called_once()
        self.assertEqual(mock_cursor.execute.call_count, 2)
        mock_cursor.execute.assert_has_calls(
            [
                call("INSERT INTO test_table (name) VALUES (%s)", ("Test",)),
                call("UPDATE test_table SET name = %s WHERE id = %s", ("Updated", 1)),
            ]
        )
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)

    @patch("app.core.database_connection.psycopg2.pool.ThreadedConnectionPool")
    def test_transaction_rollback_on_exception(self, mock_pool):
        """Test transaction rollback on exception."""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance

        # Second query raises an exception
        mock_cursor.execute.side_effect = [None, psycopg2.Error("Database error")]

        # Initialize the connection pool
        DatabaseManager.init_connection_pool(
            host="localhost",
            port=5432,
            dbname="test_db",
            user="test_user",
            password="test_password",
        )

        # Act & Assert
        instance = DatabaseManager.get_instance()
        with self.assertRaises(DatabaseError):
            with instance.transaction() as cursor:
                cursor.execute("INSERT INTO test_table (name) VALUES (%s)", ("Test",))
                cursor.execute("INVALID SQL")

        # Assert
        mock_connection.cursor.assert_called_once()
        self.assertEqual(mock_cursor.execute.call_count, 2)
        mock_connection.rollback.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_connection)


if __name__ == "__main__":
    unittest.main()
