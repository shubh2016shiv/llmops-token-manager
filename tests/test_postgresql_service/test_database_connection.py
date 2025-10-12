"""
Unit Tests for Database Connection Manager
-------------------------------------------
Tests for connection pooling, query execution, and transaction management.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import psycopg2
from psycopg2 import pool

from app.core.database_connection import DatabaseManager, initialize_db, close_db, get_db_manager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Reset singleton instance for each test
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def tearDown(self):
        """Clean up after each test."""
        # Close any open connections
        if DatabaseManager._pool is not None:
            try:
                DatabaseManager._pool.closeall()
            except:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def test_singleton_pattern(self):
        """Test that DatabaseManager implements singleton pattern."""
        manager1 = DatabaseManager()
        manager2 = DatabaseManager()
        
        self.assertIs(manager1, manager2, "DatabaseManager should be a singleton")

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    @patch('app.core.database_connection.settings')
    def test_initialize_with_default_config(self, mock_settings, mock_pool_class):
        """Test initialization with default configuration from settings."""
        # Mock settings
        mock_settings.database_host = 'localhost'
        mock_settings.database_port = 5432
        mock_settings.database_name = 'mydb'
        mock_settings.database_user = 'myuser'
        mock_settings.database_password = 'mypassword'
        mock_settings.database_pool_size = 1
        mock_settings.database_max_overflow = 4

        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        manager = DatabaseManager()
        manager.initialize()

        # Verify pool was created with correct parameters
        mock_pool_class.assert_called_once_with(
            minconn=1,
            maxconn=5,
            host='localhost',
            port=5432,
            dbname='mydb',
            user='myuser',
            password='mypassword'
        )
        self.assertEqual(manager._pool, mock_pool)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_initialize_with_custom_config(self, mock_pool_class):
        """Test initialization with custom configuration."""
        custom_config = {
            'host': 'testhost',
            'port': 5433,
            'dbname': 'testdb',
            'user': 'testuser',
            'password': 'testpass',
            'min_connections': 2,
            'max_connections': 10
        }

        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        manager = DatabaseManager()
        manager.initialize(config=custom_config)

        mock_pool_class.assert_called_once_with(
            minconn=2,
            maxconn=10,
            host='testhost',
            port=5433,
            dbname='testdb',
            user='testuser',
            password='testpass'
        )

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_initialize_already_initialized(self, mock_pool_class):
        """Test that re-initialization is prevented."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        # Try to initialize again
        manager.initialize(config=config)

        # Pool should only be created once
        self.assertEqual(mock_pool_class.call_count, 1)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_initialize_connection_error(self, mock_pool_class):
        """Test handling of connection errors during initialization."""
        mock_pool_class.side_effect = psycopg2.OperationalError("Connection failed")

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        
        with self.assertRaises(psycopg2.OperationalError):
            manager.initialize(config=config)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_close_pool(self, mock_pool_class):
        """Test closing the connection pool."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        manager.close()

        mock_pool.closeall.assert_called_once()
        self.assertIsNone(manager._pool)

    def test_close_uninitialized_pool(self):
        """Test closing an uninitialized pool (should not raise error)."""
        manager = DatabaseManager()
        manager.close()  # Should not raise any exception

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_connection_success(self, mock_pool_class):
        """Test getting a connection from the pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        conn = manager.get_connection()

        mock_pool.getconn.assert_called_once()
        self.assertEqual(conn, mock_conn)

    def test_get_connection_uninitialized(self):
        """Test getting connection from uninitialized pool raises error."""
        manager = DatabaseManager()
        
        with self.assertRaises(RuntimeError) as context:
            manager.get_connection()
        
        self.assertIn("Database not initialized", str(context.exception))

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_release_connection(self, mock_pool_class):
        """Test releasing a connection back to the pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        manager.release_connection(mock_conn)

        mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_release_connection_uninitialized(self):
        """Test releasing connection to uninitialized pool (should not crash)."""
        manager = DatabaseManager()
        mock_conn = MagicMock()
        
        # Should not raise exception
        manager.release_connection(mock_conn)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_fetch_all(self, mock_pool_class):
        """Test executing query with fetch='all'."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        # Mock data
        mock_data = [{'id': 1, 'name': 'test1'}, {'id': 2, 'name': 'test2'}]
        mock_cursor.fetchall.return_value = mock_data
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        query = "SELECT * FROM test_table"
        result = manager.execute_query(query, fetch='all')

        mock_cursor.execute.assert_called_once_with(query, None)
        mock_cursor.fetchall.assert_called_once()
        self.assertEqual(result, mock_data)
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_fetch_one(self, mock_pool_class):
        """Test executing query with fetch='one'."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_data = {'id': 1, 'name': 'test1'}
        mock_cursor.fetchone.return_value = mock_data
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        query = "SELECT * FROM test_table WHERE id = %s"
        params = (1,)
        result = manager.execute_query(query, params=params, fetch='one')

        mock_cursor.execute.assert_called_once_with(query, params)
        mock_cursor.fetchone.assert_called_once()
        self.assertEqual(result, mock_data)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_fetch_count(self, mock_pool_class):
        """Test executing query with fetch='count'."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        query = "UPDATE test_table SET name = %s WHERE id = %s"
        params = ('updated', 1)
        result = manager.execute_query(query, params=params, fetch='count')

        self.assertEqual(result, 5)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_no_fetch(self, mock_pool_class):
        """Test executing query with fetch=None (commit only)."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        query = "INSERT INTO test_table (name) VALUES (%s)"
        params = ('test',)
        result = manager.execute_query(query, params=params, fetch=None)

        mock_cursor.execute.assert_called_once_with(query, params)
        mock_conn.commit.assert_called_once()
        self.assertIsNone(result)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_error_handling(self, mock_pool_class):
        """Test error handling in execute_query."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        # Simulate database error
        mock_cursor.execute.side_effect = psycopg2.DatabaseError("Query failed")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        query = "INVALID SQL"
        
        with self.assertRaises(psycopg2.DatabaseError):
            manager.execute_query(query)
        
        # Verify rollback was called
        mock_conn.rollback.assert_called_once()
        # Verify connection was released
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_transaction_success(self, mock_pool_class):
        """Test executing multiple queries in a transaction."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        queries = [
            ("INSERT INTO test_table (name) VALUES (%s)", ('test1',)),
            ("INSERT INTO test_table (name) VALUES (%s)", ('test2',)),
            ("UPDATE test_table SET active = %s", (True,))
        ]
        
        result = manager.execute_transaction(queries)

        self.assertTrue(result)
        self.assertEqual(mock_cursor.execute.call_count, 3)
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_transaction_error(self, mock_pool_class):
        """Test transaction rollback on error."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        # First query succeeds, second fails
        mock_cursor.execute.side_effect = [None, psycopg2.DatabaseError("Query failed")]
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.side_effect = psycopg2.DatabaseError("Query failed")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)
        
        queries = [
            ("INSERT INTO test_table (name) VALUES (%s)", ('test1',)),
            ("INVALID SQL", ())
        ]
        
        result = manager.execute_transaction(queries)

        self.assertFalse(result)
        mock_pool.putconn.assert_called_once_with(mock_conn)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for module-level convenience functions."""

    def setUp(self):
        """Set up test fixtures before each test."""
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def tearDown(self):
        """Clean up after each test."""
        if DatabaseManager._pool is not None:
            try:
                DatabaseManager._pool.closeall()
            except:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_initialize_db(self, mock_pool_class):
        """Test initialize_db convenience function."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        initialize_db(config)

        mock_pool_class.assert_called_once()
        # Access pool through the singleton instance
        manager = get_db_manager()
        self.assertIsNotNone(manager._pool)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_close_db(self, mock_pool_class):
        """Test close_db convenience function."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        initialize_db(config)
        close_db()

        mock_pool.closeall.assert_called_once()
        # Access pool through the singleton instance
        manager = get_db_manager()
        self.assertIsNone(manager._pool)

    def test_get_db_manager(self):
        """Test get_db_manager convenience function."""
        manager = get_db_manager()
        
        self.assertIsInstance(manager, DatabaseManager)
        self.assertIs(manager, get_db_manager(), "Should return same instance")


class TestDatabaseManagerIntegration(unittest.TestCase):
    """Integration tests with actual database configuration."""

    def setUp(self):
        """Set up test fixtures before each test."""
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    def tearDown(self):
        """Clean up after each test."""
        if DatabaseManager._pool is not None:
            try:
                DatabaseManager._pool.closeall()
            except:
                pass
        DatabaseManager._instance = None
        DatabaseManager._pool = None

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_multiple_connections(self, mock_pool_class):
        """Test getting multiple connections from pool."""
        mock_pool = MagicMock()
        mock_conn1 = MagicMock()
        mock_conn2 = MagicMock()
        mock_pool.getconn.side_effect = [mock_conn1, mock_conn2]
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)

        conn1 = manager.get_connection()
        conn2 = manager.get_connection()

        self.assertEqual(mock_pool.getconn.call_count, 2)
        self.assertIsNot(conn1, conn2)

        manager.release_connection(conn1)
        manager.release_connection(conn2)

        self.assertEqual(mock_pool.putconn.call_count, 2)

    @patch('app.core.database_connection.psycopg2.pool.ThreadedConnectionPool')
    def test_connection_lifecycle(self, mock_pool_class):
        """Test complete connection lifecycle: get, use, release."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_cursor.fetchall.return_value = [{'count': 10}]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mydb',
            'user': 'myuser',
            'password': 'mypassword',
            'min_connections': 1,
            'max_connections': 5
        }

        manager = DatabaseManager()
        manager.initialize(config=config)

        # Execute query (which gets, uses, and releases connection)
        result = manager.execute_query("SELECT COUNT(*) as count FROM test_table")

        # Verify connection was acquired and released
        mock_pool.getconn.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)
        self.assertEqual(result, [{'count': 10}])


if __name__ == '__main__':
    unittest.main()
