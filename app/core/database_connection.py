"""
Database Connection Manager
---------------------------
Manages PostgreSQL database connections with connection pooling.
Provides direct access to database operations.
"""
import logging
import sys
from typing import Any
from typing import Dict
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""

    _instance = None
    _pool = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize database connection pool.

        Args:
            config: Optional configuration override
        """
        if self._pool is not None:
            logger.warning("Database connection pool already initialized")
            return

        # Use provided config or settings
        if config is None:
            # Lazy import to avoid circular dependencies and import issues
            from app.core.config_manager import settings

            config = {
                "host": settings.database_host,
                "port": settings.database_port,
                "dbname": settings.database_name,
                "user": settings.database_user,
                "password": settings.database_password,
                "min_connections": settings.database_pool_size,
                "max_connections": settings.database_pool_size + settings.database_max_overflow,
            }

        logger.info(f"Initializing database connection to {config.get('host')}:{config.get('port')}")

        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=config.get("min_connections", 1),
                maxconn=config.get("max_connections", 10),
                host=config.get("host", "localhost"),
                port=config.get("port", 5432),
                dbname=config.get("dbname", "mydb"),
                user=config.get("user", "myuser"),
                password=config.get("password", "mypassword"),
            )
            logger.info("Database connection pool initialized successfully")
        except psycopg2.Error as e:
            logger.error(f"Error initializing database connection pool: {e}")
            raise

    def close(self) -> None:
        """Close database connection pool."""
        if self._pool is None:
            return

        logger.info("Closing database connections")
        self._pool.closeall()
        self._pool = None
        logger.info("Database connections closed")

    def get_connection(self):
        """
        Get a connection from the pool.

        Returns:
            Database connection

        Raises:
            RuntimeError: If pool is not initialized
        """
        if self._pool is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._pool.getconn()

    def release_connection(self, conn):
        """
        Return a connection to the pool.

        Args:
            conn: Connection to release
        """
        if self._pool is None:
            logger.warning("Attempting to release connection to uninitialized pool")
            return
        self._pool.putconn(conn)

    def execute_query(self, query: str, params=None, fetch="all"):
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Result fetch mode ('all', 'one', 'count', or None)

        Returns:
            Query results based on fetch mode
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)

                if fetch == "all":
                    return cursor.fetchall()
                elif fetch == "one":
                    return cursor.fetchone()
                elif fetch == "count":
                    return cursor.rowcount
                else:
                    conn.commit()
                    return None
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_connection(conn)

    def execute_transaction(self, queries):
        """
        Execute multiple queries in a transaction.

        Args:
            queries: List of (query, params) tuples

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn:  # Automatic commit/rollback
                with conn.cursor() as cursor:
                    for query, params in queries:
                        cursor.execute(query, params)
            return True
        except psycopg2.Error as e:
            logger.error(f"Transaction error: {e}")
            return False
        finally:
            if conn:
                self.release_connection(conn)


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def initialize_db(config=None):
    """Initialize database connection pool."""
    db_manager.initialize(config)


def close_db():
    """Close all database connections."""
    db_manager.close()


def get_db_manager():
    """Get database manager instance."""
    return db_manager


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Test database connection with config
    test_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "mydb",
        "user": "myuser",
        "password": "mypassword",
        "min_connections": 1,
        "max_connections": 5,
    }

    try:
        print(f"Attempting to connect to database at {test_config['host']}:{test_config['port']}")
        initialize_db(config=test_config)

        # Test a simple query
        result = db_manager.execute_query("SELECT 1 as test", fetch="one")
        print(f"Connection test result: {result}")

        # Close connection
        close_db()
        print("Database connection test completed successfully")
    except Exception as e:
        print(f"Error testing database connection: {e}")
        sys.exit(1)
