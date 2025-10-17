"""
Database Connection Manager
---------------------------
Manages PostgreSQL database connections with SQLAlchemy async engine.
Provides both ORM and raw SQL query capabilities.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, AsyncGenerator, List, Union
from contextlib import asynccontextmanager

# SQLAlchemy imports for ORM session support
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

import asyncio
from app.core.config_manager import settings

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations using SQLAlchemy async engine.

    ARCHITECTURE NOTE:
    This manager uses SQLAlchemy's async engine with asyncpg driver for all database
    operations. This provides:

    1. High-performance connection pooling optimized for async operations
    2. Windows compatibility (no event loop issues)
    3. Both ORM and raw SQL query capabilities
    4. Automatic connection health checks and recycling

    The previous hybrid approach with psycopg AsyncConnectionPool has been removed
    to eliminate Windows event loop compatibility issues.
    """

    _instance = None
    _engine = None
    _sessionmaker = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize SQLAlchemy async engine.

        Args:
            config: Optional configuration override
        """
        if self._engine is not None:
            logger.warning("Database engine already initialized")
            return

        # Use provided config or settings
        if config is None:
            config = {
                "host": settings.database_host,
                "port": settings.database_port,
                "dbname": settings.database_name,
                "user": settings.database_user,
                "password": settings.database_password,
                "min_connections": settings.database_pool_size,
                "max_connections": settings.database_pool_size
                + settings.database_max_overflow,
            }

        logger.info(
            f"Initializing database connection to {config.get('host')}:{config.get('port')}"
        )

        try:
            # Initialize SQLAlchemy async engine with asyncpg driver
            db_url = (
                f"postgresql+asyncpg://"
                f"{config['user']}:{config['password']}@"
                f"{config['host']}:{config['port']}/"
                f"{config['dbname']}"
            )

            # Configure SQLAlchemy engine for high performance
            self._engine = create_async_engine(
                db_url,
                pool_size=config.get("min_connections", 5),
                max_overflow=config.get("max_connections", 10)
                - config.get("min_connections", 5),
                pool_pre_ping=True,  # Health checks
                pool_recycle=3600,  # Recycle connections every hour
                pool_timeout=30,  # Wait time for connection
                echo=False,  # Disable SQL echo in production
            )

            self._sessionmaker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            logger.info(
                "SQLAlchemy async engine and sessionmaker initialized successfully"
            )

        except Exception as e:
            logger.error(f"Error initializing SQLAlchemy engine: {e}")
            raise

    async def close(self) -> None:
        """Close SQLAlchemy engine."""
        if self._engine is not None:
            logger.info("Disposing SQLAlchemy engine")
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            logger.info("SQLAlchemy engine disposed")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a SQLAlchemy async session from the sessionmaker.

        This method provides ORM-based database access and raw SQL execution
        via SQLAlchemy's text() function.

        Yields:
            AsyncSession: Active SQLAlchemy session with automatic
                         commit on success or rollback on exception

        Raises:
            RuntimeError: If database not initialized

        Example:
            async with db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                value = result.scalar()
        """
        if not self._sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
            logger.debug("Session committed successfully")
        except Exception as e:
            await session.rollback()
            logger.warning(f"Session rolled back due to error: {e}")
            raise
        finally:
            await session.close()
            logger.debug("Session closed and returned to pool")

    async def execute_raw_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_mode: str = "all",
    ) -> Union[List[Dict[str, Any]], Dict[str, Any], int, None]:
        """
        Execute a raw SQL query using SQLAlchemy's text() function.

        This method provides a direct replacement for the previous execute_query()
        method but uses SQLAlchemy's async engine instead of psycopg.

        Args:
            query: SQL query string
            params: Query parameters as dictionary
            fetch_mode: Result fetch mode ('all', 'one', 'scalar', 'count', or None)

        Returns:
            Query results based on fetch mode:
            - 'all': List of dictionaries (rows)
            - 'one': Single dictionary (row)
            - 'scalar': Single value
            - 'count': Number of rows affected
            - None: No return value

        Example:
            users = await db_manager.execute_raw_query(
                "SELECT * FROM users WHERE email = :email",
                {"email": "user@example.com"}
            )
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})

            if fetch_mode == "all":
                return [dict(row) for row in result.mappings().all()]
            elif fetch_mode == "one":
                row = result.mappings().one_or_none()
                return dict(row) if row else None
            elif fetch_mode == "scalar":
                return result.scalar_one_or_none()
            elif fetch_mode == "count":
                return result.rowcount
            else:
                return None

    async def execute_transaction(self, queries: List[Dict[str, Any]]) -> bool:
        """
        Execute multiple queries in a transaction using SQLAlchemy.

        Args:
            queries: List of dictionaries with 'query' and 'params' keys

        Returns:
            True if successful, False otherwise

        Example:
            success = await db_manager.execute_transaction([
                {
                    "query": "INSERT INTO users (name, email) VALUES (:name, :email)",
                    "params": {"name": "John", "email": "john@example.com"}
                },
                {
                    "query": "UPDATE user_stats SET total_users = total_users + 1",
                    "params": {}
                }
            ])
        """
        try:
            async with self.get_session() as session:
                for query_data in queries:
                    sql_query = query_data["query"]
                    params = query_data.get("params", {})
                    await session.execute(text(sql_query), params)
                return True
        except Exception as e:
            logger.error(f"Transaction error: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
async def initialize_db(config=None):
    """Initialize database connection."""
    await db_manager.initialize(config)


async def close_db():
    """Close all database connections."""
    await db_manager.close()


def get_db_manager():
    """Get database manager instance."""
    return db_manager


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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

    async def test_connection():
        try:
            print(
                f"Attempting to connect to database at {test_config['host']}:{test_config['port']}"
            )
            await initialize_db(config=test_config)

            # Test a simple query
            result = await db_manager.execute_raw_query(
                "SELECT 1 as test", fetch_mode="one"
            )
            print(f"Connection test result: {result}")

            # Close connection
            await close_db()
            print("Database connection test completed successfully")
        except Exception as e:
            print(f"Error testing database connection: {e}")
            sys.exit(1)

    asyncio.run(test_connection())
