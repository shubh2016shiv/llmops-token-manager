"""
Database Connection Manager.

Manages PostgreSQL database connections with SQLAlchemy async engine.
Provides both ORM and raw SQL query capabilities.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import sys
from typing import Any

from sqlalchemy import text

# SQLAlchemy imports for ORM session support
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import Pool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Fallback values used only when a caller passes an explicit `config` override
# dict that omits these keys (see `initialize()`). Not application config: the
# settings-driven path always supplies database_pool_size/database_max_overflow
# from app/core/config, so these never apply to real (non-override) startup.
_DEFAULT_MIN_CONNECTIONS = 5
_DEFAULT_MAX_CONNECTIONS = 10


class DatabaseSessionManager:
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
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _resolve_connection_config() -> dict[str, Any]:
        """
        Build connection config from settings when no explicit override is given.

        Routes through PgBouncer when enabled; connects to PostgreSQL directly
        otherwise. In Docker, DATABASE_HOST is overridden to "pgbouncer" via
        docker-compose env. When running locally, pgbouncer_enabled=True routes
        to localhost:6432.
        """
        if settings.pgbouncer_enabled:
            host, port = settings.pgbouncer_host, settings.pgbouncer_port
        else:
            host, port = settings.database_host, settings.database_port
        return {
            "host": host,
            "port": port,
            "dbname": settings.database_name,
            "user": settings.database_user,
            "password": settings.database_password,
            "min_connections": settings.database_pool_size,
            "max_connections": settings.database_pool_size
            + settings.database_max_overflow,
        }

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize SQLAlchemy async engine.

        Args:
            config: Optional configuration override

        """
        if self._engine is not None:
            logger.warning("Database engine already initialized")
            return

        if config is None:
            # No override: reuse the config's own PgBouncer-aware URL builder
            # instead of re-deriving the same routing decision here.
            config = self._resolve_connection_config()
            database_url = settings.effective_database_url
        else:
            database_url = (
                f"postgresql+asyncpg://"
                f"{config['user']}:{config['password']}@"
                f"{config['host']}:{config['port']}/"
                f"{config['dbname']}"
            )

        logger.info(
            "Initializing database connection to "
            f"{config.get('host')}:{config.get('port')}"
        )

        try:
            # asyncpg caches prepared statements per logical connection, naming
            # them sequentially (__asyncpg_stmt_1__, _2__, ...). In PgBouncer
            # transaction mode, a single SQLAlchemy-pooled connection can be
            # handed to a DIFFERENT real PostgreSQL backend between queries --
            # that is the whole point of transaction-mode pooling. If the new
            # backend already has a different session's statement registered
            # under that same auto-generated name, asyncpg raises
            # DuplicatePreparedStatementError. statement_cache_size=0 disables
            # asyncpg's client-side cache so it never assumes a name survives
            # across queries. Only needed when routing through PgBouncer --
            # a direct connection has a stable 1:1 mapping to one real backend,
            # so caching there is safe and worth keeping for performance.
            connect_args = (
                {"statement_cache_size": 0} if settings.pgbouncer_enabled else {}
            )

            # Configure SQLAlchemy engine for high performance
            min_connections = config.get("min_connections", _DEFAULT_MIN_CONNECTIONS)
            max_connections = config.get("max_connections", _DEFAULT_MAX_CONNECTIONS)
            self._engine = create_async_engine(
                database_url,
                pool_size=min_connections,
                max_overflow=max_connections - min_connections,
                pool_pre_ping=True,  # Health checks
                pool_recycle=settings.database_pool_recycle_seconds,
                pool_timeout=settings.database_pool_timeout_seconds,
                echo=False,  # Disable SQL echo in production
                connect_args=connect_args,
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

    @property
    def pool(self) -> Pool | None:
        """Return the SQLAlchemy pool when the async engine is initialized."""
        if self._engine is None:
            return None
        return self._engine.pool

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
        params: dict[str, Any] | None = None,
        fetch_mode: str = "all",
    ) -> list[dict[str, Any]] | dict[str, Any] | int | None:
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
                # return result.rowcount
                return getattr(result, "rowcount", 0)
            else:
                return None

    async def execute_transaction(self, queries: list[dict[str, Any]]) -> bool:
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
db_manager = DatabaseSessionManager()


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
        """Run a local connectivity smoke test for the database manager."""
        try:
            print(
                "Attempting to connect to database at "
                f"{test_config['host']}:{test_config['port']}"
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
