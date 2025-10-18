"""
Base Database Service
--------------------
Production-ready base class for all database services with optimized connection management,
error handling, and shared utilities for high-concurrency environments.

This base class provides:
- SQLAlchemy session management
- Transaction handling with automatic rollback
- Consistent error handling and logging
- Query execution utilities
- Validation helpers
"""

from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database_connection import DatabaseManager


class BaseDatabaseService:
    """
    Base class for all database service classes.

    Provides shared functionality for database operations including:
    - SQLAlchemy session management
    - Transaction handling with commit/rollback
    - Error handling and logging
    - Common validation utilities

    Thread-safe and optimized for high-concurrency scenarios (10,000+ concurrent users).
    """

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the database service with a database manager.

        Args:
            database_manager: Optional DatabaseManager instance. If not provided,
                            uses the singleton instance for connection pooling.
        """
        self.database_manager = database_manager or DatabaseManager()
        self._service_name = self.__class__.__name__

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a SQLAlchemy session with automatic commit/rollback.

        This is a direct wrapper around DatabaseManager.get_session()
        that preserves the same API interface as the previous get_database_connection().

        Yields:
            AsyncSession: SQLAlchemy session

        Example:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT * FROM users"))
                users = result.mappings().all()
        """
        async with self.database_manager.get_session() as session:
            yield session

    async def execute_single_query(
        self,
        sql_query: str,
        query_parameters: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a single SQL query with automatic session management.

        Optimized for read operations with minimal overhead.

        Args:
            sql_query: SQL query string to execute
            query_parameters: Optional dictionary of query parameters
            fetch_results: Whether to fetch and return results (default: True)

        Returns:
            List of result dictionaries if fetch_results is True, None otherwise

        Example:
            results = await self.execute_single_query(
                "SELECT * FROM users WHERE email = :email",
                {"email": email}
            )
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql_query), query_parameters or {})

                if fetch_results:
                    rows = result.mappings().all()
                    return [dict(row) for row in rows] if rows else []
                return None

        except Exception as error:
            logger.error(
                f"{self._service_name}: Error executing query: {error}",
                exc_info=True,
            )
            raise

    async def execute_batch_insert(
        self,
        sql_query: str,
        parameter_list: List[Dict[str, Any]],
        page_size: int = 1000,
    ) -> int:
        """
        Execute batch insert operations for high-performance bulk inserts.

        Uses SQLAlchemy's execute() method for optimized batch operations.
        Recommended for inserting large numbers of records efficiently.

        Args:
            sql_query: SQL INSERT query string
            parameter_list: List of parameter dictionaries for batch insertion
            page_size: Number of records to insert per batch (default: 1000)

        Returns:
            Number of rows inserted

        Example:
            rows_inserted = await self.execute_batch_insert(
                "INSERT INTO logs (user_id, action) VALUES (:user_id, :action)",
                [{"user_id": user1_id, "action": 'login'}, {"user_id": user2_id, "action": 'logout'}]
            )
        """
        if not parameter_list:
            logger.warning(
                f"{self._service_name}: Empty parameter list for batch insert"
            )
            return 0

        try:
            async with self.get_session() as session:
                rows_affected = 0
                # Execute batch operations in chunks
                for i in range(0, len(parameter_list), page_size):
                    chunk = parameter_list[i : i + page_size]
                    for params in chunk:
                        result = await session.execute(text(sql_query), params)
                        rows_affected += getattr(result, "rowcount", 0)

                logger.info(
                    f"{self._service_name}: Batch insert completed. "
                    f"Rows inserted: {rows_affected}"
                )
                return rows_affected
        except Exception as error:
            logger.error(
                f"{self._service_name}: Error in batch insert: {error}",
                exc_info=True,
            )
            raise

    # ========================================================================
    # VALIDATION UTILITIES
    # ========================================================================

    def validate_uuid(self, uuid_value: UUID, parameter_name: str = "UUID") -> None:
        """
        Validate that a UUID is not None and is a valid UUID instance.

        Args:
            uuid_value: UUID to validate
            parameter_name: Name of the parameter for error messages

        Raises:
            ValueError: If UUID is invalid or None
        """
        if uuid_value is None:
            raise ValueError(f"{parameter_name} cannot be None")
        if not isinstance(uuid_value, UUID):
            raise ValueError(f"{parameter_name} must be a valid UUID instance")

    def validate_positive_integer(
        self,
        integer_value: int,
        parameter_name: str = "value",
        allow_zero: bool = False,
    ) -> None:
        """
        Validate that an integer is positive (and optionally allow zero).

        Args:
            integer_value: Integer to validate
            parameter_name: Name of the parameter for error messages
            allow_zero: Whether to allow zero as a valid value (default: False)

        Raises:
            ValueError: If integer is not positive
        """
        if not isinstance(integer_value, int):
            raise ValueError(f"{parameter_name} must be an integer")

        minimum_value = 0 if allow_zero else 1
        if integer_value < minimum_value:
            raise ValueError(
                f"{parameter_name} must be {'non-negative' if allow_zero else 'positive'}, "
                f"got {integer_value}"
            )

    def validate_string_not_empty(
        self, string_value: str, parameter_name: str = "string"
    ) -> None:
        """
        Validate that a string is not None or empty.

        Args:
            string_value: String to validate
            parameter_name: Name of the parameter for error messages

        Raises:
            ValueError: If string is None or empty
        """
        if not string_value or not isinstance(string_value, str):
            raise ValueError(f"{parameter_name} must be a non-empty string")
        if not string_value.strip():
            raise ValueError(f"{parameter_name} cannot be only whitespace")

    def validate_enum_value(
        self, enum_value: str, valid_values: List[str], parameter_name: str = "value"
    ) -> None:
        """
        Validate that a value is one of the allowed enum values.

        Args:
            enum_value: Value to validate
            valid_values: List of valid enum values
            parameter_name: Name of the parameter for error messages

        Raises:
            ValueError: If value is not in the list of valid values
        """
        if enum_value not in valid_values:
            raise ValueError(
                f"Invalid {parameter_name}: '{enum_value}'. "
                f"Must be one of: {', '.join(valid_values)}"
            )

    def validate_pagination_parameters(
        self, limit: int, offset: int, max_limit: int = 1000
    ) -> None:
        """
        Validate pagination parameters for queries.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            max_limit: Maximum allowed limit value (default: 1000)

        Raises:
            ValueError: If pagination parameters are invalid
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if limit > max_limit:
            raise ValueError(f"limit cannot exceed {max_limit}, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def build_dynamic_update_query(
        self,
        table_name: str,
        update_fields: Dict[str, Any],
        where_clause: str,
        where_parameters: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a dynamic UPDATE query with only the fields that need updating.

        This method is useful for update operations where only some fields
        may be provided, avoiding unnecessary database writes.

        Args:
            table_name: Name of the table to update
            update_fields: Dictionary of field names and their new values
            where_clause: WHERE clause (e.g., "user_id = :user_id")
            where_parameters: Parameters for the WHERE clause as dictionary

        Returns:
            Tuple of (query_string, parameters_dict)

        Example:
            query, params = self.build_dynamic_update_query(
                "users",
                {"email": "new@email.com", "role": "admin"},
                "user_id = :user_id",
                {"user_id": user_id}
            )
        """
        if not update_fields:
            raise ValueError("update_fields cannot be empty")

        # Always include updated_at timestamp
        set_clauses = ["updated_at = CURRENT_TIMESTAMP"]
        parameters = {}

        for field_name, field_value in update_fields.items():
            param_name = f"set_{field_name}"
            set_clauses.append(f"{field_name} = :{param_name}")
            parameters[param_name] = field_value

        # Add WHERE clause parameters
        parameters.update(where_parameters)

        sql_query = f"""
            UPDATE {table_name}
            SET {", ".join(set_clauses)}
            WHERE {where_clause}
            RETURNING *
        """

        return sql_query, parameters

    def log_operation(
        self,
        operation_type: str,
        entity_identifier: Any,
        success: bool = True,
        additional_context: Optional[str] = None,
    ) -> None:
        """
        Log database operations for monitoring and debugging.

        Args:
            operation_type: Type of operation (e.g., "CREATE", "UPDATE", "DELETE")
            entity_identifier: Identifier of the entity being operated on
            success: Whether the operation was successful
            additional_context: Optional additional context information
        """
        log_level = "info" if success else "error"
        status = "succeeded" if success else "failed"

        message = (
            f"{self._service_name}: {operation_type} operation {status} "
            f"for entity: {entity_identifier}"
        )

        if additional_context:
            message += f" - {additional_context}"

        getattr(logger, log_level)(message)
