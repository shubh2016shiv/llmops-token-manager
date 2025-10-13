"""
PostgreSQL CRUD Operations for Users Management
-----------------------------------------------
Production-ready database service for user management operations including:
- User creation, retrieval, updates, and deletion
- Role and status management
- User filtering and pagination
- Optimized for high-concurrency environments (10,000+ concurrent users)
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from loguru import logger

from app.core.database_connection import DatabaseManager
from app.psql_db_services.base_service import BaseDatabaseService


class UsersService(BaseDatabaseService):
    """
    Production-ready service for user database operations.

    Inherits from BaseDatabaseService for optimized connection pooling,
    transaction management, and error handling.

    Supports:
    - CRUD operations for user records
    - Role-based access control (owner, admin, developer, viewer)
    - User status management (active, suspended, inactive)
    - Efficient pagination and filtering
    - Thread-safe operations for high-concurrency scenarios
    """

    # Define valid enum values as class constants for reusability
    VALID_USER_ROLES = ["owner", "admin", "developer", "viewer"]
    VALID_USER_STATUSES = ["active", "suspended", "inactive"]

    DEFAULT_USER_ROLE = "developer"
    DEFAULT_USER_STATUS = "active"

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the users service with database manager.

        Args:
            database_manager: Optional DatabaseManager instance (uses singleton if not provided)
        """
        super().__init__(database_manager)

    def validate_user_role(self, user_role: str) -> None:
        """
        Validate that a user role is one of the allowed values.

        Args:
            user_role: User role to validate

        Raises:
            ValueError: If role is not in the list of valid roles
        """
        self.validate_enum_value(user_role, self.VALID_USER_ROLES, "user role")

    def validate_user_status(self, user_status: str) -> None:
        """
        Validate that a user status is one of the allowed values.

        Args:
            user_status: User status to validate

        Raises:
            ValueError: If status is not in the list of valid statuses
        """
        self.validate_enum_value(user_status, self.VALID_USER_STATUSES, "user status")

    def validate_email_address(self, email_address: str) -> None:
        """
        Validate that an email address is properly formatted.

        Args:
            email_address: Email address to validate

        Raises:
            ValueError: If email is invalid or empty
        """
        self.validate_string_not_empty(email_address, "email address")

        # Add email format validation
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email_address):
            raise ValueError(f"Invalid email address format: {email_address}")

    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================

    async def create_user(
        self,
        email_address: str,
        user_role: str = DEFAULT_USER_ROLE,
        user_status: str = DEFAULT_USER_STATUS,
    ) -> Dict[str, Any]:
        """
        Create a new user record in the database.

        Args:
            email_address: User's email address (must be unique)
            user_role: User role (owner, admin, developer, viewer). Defaults to 'developer'
            user_status: User status (active, suspended, inactive). Defaults to 'active'

        Returns:
            Dictionary containing the created user record with all fields

        Raises:
            psycopg.IntegrityError: If email already exists (unique constraint violation)
            psycopg.Error: On other database errors
            ValueError: On invalid input parameters
        """
        self.validate_email_address(email_address)
        self.validate_user_role(user_role)
        self.validate_user_status(user_status)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        INSERT INTO users (email, role, status)
                        VALUES (%s, %s, %s)
                        RETURNING *
                    """

                    await cursor.execute(
                        sql_query, (email_address, user_role, user_status)
                    )
                    created_user = await cursor.fetchone()

                    if not created_user:
                        raise RuntimeError("Failed to create user record")

                    self.log_operation("CREATE", email_address, success=True)
                    return dict(created_user)
        except psycopg.IntegrityError as integrity_error:
            logger.error(
                f"Integrity error creating user {email_address}: {integrity_error}"
            )
            raise
        except psycopg.Error as database_error:
            logger.error(
                f"Database error creating user {email_address}: {database_error}"
            )
            raise

    # ========================================================================
    # READ OPERATIONS
    # ========================================================================

    async def get_user_by_id(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by their unique identifier.

        Args:
            user_id: User's unique UUID identifier

        Returns:
            Dictionary containing user record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If user_id is invalid
        """
        self.validate_uuid(user_id, "user_id")

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        SELECT * FROM users
                        WHERE user_id = %s
                    """
                    await cursor.execute(sql_query, (user_id,))
                    user_record = await cursor.fetchone()
                    return dict(user_record) if user_record else None
        except psycopg.Error as database_error:
            logger.error(f"Error fetching user {user_id}: {database_error}")
            raise

    async def get_user_by_email(self, email_address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by their email address.

        Args:
            email_address: User's email address

        Returns:
            Dictionary containing user record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If email is invalid
        """
        self.validate_email_address(email_address)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        SELECT * FROM users
                        WHERE email = %s
                    """
                    await cursor.execute(sql_query, (email_address,))
                    user_record = await cursor.fetchone()
                    return dict(user_record) if user_record else None
        except psycopg.Error as database_error:
            logger.error(
                f"Error fetching user by email {email_address}: {database_error}"
            )
            raise

    async def get_all_users(
        self,
        role_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all users with optional filtering and pagination.

        Optimized for high-concurrency scenarios with indexed queries.

        Args:
            role_filter: Optional role to filter by (owner, admin, developer, viewer)
            status_filter: Optional status to filter by (active, suspended, inactive)
            limit: Maximum number of records to return (default: 100, max: 1000)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of user records ordered by creation date (newest first)

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid pagination or filter parameters
        """
        self.validate_pagination_parameters(limit, offset)

        if role_filter:
            self.validate_user_role(role_filter)
        if status_filter:
            self.validate_user_status(status_filter)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = "SELECT * FROM users WHERE 1=1"
                    query_parameters = []

                    if role_filter:
                        sql_query += " AND role = %s"
                        query_parameters.append(role_filter)

                    if status_filter:
                        sql_query += " AND status = %s"
                        query_parameters.append(status_filter)

                    sql_query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                    query_parameters.extend([limit, offset])

                    await cursor.execute(sql_query, query_parameters)
                    user_records = await cursor.fetchall()
                    logger.debug(f"Retrieved {len(user_records)} users")
                    return [dict(row) for row in user_records]
        except psycopg.Error as database_error:
            logger.error(f"Error fetching users: {database_error}")
            raise

    async def get_users_by_role(
        self, user_role: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all users with a specific role.

        Args:
            user_role: User role to filter by (owner, admin, developer, viewer)
            limit: Maximum number of records to return (default: 100)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of user records with the specified role

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid parameters
        """
        return await self.get_all_users(
            role_filter=user_role, limit=limit, offset=offset
        )

    async def get_active_users(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all active users.

        Args:
            limit: Maximum number of records to return (default: 100)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of active user records

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid pagination parameters
        """
        return await self.get_all_users(
            status_filter="active", limit=limit, offset=offset
        )

    async def count_users_by_status(self, user_status: str) -> int:
        """
        Count the number of users with a specific status.

        Args:
            user_status: User status to count (active, suspended, inactive)

        Returns:
            Number of users with the specified status

        Raises:
            psycopg.Error: On database errors
            ValueError: If status is invalid
        """
        self.validate_user_status(user_status)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor() as cursor:
                    sql_query = """
                        SELECT COUNT(*) FROM users
                        WHERE status = %s
                    """
                    await cursor.execute(sql_query, (user_status,))
                    count_result = await cursor.fetchone()
                    return count_result[0] if count_result else 0
        except psycopg.Error as database_error:
            logger.error(
                f"Error counting users by status {user_status}: {database_error}"
            )
            raise

    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================

    async def update_user(
        self,
        user_id: UUID,
        email_address: Optional[str] = None,
        user_role: Optional[str] = None,
        user_status: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update user information with dynamic field updates.

        Only provided fields will be updated, optimizing database writes.

        Args:
            user_id: User's unique UUID identifier
            email_address: Optional new email address
            user_role: Optional new role (owner, admin, developer, viewer)
            user_status: Optional new status (active, suspended, inactive)

        Returns:
            Updated user record dictionary or None if user not found

        Raises:
            psycopg.IntegrityError: If email already exists
            psycopg.Error: On other database errors
            ValueError: On invalid input parameters
        """
        self.validate_uuid(user_id, "user_id")

        if email_address:
            self.validate_email_address(email_address)
        if user_role:
            self.validate_user_role(user_role)
        if user_status:
            self.validate_user_status(user_status)

        # Build update fields dictionary
        update_fields_dict = {}
        if email_address is not None:
            update_fields_dict["email"] = email_address
        if user_role is not None:
            update_fields_dict["role"] = user_role
        if user_status is not None:
            update_fields_dict["status"] = user_status

        if not update_fields_dict:
            logger.warning(f"No fields to update for user {user_id}")
            return await self.get_user_by_id(user_id)

        try:
            sql_query, query_parameters = self.build_dynamic_update_query(
                table_name="users",
                update_fields=update_fields_dict,
                where_clause="user_id = %s",
                where_parameters=(user_id,),
            )

            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(sql_query, query_parameters)
                    updated_user = await cursor.fetchone()

                    if updated_user:
                        self.log_operation("UPDATE", user_id, success=True)
                        return dict(updated_user)

                    logger.warning(f"User {user_id} not found for update")
                    return None
        except psycopg.IntegrityError as integrity_error:
            logger.error(f"Integrity error updating user {user_id}: {integrity_error}")
            raise
        except psycopg.Error as database_error:
            logger.error(f"Error updating user {user_id}: {database_error}")
            raise

    async def update_user_role(
        self, user_id: UUID, new_user_role: str
    ) -> Optional[Dict[str, Any]]:
        """
        Update a user's role.

        Args:
            user_id: User's unique UUID identifier
            new_user_role: New role to assign (owner, admin, developer, viewer)

        Returns:
            Updated user record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid parameters
        """
        return await self.update_user(user_id=user_id, user_role=new_user_role)

    async def update_user_status(
        self, user_id: UUID, new_user_status: str
    ) -> Optional[Dict[str, Any]]:
        """
        Update a user's status.

        Args:
            user_id: User's unique UUID identifier
            new_user_status: New status to set (active, suspended, inactive)

        Returns:
            Updated user record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid parameters
        """
        return await self.update_user(user_id=user_id, user_status=new_user_status)

    async def suspend_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Suspend a user account by setting status to 'suspended'.

        Args:
            user_id: User's unique UUID identifier

        Returns:
            Updated user record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If user_id is invalid
        """
        return await self.update_user_status(
            user_id=user_id, new_user_status="suspended"
        )

    async def activate_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Activate a user account by setting status to 'active'.

        Args:
            user_id: User's unique UUID identifier

        Returns:
            Updated user record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If user_id is invalid
        """
        return await self.update_user_status(user_id=user_id, new_user_status="active")

    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================

    async def delete_user(self, user_id: UUID) -> bool:
        """
        Delete a user record from the database.

        Note: This operation may cascade to related records depending on
        database foreign key constraints.

        Args:
            user_id: User's unique UUID identifier

        Returns:
            True if user was deleted, False if user was not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If user_id is invalid
        """
        self.validate_uuid(user_id, "user_id")

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor() as cursor:
                    sql_query = """
                        DELETE FROM users
                        WHERE user_id = %s
                    """
                    await cursor.execute(sql_query, (user_id,))
                    was_deleted = cursor.rowcount > 0

                    if was_deleted:
                        self.log_operation("DELETE", user_id, success=True)
                    else:
                        logger.debug(f"User not found for deletion: {user_id}")

                    return was_deleted
        except psycopg.Error as database_error:
            logger.error(f"Error deleting user {user_id}: {database_error}")
            raise

    async def delete_user_by_email(self, email_address: str) -> bool:
        """
        Delete a user by their email address.

        Note: This operation may cascade to related records depending on
        database foreign key constraints.

        Args:
            email_address: User's email address

        Returns:
            True if user was deleted, False if user was not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If email is invalid
        """
        self.validate_email_address(email_address)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor() as cursor:
                    sql_query = """
                        DELETE FROM users
                        WHERE email = %s
                    """
                    await cursor.execute(sql_query, (email_address,))
                    was_deleted = cursor.rowcount > 0

                    if was_deleted:
                        self.log_operation("DELETE", email_address, success=True)
                    else:
                        logger.debug(
                            f"User with email {email_address} not found for deletion"
                        )

                    return was_deleted
        except psycopg.Error as database_error:
            logger.error(
                f"Error deleting user by email {email_address}: {database_error}"
            )
            raise
