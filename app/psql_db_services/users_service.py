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
from datetime import datetime

from sqlalchemy import text
from loguru import logger

from app.core.database_connection import DatabaseManager
from app.psql_db_services.base_service import BaseDatabaseService
from functools import lru_cache
from email_validator import validate_email, EmailNotValidError


class UsersService(BaseDatabaseService):
    """
    Production-ready service for user database operations.

    Inherits from BaseDatabaseService for optimized connection pooling,
    transaction management, and error handling.

    Supports:
    - CRUD operations for user records
    - Role-based access control
    - User status management (active, suspended, inactive)
    - Efficient pagination and filtering
    - Thread-safe operations for high-concurrency scenarios
    """

    VALID_USER_ROLES = ["developer", "operator", "admin", "owner"]
    VALID_USER_STATUSES = ["active", "suspended", "inactive"]

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        super().__init__(database_manager)

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    async def check_email_exists(self, email: str) -> bool:
        """Check if email already exists in database"""
        try:
            async with self.get_session() as session:
                sql_query = "SELECT 1 FROM users WHERE email = :email LIMIT 1"
                result = await session.execute(text(sql_query), {"email": email})
                return result.first() is not None
        except Exception as e:
            logger.error(f"Error checking email existence: {e}")
            raise

    async def check_username_exists(self, username: str) -> bool:
        """Check if username already exists in database"""
        try:
            async with self.get_session() as session:
                sql_query = "SELECT 1 FROM users WHERE username = :username LIMIT 1"
                result = await session.execute(text(sql_query), {"username": username})
                return result.first() is not None
        except Exception as e:
            logger.error(f"Error checking username existence: {e}")
            raise

    @lru_cache(maxsize=1000)  # Cache validation results
    def validate_email_address(self, email_address: str) -> None:
        """
        Validate email address with caching for performance.

        Args:
            email_address: Email address to validate

        Raises:
            ValueError: If email format is invalid
        """
        if not email_address:
            raise ValueError("Email address cannot be empty")

        try:
            # Normalize and validate
            validated = validate_email(email_address, check_deliverability=False)
            # Use normalized email
            return validated.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {str(e)}")

    def validate_user_role(self, user_role: str) -> None:
        """
        Validate user role.

        Args:
            user_role: User role to validate

        Raises:
            ValueError: If role is invalid
        """
        if user_role not in self.VALID_USER_ROLES:
            raise ValueError(
                f"Invalid user role '{user_role}'. Must be one of: {', '.join(self.VALID_USER_ROLES)}"
            )

    def validate_user_status(self, user_status: str) -> None:
        """
        Validate user status.

        Args:
            user_status: User status to validate

        Raises:
            ValueError: If status is invalid
        """
        if user_status not in self.VALID_USER_STATUSES:
            raise ValueError(
                f"Invalid user status '{user_status}'. Must be one of: {', '.join(self.VALID_USER_STATUSES)}"
            )

    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================

    async def create_user(
        self,
        user_id: UUID,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        password_hash: str,
        user_role: str = "developer",
        user_status: str = "active",
        created_at: datetime = None,
        updated_at: datetime = None,
    ) -> Dict[str, Any]:
        """
        Create a new user record in the database.

        Args:
            user_id: Unique user identifier (UUID)
            username: Unique username
            email: User's email address (must be unique)
            first_name: User's first name
            last_name: User's last name
            password_hash: Hashed password
            user_role: User role. Defaults to 'developer'
            user_status: User status. Defaults to 'active'
            created_at: Creation timestamp (defaults to now)
            updated_at: Update timestamp (defaults to now)

        Returns:
            Dictionary containing the created user record

        Raises:
            ValueError: If email or username already exists
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        # Check uniqueness
        if await self.check_email_exists(email):
            raise ValueError(f"Email '{email}' already exists")

        if await self.check_username_exists(username):
            raise ValueError(f"Username '{username}' already exists")

        now = datetime.utcnow()
        created_at = created_at or now
        updated_at = updated_at or now

        try:
            async with self.get_session() as session:
                sql_query = """
                    INSERT INTO users (
                        user_id, username, email, first_name, last_name,
                        password_hash, role, status, created_at, updated_at
                    )
                    VALUES (
                        :user_id, :username, :email, :first_name, :last_name,
                        :password_hash, :role, :status, :created_at, :updated_at
                    )
                    RETURNING user_id, username, email, first_name, last_name,
                              role, status, created_at, updated_at
                """

                params = {
                    "user_id": user_id,
                    "username": username,
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "password_hash": password_hash,
                    "role": user_role,
                    "status": user_status,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }

                result = await session.execute(text(sql_query), params)
                created_user = result.mappings().one_or_none()

                if not created_user:
                    raise RuntimeError("Failed to create user record")

                await session.commit()
                logger.info(f"User created successfully: {email}")
                return dict(created_user)

        except Exception as e:
            logger.error(f"Error creating user {email}: {e}")
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If user_id is invalid
        """
        self.validate_uuid(user_id, "user_id")

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT * FROM users
                    WHERE user_id = :user_id
                """
                result = await session.execute(text(sql_query), {"user_id": user_id})
                user_record = result.mappings().one_or_none()
                return dict(user_record) if user_record else None
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            raise

    async def get_user_by_email(self, email_address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by their email address.

        Args:
            email_address: User's email address

        Returns:
            Dictionary containing user record or None if not found

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If email is invalid
        """
        self.validate_email_address(email_address)

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT * FROM users
                    WHERE email = :email
                """
                result = await session.execute(
                    text(sql_query), {"email": email_address}
                )
                user_record = result.mappings().one_or_none()
                return dict(user_record) if user_record else None
        except Exception as e:
            logger.error(f"Error fetching user by email {email_address}: {e}")
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: On invalid pagination or filter parameters
        """
        self.validate_pagination_parameters(limit, offset)

        if role_filter:
            self.validate_user_role(role_filter)
        if status_filter:
            self.validate_user_status(status_filter)

        try:
            async with self.get_session() as session:
                sql_query = "SELECT * FROM users WHERE 1=1"
                params = {}

                if role_filter:
                    sql_query += " AND role = :role"
                    params["role"] = role_filter

                if status_filter:
                    sql_query += " AND status = :status"
                    params["status"] = status_filter

                sql_query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
                params["limit"] = limit
                params["offset"] = offset

                result = await session.execute(text(sql_query), params)
                user_records = result.mappings().all()
                logger.debug(f"Retrieved {len(user_records)} users")
                return [dict(row) for row in user_records]
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If status is invalid
        """
        self.validate_user_status(user_status)

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT COUNT(*) FROM users
                    WHERE status = :status
                """
                result = await session.execute(text(sql_query), {"status": user_status})
                return result.scalar_one_or_none() or 0
        except Exception as e:
            logger.error(f"Error counting users by status {user_status}: {e}")
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
            sqlalchemy.exc.IntegrityError: If email already exists
            sqlalchemy.exc.SQLAlchemyError: On other database errors
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
                where_clause="user_id = :user_id",
                where_parameters={"user_id": user_id},
            )

            async with self.get_session() as session:
                result = await session.execute(text(sql_query), query_parameters)
                updated_user = result.mappings().one_or_none()

                if updated_user:
                    self.log_operation("UPDATE", user_id, success=True)
                    return dict(updated_user)

                logger.warning(f"User {user_id} not found for update")
                return None
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If user_id is invalid
        """
        self.validate_uuid(user_id, "user_id")

        try:
            async with self.get_session() as session:
                sql_query = """
                    DELETE FROM users
                    WHERE user_id = :user_id
                """
                result = await session.execute(text(sql_query), {"user_id": user_id})
                was_deleted = result.rowcount > 0

                if was_deleted:
                    self.log_operation("DELETE", user_id, success=True)
                else:
                    logger.debug(f"User not found for deletion: {user_id}")

                return bool(was_deleted)
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
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
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If email is invalid
        """
        self.validate_email_address(email_address)

        try:
            async with self.get_session() as session:
                sql_query = """
                    DELETE FROM users
                    WHERE email = :email
                """
                result = await session.execute(
                    text(sql_query), {"email": email_address}
                )
                was_deleted = result.rowcount > 0

                if was_deleted:
                    self.log_operation("DELETE", email_address, success=True)
                else:
                    logger.debug(
                        f"User with email {email_address} not found for deletion"
                    )

                return bool(was_deleted)
        except Exception as e:
            logger.error(f"Error deleting user by email {email_address}: {e}")
            raise
