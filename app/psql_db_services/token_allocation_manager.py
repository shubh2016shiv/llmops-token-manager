"""
PostgreSQL CRUD Operations for Token Allocation Management
----------------------------------------------------------
Production-ready database service for token allocation, tracking, and lifecycle management including:
- Token allocation creation and tracking
- Load balancing across model deployments
- Allocation status management (ACQUIRED, WAITING, PAUSED, RELEASED, EXPIRED)
- Usage statistics and analytics
- Optimized for high-concurrency environments (10,000+ concurrent users)
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import uuid
from uuid import UUID

from sqlalchemy import text
from loguru import logger

from app.core.database_connection import DatabaseManager
from app.psql_db_services.base_service import BaseDatabaseService


class TokenAllocationService(BaseDatabaseService):
    """
    Production-ready service for token allocation database operations.

    Inherits from BaseDatabaseService for optimized connection pooling,
    transaction management, and error handling.

    Supports:
    - CRUD operations for token allocations
    - Load balancing and least-loaded endpoint selection
    - Allocation lifecycle management (acquire, pause, resume, release)
    - Expiration tracking and cleanup
    - Usage analytics and reporting
    - Thread-safe operations for high-concurrency scenarios
    """

    # Define valid allocation statuses as class constants
    VALID_ALLOCATION_STATUSES = [
        "ACQUIRED",
        "WAITING",
        "PAUSED",
        "RELEASED",
        "EXPIRED",
        "FAILED",
    ]

    DEFAULT_ALLOCATION_STATUS = "ACQUIRED"

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the token allocation service with database manager.

        Args:
            database_manager: Optional DatabaseManager instance (uses singleton if not provided)
        """
        super().__init__(database_manager)

    def validate_allocation_status(self, allocation_status: str) -> None:
        """
        Validate that an allocation status is one of the allowed values.

        Args:
            allocation_status: Allocation status to validate

        Raises:
            ValueError: If status is not in the list of valid statuses
        """
        self.validate_enum_value(
            allocation_status, self.VALID_ALLOCATION_STATUSES, "allocation status"
        )

    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================

    async def create_token_allocation(
        self,
        token_request_identifier: str,
        user_id: UUID,
        model_name: str,
        token_count: int,
        allocation_status: str = DEFAULT_ALLOCATION_STATUS,
        allocation_timestamp: Optional[datetime] = None,
        expiration_timestamp: Optional[datetime] = None,
        deployment_name: Optional[str] = None,
        cloud_provider_name: Optional[str] = None,
        api_endpoint_url: Optional[str] = None,
        deployment_region: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a new token allocation record in the database.

        This method is optimized for high-concurrency scenarios with proper
        validation and error handling.

        Args:
            token_request_identifier: Unique identifier for this allocation request
            user_id: UUID of the user requesting tokens
            model_name: Name of the LLM model
            token_count: Number of tokens to allocate (must be positive)
            allocation_status: Status (ACQUIRED, WAITING, PAUSED, etc.). Defaults to 'ACQUIRED'
            allocation_timestamp: When allocation was made (defaults to current time)
            expiration_timestamp: When allocation expires (optional)
            deployment_name: Optional deployment identifier
            cloud_provider_name: Optional cloud provider name
            api_endpoint_url: Optional API endpoint URL
            deployment_region: Optional geographic region identifier
            request_metadata: Optional JSON metadata for additional context
            temperature: Optional temperature setting for this request
            top_p: Optional top P (nucleus sampling) parameter for this request
            seed: Optional seed value for reproducible LLM outputs

        Returns:
            Dictionary containing the created allocation record with all fields

        Raises:
            sqlalchemy.exc.IntegrityError: If allocation with same ID already exists
            sqlalchemy.exc.SQLAlchemyError: On other database errors
            ValueError: On invalid input parameters
        """
        self.validate_string_not_empty(
            token_request_identifier, "token_request_identifier"
        )
        self.validate_uuid(user_id, "user_id")
        self.validate_string_not_empty(model_name, "model_name")
        self.validate_positive_integer(token_count, "token_count")
        self.validate_allocation_status(allocation_status)

        try:
            async with self.get_session() as session:
                sql_query = """
                    INSERT INTO token_manager (
                        token_request_id, user_id, llm_model_name,
                        deployment_name, cloud_provider, api_endpoint_url, region,
                        token_count, allocation_status, allocated_at, expires_at,
                        request_context, temperature, top_p, seed
                    ) VALUES (
                        :token_request_id, :user_id, :model_name,
                        :deployment_name, :cloud_provider, :api_endpoint_url, :region,
                        :token_count, :allocation_status, :allocated_at, :expires_at,
                        :request_context, :temperature, :top_p, :seed
                    )
                    RETURNING *
                """

                # Convert dict to JSON string for JSONB column
                request_context_json = (
                    json.dumps(request_metadata) if request_metadata else None
                )

                params = {
                    "token_request_id": token_request_identifier,
                    "user_id": user_id,
                    "model_name": model_name,
                    "deployment_name": deployment_name,
                    "cloud_provider": cloud_provider_name,
                    "api_endpoint_url": api_endpoint_url,
                    "region": deployment_region,
                    "token_count": token_count,
                    "allocation_status": allocation_status,
                    "allocated_at": allocation_timestamp or datetime.now(),
                    "expires_at": expiration_timestamp,
                    "request_context": request_context_json,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                }

                result = await session.execute(text(sql_query), params)
                created_allocation = result.mappings().one_or_none()

                if not created_allocation:
                    raise RuntimeError("Failed to create allocation record")

                self.log_operation(
                    "CREATE",
                    token_request_identifier,
                    success=True,
                    additional_context=f"{token_count} tokens for {model_name}",
                )
                return dict(created_allocation)
        except Exception as e:
            logger.error(f"Error creating allocation {token_request_identifier}: {e}")
            raise

    # ========================================================================
    # READ OPERATIONS
    # ========================================================================

    async def get_allocation_by_request_id(
        self, token_request_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a token allocation by its unique request identifier.

        Args:
            token_request_identifier: Unique token request identifier

        Returns:
            Dictionary containing allocation record or None if not found

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If token_request_identifier is invalid
        """
        self.validate_string_not_empty(
            token_request_identifier, "token_request_identifier"
        )

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT * FROM token_manager
                    WHERE token_request_id = :token_request_id
                """
                result = await session.execute(
                    text(sql_query), {"token_request_id": token_request_identifier}
                )
                allocation_record = result.mappings().one_or_none()
                return dict(allocation_record) if allocation_record else None
        except Exception as e:
            logger.error(f"Error fetching allocation {token_request_identifier}: {e}")
            raise

    async def get_total_allocated_tokens_by_model(
        self, model_name: str, included_statuses: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get total allocated tokens grouped by model and API endpoint.

        This method is critical for load balancing - it returns endpoints
        sorted by total allocated tokens (least loaded first).

        Optimized for high-concurrency scenarios with proper indexing.

        Args:
            model_name: LLM model name to query
            included_statuses: List of statuses to include (default: ACQUIRED, PAUSED)

        Returns:
            List of dictionaries with aggregated token counts per endpoint,
            sorted by total_tokens ascending (least loaded first)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If model_name is invalid
        """
        self.validate_string_not_empty(model_name, "model_name")

        if included_statuses is None:
            included_statuses = ["ACQUIRED", "PAUSED"]

        # Validate all statuses
        for status in included_statuses:
            self.validate_allocation_status(status)

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT
                        model_name,
                        api_endpoint_url,
                        region,
                        cloud_provider,
                        SUM(token_count) as total_tokens,
                        COUNT(*) as allocation_count
                    FROM token_manager
                    WHERE
                        model_name = :model_name
                        AND allocation_status = ANY(:included_statuses)
                        AND (expires_at IS NULL OR expires_at > NOW())
                    GROUP BY model_name, api_endpoint_url, region, cloud_provider
                    ORDER BY total_tokens ASC
                """

                result = await session.execute(
                    text(sql_query),
                    {"model_name": model_name, "included_statuses": included_statuses},
                )
                endpoint_statistics = result.mappings().all()

                logger.debug(
                    f"Found {len(endpoint_statistics)} endpoints for model {model_name}"
                )
                return [dict(row) for row in endpoint_statistics]
        except Exception as e:
            logger.error(f"Error fetching allocations for model {model_name}: {e}")
            raise

    async def get_total_allocated_tokens_for_endpoint(
        self, model_name: str, api_endpoint_url: str
    ) -> int:
        """
        Get total allocated tokens for a specific model and endpoint.

        This method is used for real-time load checking before allocation.

        Args:
            model_name: LLM model name
            api_endpoint_url: API endpoint URL

        Returns:
            Total number of allocated tokens (0 if none found)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If parameters are invalid
        """
        self.validate_string_not_empty(model_name, "model_name")
        self.validate_string_not_empty(api_endpoint_url, "api_endpoint_url")

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT COALESCE(SUM(token_count), 0) as total_tokens
                    FROM token_manager
                    WHERE
                        llm_model_name = :llm_model_name
                        AND api_endpoint_url = :api_endpoint_url
                        AND allocation_status IN ('ACQUIRED', 'PAUSED')
                        AND (expires_at IS NULL OR expires_at > NOW())
                """

                result = await session.execute(
                    text(sql_query),
                    {
                        "llm_model_name": model_name,
                        "api_endpoint_url": api_endpoint_url,
                    },
                )
                count_result = result.scalar_one_or_none()
                return count_result if count_result else 0
        except Exception as e:
            logger.error(f"Error fetching tokens for endpoint {api_endpoint_url}: {e}")
            raise

    async def get_user_allocations(
        self, user_id: UUID, status_filter: Optional[List[str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all allocations for a specific user

        Args:
            user_id: User UUID
            status_filter: Optional list of statuses to filter by
            limit: Maximum number of records to return (default: 100)

        Returns:
            List of allocation records ordered by most recent first

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                if status_filter:
                    query = """
                        SELECT * FROM token_manager
                        WHERE user_id = :user_id AND allocation_status = ANY(:status_filter)
                        ORDER BY allocated_at DESC
                        LIMIT :limit
                    """
                    result = await session.execute(
                        text(query),
                        {
                            "user_id": user_id,
                            "status_filter": status_filter,
                            "limit": limit,
                        },
                    )
                else:
                    query = """
                        SELECT * FROM token_manager
                        WHERE user_id = :user_id
                        ORDER BY allocated_at DESC
                        LIMIT :limit
                    """
                    result = await session.execute(
                        text(query), {"user_id": user_id, "limit": limit}
                    )

                results = result.mappings().all()
                logger.debug(f"Found {len(results)} allocations for user {user_id}")
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error fetching user allocations for {user_id}: {e}")
            raise

    async def get_active_allocations_count_by_model(self, model_name: str) -> int:
        """
        Get count of active allocations for a model

        Args:
            model_name: LLM model name

        Returns:
            Count of active allocations (0 if none found)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                query = """
                    SELECT COUNT(*)
                    FROM token_manager
                    WHERE
                        llm_model_name = :llm_model_name
                        AND allocation_status IN ('ACQUIRED', 'PAUSED')
                        AND (expires_at IS NULL OR expires_at > NOW())
                """
                result = await session.execute(
                    text(query), {"llm_model_name": model_name}
                )
                return result.scalar_one_or_none() or 0
        except Exception as e:
            logger.error(f"Error counting active allocations for {model_name}: {e}")
            raise

    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================

    async def update_allocation_status(
        self,
        token_request_id: str,
        new_status: str,
        api_endpoint: Optional[str] = None,
        region: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        latency_ms: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update allocation status and related fields

        Args:
            token_request_id: Unique token request identifier
            new_status: New status to set (ACQUIRED, WAITING, PAUSED, RELEASED, FAILED)
            api_endpoint: Optional endpoint to update
            region: Optional region to update
            expires_at: Optional new expiration time
            completed_at: Optional completion timestamp
            latency_ms: Optional latency in milliseconds

        Returns:
            Updated record or None if not found

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                # Build dynamic update query
                update_fields = ["allocation_status = :new_status"]
                params: Dict[str, Any] = {
                    "new_status": new_status,
                    "token_request_id": token_request_id,
                }

                if api_endpoint is not None:
                    update_fields.append("api_endpoint_url = :api_endpoint_url")
                    params["api_endpoint_url"] = api_endpoint

                if region is not None:
                    update_fields.append("region = :region")
                    params["region"] = region

                if expires_at is not None:
                    update_fields.append("expires_at = :expires_at")
                    params["expires_at"] = expires_at

                if completed_at is not None:
                    update_fields.append("completed_at = :completed_at")
                    params["completed_at"] = completed_at

                if latency_ms is not None:
                    update_fields.append("latency_ms = :latency_ms")
                    params["latency_ms"] = latency_ms

                query = f"""
                    UPDATE token_manager
                    SET {", ".join(update_fields)}
                    WHERE token_request_id = :token_request_id
                    RETURNING *
                """

                result = await session.execute(text(query), params)
                updated_record = result.mappings().one_or_none()

                if updated_record:
                    logger.info(
                        f"Updated allocation {token_request_id} to status {new_status}"
                    )
                    return dict(updated_record)

                logger.warning(f"Allocation {token_request_id} not found for update")
                return None
        except Exception as e:
            logger.error(f"Error updating allocation {token_request_id}: {e}")
            raise

    async def transition_waiting_to_acquired(
        self,
        token_request_id: str,
        api_endpoint: str,
        region: str,
        expires_at: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Atomically transition allocation from WAITING to ACQUIRED
        Only succeeds if current status is WAITING (prevents race conditions)

        Args:
            token_request_id: Unique token request identifier
            api_endpoint: API endpoint to assign
            region: Region to assign
            expires_at: New expiration time

        Returns:
            Updated record or None if transition failed (not in WAITING state)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                query = """
                    UPDATE token_manager
                    SET
                        allocation_status = 'ACQUIRED',
                        api_endpoint_url = :api_endpoint_url,
                        region = :region,
                        expires_at = :expires_at
                    WHERE
                        token_request_id = :token_request_id
                        AND allocation_status = 'WAITING'
                    RETURNING *
                """

                result = await session.execute(
                    text(query),
                    {
                        "api_endpoint_url": api_endpoint,
                        "region": region,
                        "expires_at": expires_at,
                        "token_request_id": token_request_id,
                    },
                )
                updated_record = result.mappings().one_or_none()

                if updated_record:
                    logger.info(
                        f"Transitioned {token_request_id} from WAITING to ACQUIRED"
                    )
                    return dict(updated_record)

                logger.debug(
                    f"Transition failed for {token_request_id} (not in WAITING state)"
                )
                return None
        except Exception as e:
            logger.error(f"Error transitioning allocation {token_request_id}: {e}")
            raise

    # async def release_allocated_token(
    #     self, token_request_id: str, latency_ms: Optional[int] = None
    # ) -> Optional[Dict[str, Any]]:
    #     """
    #     Mark allocation as completed (RELEASED status) and calculate latency
    #
    #     Args:
    #         token_request_id: Unique token request identifier
    #         latency_ms: Optional pre-calculated latency in milliseconds
    #
    #     Returns:
    #         Updated record or None if not found
    #
    #     Raises:
    #         sqlalchemy.exc.SQLAlchemyError: On database errors
    #     """
    #     try:
    #         async with self.get_session() as session:
    #             if latency_ms is None:
    #                 # Calculate latency from allocated_at to now
    #                 query = """
    #                     UPDATE token_manager
    #                     SET
    #                         allocation_status = 'RELEASED',
    #                         completed_at = NOW(),
    #                         latency_ms = EXTRACT(EPOCH FROM (NOW() - allocated_at)) * 1000
    #                     WHERE token_request_id = :token_request_id
    #                     RETURNING *
    #                 """
    #                 result = await session.execute(
    #                     text(query), {"token_request_id": token_request_id}
    #                 )
    #             else:
    #                 query = """
    #                     UPDATE token_manager
    #                     SET
    #                         allocation_status = 'RELEASED',
    #                         completed_at = NOW(),
    #                         latency_ms = :latency_ms
    #                     WHERE token_request_id = :token_request_id
    #                     RETURNING *
    #                 """
    #                 result = await session.execute(
    #                     text(query),
    #                     {
    #                         "latency_ms": latency_ms,
    #                         "token_request_id": token_request_id,
    #                     },
    #                 )
    #
    #             updated_record = result.mappings().one_or_none()
    #             if updated_record:
    #                 logger.info(f"Completed allocation {token_request_id}")
    #             return dict(updated_record) if updated_record else None
    #     except Exception as e:
    #         logger.error(f"Error completing allocation {token_request_id}: {e}")
    #         raise

    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================

    async def delete_allocation(self, token_request_id: str) -> bool:
        """
        Delete a token allocation (release tokens permanently)

        Args:
            token_request_id: Unique token request identifier

        Returns:
            True if deleted, False if not found

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                query = """
                    DELETE FROM token_manager
                    WHERE token_request_id = :token_request_id
                """
                result = await session.execute(
                    text(query), {"token_request_id": token_request_id}
                )
                deleted = getattr(result, "rowcount", 0) > 0

                if deleted:
                    logger.info(f"Deleted allocation: {token_request_id}")
                else:
                    logger.debug(
                        f"Allocation not found for deletion: {token_request_id}"
                    )

                return bool(deleted)
        except Exception as e:
            logger.error(f"Error deleting allocation {token_request_id}: {e}")
            raise

    async def delete_expired_allocations(self) -> int:
        """
        Clean up expired allocations (batch cleanup operation)

        Returns:
            Number of deleted records

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                query = """
                    DELETE FROM token_manager
                    WHERE
                        expires_at IS NOT NULL
                        AND expires_at < NOW()
                        AND allocation_status IN ('ACQUIRED', 'PAUSED', 'WAITING')
                """
                result = await session.execute(text(query))
                deleted_count = getattr(result, "rowcount", 0)

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired allocations")
                else:
                    logger.debug("No expired allocations to clean up")

                return int(deleted_count)
        except Exception as e:
            logger.error(f"Error deleting expired allocations: {e}")
            raise

    async def delete_allocations_by_user(
        self, user_id: UUID, status: Optional[str] = None
    ) -> int:
        """
        Delete all allocations for a user (optional: filter by status)

        Args:
            user_id: User UUID
            status: Optional status filter

        Returns:
            Number of deleted records

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                if status:
                    query = """
                        DELETE FROM token_manager
                        WHERE user_id = :user_id AND allocation_status = :status
                    """
                    result = await session.execute(
                        text(query), {"user_id": user_id, "status": status}
                    )
                else:
                    query = """
                        DELETE FROM token_manager
                        WHERE user_id = :user_id
                    """
                    result = await session.execute(text(query), {"user_id": user_id})

                deleted_count = getattr(result, "rowcount", 0)
                logger.info(f"Deleted {deleted_count} allocations for user {user_id}")
                return int(deleted_count)
        except Exception as e:
            logger.error(f"Error deleting allocations for user {user_id}: {e}")
            raise

    # ========================================================================
    # SPECIALIZED OPERATIONS (Business Logic Support)
    # ========================================================================

    async def pause_deployment(
        self,
        user_id: UUID,
        llm_provider: str,
        model_name: str,
        api_endpoint: str,
        pause_reason: str = "",
        pause_duration_minutes: int = 30,
    ) -> Dict[str, Any]:
        """
        Pause a deployment by creating a PAUSED allocation
        Similar to MongoDB's pause_llm_deployment method

        Args:
            user_id: User requesting the pause
            llm_provider: LLM provider name
            model_name: Model name
            api_endpoint: API endpoint URL to pause
            pause_reason: Reason for pausing
            pause_duration_minutes: Duration to pause for

        Returns:
            Dictionary with pause details

        Raises:
            ValueError: If model or deployment not found
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                # Check for existing active pause to prevent race conditions
                pause_check_query = """
                    SELECT 1 FROM token_manager
                    WHERE llm_model_name = :llm_model_name
                      AND api_endpoint_url = :api_endpoint_url
                      AND allocation_status = 'PAUSED'
                      AND expires_at > NOW()
                """
                existing_pause = await session.execute(
                    text(pause_check_query),
                    {"llm_model_name": model_name, "api_endpoint_url": api_endpoint},
                )
                if existing_pause.scalar_one_or_none():
                    logger.warning(
                        f"Deployment {model_name} at {api_endpoint} is already paused."
                    )
                    return {
                        "alloc_status": "ALREADY_PAUSED",
                        "llm_model_name": model_name,
                        "api_endpoint_url": api_endpoint,
                        "reason": "Deployment is already in a paused state.",
                    }

                # Find the model configuration for this deployment
                query = """
                    SELECT *
                    FROM llm_models
                    WHERE llm_model_name = :llm_model_name AND api_endpoint_url = :api_endpoint_url AND is_active_status = TRUE
                """
                result = await session.execute(
                    text(query),
                    {"llm_model_name": model_name, "api_endpoint_url": api_endpoint},
                )
                chosen_model_config = result.mappings().one_or_none()

                if not chosen_model_config:
                    logger.warning(
                        f"Deployment not found: {model_name} at {api_endpoint}"
                    )
                    return {
                        "alloc_status": "NOT_FOUND",
                        "llm_model_name": model_name,
                        "api_endpoint_url": api_endpoint,
                        "reason": "Deployment not found",
                    }

            # Get required properties from model config
            max_token_limit = chosen_model_config.get("max_tokens", 100000)
            provider_name = chosen_model_config.get("provider_name")
            region = chosen_model_config.get("deployment_region", "unknown")
            deployment_name = chosen_model_config.get("deployment_name", "")

            # Create a token request ID for the pause allocation
            token_request_id = f"pause_{uuid.uuid4().hex}"

            # Create the pause allocation
            return await self.create_pause_allocation(
                token_request_id=token_request_id,
                user_id=user_id,
                model_name=model_name,
                api_endpoint=api_endpoint,
                region=region,
                max_token_limit=max_token_limit,
                pause_duration_minutes=pause_duration_minutes,
                cloud_provider=provider_name,
                deployment_name=deployment_name,
                reason=pause_reason,
            )

        except ValueError as e:
            logger.error(f"Value error in pause_deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error in pause_deployment: {e}")
            raise

    async def create_pause_allocation(
        self,
        token_request_id: str,
        user_id: UUID,
        model_name: str,
        api_endpoint: str,
        region: str,
        max_token_limit: int,
        pause_duration_minutes: int,
        cloud_provider: Optional[str] = None,
        deployment_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a PAUSED allocation to block an entire deployment
        Used for failover scenarios and deployment maintenance

        Args:
            token_request_id: Unique identifier for pause allocation
            model_name: Model to pause
            api_endpoint: Endpoint to pause
            region: Region to pause
            max_token_limit: Full token limit to block
            pause_duration_minutes: How long to pause (in minutes)
            cloud_provider: Optional cloud provider name
            deployment_name: Optional deployment identifier
            reason: Optional reason for pausing

        Returns:
            Created allocation record

        Raises:
            ValueError: On invalid input parameters
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        if max_token_limit <= 0:
            raise ValueError(f"Token limit must be positive, got {max_token_limit}")
        if pause_duration_minutes <= 0:
            raise ValueError(
                f"Pause duration must be positive, got {pause_duration_minutes}"
            )

        # Calculate expiration and create context object
        expiration_timestamp = datetime.now() + timedelta(
            minutes=pause_duration_minutes
        )
        context = (
            {"reason": reason, "operation": "pause_deployment"}
            if reason
            else {"operation": "pause_deployment"}
        )

        logger.info(
            f"Creating pause allocation for {model_name} at {api_endpoint} for {pause_duration_minutes}m"
        )

        return await self.create_token_allocation(
            token_request_identifier=token_request_id,
            user_id=user_id,
            model_name=model_name,
            token_count=max_token_limit,
            allocation_status="PAUSED",
            expiration_timestamp=expiration_timestamp,
            api_endpoint_url=api_endpoint,
            cloud_provider_name=cloud_provider,
            deployment_name=deployment_name,
            request_metadata=context,
            deployment_region=region,
        )

    async def get_allocation_summary_by_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of allocations for a model

        Args:
            model_name: Model name to summarize

        Returns:
            Dictionary with counts and totals by status

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                query = """
                    SELECT
                        allocation_status,
                        COUNT(*) as count,
                        SUM(token_count) as total_tokens,
                        AVG(token_count) as avg_tokens
                    FROM token_manager
                    WHERE
                        model_name = :model_name
                        AND (expires_at IS NULL OR expires_at > NOW())
                    GROUP BY allocation_status
                """
                result = await session.execute(text(query), {"model_name": model_name})
                results = result.mappings().all()

                summary = {
                    "model_name": model_name,
                    "by_status": [dict(row) for row in results],
                }

                logger.debug(
                    f"Generated summary for model {model_name}: {len(results)} statuses"
                )
                return summary
        except Exception as e:
            logger.error(f"Error generating summary for model {model_name}: {e}")
            raise

    async def get_user_token_usage_stats(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get token usage statistics for a user

        Args:
            user_id: User UUID

        Returns:
            Dictionary with usage statistics (empty dict if no data)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            async with self.get_session() as session:
                query = """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(token_count) as total_tokens,
                        AVG(token_count) as avg_tokens_per_request,
                        AVG(latency_ms) as avg_latency_ms,
                        COUNT(CASE WHEN allocation_status = 'RELEASED' THEN 1 END) as completed_requests,
                        COUNT(CASE WHEN allocation_status = 'FAILED' THEN 1 END) as failed_requests
                    FROM token_manager
                    WHERE user_id = :user_id
                """
                result = await session.execute(text(query), {"user_id": user_id})
                result_row = result.mappings().one_or_none()

                stats = dict(result_row) if result_row else {}
                logger.debug(f"Generated usage stats for user {user_id}")
                return stats
        except Exception as e:
            logger.error(f"Error getting usage stats for user {user_id}: {e}")
            raise

    # ========================================================================
    # ALLOCATION LOGIC OPERATIONS (Core Business Logic)
    # ========================================================================

    async def retry_acquire_tokens(
        self, token_request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retry acquiring tokens for a waiting request
        Similar to MongoDB's retry_acquire method

        Args:
            token_request_id: Token request ID

        Returns:
            Updated allocation or None if not possible

        Raises:
            ValueError: If token request not found or invalid
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        try:
            # Get the allocation record
            allocation = await self.get_allocation_by_request_id(token_request_id)
            if not allocation:
                logger.warning(f"Token request not found: {token_request_id}")
                return {"error": f"Invalid token_request_id = {token_request_id}"}

            # Check if it's in WAITING status
            if allocation["allocation_status"] != "WAITING":
                logger.warning(
                    f"Token request {token_request_id} is not in WAITING status"
                )
                return {
                    "error": f"Token request {token_request_id} is not in WAITING status"
                }

            # Get model name and token count
            model_name = allocation["model_name"]
            token_count = allocation["token_count"]

            # Get least loaded deployment
            (
                total_allocated_tokens,
                chosen_model_config,
            ) = await self.get_least_loaded_deployment(model_name)
            max_token_limit = chosen_model_config.get("max_tokens", 100000)
            max_token_lock_time_secs = chosen_model_config.get(
                "max_token_lock_time_secs", 70
            )

            # Check if we can allocate now
            if total_allocated_tokens + token_count > max_token_limit:
                logger.debug(
                    f"Total allocated tokens: {total_allocated_tokens} still exceeds limit for model {model_name}"
                )
                return {
                    "alloc_status": "WAITING",
                    "token_request_id": token_request_id,
                    "model_name": model_name,
                    "token_count": token_count,
                }

            # Update the allocation to ACQUIRED
            expires_at = datetime.now() + timedelta(seconds=max_token_lock_time_secs)
            api_endpoint = chosen_model_config.get("api_endpoint_url", "")
            region = chosen_model_config.get("region", "")

            updated_allocation = await self.transition_waiting_to_acquired(
                token_request_id=token_request_id,
                api_endpoint=api_endpoint,
                region=region,
                expires_at=expires_at,
            )

            if updated_allocation:
                # Add additional fields for response
                updated_allocation["api_version"] = chosen_model_config.get(
                    "api_version", ""
                )
                updated_allocation["api_keyv_id"] = chosen_model_config.get(
                    "api_keyv_id", ""
                )
                updated_allocation["temperature"] = chosen_model_config.get(
                    "temperature", 0.0
                )
                updated_allocation["seed"] = chosen_model_config.get("seed", 42)
                return updated_allocation
            else:
                return {
                    "error": f"Failed to acquire tokens for request {token_request_id}"
                }

        except ValueError as e:
            logger.error(f"Value error in retry_acquire_tokens: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error in retry_acquire_tokens: {e}")
            raise

    async def acquire_tokens(
        self,
        user_id: UUID,
        llm_provider: str,
        model_name: str,
        token_count: int,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Acquire tokens for a model

        Args:
            user_id: User requesting tokens
            llm_provider: LLM provider name
            model_name: Model name
            token_count: Number of tokens to allocate
            request_context: Optional request context

        Returns:
            Dictionary with allocation details

        Raises:
            ValueError: If token count exceeds limit or no deployments found
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        if token_count <= 0:
            raise ValueError(f"Token count must be positive, got {token_count}")

        try:
            # Get least loaded deployment
            (
                total_allocated_tokens,
                chosen_model_config,
            ) = await self.get_least_loaded_deployment(model_name)

            # Extract max token limit and lock time
            max_token_limit = chosen_model_config.get("max_tokens")
            if not max_token_limit:
                max_token_limit = 100000  # Default if not specified

            max_token_lock_time_secs = chosen_model_config.get(
                "max_token_lock_time_secs", 70
            )

            # Check if token count exceeds limit
            if token_count > max_token_limit:
                logger.warning(
                    f"Token count {token_count} exceeds limit {max_token_limit} for model {model_name}"
                )
                return {
                    "error": f"Invalid token count, max limit exceeded for model {model_name} is {max_token_limit} for region {chosen_model_config.get('region', 'unknown')}"
                }

            logger.info(
                f"Total allocated tokens for {model_name}: {total_allocated_tokens}"
            )

            # Create token request ID
            token_request_id = f"req_{uuid.uuid4().hex}"
            now = datetime.now()
            expires_at = now + timedelta(seconds=max_token_lock_time_secs)

            # Initialize with WAITING status
            allocation_status = "WAITING"
            # api_version = ""
            deployment_name = ""
            api_endpoint = ""
            region = ""
            # api_keyv_id = ""
            temperature = 0.0
            seed = 0

            # Check if we can allocate immediately
            if (
                total_allocated_tokens + token_count <= max_token_limit
            ):  # TODO: Correct this
                # Immediate allocation (ACQUIRED)
                allocation_status = "ACQUIRED"
                # api_version = chosen_model_config.get("api_version", "")
                deployment_name = chosen_model_config.get("deployment_name", "")
                api_endpoint = chosen_model_config.get("api_endpoint_url", "")
                region = chosen_model_config.get("region", "")
                # api_keyv_id = chosen_model_config.get("api_keyv_id", "")
                temperature = chosen_model_config.get("temperature", 0.0)
                seed = chosen_model_config.get("seed", 42)

            # Create the allocation record
            allocation = await self.create_token_allocation(
                token_request_identifier=token_request_id,
                user_id=user_id,
                model_name=model_name,
                token_count=token_count,
                allocation_status=allocation_status,
                expiration_timestamp=expires_at,
                deployment_name=deployment_name,
                cloud_provider_name="azure"
                if "azure" in api_endpoint.lower()
                else "openai",
                api_endpoint_url=api_endpoint,
                deployment_region=region,
                request_metadata=request_context,
            )

            # Add additional fields for response
            allocation["temperature"] = temperature
            allocation["seed"] = seed

            return allocation

        except ValueError as e:
            logger.error(f"Value error in acquire_tokens: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error in acquire_tokens: {e}")
            raise

    async def get_least_loaded_deployment(
        self, model_name: str
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Get the least loaded deployment for a model
        Similar to MongoDB's _get_total_allocated_tokens

        Args:
            model_name: Name of the model to get deployments for

        Returns:
            Tuple of (total_allocated_tokens, chosen_model_config)

        Raises:
            ValueError: If no deployments found for model
        """
        try:
            # Get all active deployments for this model
            async with self.get_session() as session:
                # First, get all active deployments for this model
                deployments_query = """
                    SELECT *
                    FROM llm_models
                    WHERE llm_model_name = :llm_model_name AND is_active_status = TRUE
                """
                result = await session.execute(
                    text(deployments_query), {"llm_model_name": model_name}
                )
                model_deployments = result.mappings().all()

                if not model_deployments:
                    raise ValueError(
                        f"No model deployments found for llm_model_name = {model_name}"
                    )

                # Get current allocations grouped by llm_model_name and api_endpoint_url
                allocations_query = """
                    SELECT
                        llm_model_name,
                        api_endpoint_url,
                        SUM(token_count) as total_tokens
                    FROM token_manager
                    WHERE
                        llm_model_name = :llm_model_name
                        AND allocation_status IN ('ACQUIRED', 'PAUSED')
                        AND (expires_at IS NULL OR expires_at > NOW())
                    GROUP BY llm_model_name, api_endpoint_url
                    ORDER BY total_tokens ASC
                """
                result = await session.execute(
                    text(allocations_query), {"llm_model_name": model_name}
                )
                allocation_results = result.mappings().all()

                chosen_model_config = None

                # Check if any deployment's api_endpoint_url is not in the allocation results
                # This means it's unused and can be chosen immediately
                if allocation_results:
                    used_endpoints = [r["api_endpoint_url"] for r in allocation_results]
                    unused_deployments = [
                        m
                        for m in model_deployments
                        if m["api_endpoint_url"] not in used_endpoints
                        and m["api_endpoint_url"] is not None
                    ]

                    if unused_deployments:
                        # Choose the first unused deployment
                        chosen_model_config = dict(unused_deployments[0])
                        return 0, chosen_model_config

                # If no allocations found or no unused deployments, choose the first deployment
                if not allocation_results:
                    chosen_model_config = dict(model_deployments[0])
                    return 0, chosen_model_config

                # Otherwise, get the deployment with the lowest token count
                least_loaded = allocation_results[0]
                total_allocated_tokens = least_loaded["total_tokens"]

                # Find the matching deployment config
                for deployment in model_deployments:
                    if (
                        deployment["api_endpoint_url"]
                        == least_loaded["api_endpoint_url"]
                    ):
                        chosen_model_config = dict(deployment)
                        break

                # If no match found (shouldn't happen), use the first deployment
                if not chosen_model_config:
                    chosen_model_config = dict(model_deployments[0])
                    logger.warning(
                        f"No matching deployment found for endpoint {least_loaded['api_endpoint_url']}"
                    )

                return total_allocated_tokens, chosen_model_config

        except Exception as e:
            logger.error(f"Error finding least loaded deployment for {model_name}: {e}")
            raise


# ============================================================================
# CONVENIENCE FUNCTION FOR REPOSITORY INITIALIZATION
# ============================================================================


def get_token_allocation_repository(
    db_manager: Optional[DatabaseManager] = None,
) -> TokenAllocationService:
    """
    Factory function to get a TokenAllocationRepository instance

    Args:
        db_manager: Optional DatabaseManager instance (uses singleton if not provided)

    Returns:
        TokenAllocationRepository instance

    Example:
        >>> from app.core.database_connection import get_db_manager
        >>> repo = get_token_allocation_repository()
        >>> allocation = repo.create_token_allocation(
        ...     token_request_id="req_123",
        ...     user_id=UUID('12345678-1234-1234-1234-123456789012'),
        ...     model_name="gpt-4",
        ...     token_count=1000
        ... )
    """
    return TokenAllocationService(db_manager)
