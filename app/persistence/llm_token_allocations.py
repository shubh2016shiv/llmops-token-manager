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

from datetime import datetime, timedelta
import json
from typing import Any
import uuid
from uuid import UUID

from loguru import logger
from sqlalchemy import text

from app.core.database import DatabaseSessionManager
from app.models.response_models import VALID_CLOUD_PROVIDERS, VALID_LLM_PROVIDERS
from app.persistence.base import BasePersistence
from app.persistence.queries.llm_token_allocation_queries import (
    CHECK_ACTIVE_PAUSE_ALLOCATION_EXISTS_SQL,
    COUNT_ACTIVE_ALLOCATIONS_BY_MODEL_SQL,
    CREATE_TOKEN_ALLOCATION_SQL,
    CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL,
    DELETE_EXPIRED_TOKEN_ALLOCATIONS_SQL,
    DELETE_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL,
    DELETE_USER_ALLOCATIONS_BY_STATUS_SQL,
    DELETE_USER_ALLOCATIONS_SQL,
    GET_ACTIVE_DEPLOYMENT_BY_MODEL_AND_ENDPOINT_LOCKED_SQL,
    GET_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL,
    GET_TOTAL_ALLOCATED_TOKENS_FOR_ENDPOINT_SQL,
    GET_USER_TOKEN_USAGE_STATS_SQL,
    LIST_ACTIVE_MODEL_DEPLOYMENTS_SQL,
    LIST_LEAST_LOADED_ALLOCATIONS_BY_MODEL_SQL,
    LIST_TOKEN_ALLOCATION_SUMMARY_BY_MODEL_SQL,
    LIST_TOTAL_ALLOCATED_TOKENS_BY_MODEL_SQL,
    LIST_USER_ALLOCATIONS_BY_STATUS_SQL,
    LIST_USER_ALLOCATIONS_SQL,
    TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL,
)


class LLMTokenAllocationPersistence(BasePersistence):
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

    def __init__(self, database_manager: DatabaseSessionManager | None = None):
        """
        Initialize the token allocation service with database manager.

        Args:
            database_manager: Optional DatabaseManager instance (uses singleton if not provided)

        """
        super().__init__(database_manager)

    def _require_configured_max_tokens(
        self,
        chosen_model_config: dict[str, Any],
        llm_model_name: str,
        api_endpoint_url: str,
    ) -> int:
        """Require configured capacity for active deployment operations."""
        max_token_limit = chosen_model_config.get("max_tokens")
        if max_token_limit is None:
            logger.error(
                "Active deployment is missing max_tokens and cannot participate in allocation flows",
                llm_model_name=llm_model_name,
                api_endpoint_url=api_endpoint_url,
                deployment_name=chosen_model_config.get("deployment_name"),
                deployment_region=chosen_model_config.get("deployment_region"),
            )
            raise ValueError(
                "Active deployment is missing max_tokens and cannot serve requests"
            )
        return int(max_token_limit)

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

    def validate_llm_provider(self, llm_provider: str) -> None:
        """
        Validate that llm_provider is a recognised provider name.

        Uses the same canonical list as TokenAllocationResponse so that the
        persistence layer and the API response model always agree on valid values.

        Raises:
            ValueError: If llm_provider is not in the allowed set.
        """
        self.validate_enum_value(llm_provider, VALID_LLM_PROVIDERS, "llm_provider")

    def validate_cloud_provider(self, cloud_provider: str | None) -> None:
        """
        Validate an optional cloud_provider value.

        None is always valid (direct/on-premise deployment with no cloud wrapper).
        A non-None value must be in the canonical VALID_CLOUD_PROVIDERS set.

        Raises:
            ValueError: If cloud_provider is not None and not in the allowed set.
        """
        if cloud_provider is not None:
            self.validate_enum_value(
                cloud_provider, VALID_CLOUD_PROVIDERS, "cloud_provider"
            )

    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================

    async def create_token_allocation(
        self,
        token_request_identifier: str,
        user_id: UUID,
        llm_provider: str,
        llm_model_name: str,
        token_count: int,
        api_endpoint_url: str,
        allocation_status: str = DEFAULT_ALLOCATION_STATUS,
        allocation_timestamp: datetime | None = None,
        expiration_timestamp: datetime | None = None,
        deployment_name: str | None = None,
        cloud_provider_name: str | None = None,
        deployment_region: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a new token allocation record in the database.

        This method is optimized for high-concurrency scenarios with proper
        validation and error handling.

        Args:
            token_request_identifier: Unique identifier for this allocation request
            user_id: UUID of the user requesting tokens
            llm_provider: LLM provider name (e.g., openai, anthropic, gemini)
            llm_model_name: Name of the LLM model (e.g., 'gpt-4o')
            token_count: Number of tokens to allocate (must be positive)
            api_endpoint_url: Required API endpoint URL
            allocation_status: Status (ACQUIRED, WAITING, PAUSED, etc.). Defaults to 'ACQUIRED'
            allocation_timestamp: When allocation was made (defaults to current time)
            expiration_timestamp: When allocation expires (optional)
            deployment_name: Optional deployment identifier
            cloud_provider_name: Optional cloud provider name
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
        self.validate_string_not_empty(llm_provider, "llm_provider")
        self.validate_llm_provider(llm_provider)
        self.validate_string_not_empty(llm_model_name, "llm_model_name")
        self.validate_string_not_empty(api_endpoint_url, "api_endpoint_url")
        self.validate_positive_integer(token_count, "token_count")
        self.validate_allocation_status(allocation_status)
        self.validate_cloud_provider(cloud_provider_name)
        # Serialise before opening the session so a non-serialisable payload
        # raises a clear ValueError instead of an opaque DB-layer TypeError.
        request_context_json = self._validate_and_serialize_json(
            request_metadata, "request_metadata"
        )

        try:
            async with self.get_session() as session:
                params = {
                    "token_request_id": token_request_identifier,
                    "user_id": user_id,
                    "llm_provider": llm_provider,
                    "llm_model_name": llm_model_name,
                    "deployment_name": deployment_name,
                    "cloud_provider": cloud_provider_name,
                    "api_endpoint_url": api_endpoint_url,
                    "deployment_region": deployment_region,
                    "token_count": token_count,
                    "allocation_status": allocation_status,
                    "allocated_at": allocation_timestamp or datetime.now(),
                    "expires_at": expiration_timestamp,
                    "request_context": request_context_json,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                }

                result = await session.execute(
                    text(CREATE_TOKEN_ALLOCATION_SQL), params
                )
                created_allocation = result.mappings().one_or_none()

                if not created_allocation:
                    raise RuntimeError("Failed to create allocation record")

                self.log_operation(
                    "CREATE",
                    token_request_identifier,
                    success=True,
                    additional_context=f"{token_count} tokens for {llm_model_name}",
                )
                return dict(created_allocation)
        except Exception as e:
            logger.error(f"Error creating allocation {token_request_identifier}: {e}")
            raise

    async def create_token_allocation_with_capacity_check(
        self,
        token_request_identifier: str,
        user_id: UUID,
        llm_provider: str,
        llm_model_name: str,
        token_count: int,
        api_endpoint_url: str,
        allocation_timestamp: datetime | None = None,
        expiration_timestamp: datetime | None = None,
        deployment_name: str | None = None,
        cloud_provider_name: str | None = None,
        deployment_region: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Create an allocation after an atomic capacity check.

        This is the DB-fallback allocation primitive. It locks the selected
        active deployment row, recomputes current active load, decides
        ACQUIRED vs WAITING, and inserts the allocation in the same transaction.
        That prevents concurrent DB fallback requests from all observing the
        same stale capacity snapshot and over-allocating the endpoint.
        """
        self.validate_string_not_empty(
            token_request_identifier, "token_request_identifier"
        )
        self.validate_uuid(user_id, "user_id")
        self.validate_string_not_empty(llm_provider, "llm_provider")
        self.validate_llm_provider(llm_provider)
        self.validate_string_not_empty(llm_model_name, "llm_model_name")
        self.validate_positive_integer(token_count, "token_count")
        self.validate_string_not_empty(api_endpoint_url, "api_endpoint_url")
        self.validate_cloud_provider(cloud_provider_name)
        request_context_json = self._validate_and_serialize_json(
            request_metadata, "request_metadata"
        )

        try:
            async with self.get_session() as session:
                params = {
                    "token_request_id": token_request_identifier,
                    "user_id": user_id,
                    "llm_provider": llm_provider,
                    "llm_model_name": llm_model_name,
                    "deployment_name": deployment_name,
                    "cloud_provider": cloud_provider_name,
                    "api_endpoint_url": api_endpoint_url,
                    "deployment_region": deployment_region,
                    "token_count": token_count,
                    "allocated_at": allocation_timestamp or datetime.now(),
                    "expires_at": expiration_timestamp,
                    "request_context": request_context_json,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                }

                result = await session.execute(
                    text(CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL), params
                )
                created_allocation = result.mappings().one_or_none()
                if not created_allocation:
                    raise ValueError(
                        "No active deployment with enough single-request capacity "
                        f"found for {llm_provider}/{llm_model_name} at {api_endpoint_url}"
                    )

                allocation = dict(created_allocation)
                self.log_operation(
                    "CREATE",
                    token_request_identifier,
                    success=True,
                    additional_context=(
                        f"{token_count} tokens for {llm_model_name} "
                        f"as {allocation.get('allocation_status')}"
                    ),
                )
                return allocation
        except Exception as e:
            logger.error(
                f"Error atomically creating allocation {token_request_identifier}: {e}"
            )
            raise

    # ========================================================================
    # READ OPERATIONS
    # ========================================================================

    async def get_allocation_by_request_id(
        self, token_request_identifier: str
    ) -> dict[str, Any] | None:
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
                result = await session.execute(
                    text(GET_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL),
                    {"token_request_id": token_request_identifier},
                )
                allocation_record = result.mappings().one_or_none()
                return dict(allocation_record) if allocation_record else None
        except Exception as e:
            logger.error(f"Error fetching allocation {token_request_identifier}: {e}")
            raise

    async def get_total_allocated_tokens_by_model(
        self, llm_model_name: str, included_statuses: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Get total allocated tokens grouped by model and API endpoint.

        This method is critical for load balancing - it returns endpoints
        sorted by total allocated tokens (least loaded first).

        Optimized for high-concurrency scenarios with proper indexing.

        Args:
            llm_model_name: LLM model name to query
            included_statuses: List of statuses to include (default: ACQUIRED, PAUSED)

        Returns:
            List of dictionaries with aggregated token counts per endpoint,
            sorted by total_tokens ascending (least loaded first)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If llm_model_name is invalid

        """
        self.validate_string_not_empty(llm_model_name, "llm_model_name")

        if included_statuses is None:
            included_statuses = ["ACQUIRED", "PAUSED"]

        # Validate all statuses
        for status in included_statuses:
            self.validate_allocation_status(status)

        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(LIST_TOTAL_ALLOCATED_TOKENS_BY_MODEL_SQL),
                    {
                        "llm_model_name": llm_model_name,
                        "included_statuses": included_statuses,
                    },
                )
                endpoint_statistics = result.mappings().all()

                logger.debug(
                    f"Found {len(endpoint_statistics)} endpoints for model {llm_model_name}"
                )
                return [dict(row) for row in endpoint_statistics]
        except Exception as e:
            logger.error(f"Error fetching allocations for model {llm_model_name}: {e}")
            raise

    async def get_total_allocated_tokens_for_endpoint(
        self, llm_model_name: str, api_endpoint_url: str
    ) -> int:
        """
        Get total allocated tokens for a specific model and endpoint.

        This method is used for real-time load checking before allocation.

        Args:
            llm_model_name: LLM model name
            api_endpoint_url: API endpoint URL

        Returns:
            Total number of allocated tokens (0 if none found)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors
            ValueError: If parameters are invalid

        """
        self.validate_string_not_empty(llm_model_name, "llm_model_name")
        self.validate_string_not_empty(api_endpoint_url, "api_endpoint_url")

        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(GET_TOTAL_ALLOCATED_TOKENS_FOR_ENDPOINT_SQL),
                    {
                        "llm_model_name": llm_model_name,
                        "api_endpoint_url": api_endpoint_url,
                    },
                )
                count_result = result.scalar_one_or_none()
                return count_result if count_result else 0
        except Exception as e:
            logger.error(f"Error fetching tokens for endpoint {api_endpoint_url}: {e}")
            raise

    async def get_user_allocations(
        self, user_id: UUID, status_filter: list[str] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
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
                    result = await session.execute(
                        text(LIST_USER_ALLOCATIONS_BY_STATUS_SQL),
                        {
                            "user_id": user_id,
                            "status_filter": status_filter,
                            "limit": limit,
                        },
                    )
                else:
                    result = await session.execute(
                        text(LIST_USER_ALLOCATIONS_SQL),
                        {"user_id": user_id, "limit": limit},
                    )

                results = result.mappings().all()
                logger.debug(f"Found {len(results)} allocations for user {user_id}")
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error fetching user allocations for {user_id}: {e}")
            raise

    async def get_active_allocations_count_by_model(self, llm_model_name: str) -> int:
        """
        Get count of active allocations for a model

        Args:
            llm_model_name: LLM model name

        Returns:
            Count of active allocations (0 if none found)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors

        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(COUNT_ACTIVE_ALLOCATIONS_BY_MODEL_SQL),
                    {"llm_model_name": llm_model_name},
                )
                return result.scalar_one_or_none() or 0
        except Exception as e:
            logger.error(f"Error counting active allocations for {llm_model_name}: {e}")
            raise

    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================

    async def update_allocation_status(
        self,
        token_request_id: str,
        new_status: str,
        api_endpoint: str | None = None,
        deployment_region: str | None = None,
        expires_at: datetime | None = None,
        completed_at: datetime | None = None,
        latency_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Update allocation status and related fields

        Args:
            token_request_id: Unique token request identifier
            new_status: New status to set (ACQUIRED, WAITING, PAUSED, RELEASED, FAILED)
            api_endpoint: Optional endpoint to update
            deployment_region: Optional deployment region to update
            expires_at: Optional new expiration time
            completed_at: Optional completion timestamp
            latency_ms: Optional latency in milliseconds

        Returns:
            Updated record or None if not found

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors

        """
        self.validate_string_not_empty(token_request_id, "token_request_id")
        self.validate_allocation_status(new_status)
        if api_endpoint is not None:
            self.validate_string_not_empty(api_endpoint, "api_endpoint")
        if latency_ms is not None:
            self.validate_positive_integer(latency_ms, "latency_ms", allow_zero=True)

        try:
            async with self.get_session() as session:
                # Build dynamic update query
                update_fields = ["allocation_status = :new_status"]
                params: dict[str, Any] = {
                    "new_status": new_status,
                    "token_request_id": token_request_id,
                }

                if api_endpoint is not None:
                    update_fields.append("api_endpoint_url = :api_endpoint_url")
                    params["api_endpoint_url"] = api_endpoint

                if deployment_region is not None:
                    update_fields.append("deployment_region = :deployment_region")
                    params["deployment_region"] = deployment_region

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
        deployment_region: str,
        expires_at: datetime,
    ) -> dict[str, Any] | None:
        """
        Atomically transition allocation from WAITING to ACQUIRED
        Only succeeds if current status is WAITING (prevents race conditions)

        Args:
            token_request_id: Unique token request identifier
            api_endpoint: API endpoint to assign
            deployment_region: Deployment region to assign
            expires_at: New expiration time

        Returns:
            Updated record or None if transition failed (not in WAITING state)

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors

        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL),
                    {
                        "api_endpoint_url": api_endpoint,
                        "deployment_region": deployment_region,
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
                result = await session.execute(
                    text(DELETE_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL),
                    {"token_request_id": token_request_id},
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
                result = await session.execute(
                    text(DELETE_EXPIRED_TOKEN_ALLOCATIONS_SQL)
                )
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
        self, user_id: UUID, status: str | None = None
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
                    result = await session.execute(
                        text(DELETE_USER_ALLOCATIONS_BY_STATUS_SQL),
                        {"user_id": user_id, "status": status},
                    )
                else:
                    result = await session.execute(
                        text(DELETE_USER_ALLOCATIONS_SQL), {"user_id": user_id}
                    )

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
        llm_model_name: str,
        api_endpoint: str,
        pause_reason: str = "",
        pause_duration_minutes: int = 30,
    ) -> dict[str, Any]:
        """
        Atomically pause a deployment by creating a PAUSED capacity-blocker allocation.

        The method serializes concurrent pause requests through a PostgreSQL row lock
        on the target llm_models row (FOR UPDATE).  The lock is held for the entire
        transaction, so a second concurrent caller blocks at the deployment lookup
        until the first commits.  Once the first caller commits the INSERT, the second
        caller finds the existing PAUSED row in the check and returns ALREADY_PAUSED
        without a duplicate insert.

        All three operations — lock, check, insert — execute within the same session
        and therefore the same transaction.  No cross-session TOCTOU window exists.

        Args:
            user_id: User requesting the pause
            llm_provider: LLM provider name
            llm_model_name: Model name
            api_endpoint: API endpoint URL to pause
            pause_reason: Reason for pausing (stored in request_context)
            pause_duration_minutes: Duration to pause for (must be > 0)

        Returns:
            Allocation record dict (allocation_status = 'PAUSED') on success.
            Sentinel dict with alloc_status = 'NOT_FOUND' or 'ALREADY_PAUSED'
            when those conditions are detected.

        Raises:
            ValueError: If pause_duration_minutes <= 0 or deployment is missing max_tokens
            sqlalchemy.exc.SQLAlchemyError: On database errors
        """
        if pause_duration_minutes <= 0:
            raise ValueError(
                f"Pause duration must be positive, got {pause_duration_minutes}"
            )

        try:
            async with self.get_session() as session:
                # Step 1 — Lock the deployment row.
                # FOR UPDATE blocks any concurrent transaction that tries to lock the
                # same llm_models row, serialising all pause requests per deployment.
                result = await session.execute(
                    text(GET_ACTIVE_DEPLOYMENT_BY_MODEL_AND_ENDPOINT_LOCKED_SQL),
                    {
                        "llm_model_name": llm_model_name,
                        "api_endpoint_url": api_endpoint,
                    },
                )
                row = result.mappings().one_or_none()

                if not row:
                    logger.warning(
                        f"Deployment not found: {llm_model_name} at {api_endpoint}"
                    )
                    return {
                        "alloc_status": "NOT_FOUND",
                        "llm_model_name": llm_model_name,
                        "api_endpoint_url": api_endpoint,
                        "reason": "Deployment not found",
                    }

                chosen_model_config: dict[str, Any] = dict(row)

                # Step 2 — Check for an existing active pause under the lock.
                # Safe from TOCTOU: the deployment row lock prevents a concurrent
                # caller from reaching this point until we commit or roll back.
                existing_pause = await session.execute(
                    text(CHECK_ACTIVE_PAUSE_ALLOCATION_EXISTS_SQL),
                    {
                        "llm_model_name": llm_model_name,
                        "api_endpoint_url": api_endpoint,
                    },
                )
                if existing_pause.scalar_one_or_none():
                    logger.warning(
                        f"Deployment {llm_model_name} at {api_endpoint} is already paused."
                    )
                    return {
                        "alloc_status": "ALREADY_PAUSED",
                        "llm_model_name": llm_model_name,
                        "api_endpoint_url": api_endpoint,
                        "reason": "Deployment is already in a paused state.",
                    }

                # Step 3 — Extract deployment fields and insert within the same transaction.
                # cloud_provider reads the correct schema column name (not provider_name).
                max_token_limit = self._require_configured_max_tokens(
                    chosen_model_config, llm_model_name, api_endpoint
                )
                cloud_provider = chosen_model_config.get("cloud_provider")
                deployment_region = (
                    chosen_model_config.get("deployment_region") or "unknown"
                )
                deployment_name = chosen_model_config.get("deployment_name") or ""

                token_request_id = f"pause_{uuid.uuid4().hex}"
                expiration_timestamp = datetime.now() + timedelta(
                    minutes=pause_duration_minutes
                )
                context: dict[str, Any] = {"operation": "pause_deployment"}
                if pause_reason:
                    context["reason"] = pause_reason

                params = {
                    "token_request_id": token_request_id,
                    "user_id": user_id,
                    "llm_provider": llm_provider,
                    "llm_model_name": llm_model_name,
                    "deployment_name": deployment_name or None,
                    "cloud_provider": cloud_provider,
                    "api_endpoint_url": api_endpoint,
                    "deployment_region": deployment_region or None,
                    "token_count": max_token_limit,
                    "allocation_status": "PAUSED",
                    "allocated_at": datetime.now(),
                    "expires_at": expiration_timestamp,
                    "request_context": json.dumps(context),
                    "temperature": None,
                    "top_p": None,
                    "seed": None,
                }

                insert_result = await session.execute(
                    text(CREATE_TOKEN_ALLOCATION_SQL), params
                )
                created_allocation = insert_result.mappings().one_or_none()

                if not created_allocation:
                    raise RuntimeError("Failed to create pause allocation record")

                logger.info(
                    f"Created pause allocation for {llm_model_name} at {api_endpoint} "
                    f"for {pause_duration_minutes}m"
                )
                return dict(created_allocation)

        except ValueError as e:
            logger.error(f"Value error in pause_deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error in pause_deployment: {e}")
            raise

    async def get_allocation_summary_by_model(
        self, llm_model_name: str
    ) -> dict[str, Any]:
        """
        Get comprehensive summary of allocations for a model

        Args:
            llm_model_name: Model name to summarize

        Returns:
            Dictionary with counts and totals by status

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors

        """
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(LIST_TOKEN_ALLOCATION_SUMMARY_BY_MODEL_SQL),
                    {"llm_model_name": llm_model_name},
                )
                results = result.mappings().all()

                summary = {
                    "llm_model_name": llm_model_name,
                    "by_status": [dict(row) for row in results],
                }

                logger.debug(
                    f"Generated summary for model {llm_model_name}: {len(results)} statuses"
                )
                return summary
        except Exception as e:
            logger.error(f"Error generating summary for model {llm_model_name}: {e}")
            raise

    async def get_user_token_usage_stats(self, user_id: UUID) -> dict[str, Any]:
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
                result = await session.execute(
                    text(GET_USER_TOKEN_USAGE_STATS_SQL), {"user_id": user_id}
                )
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
    ) -> dict[str, Any] | None:
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
            llm_provider = allocation["llm_provider"]
            llm_model_name = allocation["llm_model_name"]
            token_count = allocation["token_count"]

            # Get least loaded deployment
            (
                total_allocated_tokens,
                chosen_model_config,
            ) = await self.get_least_loaded_deployment(llm_provider, llm_model_name)
            max_token_limit = self._require_configured_max_tokens(
                chosen_model_config,
                llm_model_name,
                chosen_model_config.get("api_endpoint_url", ""),
            )
            max_token_lock_time_secs = chosen_model_config.get(
                "max_token_lock_time_secs", 70
            )

            # Check if we can allocate now
            if total_allocated_tokens + token_count > max_token_limit:
                logger.debug(
                    f"Total allocated tokens: {total_allocated_tokens} still exceeds limit for model {llm_model_name}"
                )
                return {
                    "alloc_status": "WAITING",
                    "token_request_id": token_request_id,
                    "llm_model_name": llm_model_name,
                    "token_count": token_count,
                }

            # Update the allocation to ACQUIRED
            expires_at = datetime.now() + timedelta(seconds=max_token_lock_time_secs)
            api_endpoint = chosen_model_config.get("api_endpoint_url", "")
            deployment_region = chosen_model_config.get("deployment_region", "")

            updated_allocation = await self.transition_waiting_to_acquired(
                token_request_id=token_request_id,
                api_endpoint=api_endpoint,
                deployment_region=deployment_region,
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
                latest_allocation = await self.get_allocation_by_request_id(
                    token_request_id
                )
                if (
                    latest_allocation
                    and latest_allocation.get("allocation_status") == "WAITING"
                ):
                    return {
                        "alloc_status": "WAITING",
                        "token_request_id": token_request_id,
                        "llm_model_name": llm_model_name,
                        "token_count": token_count,
                    }
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
        llm_model_name: str,
        token_count: int,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Acquire tokens for a model

        Args:
            user_id: User requesting tokens
            llm_provider: LLM provider name
            llm_model_name: Model name
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
            ) = await self.get_least_loaded_deployment(llm_provider, llm_model_name)

            # Extract max token limit and lock time
            max_token_limit = self._require_configured_max_tokens(
                chosen_model_config,
                llm_model_name,
                chosen_model_config.get("api_endpoint_url", ""),
            )

            max_token_lock_time_secs = chosen_model_config.get(
                "max_token_lock_time_secs", 70
            )

            # Check if token count exceeds limit
            if token_count > max_token_limit:
                logger.warning(
                    f"Token count {token_count} exceeds limit {max_token_limit} for model {llm_model_name}"
                )
                return {
                    "error": f"Invalid token count, max limit exceeded for model {llm_model_name} is {max_token_limit} for region {{chosen_model_config.get('deployment_region', 'unknown')}}"
                }

            logger.info(
                f"Total allocated tokens for {llm_model_name}: {total_allocated_tokens}"
            )

            # Create token request ID
            token_request_id = f"req_{uuid.uuid4().hex}"
            now = datetime.now()
            expires_at = now + timedelta(seconds=max_token_lock_time_secs)

            deployment_name = chosen_model_config.get("deployment_name", "")
            api_endpoint = chosen_model_config.get("api_endpoint_url", "")
            deployment_region = chosen_model_config.get("deployment_region", "")
            temperature = chosen_model_config.get("temperature", 0.0)
            seed = chosen_model_config.get(
                "seed", chosen_model_config.get("random_seed", 42)
            )

            # Create the allocation record
            allocation = await self.create_token_allocation_with_capacity_check(
                token_request_identifier=token_request_id,
                user_id=user_id,
                llm_provider=llm_provider,
                llm_model_name=llm_model_name,
                token_count=token_count,
                expiration_timestamp=expires_at,
                deployment_name=deployment_name,
                cloud_provider_name=chosen_model_config.get("cloud_provider"),
                api_endpoint_url=api_endpoint,
                deployment_region=deployment_region,
                request_metadata=request_context,
                temperature=temperature,
                top_p=chosen_model_config.get("top_p"),
                seed=seed,
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
        self, llm_provider: str, llm_model_name: str
    ) -> tuple[int, dict[str, Any]]:
        """
        Get the least loaded deployment for a provider/model pair.
        Similar to MongoDB's _get_total_allocated_tokens

        Args:
            llm_provider: Provider name to constrain deployment selection
            llm_model_name: Name of the model to get deployments for

        Returns:
            Tuple of (total_allocated_tokens, chosen_model_config)

        Raises:
            ValueError: If no deployments found for model

        """
        self.validate_string_not_empty(llm_provider, "llm_provider")
        self.validate_string_not_empty(llm_model_name, "llm_model_name")

        try:
            # Get all active deployments for this provider/model pair.
            async with self.get_session() as session:
                result = await session.execute(
                    text(LIST_ACTIVE_MODEL_DEPLOYMENTS_SQL),
                    {
                        "llm_provider": llm_provider,
                        "llm_model_name": llm_model_name,
                    },
                )
                model_deployments = []
                for row in result.mappings().all():
                    deployment = dict(row)
                    # Support older fixtures that still expose api_base instead of
                    # api_endpoint_url while normalizing downstream logic.
                    deployment["api_endpoint_url"] = deployment.get(
                        "api_endpoint_url"
                    ) or deployment.get("api_base")
                    model_deployments.append(deployment)

                if not model_deployments:
                    raise ValueError(
                        "No model deployments found for "
                        f"llm_provider = {llm_provider}, "
                        f"llm_model_name = {llm_model_name}"
                    )

                valid_model_deployments = [
                    deployment
                    for deployment in model_deployments
                    if deployment.get("max_tokens") is not None
                    and deployment.get("api_endpoint_url") is not None
                ]
                invalid_model_deployments = [
                    deployment
                    for deployment in model_deployments
                    if deployment not in valid_model_deployments
                ]
                for invalid_deployment in invalid_model_deployments:
                    logger.error(
                        "Active deployment is missing max_tokens and is excluded from least-loaded selection",
                        llm_provider=invalid_deployment.get("llm_provider"),
                        llm_model_name=invalid_deployment.get("llm_model_name"),
                        api_endpoint_url=invalid_deployment.get("api_endpoint_url"),
                        deployment_name=invalid_deployment.get("deployment_name"),
                        deployment_region=invalid_deployment.get("deployment_region"),
                    )
                if not valid_model_deployments:
                    raise ValueError(
                        "No valid active deployments with max_tokens found for "
                        f"llm_provider = {llm_provider}, "
                        f"llm_model_name = {llm_model_name}"
                    )

                # Get current allocations for the same provider/model pair.
                result = await session.execute(
                    text(LIST_LEAST_LOADED_ALLOCATIONS_BY_MODEL_SQL),
                    {
                        "llm_provider": llm_provider,
                        "llm_model_name": llm_model_name,
                    },
                )
                allocation_results = result.mappings().all()

                chosen_model_config = None

                # Check if any deployment's api_endpoint_url is not in the allocation results
                # This means it's unused and can be chosen immediately
                if allocation_results:
                    used_endpoints = [r["api_endpoint_url"] for r in allocation_results]
                    unused_deployments = [
                        m
                        for m in valid_model_deployments
                        if m["api_endpoint_url"] not in used_endpoints
                    ]

                    if unused_deployments:
                        # Choose the first unused deployment
                        chosen_model_config = unused_deployments[0]
                        return 0, chosen_model_config

                # If no allocations found or no unused deployments, choose the first deployment
                if not allocation_results:
                    chosen_model_config = valid_model_deployments[0]
                    return 0, chosen_model_config

                # Otherwise, get the deployment with the lowest token count
                least_loaded = allocation_results[0]
                total_allocated_tokens = least_loaded["total_tokens"]

                # Find the matching deployment config
                for deployment in valid_model_deployments:
                    if (
                        deployment["api_endpoint_url"]
                        == least_loaded["api_endpoint_url"]
                    ):
                        chosen_model_config = deployment
                        break

                # If no match found (shouldn't happen), use the first deployment
                if not chosen_model_config:
                    chosen_model_config = valid_model_deployments[0]
                    logger.warning(
                        f"No matching deployment found for endpoint {least_loaded['api_endpoint_url']}"
                    )

                return total_allocated_tokens, chosen_model_config

        except Exception as e:
            logger.error(
                "Error finding least loaded deployment for "
                f"{llm_provider}/{llm_model_name}: {e}"
            )
            raise


# ============================================================================
# CONVENIENCE FUNCTION FOR REPOSITORY INITIALIZATION
# ============================================================================


def get_token_allocation_repository(
    db_manager: DatabaseSessionManager | None = None,
) -> LLMTokenAllocationPersistence:
    """
    Factory function to get a TokenAllocationRepository instance

    Args:
        db_manager: Optional DatabaseManager instance (uses singleton if not provided)

    Returns:
        TokenAllocationRepository instance

    Example:
        >>> from app.core.database import get_db_manager
        >>> db_mgr = get_db_manager()
        >>> repo = get_token_allocation_repository(db_mgr)
        >>> allocation = repo.create_token_allocation(
        ...     token_request_id="req_123",
        ...     user_id=UUID('12345678-1234-1234-1234-123456789012'),
        ...     llm_provider="openai",
        ...     llm_model_name="gpt-4",
        ...     token_count=1000
        ... )

    """
    return LLMTokenAllocationPersistence(db_manager)
