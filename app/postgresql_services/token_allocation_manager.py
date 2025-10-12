"""
PostgreSQL CRUD Operations for Token Allocation Management
Provides database operations for token allocation, tracking, and lifecycle management
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid
from uuid import UUID
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from loguru import logger

from app.core.database_connection import DatabaseManager


class TokenAllocationService:
    """
    Repository for token allocation database operations
    Uses DatabaseManager for connection pooling and transaction management
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize repository with database manager

        Args:
            db_manager: Optional DatabaseManager instance (uses singleton if not provided)
        """
        self.db_manager = db_manager or DatabaseManager()

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic commit/rollback

        Yields:
            Database connection from pool

        Raises:
            psycopg2.Error: On database errors
        """
        conn = None
        try:
            conn = self.db_manager.get_connection()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database transaction error: {e}", exc_info=True)
            raise
        finally:
            if conn:
                self.db_manager.release_connection(conn)

    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================

    def create_token_allocation(
        self,
        token_request_id: str,
        user_id: UUID,
        model_name: str,
        token_count: int,
        allocation_status: str = "ACQUIRED",
        allocated_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        model_id: Optional[UUID] = None,
        deployment_name: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        region: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new token allocation record

        Args:
            token_request_id: Unique identifier for this allocation
            user_id: User requesting tokens
            model_name: LLM model name
            token_count: Number of tokens to allocate
            allocation_status: Status (ACQUIRED, WAITING, PAUSED, etc.)
            allocated_at: When allocation was made (defaults to now)
            expires_at: When allocation expires
            model_id: Optional model UUID reference
            deployment_name: Optional deployment identifier
            cloud_provider: Optional cloud provider name
            api_endpoint: Optional API endpoint URL
            region: Optional region identifier
            request_context: Optional JSON metadata

        Returns:
            Dictionary containing the created allocation record

        Raises:
            psycopg2.Error: On database errors
            ValueError: On invalid input parameters
        """
        if token_count <= 0:
            raise ValueError(f"Token count must be positive, got {token_count}")

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        INSERT INTO token_allocations (
                            token_request_id, user_id, model_name, model_id,
                            deployment_name, cloud_provider, api_endpoint, region,
                            token_count, allocation_status, allocated_at, expires_at,
                            request_context
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING *
                    """

                    cur.execute(
                        query,
                        (
                            token_request_id,
                            user_id,
                            model_name,
                            model_id,
                            deployment_name,
                            cloud_provider,
                            api_endpoint,
                            region,
                            token_count,
                            allocation_status,
                            allocated_at or datetime.now(),
                            expires_at,
                            psycopg2.extras.Json(request_context) if request_context else None,
                        ),
                    )

                    result = cur.fetchone()
                    if not result:
                        raise RuntimeError("Failed to create allocation record")

                    logger.info(f"Created token allocation: {token_request_id} ({token_count} tokens)")
                    return dict(result)
        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity error creating allocation {token_request_id}: {e}")
            raise
        except psycopg2.Error as e:
            logger.error(f"Database error creating allocation {token_request_id}: {e}")
            raise

    # ========================================================================
    # READ OPERATIONS
    # ========================================================================

    def get_allocation_by_request_id(self, token_request_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a token allocation by its request ID

        Args:
            token_request_id: Unique token request identifier

        Returns:
            Dictionary containing allocation record or None if not found

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT * FROM token_allocations
                        WHERE token_request_id = %s
                    """
                    cur.execute(query, (token_request_id,))
                    result = cur.fetchone()
                    return dict(result) if result else None
        except psycopg2.Error as e:
            logger.error(f"Error fetching allocation {token_request_id}: {e}")
            raise

    def get_total_allocated_tokens_by_model(
        self, model_name: str, include_statuses: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get total allocated tokens grouped by model and API endpoint
        Useful for finding the least-loaded deployment

        Args:
            model_name: LLM model name to query
            include_statuses: List of statuses to include (default: ACQUIRED, PAUSED)

        Returns:
            List of dictionaries with aggregated token counts per endpoint
            Sorted by total_tokens ascending (least loaded first)

        Raises:
            psycopg2.Error: On database errors
        """
        if include_statuses is None:
            include_statuses = ["ACQUIRED", "PAUSED"]

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT
                            model_name,
                            api_endpoint,
                            region,
                            cloud_provider,
                            SUM(token_count) as total_tokens,
                            COUNT(*) as allocation_count
                        FROM token_allocations
                        WHERE
                            model_name = %s
                            AND allocation_status = ANY(%s)
                            AND (expires_at IS NULL OR expires_at > NOW())
                        GROUP BY model_name, api_endpoint, region, cloud_provider
                        ORDER BY total_tokens ASC
                    """

                    cur.execute(query, (model_name, include_statuses))
                    results = cur.fetchall()

                    logger.debug(f"Found {len(results)} endpoints for model {model_name}")
                    return [dict(row) for row in results]
        except psycopg2.Error as e:
            logger.error(f"Error fetching allocations for model {model_name}: {e}")
            raise

    def get_total_allocated_tokens_for_endpoint(self, model_name: str, api_endpoint: str) -> int:
        """
        Get total allocated tokens for a specific model and endpoint

        Args:
            model_name: LLM model name
            api_endpoint: API endpoint URL

        Returns:
            Total number of allocated tokens (0 if none found)

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT COALESCE(SUM(token_count), 0) as total_tokens
                        FROM token_allocations
                        WHERE
                            model_name = %s
                            AND api_endpoint = %s
                            AND allocation_status IN ('ACQUIRED', 'PAUSED')
                            AND (expires_at IS NULL OR expires_at > NOW())
                    """

                    cur.execute(query, (model_name, api_endpoint))
                    result = cur.fetchone()
                    return result[0] if result else 0
        except psycopg2.Error as e:
            logger.error(f"Error fetching tokens for endpoint {api_endpoint}: {e}")
            raise

    def get_user_allocations(
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
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if status_filter:
                        query = """
                            SELECT * FROM token_allocations
                            WHERE user_id = %s AND allocation_status = ANY(%s)
                            ORDER BY allocated_at DESC
                            LIMIT %s
                        """
                        cur.execute(query, (user_id, status_filter, limit))
                    else:
                        query = """
                            SELECT * FROM token_allocations
                            WHERE user_id = %s
                            ORDER BY allocated_at DESC
                            LIMIT %s
                        """
                        cur.execute(query, (user_id, limit))

                    results = cur.fetchall()
                    logger.debug(f"Found {len(results)} allocations for user {user_id}")
                    return [dict(row) for row in results]
        except psycopg2.Error as e:
            logger.error(f"Error fetching user allocations for {user_id}: {e}")
            raise

    def get_active_allocations_count_by_model(self, model_name: str) -> int:
        """
        Get count of active allocations for a model

        Args:
            model_name: LLM model name

        Returns:
            Count of active allocations (0 if none found)

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT COUNT(*)
                        FROM token_allocations
                        WHERE
                            model_name = %s
                            AND allocation_status IN ('ACQUIRED', 'PAUSED')
                            AND (expires_at IS NULL OR expires_at > NOW())
                    """
                    cur.execute(query, (model_name,))
                    result = cur.fetchone()
                    return result[0] if result else 0
        except psycopg2.Error as e:
            logger.error(f"Error counting active allocations for {model_name}: {e}")
            raise

    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================

    def update_allocation_status(
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
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Build dynamic update query
                    update_fields = ["allocation_status = %s"]
                    params = [new_status]

                    if api_endpoint is not None:
                        update_fields.append("api_endpoint = %s")
                        params.append(api_endpoint)

                    if region is not None:
                        update_fields.append("region = %s")
                        params.append(region)

                    if expires_at is not None:
                        update_fields.append("expires_at = %s")
                        params.append(expires_at)

                    if completed_at is not None:
                        update_fields.append("completed_at = %s")
                        params.append(completed_at)

                    if latency_ms is not None:
                        update_fields.append("latency_ms = %s")
                        params.append(latency_ms)

                    params.append(token_request_id)

                    query = f"""
                        UPDATE token_allocations
                        SET {', '.join(update_fields)}
                        WHERE token_request_id = %s
                        RETURNING *
                    """

                    cur.execute(query, params)
                    result = cur.fetchone()

                    if result:
                        logger.info(f"Updated allocation {token_request_id} to status {new_status}")
                        return dict(result)

                    logger.warning(f"Allocation {token_request_id} not found for update")
                    return None
        except psycopg2.Error as e:
            logger.error(f"Error updating allocation {token_request_id}: {e}")
            raise

    def transition_waiting_to_acquired(
        self, token_request_id: str, api_endpoint: str, region: str, expires_at: datetime
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
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        UPDATE token_allocations
                        SET
                            allocation_status = 'ACQUIRED',
                            api_endpoint = %s,
                            region = %s,
                            expires_at = %s
                        WHERE
                            token_request_id = %s
                            AND allocation_status = 'WAITING'
                        RETURNING *
                    """

                    cur.execute(query, (api_endpoint, region, expires_at, token_request_id))
                    result = cur.fetchone()

                    if result:
                        logger.info(f"Transitioned {token_request_id} from WAITING to ACQUIRED")
                        return dict(result)

                    logger.debug(f"Transition failed for {token_request_id} (not in WAITING state)")
                    return None
        except psycopg2.Error as e:
            logger.error(f"Error transitioning allocation {token_request_id}: {e}")
            raise

    def update_allocation_completed(
        self, token_request_id: str, latency_ms: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Mark allocation as completed (RELEASED status) and calculate latency

        Args:
            token_request_id: Unique token request identifier
            latency_ms: Optional pre-calculated latency in milliseconds

        Returns:
            Updated record or None if not found

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if latency_ms is None:
                        # Calculate latency from allocated_at to now
                        query = """
                            UPDATE token_allocations
                            SET
                                allocation_status = 'RELEASED',
                                completed_at = NOW(),
                                latency_ms = EXTRACT(EPOCH FROM (NOW() - allocated_at)) * 1000
                            WHERE token_request_id = %s
                            RETURNING *
                        """
                        cur.execute(query, (token_request_id,))
                    else:
                        query = """
                            UPDATE token_allocations
                            SET
                                allocation_status = 'RELEASED',
                                completed_at = NOW(),
                                latency_ms = %s
                            WHERE token_request_id = %s
                            RETURNING *
                        """
                        cur.execute(query, (latency_ms, token_request_id))

                    result = cur.fetchone()
                    if result:
                        logger.info(f"Completed allocation {token_request_id}")
                    return dict(result) if result else None
        except psycopg2.Error as e:
            logger.error(f"Error completing allocation {token_request_id}: {e}")
            raise

    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================

    def delete_allocation(self, token_request_id: str) -> bool:
        """
        Delete a token allocation (release tokens permanently)

        Args:
            token_request_id: Unique token request identifier

        Returns:
            True if deleted, False if not found

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        DELETE FROM token_allocations
                        WHERE token_request_id = %s
                    """
                    cur.execute(query, (token_request_id,))
                    deleted = cur.rowcount > 0

                    if deleted:
                        logger.info(f"Deleted allocation: {token_request_id}")
                    else:
                        logger.debug(f"Allocation not found for deletion: {token_request_id}")

                    return deleted
        except psycopg2.Error as e:
            logger.error(f"Error deleting allocation {token_request_id}: {e}")
            raise

    def delete_expired_allocations(self) -> int:
        """
        Clean up expired allocations (batch cleanup operation)

        Returns:
            Number of deleted records

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        DELETE FROM token_allocations
                        WHERE
                            expires_at IS NOT NULL
                            AND expires_at < NOW()
                            AND allocation_status IN ('ACQUIRED', 'PAUSED', 'WAITING')
                    """
                    cur.execute(query)
                    deleted_count = cur.rowcount

                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} expired allocations")
                    else:
                        logger.debug("No expired allocations to clean up")

                    return deleted_count
        except psycopg2.Error as e:
            logger.error(f"Error deleting expired allocations: {e}")
            raise

    def delete_allocations_by_user(self, user_id: UUID, status: Optional[str] = None) -> int:
        """
        Delete all allocations for a user (optional: filter by status)

        Args:
            user_id: User UUID
            status: Optional status filter

        Returns:
            Number of deleted records

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if status:
                        query = """
                            DELETE FROM token_allocations
                            WHERE user_id = %s AND allocation_status = %s
                        """
                        cur.execute(query, (user_id, status))
                    else:
                        query = """
                            DELETE FROM token_allocations
                            WHERE user_id = %s
                        """
                        cur.execute(query, (user_id,))

                    deleted_count = cur.rowcount
                    logger.info(f"Deleted {deleted_count} allocations for user {user_id}")
                    return deleted_count
        except psycopg2.Error as e:
            logger.error(f"Error deleting allocations for user {user_id}: {e}")
            raise

    # ========================================================================
    # SPECIALIZED OPERATIONS (Business Logic Support)
    # ========================================================================

    def pause_deployment(
        self, model_name: str, api_endpoint: str, pause_reason: str = "", pause_duration_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Pause a deployment by creating a PAUSED allocation
        Similar to MongoDB's pause_llm_deployment method

        Args:
            model_name: Model name
            api_endpoint: API endpoint to pause
            pause_reason: Reason for pausing
            pause_duration_minutes: Duration to pause for

        Returns:
            Dictionary with pause details

        Raises:
            ValueError: If model or deployment not found
            psycopg2.Error: On database errors
        """
        try:
            # Find the model configuration for this deployment
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT *
                        FROM llm_models
                        WHERE model_name = %s AND api_base = %s AND is_active = TRUE
                    """
                    cur.execute(query, (model_name, api_endpoint))
                    chosen_model_config = cur.fetchone()

                    if not chosen_model_config:
                        logger.warning(f"Deployment not found: {model_name} at {api_endpoint}")
                        return {
                            "alloc_status": "NOT_FOUND",
                            "model_name": model_name,
                            "api_base": api_endpoint,
                            "reason": "Deployment not found",
                        }

            # Get the max token limit for this model
            max_token_limit = chosen_model_config.get("max_tokens", 100000)
            region = chosen_model_config.get("region", "unknown")
            deployment_name = chosen_model_config.get("deployment_name", "")

            # Create a token request ID for the pause allocation
            token_request_id = f"pause_{uuid.uuid4().hex}"

            # Create the pause allocation
            self.create_pause_allocation(
                token_request_id=token_request_id,
                model_name=model_name,
                api_endpoint=api_endpoint,
                region=region,
                max_token_limit=max_token_limit,
                pause_duration_minutes=pause_duration_minutes,
                cloud_provider="azure" if "azure" in api_endpoint.lower() else "openai",
                deployment_name=deployment_name,
                reason=pause_reason,
            )

            logger.info(f"Paused deployment {model_name} at {api_endpoint} for {pause_duration_minutes} minutes")

            return {
                "alloc_status": "PAUSED",
                "model_name": model_name,
                "api_base": api_endpoint,
                "reason": pause_reason,
            }

        except ValueError as e:
            logger.error(f"Value error in pause_deployment: {e}")
            raise
        except psycopg2.Error as e:
            logger.error(f"Database error in pause_deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in pause_deployment: {e}")
            raise ValueError(f"Failed to pause deployment: {e}")

    def create_pause_allocation(
        self,
        token_request_id: str,
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
            psycopg2.Error: On database errors
        """
        if max_token_limit <= 0:
            raise ValueError(f"Token limit must be positive, got {max_token_limit}")
        if pause_duration_minutes <= 0:
            raise ValueError(f"Pause duration must be positive, got {pause_duration_minutes}")

        context = {"reason": reason, "operation": "pause_deployment"} if reason else {"operation": "pause_deployment"}

        logger.info(f"Creating pause allocation for {model_name} at {api_endpoint} for {pause_duration_minutes}m")

        return self.create_token_allocation(
            token_request_id=token_request_id,
            user_id=UUID("00000000-0000-0000-0000-000000000000"),  # System user for pauses
            model_name=model_name,
            token_count=max_token_limit,
            allocation_status="PAUSED",
            api_endpoint=api_endpoint,
            region=region,
            cloud_provider=cloud_provider,
            deployment_name=deployment_name,
            allocated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=pause_duration_minutes),
            request_context=context,
        )

    def get_allocation_summary_by_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of allocations for a model

        Args:
            model_name: Model name to summarize

        Returns:
            Dictionary with counts and totals by status

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT
                            allocation_status,
                            COUNT(*) as count,
                            SUM(token_count) as total_tokens,
                            AVG(token_count) as avg_tokens
                        FROM token_allocations
                        WHERE
                            model_name = %s
                            AND (expires_at IS NULL OR expires_at > NOW())
                        GROUP BY allocation_status
                    """
                    cur.execute(query, (model_name,))
                    results = cur.fetchall()

                    summary = {"model_name": model_name, "by_status": [dict(row) for row in results]}

                    logger.debug(f"Generated summary for model {model_name}: {len(results)} statuses")
                    return summary
        except psycopg2.Error as e:
            logger.error(f"Error generating summary for model {model_name}: {e}")
            raise

    def get_user_token_usage_stats(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get token usage statistics for a user

        Args:
            user_id: User UUID

        Returns:
            Dictionary with usage statistics (empty dict if no data)

        Raises:
            psycopg2.Error: On database errors
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT
                            COUNT(*) as total_requests,
                            SUM(token_count) as total_tokens,
                            AVG(token_count) as avg_tokens_per_request,
                            AVG(latency_ms) as avg_latency_ms,
                            COUNT(CASE WHEN allocation_status = 'RELEASED' THEN 1 END) as completed_requests,
                            COUNT(CASE WHEN allocation_status = 'FAILED' THEN 1 END) as failed_requests
                        FROM token_allocations
                        WHERE user_id = %s
                    """
                    cur.execute(query, (user_id,))
                    result = cur.fetchone()

                    stats = dict(result) if result else {}
                    logger.debug(f"Generated usage stats for user {user_id}")
                    return stats
        except psycopg2.Error as e:
            logger.error(f"Error getting usage stats for user {user_id}: {e}")
            raise

    # ========================================================================
    # ALLOCATION LOGIC OPERATIONS (Core Business Logic)
    # ========================================================================

    def retry_acquire_tokens(self, token_request_id: str) -> Optional[Dict[str, Any]]:
        """
        Retry acquiring tokens for a waiting request
        Similar to MongoDB's retry_acquire method

        Args:
            token_request_id: Token request ID

        Returns:
            Updated allocation or None if not possible

        Raises:
            ValueError: If token request not found or invalid
            psycopg2.Error: On database errors
        """
        try:
            # Get the allocation record
            allocation = self.get_allocation_by_request_id(token_request_id)
            if not allocation:
                logger.warning(f"Token request not found: {token_request_id}")
                return {"error": f"Invalid token_req_id = {token_request_id}"}

            # Check if it's in WAITING status
            if allocation["allocation_status"] != "WAITING":
                logger.warning(f"Token request {token_request_id} is not in WAITING status")
                return {"error": f"Token request {token_request_id} is not in WAITING status"}

            # Get model name and token count
            model_name = allocation["model_name"]
            token_count = allocation["token_count"]

            # Get least loaded deployment
            total_allocated_tokens, chosen_model_config = self.get_least_loaded_deployment(model_name)
            max_token_limit = chosen_model_config.get("max_tokens", 100000)
            max_token_lock_time_secs = chosen_model_config.get("max_token_lock_time_secs", 70)

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
            api_endpoint = chosen_model_config.get("api_base", "")
            region = chosen_model_config.get("region", "")

            updated_allocation = self.transition_waiting_to_acquired(
                token_request_id=token_request_id, api_endpoint=api_endpoint, region=region, expires_at=expires_at
            )

            if updated_allocation:
                # Add additional fields for response
                updated_allocation["api_version"] = chosen_model_config.get("api_version", "")
                updated_allocation["api_keyv_id"] = chosen_model_config.get("api_keyv_id", "")
                updated_allocation["temperature"] = chosen_model_config.get("temperature", 0.0)
                updated_allocation["seed"] = chosen_model_config.get("seed", 42)
                return updated_allocation
            else:
                return {"error": f"Failed to acquire tokens for request {token_request_id}"}

        except ValueError as e:
            logger.error(f"Value error in retry_acquire_tokens: {e}")
            raise
        except psycopg2.Error as e:
            logger.error(f"Database error in retry_acquire_tokens: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in retry_acquire_tokens: {e}")
            raise ValueError(f"Failed to retry acquire tokens: {e}")

    def acquire_tokens(
        self, user_id: UUID, model_name: str, token_count: int, request_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Acquire tokens for a model
        Similar to MongoDB's acquire method

        Args:
            user_id: User requesting tokens
            model_name: Model name
            token_count: Number of tokens to allocate
            request_context: Optional request context

        Returns:
            Dictionary with allocation details

        Raises:
            ValueError: If token count exceeds limit or no deployments found
            psycopg2.Error: On database errors
        """
        if token_count <= 0:
            raise ValueError(f"Token count must be positive, got {token_count}")

        try:
            # Get least loaded deployment
            total_allocated_tokens, chosen_model_config = self.get_least_loaded_deployment(model_name)

            # Extract max token limit and lock time
            max_token_limit = chosen_model_config.get("max_tokens")
            if not max_token_limit:
                max_token_limit = 100000  # Default if not specified

            max_token_lock_time_secs = chosen_model_config.get("max_token_lock_time_secs", 70)

            # Check if token count exceeds limit
            if token_count > max_token_limit:
                logger.warning(f"Token count {token_count} exceeds limit {max_token_limit} for model {model_name}")
                return {
                    "error": f"Invalid token count, max limit exceeded for model {model_name} is {max_token_limit} for region {chosen_model_config.get('region', 'unknown')}"
                }

            logger.info(f"Total allocated tokens for {model_name}: {total_allocated_tokens}")

            # Create token request ID
            token_request_id = f"req_{uuid.uuid4().hex}"
            now = datetime.now()
            expires_at = now + timedelta(seconds=max_token_lock_time_secs)

            # Initialize with WAITING status
            allocation_status = "WAITING"
            api_version = ""
            deployment_name = ""
            api_endpoint = ""
            region = ""
            api_keyv_id = ""
            temperature = 0.0
            seed = 0

            # Check if we can allocate immediately
            if total_allocated_tokens + token_count <= max_token_limit:
                # Immediate allocation (ACQUIRED)
                allocation_status = "ACQUIRED"
                api_version = chosen_model_config.get("api_version", "")
                deployment_name = chosen_model_config.get("deployment_name", "")
                api_endpoint = chosen_model_config.get("api_base", "")
                region = chosen_model_config.get("region", "")
                api_keyv_id = chosen_model_config.get("api_keyv_id", "")
                temperature = chosen_model_config.get("temperature", 0.0)
                seed = chosen_model_config.get("seed", 42)

            # Create the allocation record
            allocation = self.create_token_allocation(
                token_request_id=token_request_id,
                user_id=user_id,
                model_name=model_name,
                token_count=token_count,
                allocation_status=allocation_status,
                allocated_at=now,
                expires_at=expires_at,
                model_id=chosen_model_config.get("model_id"),
                deployment_name=deployment_name,
                cloud_provider="azure" if "azure" in api_endpoint.lower() else "openai",
                api_endpoint=api_endpoint,
                region=region,
                request_context=request_context,
            )

            # Add additional fields for response
            allocation["api_version"] = api_version
            allocation["api_keyv_id"] = api_keyv_id
            allocation["temperature"] = temperature
            allocation["seed"] = seed

            return allocation

        except ValueError as e:
            logger.error(f"Value error in acquire_tokens: {e}")
            raise
        except psycopg2.Error as e:
            logger.error(f"Database error in acquire_tokens: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in acquire_tokens: {e}")
            raise ValueError(f"Failed to acquire tokens: {e}")

    def get_least_loaded_deployment(self, model_name: str) -> Tuple[int, Dict[str, Any]]:
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
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First, get all active deployments for this model
                    deployments_query = """
                        SELECT *
                        FROM llm_models
                        WHERE model_name = %s AND is_active = TRUE
                    """
                    cur.execute(deployments_query, (model_name,))
                    model_deployments = cur.fetchall()

                    if not model_deployments:
                        raise ValueError(f"No model deployments found for model_name = {model_name}")

                    # Get current allocations grouped by model_name and api_base
                    allocations_query = """
                        SELECT
                            model_name,
                            api_endpoint,
                            SUM(token_count) as total_tokens
                        FROM token_allocations
                        WHERE
                            model_name = %s
                            AND allocation_status IN ('ACQUIRED', 'PAUSED')
                            AND (expires_at IS NULL OR expires_at > NOW())
                        GROUP BY model_name, api_endpoint
                        ORDER BY total_tokens ASC
                    """
                    cur.execute(allocations_query, (model_name,))
                    allocation_results = cur.fetchall()

                    chosen_model_config = None

                    # Check if any deployment's api_base is not in the allocation results
                    # This means it's unused and can be chosen immediately
                    if allocation_results:
                        used_endpoints = [r["api_endpoint"] for r in allocation_results]
                        unused_deployments = [
                            m
                            for m in model_deployments
                            if m["api_base"] not in used_endpoints and m["api_base"] is not None
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
                        if deployment["api_base"] == least_loaded["api_endpoint"]:
                            chosen_model_config = dict(deployment)
                            break

                    # If no match found (shouldn't happen), use the first deployment
                    if not chosen_model_config:
                        chosen_model_config = dict(model_deployments[0])
                        logger.warning(f"No matching deployment found for endpoint {least_loaded['api_endpoint']}")

                    return total_allocated_tokens, chosen_model_config

        except psycopg2.Error as e:
            logger.error(f"Database error in get_least_loaded_deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Error finding least loaded deployment for {model_name}: {e}")
            raise ValueError(f"Failed to get deployment for model {model_name}: {e}")


# ============================================================================
# CONVENIENCE FUNCTION FOR REPOSITORY INITIALIZATION
# ============================================================================


def get_token_allocation_repository(db_manager: Optional[DatabaseManager] = None) -> TokenAllocationService:
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
