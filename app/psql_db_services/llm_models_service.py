"""
PostgreSQL CRUD Operations for LLM Models Management
----------------------------------------------------
Production-ready database service for LLM model configuration and tracking including:
- Model creation, retrieval, updates, and deletion
- Provider management (OpenAI, Gemini, Anthropic)
- Rate limiting configuration
- Usage statistics tracking
- Optimized for high-concurrency environments (10,000+ concurrent users)
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from loguru import logger

from app.core.database_connection import DatabaseManager
from app.psql_db_services.base_service import BaseDatabaseService


class LLMModelsService(BaseDatabaseService):
    """
    Production-ready service for LLM model database operations.

    Inherits from BaseDatabaseService for optimized connection pooling,
    transaction management, and error handling.

    Supports:
    - CRUD operations for LLM model configurations
    - Multi-provider support (OpenAI, Gemini, Anthropic)
    - Rate limiting and quota management
    - Usage tracking and statistics
    - Thread-safe operations for high-concurrency scenarios
    """

    # Define valid enum values as class constants
    VALID_LLM_PROVIDERS = ["openai", "gemini", "anthropic"]

    # Temperature constraints for LLM models
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the LLM models service with database manager.

        Args:
            database_manager: Optional DatabaseManager instance (uses singleton if not provided)
        """
        super().__init__(database_manager)

    def validate_llm_provider(self, provider_name: str) -> None:
        """
        Validate that an LLM provider is one of the supported providers.

        Args:
            provider_name: Provider name to validate

        Raises:
            ValueError: If provider is not in the list of valid providers
        """
        self.validate_enum_value(
            provider_name, self.VALID_LLM_PROVIDERS, "LLM provider"
        )

    def validate_model_numerical_parameters(
        self,
        maximum_tokens: Optional[int] = None,
        tokens_per_minute_limit: Optional[int] = None,
        requests_per_minute_limit: Optional[int] = None,
        temperature_value: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Validate numerical parameters for LLM model configuration.

        Args:
            maximum_tokens: Optional maximum tokens per request
            tokens_per_minute_limit: Optional token rate limit per minute
            requests_per_minute_limit: Optional request rate limit per minute
            temperature_value: Optional temperature setting (0.0 to 2.0)
            random_seed: Optional random seed for reproducibility

        Raises:
            ValueError: On invalid numerical values
        """
        if maximum_tokens is not None:
            self.validate_positive_integer(maximum_tokens, "maximum_tokens")

        if tokens_per_minute_limit is not None:
            self.validate_positive_integer(
                tokens_per_minute_limit, "tokens_per_minute_limit"
            )

        if requests_per_minute_limit is not None:
            self.validate_positive_integer(
                requests_per_minute_limit, "requests_per_minute_limit"
            )

        if temperature_value is not None:
            if not isinstance(temperature_value, (int, float)):
                raise ValueError("temperature_value must be a number")
            if not (self.MIN_TEMPERATURE <= temperature_value <= self.MAX_TEMPERATURE):
                raise ValueError(
                    f"temperature_value must be between {self.MIN_TEMPERATURE} and {self.MAX_TEMPERATURE}, "
                    f"got {temperature_value}"
                )

        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer")

    # ========================================================================
    # CREATE OPERATIONS
    # ========================================================================

    async def create_llm_model(
        self,
        provider_name: str,
        model_name: str,
        deployment_name: Optional[str] = None,
        api_key_vault_identifier: Optional[str] = None,
        api_endpoint_url: Optional[str] = None,
        model_version: Optional[str] = None,
        maximum_tokens: Optional[int] = None,
        tokens_per_minute_limit: Optional[int] = None,
        requests_per_minute_limit: Optional[int] = None,
        is_active_status: bool = True,
        temperature_value: Optional[float] = None,
        random_seed: Optional[int] = None,
        geographic_region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new LLM model configuration in the database.

        Args:
            provider_name: LLM provider (openai, gemini, anthropic)
            model_name: Name of the model (required)
            deployment_name: Optional deployment identifier
            api_key_vault_identifier: Optional API key vault reference
            api_endpoint_url: Optional API endpoint URL
            model_version: Optional model version string
            maximum_tokens: Optional maximum tokens per request
            tokens_per_minute_limit: Optional token rate limit per minute
            requests_per_minute_limit: Optional request rate limit per minute
            is_active_status: Whether model is active (default: True)
            temperature_value: Optional temperature setting (0.0 to 2.0)
            random_seed: Optional seed for reproducibility
            geographic_region: Optional geographic region identifier

        Returns:
            Dictionary containing the created model record with all fields

        Raises:
            psycopg.IntegrityError: If model configuration already exists
            psycopg.Error: On other database errors
            ValueError: On invalid input parameters
        """
        self.validate_llm_provider(provider_name)
        self.validate_string_not_empty(model_name, "model_name")
        self.validate_model_numerical_parameters(
            maximum_tokens=maximum_tokens,
            tokens_per_minute_limit=tokens_per_minute_limit,
            requests_per_minute_limit=requests_per_minute_limit,
            temperature_value=temperature_value,
            random_seed=random_seed,
        )

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        INSERT INTO llm_models (
                            provider, model_name, deployment_name, api_key_vault_id,
                            api_endpoint, model_version, max_tokens, tokens_per_minute_limit,
                            requests_per_minute_limit, is_active, temperature, seed, region
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING *
                    """

                    await cursor.execute(
                        sql_query,
                        (
                            provider_name,
                            model_name,
                            deployment_name,
                            api_key_vault_identifier,
                            api_endpoint_url,
                            model_version,
                            maximum_tokens,
                            tokens_per_minute_limit,
                            requests_per_minute_limit,
                            is_active_status,
                            temperature_value,
                            random_seed,
                            geographic_region,
                        ),
                    )

                    created_model = await cursor.fetchone()
                    if not created_model:
                        raise RuntimeError("Failed to create model record")

                    self.log_operation(
                        "CREATE", f"{provider_name}/{model_name}", success=True
                    )
                    return dict(created_model)
        except psycopg.IntegrityError as integrity_error:
            logger.error(
                f"Integrity error creating model {model_name}: {integrity_error}"
            )
            raise
        except psycopg.Error as database_error:
            logger.error(
                f"Database error creating model {model_name}: {database_error}"
            )
            raise

    # ========================================================================
    # READ OPERATIONS
    # ========================================================================

    async def get_llm_model_by_id(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve an LLM model by its unique identifier.

        Args:
            model_id: Model's unique UUID identifier

        Returns:
            Dictionary containing model record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If model_id is invalid
        """
        self.validate_uuid(model_id, "model_id")

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        SELECT * FROM llm_models
                        WHERE model_id = %s
                    """
                    await cursor.execute(sql_query, (model_id,))
                    model_record = await cursor.fetchone()
                    return dict(model_record) if model_record else None
        except psycopg.Error as database_error:
            logger.error(f"Error fetching model {model_id}: {database_error}")
            raise

    async def get_llm_model_by_name_and_endpoint(
        self, model_name: str, api_endpoint_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an LLM model by its name and API endpoint.

        Useful for finding specific model deployments.

        Args:
            model_name: Model name to search for
            api_endpoint_url: API endpoint URL

        Returns:
            Dictionary containing model record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If parameters are invalid
        """
        self.validate_string_not_empty(model_name, "model_name")
        self.validate_string_not_empty(api_endpoint_url, "api_endpoint_url")

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        SELECT * FROM llm_models
                        WHERE model_name = %s AND api_endpoint = %s
                    """
                    await cursor.execute(sql_query, (model_name, api_endpoint_url))
                    model_record = await cursor.fetchone()
                    return dict(model_record) if model_record else None
        except psycopg.Error as database_error:
            logger.error(
                f"Error fetching model {model_name} at {api_endpoint_url}: {database_error}"
            )
            raise

    async def get_all_llm_models(
        self,
        provider_filter: Optional[str] = None,
        active_status_filter: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all LLM models with optional filtering and pagination.

        Optimized for high-concurrency scenarios with indexed queries.

        Args:
            provider_filter: Optional provider to filter by (openai, gemini, anthropic)
            active_status_filter: Optional active status filter (True/False)
            limit: Maximum number of records to return (default: 100, max: 1000)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of model records ordered by creation date (newest first)

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid pagination or filter parameters
        """
        self.validate_pagination_parameters(limit, offset)

        if provider_filter:
            self.validate_llm_provider(provider_filter)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = "SELECT * FROM llm_models WHERE 1=1"
                    query_parameters = []

                    if provider_filter:
                        sql_query += " AND provider = %s"
                        query_parameters.append(provider_filter)

                    if active_status_filter is not None:
                        sql_query += " AND is_active = %s"
                        query_parameters.append(active_status_filter)

                    sql_query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                    query_parameters.extend([limit, offset])

                    await cursor.execute(sql_query, query_parameters)
                    model_records = await cursor.fetchall()
                    logger.debug(f"Retrieved {len(model_records)} models")
                    return [dict(row) for row in model_records]
        except psycopg.Error as database_error:
            logger.error(f"Error fetching models: {database_error}")
            raise

    async def get_active_llm_models(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all active LLM models.

        Args:
            limit: Maximum number of records to return (default: 100)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of active model records

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid pagination parameters
        """
        return await self.get_all_llm_models(
            active_status_filter=True, limit=limit, offset=offset
        )

    async def get_llm_models_by_provider(
        self, provider_name: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all models for a specific LLM provider.

        Args:
            provider_name: Provider name to filter by (openai, gemini, anthropic)
            limit: Maximum number of records to return (default: 100)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of model records for the specified provider

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid parameters
        """
        return await self.get_all_llm_models(
            provider_filter=provider_name, limit=limit, offset=offset
        )

    async def get_llm_models_by_name(
        self, model_name: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all deployments/configurations for a specific model name.

        Args:
            model_name: Model name to search for
            limit: Maximum number of records to return (default: 100)
            offset: Number of records to skip for pagination (default: 0)

        Returns:
            List of model records with the same name

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid parameters
        """
        self.validate_string_not_empty(model_name, "model_name")
        self.validate_pagination_parameters(limit, offset)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        SELECT * FROM llm_models
                        WHERE model_name = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    await cursor.execute(sql_query, (model_name, limit, offset))
                    model_records = await cursor.fetchall()
                    logger.debug(
                        f"Found {len(model_records)} configurations for model {model_name}"
                    )
                    return [dict(row) for row in model_records]
        except psycopg.Error as database_error:
            logger.error(
                f"Error fetching models by name {model_name}: {database_error}"
            )
            raise

    async def count_llm_models_by_provider(self, provider_name: str) -> int:
        """
        Count the number of models for a specific provider.

        Args:
            provider_name: Provider name (openai, gemini, anthropic)

        Returns:
            Number of models for the specified provider

        Raises:
            psycopg.Error: On database errors
            ValueError: If provider is invalid
        """
        self.validate_llm_provider(provider_name)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor() as cursor:
                    sql_query = """
                        SELECT COUNT(*) FROM llm_models
                        WHERE provider = %s
                    """
                    await cursor.execute(sql_query, (provider_name,))
                    count_result = await cursor.fetchone()
                    return count_result[0] if count_result else 0
        except psycopg.Error as database_error:
            logger.error(
                f"Error counting models for provider {provider_name}: {database_error}"
            )
            raise

    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================

    async def update_llm_model(
        self,
        model_id: UUID,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_key_vault_identifier: Optional[str] = None,
        api_endpoint_url: Optional[str] = None,
        model_version: Optional[str] = None,
        maximum_tokens: Optional[int] = None,
        tokens_per_minute_limit: Optional[int] = None,
        requests_per_minute_limit: Optional[int] = None,
        is_active_status: Optional[bool] = None,
        temperature_value: Optional[float] = None,
        random_seed: Optional[int] = None,
        geographic_region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update LLM model configuration with dynamic field updates.

        Only provided fields will be updated, optimizing database writes.

        Args:
            model_id: Model's unique UUID identifier
            provider_name: Optional new provider (openai, gemini, anthropic)
            model_name: Optional new model name
            deployment_name: Optional new deployment identifier
            api_key_vault_identifier: Optional new API key vault reference
            api_endpoint_url: Optional new API endpoint URL
            model_version: Optional new model version
            maximum_tokens: Optional new maximum tokens per request
            tokens_per_minute_limit: Optional new token rate limit
            requests_per_minute_limit: Optional new request rate limit
            is_active_status: Optional new active status
            temperature_value: Optional new temperature (0.0 to 2.0)
            random_seed: Optional new seed for reproducibility
            geographic_region: Optional new geographic region

        Returns:
            Updated model record dictionary or None if model not found

        Raises:
            psycopg.IntegrityError: If update violates constraints
            psycopg.Error: On other database errors
            ValueError: On invalid input parameters
        """
        self.validate_uuid(model_id, "model_id")

        if provider_name:
            self.validate_llm_provider(provider_name)
        self.validate_model_numerical_parameters(
            maximum_tokens=maximum_tokens,
            tokens_per_minute_limit=tokens_per_minute_limit,
            requests_per_minute_limit=requests_per_minute_limit,
            temperature_value=temperature_value,
            random_seed=random_seed,
        )

        # Build update fields dictionary
        update_fields_dict = {}
        if provider_name is not None:
            update_fields_dict["provider"] = provider_name
        if model_name is not None:
            update_fields_dict["model_name"] = model_name
        if deployment_name is not None:
            update_fields_dict["deployment_name"] = deployment_name
        if api_key_vault_identifier is not None:
            update_fields_dict["api_key_vault_id"] = api_key_vault_identifier
        if api_endpoint_url is not None:
            update_fields_dict["api_endpoint"] = api_endpoint_url
        if model_version is not None:
            update_fields_dict["model_version"] = model_version
        if maximum_tokens is not None:
            update_fields_dict["max_tokens"] = maximum_tokens
        if tokens_per_minute_limit is not None:
            update_fields_dict["tokens_per_minute_limit"] = tokens_per_minute_limit
        if requests_per_minute_limit is not None:
            update_fields_dict["requests_per_minute_limit"] = requests_per_minute_limit
        if is_active_status is not None:
            update_fields_dict["is_active"] = is_active_status
        if temperature_value is not None:
            update_fields_dict["temperature"] = temperature_value
        if random_seed is not None:
            update_fields_dict["seed"] = random_seed
        if geographic_region is not None:
            update_fields_dict["region"] = geographic_region

        if not update_fields_dict:
            logger.warning(f"No fields to update for model {model_id}")
            return await self.get_llm_model_by_id(model_id)

        try:
            sql_query, query_parameters = self.build_dynamic_update_query(
                table_name="llm_models",
                update_fields=update_fields_dict,
                where_clause="model_id = %s",
                where_parameters=(model_id,),
            )

            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(sql_query, query_parameters)
                    updated_model = await cursor.fetchone()

                    if updated_model:
                        self.log_operation("UPDATE", model_id, success=True)
                        return dict(updated_model)

                    logger.warning(f"Model {model_id} not found for update")
                    return None
        except psycopg.IntegrityError as integrity_error:
            logger.error(
                f"Integrity error updating model {model_id}: {integrity_error}"
            )
            raise
        except psycopg.Error as database_error:
            logger.error(f"Error updating model {model_id}: {database_error}")
            raise

    async def update_llm_model_status(
        self, model_id: UUID, is_active_status: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Update a model's active status.

        Args:
            model_id: Model's unique UUID identifier
            is_active_status: New active status (True/False)

        Returns:
            Updated model record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If model_id is invalid
        """
        return await self.update_llm_model(
            model_id=model_id, is_active_status=is_active_status
        )

    async def activate_llm_model(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Activate a model by setting is_active to True.

        Args:
            model_id: Model's unique UUID identifier

        Returns:
            Updated model record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If model_id is invalid
        """
        return await self.update_llm_model_status(
            model_id=model_id, is_active_status=True
        )

    async def deactivate_llm_model(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Deactivate a model by setting is_active to False.

        Args:
            model_id: Model's unique UUID identifier

        Returns:
            Updated model record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If model_id is invalid
        """
        return await self.update_llm_model_status(
            model_id=model_id, is_active_status=False
        )

    async def update_model_usage_statistics(
        self,
        model_id: UUID,
        request_count_increment: int = 1,
        token_count_increment: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Update model usage statistics atomically.

        Optimized for high-concurrency scenarios with atomic increments.

        Args:
            model_id: Model's unique UUID identifier
            request_count_increment: Number of requests to add (default: 1)
            token_count_increment: Number of tokens to add (default: 0)

        Returns:
            Updated model record or None if not found

        Raises:
            psycopg.Error: On database errors
            ValueError: On invalid increment values or model_id
        """
        self.validate_uuid(model_id, "model_id")
        self.validate_positive_integer(
            request_count_increment, "request_count_increment", allow_zero=True
        )
        self.validate_positive_integer(
            token_count_increment, "token_count_increment", allow_zero=True
        )

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor(row_factory=dict_row) as cursor:
                    sql_query = """
                        UPDATE llm_models
                        SET
                            total_requests = total_requests + %s,
                            total_tokens_processed = total_tokens_processed + %s,
                            last_used_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE model_id = %s
                        RETURNING *
                    """

                    await cursor.execute(
                        sql_query,
                        (request_count_increment, token_count_increment, model_id),
                    )
                    updated_model = await cursor.fetchone()

                    if updated_model:
                        logger.debug(f"Updated usage stats for model {model_id}")
                        return dict(updated_model)

                    logger.warning(f"Model {model_id} not found for usage update")
                    return None
        except psycopg.Error as database_error:
            logger.error(
                f"Error updating usage stats for model {model_id}: {database_error}"
            )
            raise

    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================

    async def delete_llm_model(self, model_id: UUID) -> bool:
        """
        Delete an LLM model configuration from the database.

        Note: This operation may cascade to related records depending on
        database foreign key constraints.

        Args:
            model_id: Model's unique UUID identifier

        Returns:
            True if model was deleted, False if model was not found

        Raises:
            psycopg.Error: On database errors
            ValueError: If model_id is invalid
        """
        self.validate_uuid(model_id, "model_id")

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor() as cursor:
                    sql_query = """
                        DELETE FROM llm_models
                        WHERE model_id = %s
                    """
                    await cursor.execute(sql_query, (model_id,))
                    was_deleted = cursor.rowcount > 0

                    if was_deleted:
                        self.log_operation("DELETE", model_id, success=True)
                    else:
                        logger.debug(f"Model not found for deletion: {model_id}")

                    return was_deleted
        except psycopg.Error as database_error:
            logger.error(f"Error deleting model {model_id}: {database_error}")
            raise

    async def delete_llm_models_by_provider(self, provider_name: str) -> int:
        """
        Delete all models for a specific LLM provider.

        Note: This is a bulk operation that may cascade to related records.

        Args:
            provider_name: Provider name (openai, gemini, anthropic)

        Returns:
            Number of deleted model records

        Raises:
            psycopg.Error: On database errors
            ValueError: If provider is invalid
        """
        self.validate_llm_provider(provider_name)

        try:
            async with self.get_database_connection() as database_connection:
                async with database_connection.cursor() as cursor:
                    sql_query = """
                        DELETE FROM llm_models
                        WHERE provider = %s
                    """
                    await cursor.execute(sql_query, (provider_name,))
                    deleted_count = cursor.rowcount

                    if deleted_count > 0:
                        self.log_operation(
                            "DELETE_BULK",
                            provider_name,
                            success=True,
                            additional_context=f"{deleted_count} models deleted",
                        )
                    else:
                        logger.debug(f"No models found for provider {provider_name}")

                    return deleted_count
        except psycopg.Error as database_error:
            logger.error(
                f"Error deleting models for provider {provider_name}: {database_error}"
            )
            raise
