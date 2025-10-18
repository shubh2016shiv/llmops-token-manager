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

from sqlalchemy import text
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

    # Temperature constraints for LLM models
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0
    VALID_LLM_PROVIDER_NAMES = [
        "openai",
        "gemini",
        "anthropic",
        "mistral",
        "cohere",
        "xai",
        "deepseek",
        "meta",
    ]

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the LLM models service with database manager.

        Args:
            database_manager: Optional DatabaseManager instance (uses singleton if not provided)
        """
        super().__init__(database_manager)

    def validate_llm_provider_name(self, provider_name: str) -> None:
        """
        Validate that an LLM provider name is one of the supported providers.
        """
        self.validate_enum_value(
            provider_name, self.VALID_LLM_PROVIDER_NAMES, "LLM provider name"
        )

    def validate_llm_model_numerical_parameters(
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
        llm_model_name: str,
        api_key_variable_name: str,
        llm_model_version: str,
        max_tokens: int,
        tokens_per_minute_limit: int,
        requests_per_minute_limit: int,
        deployment_name: Optional[str] = None,
        api_endpoint_url: Optional[str] = None,
        is_active_status: bool = True,
        temperature: Optional[float] = None,
        random_seed: Optional[int] = None,
        deployment_region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new LLM model configuration in the database.

        Args:
            provider_name: LLM provider (openai, gemini, anthropic)
            llm_model_name: Name of the model (required)
            deployment_name: Optional deployment identifier
            api_key_variable_name: Optional API key vault reference
            api_endpoint_url: Optional API endpoint URL
            llm_model_version: Optional model version string
            max_tokens: Optional maximum tokens per request
            tokens_per_minute_limit: Optional token rate limit per minute
            requests_per_minute_limit: Optional request rate limit per minute
            is_active_status: Whether model is active (default: True)
            temperature: Optional temperature setting (0.0 to 2.0)
            random_seed: Optional seed for reproducibility
            deployment_region: Optional geographic region identifier

        Returns:
            Dictionary containing the created model record with all fields

        Raises:
            sqlalchemy.exc.IntegrityError: If model configuration already exists
            sqlalchemy.exc.SQLAlchemyError: On other database errors
            ValueError: On invalid input parameters
        """

        self.validate_string_not_empty(llm_model_name, "model_name")
        self.validate_llm_model_numerical_parameters(
            maximum_tokens=max_tokens,
            tokens_per_minute_limit=tokens_per_minute_limit,
            requests_per_minute_limit=requests_per_minute_limit,
            temperature_value=temperature,
            random_seed=random_seed,
        )

        try:
            async with self.get_session() as session:
                sql_query = """
                    INSERT INTO llm_models (
                        provider_name, llm_model_name, deployment_name, api_key_variable_name,
                        api_endpoint_url, llm_model_version, max_tokens, tokens_per_minute_limit,
                        requests_per_minute_limit, is_active_status, temperature, random_seed, deployment_region
                    ) VALUES (
                        :provider_name, :llm_model_name, :deployment_name, :api_key_variable_name,
                        :api_endpoint_url, :llm_model_version, :max_tokens, :tokens_per_minute_limit,
                        :requests_per_minute_limit, :is_active_status, :temperature, :random_seed, :deployment_region
                    )
                    RETURNING *
                """

                params = {
                    "provider_name": provider_name,
                    "llm_model_name": llm_model_name,
                    "deployment_name": deployment_name,
                    "api_key_variable_name": api_key_variable_name,
                    "api_endpoint_url": api_endpoint_url,
                    "llm_model_version": llm_model_version,
                    "max_tokens": max_tokens,
                    "tokens_per_minute_limit": tokens_per_minute_limit,
                    "requests_per_minute_limit": requests_per_minute_limit,
                    "is_active_status": is_active_status,
                    "temperature": temperature,
                    "random_seed": random_seed,
                    "deployment_region": deployment_region,
                }

                result = await session.execute(text(sql_query), params)
                created_model = result.mappings().one_or_none()

                if not created_model:
                    raise RuntimeError("Failed to create model record")

                self.log_operation(
                    "CREATE", f"{provider_name}/{llm_model_name}", success=True
                )
                return dict(created_model)
        except Exception as e:
            logger.error(f"Error creating model {llm_model_name}: {e}")
            raise

    # ========================================================================
    # READ OPERATIONS
    # ========================================================================

    async def get_llm_model_by_provider_and_model(
        self,
        provider_name: str,
        llm_model_name: str,
        llm_model_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an LLM model by its provider, model name, and optional version.

        This is the primary method for fetching a specific model configuration using the composite key.

        Args:
            provider_name: Provider name (e.g., 'openai', 'anthropic').
            llm_model_name: Model name (e.g., 'gpt-4o').
            llm_model_version: Optional model version (e.g., '2024-08'), or None if not versioned.

        Returns:
            Dictionary containing the model record or None if not found.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name or llm_model_name is invalid or empty.
        """
        self.validate_llm_provider_name(provider_name)
        self.validate_string_not_empty(llm_model_name, "llm_model_name")

        try:
            async with self.get_session() as session:
                where_conditions = "provider_name = :provider_name AND llm_model_name = :llm_model_name"
                params = {
                    "provider_name": provider_name,
                    "llm_model_name": llm_model_name,
                }
                if llm_model_version is not None:
                    where_conditions += " AND llm_model_version = :llm_model_version"
                    params["llm_model_version"] = llm_model_version
                else:
                    where_conditions += " AND llm_model_version IS NULL"

                sql_query = f"""
                    SELECT * FROM llm_models
                    WHERE {where_conditions}
                """
                result = await session.execute(text(sql_query), params)
                model_record = result.mappings().one_or_none()
                return dict(model_record) if model_record else None
        except Exception as e:
            logger.error(
                f"Error fetching model ({provider_name}, {llm_model_name}, {llm_model_version}): {e}"
            )
            raise

    async def get_llm_models_by_provider(
        self,
        provider_name: str,
        active_only: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all LLM models for a specific provider with optional active status filter and pagination.

        This is the core method for listing models within a provider, optimized for business-critical
        discovery and management workflows.

        Args:
            provider_name: Provider name to filter by (e.g., 'openai', 'anthropic').
            active_only: Optional filter for active models only (True/False); defaults to all.
            limit: Maximum number of records to return (default: 100, max: 1000).
            offset: Number of records to skip for pagination (default: 0).

        Returns:
            List of model records ordered by creation date (newest first).

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: On invalid provider_name or pagination parameters.
        """
        self.validate_llm_provider_name(provider_name)
        self.validate_pagination_parameters(limit, offset)

        try:
            async with self.get_session() as session:
                sql_query = (
                    "SELECT * FROM llm_models WHERE provider_name = :provider_name"
                )
                params: Dict[str, Any] = {"provider_name": provider_name}

                if active_only is not None:
                    sql_query += " AND is_active_status = :is_active_status"
                    params["is_active_status"] = str(active_only)

                sql_query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
                params["limit"] = str(limit)
                params["offset"] = str(offset)

                result = await session.execute(text(sql_query), params)
                model_records = result.mappings().all()
                logger.debug(
                    f"Retrieved {len(model_records)} models for provider {provider_name}"
                )
                return [dict(row) for row in model_records]
        except Exception as e:
            logger.error(f"Error fetching models for provider {provider_name}: {e}")
            raise

    async def get_active_llm_models_by_provider(
        self,
        provider_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all active LLM models for a specific provider.

        Convenience wrapper for business-critical active model discovery per provider.

        Args:
            provider_name: Provider name to filter by (e.g., 'openai', 'anthropic').
            limit: Maximum number of records to return (default: 100).
            offset: Number of records to skip for pagination (default: 0).

        Returns:
            List of active model records ordered by creation date (newest first).

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: On invalid provider_name or pagination parameters.
        """
        return await self.get_llm_models_by_provider(
            provider_name=provider_name,
            active_only=True,
            limit=limit,
            offset=offset,
        )

    async def count_llm_models_by_provider(self, provider_name: str) -> int:
        """
        Count the number of LLM models for a specific provider.

        Useful for dashboard metrics and capacity planning.

        Args:
            provider_name: Provider name (e.g., 'openai', 'anthropic').

        Returns:
            Number of models for the specified provider.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name is invalid.
        """
        self.validate_llm_provider_name(provider_name)

        try:
            async with self.get_session() as session:
                sql_query = """
                    SELECT COUNT(*) FROM llm_models
                    WHERE provider_name = :provider_name
                """
                result = await session.execute(
                    text(sql_query), {"provider_name": provider_name}
                )
                return result.scalar_one_or_none() or 0
        except Exception as e:
            logger.error(f"Error counting models for provider {provider_name}: {e}")
            raise

    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================

    async def update_llm_model(
        self,
        provider_name: str,
        llm_model_name: str,
        llm_model_version: Optional[str] = None,
        new_provider_name: Optional[str] = None,
        new_llm_model_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_key_variable_name: Optional[str] = None,
        api_endpoint_url: Optional[str] = None,
        new_llm_model_version: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tokens_per_minute_limit: Optional[int] = None,
        requests_per_minute_limit: Optional[int] = None,
        is_active_status: Optional[bool] = None,
        temperature: Optional[float] = None,
        random_seed: Optional[int] = None,
        deployment_region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update an LLM model configuration with dynamic field updates.

        Only provided fields will be updated, optimizing database writes. The model is identified by
        its composite key (provider_name, llm_model_name, llm_model_version). Updates to the composite
        key fields (new_provider_name, new_llm_model_name, new_llm_model_version) will change the
        model's identity in the table.

        Args:
            provider_name: Current provider name identifying the model (e.g., 'openai').
            llm_model_name: Current model name identifying the model (e.g., 'gpt-4o').
            llm_model_version: Current model version identifying the model (e.g., '2024-08'), or None if not versioned.
            new_provider_name: Optional new provider name (e.g., 'openai', 'anthropic').
            new_llm_model_name: Optional new model name (e.g., 'gpt-4o', 'claude-3.5-sonnet').
            deployment_name: Optional new deployment name (e.g., 'gpt-4o').
            api_key_variable_name: Optional new API key variable name (e.g., 'OPENAI_API_KEY_GPT4O').
            api_endpoint_url: Optional new API endpoint URL.
            new_llm_model_version: Optional new model version (e.g., '2024-10').
            max_tokens: Optional new maximum tokens per request.
            tokens_per_minute_limit: Optional new token rate limit per minute.
            requests_per_minute_limit: Optional new request rate limit per minute.
            is_active_status: Optional new active status (True/False).
            temperature: Optional new temperature value (0.0 to 2.0).
            random_seed: Optional new random seed for reproducibility.
            deployment_region: Optional new deployment region (e.g., 'eastus2').

        Returns:
            Updated model record as a dictionary, or None if the model is not found.

        Raises:
            sqlalchemy.exc.IntegrityError: If the update violates constraints (e.g., duplicate composite key).
            sqlalchemy.exc.SQLAlchemyError: On other database errors.
            ValueError: On invalid input parameters (e.g., invalid provider name).
        """
        # Validate composite key inputs
        if not provider_name or not llm_model_name:
            raise ValueError(
                "provider_name and llm_model_name must be provided to identify the model"
            )

        # Validate new provider name if provided
        if new_provider_name:
            self.validate_llm_provider_name(new_provider_name)

        # Build update fields dictionary
        update_fields_dict: Dict[str, Any] = {}
        if new_provider_name is not None:
            update_fields_dict["provider_name"] = new_provider_name
        if new_llm_model_name is not None:
            update_fields_dict["llm_model_name"] = new_llm_model_name
        if deployment_name is not None:
            update_fields_dict["deployment_name"] = deployment_name
        if api_key_variable_name is not None:
            update_fields_dict["api_key_variable_name"] = api_key_variable_name
        if api_endpoint_url is not None:
            update_fields_dict["api_endpoint_url"] = api_endpoint_url
        if new_llm_model_version is not None:
            update_fields_dict["llm_model_version"] = new_llm_model_version
        if max_tokens is not None:
            update_fields_dict["max_tokens"] = max_tokens
        if tokens_per_minute_limit is not None:
            update_fields_dict["tokens_per_minute_limit"] = tokens_per_minute_limit
        if requests_per_minute_limit is not None:
            update_fields_dict["requests_per_minute_limit"] = requests_per_minute_limit
        if is_active_status is not None:
            update_fields_dict["is_active_status"] = is_active_status
        if temperature is not None:
            update_fields_dict["temperature"] = temperature
        if random_seed is not None:
            update_fields_dict["random_seed"] = random_seed
        if deployment_region is not None:
            update_fields_dict["deployment_region"] = deployment_region

        # If no fields to update, return current model
        if not update_fields_dict:
            logger.warning(
                f"No fields to update for model ({provider_name}, {llm_model_name}, {llm_model_version})"
            )
            return await self.get_llm_model_by_provider_and_model(
                provider_name, llm_model_name, llm_model_version
            )

        try:
            # Build where clause for composite key
            where_clause = (
                "provider_name = :provider_name AND llm_model_name = :llm_model_name"
            )
            where_parameters = {
                "provider_name": provider_name,
                "llm_model_name": llm_model_name,
            }
            if llm_model_version is not None:
                where_clause += " AND llm_model_version = :llm_model_version"
                where_parameters["llm_model_version"] = llm_model_version
            else:
                where_clause += " AND llm_model_version IS NULL"

            sql_query, query_parameters = self.build_dynamic_update_query(
                table_name="llm_models",
                update_fields=update_fields_dict,
                where_clause=where_clause,
                where_parameters=where_parameters,
            )

            async with self.get_session() as session:
                result = await session.execute(text(sql_query), query_parameters)
                updated_model = result.mappings().one_or_none()

                if updated_model:
                    self.log_operation(
                        "UPDATE",
                        f"({provider_name}, {llm_model_name}, {llm_model_version})",
                        success=True,
                    )
                    return dict(updated_model)

                logger.warning(
                    f"Model ({provider_name}, {llm_model_name}, {llm_model_version}) not found for update"
                )
                return None
        except Exception as e:
            logger.error(
                f"Error updating model ({provider_name}, {llm_model_name}, {llm_model_version}): {e}"
            )
            raise

    async def update_llm_model_status(
        self,
        provider_name: str,
        llm_model_name: str,
        llm_model_version: Optional[str] = None,
        is_active_status: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Update a model's active status.

        Args:
            provider_name: Current provider name identifying the model (e.g., 'openai').
            llm_model_name: Current model name identifying the model (e.g., 'gpt-4o').
            llm_model_version: Current model version identifying the model (e.g., '2024-08'), or None if not versioned.
            is_active_status: New active status (True for active, False for inactive).

        Returns:
            Updated model record as a dictionary, or None if the model is not found.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name or llm_model_name is empty.
        """
        if not provider_name or not llm_model_name:
            raise ValueError(
                "provider_name and llm_model_name must be provided to identify the model"
            )

        return await self.update_llm_model(
            provider_name=provider_name,
            llm_model_name=llm_model_name,
            llm_model_version=llm_model_version,
            is_active_status=is_active_status,
        )

    async def activate_llm_model(
        self,
        provider_name: str,
        llm_model_name: str,
        llm_model_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Activate a model by setting its active status to True.

        Args:
            provider_name: Current provider name identifying the model (e.g., 'openai').
            llm_model_name: Current model name identifying the model (e.g., 'gpt-4o').
            llm_model_version: Current model version identifying the model (e.g., '2024-08'), or None if not versioned.

        Returns:
            Updated model record as a dictionary, or None if the model is not found.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name or llm_model_name is empty.
        """
        return await self.update_llm_model_status(
            provider_name=provider_name,
            llm_model_name=llm_model_name,
            llm_model_version=llm_model_version,
            is_active_status=True,
        )

    async def deactivate_llm_model(
        self,
        provider_name: str,
        llm_model_name: str,
        llm_model_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Deactivate a model by setting its active status to False.

        Args:
            provider_name: Current provider name identifying the model (e.g., 'openai').
            llm_model_name: Current model name identifying the model (e.g., 'gpt-4o').
            llm_model_version: Current model version identifying the model (e.g., '2024-08'), or None if not versioned.

        Returns:
            Updated model record as a dictionary, or None if the model is not found.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name or llm_model_name is empty.
        """
        return await self.update_llm_model_status(
            provider_name=provider_name,
            llm_model_name=llm_model_name,
            llm_model_version=llm_model_version,
            is_active_status=False,
        )

    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================

    async def delete_llm_model(
        self,
        provider_name: str,
        llm_model_name: str,
        llm_model_version: Optional[str] = None,
    ) -> bool:
        """
        Delete an LLM model configuration from the database.

        Note: This operation may cascade to related records depending on
        database foreign key constraints.

        Args:
            provider_name: Provider name identifying the model (e.g., 'openai').
            llm_model_name: Model name identifying the model (e.g., 'gpt-4o').
            llm_model_version: Model version identifying the model (e.g., '2024-08'), or None if not versioned.

        Returns:
            True if the model was deleted, False if the model was not found.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name or llm_model_name is empty.
        """
        if not provider_name or not llm_model_name:
            raise ValueError(
                "provider_name and llm_model_name must be provided to identify the model"
            )

        if provider_name:
            self.validate_llm_provider_name(provider_name)

        try:
            async with self.get_session() as session:
                where_conditions = "provider_name = :provider_name AND llm_model_name = :llm_model_name"
                params = {
                    "provider_name": provider_name,
                    "llm_model_name": llm_model_name,
                }
                if llm_model_version is not None:
                    where_conditions += " AND llm_model_version = :llm_model_version"
                    params["llm_model_version"] = llm_model_version
                else:
                    where_conditions += " AND llm_model_version IS NULL"

                sql_query = f"""
                    DELETE FROM llm_models
                    WHERE {where_conditions}
                """
                result = await session.execute(text(sql_query), params)
                # was_deleted = result.rowcount > 0
                was_deleted = getattr(result, "rowcount", 0) > 0

                if was_deleted:
                    self.log_operation(
                        "DELETE",
                        f"({provider_name}, {llm_model_name}, {llm_model_version})",
                        success=True,
                    )
                else:
                    logger.debug(
                        f"Model not found for deletion: ({provider_name}, {llm_model_name}, {llm_model_version})"
                    )

                return bool(was_deleted)
        except Exception as e:
            logger.error(
                f"Error deleting model ({provider_name}, {llm_model_name}, {llm_model_version}): {e}"
            )
            raise

    async def delete_llm_models_by_provider(self, provider_name: str) -> int:
        """
        Delete all models for a specific LLM provider.

        Note: This is a bulk operation that may cascade to related records.

        Args:
            provider_name: Provider name (e.g., 'openai', 'anthropic').

        Returns:
            Number of deleted model records.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
            ValueError: If provider_name is invalid.
        """
        self.validate_llm_provider_name(provider_name)

        try:
            async with self.get_session() as session:
                sql_query = """
                    DELETE FROM llm_models
                    WHERE provider_name = :provider_name
                """
                result = await session.execute(
                    text(sql_query), {"provider_name": provider_name}
                )
                deleted_count = getattr(result, "rowcount", 0)

                if deleted_count > 0:
                    self.log_operation(
                        "DELETE_BULK",
                        provider_name,
                        success=True,
                        additional_context=f"{deleted_count} models deleted",
                    )
                else:
                    logger.debug(f"No models found for provider {provider_name}")

                return int(deleted_count)
        except Exception as e:
            logger.error(f"Error deleting models for provider {provider_name}: {e}")
            raise
