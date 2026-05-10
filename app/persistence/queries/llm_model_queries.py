"""
LLM model persistence SQL definitions.

This module stores named SQL constants for static LLM model queries.
Dynamic query builders remain in the persistence layer for later migration.
"""

CREATE_LLM_MODEL_SQL = """
    INSERT INTO llm_models (
        llm_provider, llm_model_name, deployment_name, api_key_variable_name,
        api_endpoint_url, llm_model_version, max_tokens, tokens_per_minute_limit,
        requests_per_minute_limit, is_active_status, temperature, top_p, random_seed, deployment_region
    ) VALUES (
        :llm_provider, :llm_model_name, :deployment_name, :api_key_variable_name,
        :api_endpoint_url, :llm_model_version, :max_tokens, :tokens_per_minute_limit,
        :requests_per_minute_limit, :is_active_status, :temperature, :top_p, :random_seed, :deployment_region
    )
    RETURNING *
"""

GET_LLM_MODEL_BY_PROVIDER_AND_MODEL_WITH_VERSION_SQL = """
    SELECT * FROM llm_models
    WHERE llm_provider = :llm_provider
      AND llm_model_name = :llm_model_name
      AND llm_model_version = :llm_model_version
"""

GET_LLM_MODEL_BY_PROVIDER_AND_MODEL_WITHOUT_VERSION_SQL = """
    SELECT * FROM llm_models
    WHERE llm_provider = :llm_provider
      AND llm_model_name = :llm_model_name
      AND llm_model_version IS NULL
"""

LIST_LLM_MODELS_BY_PROVIDER_BASE_SQL = """
    SELECT * FROM llm_models
    WHERE llm_provider = :llm_provider
"""

COUNT_LLM_MODELS_BY_PROVIDER_BASE_SQL = """
    SELECT COUNT(*) FROM llm_models
    WHERE llm_provider = :llm_provider
"""

DELETE_LLM_MODEL_BY_PROVIDER_AND_MODEL_WITH_VERSION_SQL = """
    DELETE FROM llm_models
    WHERE llm_provider = :llm_provider
      AND llm_model_name = :llm_model_name
      AND llm_model_version = :llm_model_version
"""

DELETE_LLM_MODEL_BY_PROVIDER_AND_MODEL_WITHOUT_VERSION_SQL = """
    DELETE FROM llm_models
    WHERE llm_provider = :llm_provider
      AND llm_model_name = :llm_model_name
      AND llm_model_version IS NULL
"""

DELETE_LLM_MODELS_BY_PROVIDER_SQL = """
    DELETE FROM llm_models
    WHERE llm_provider = :llm_provider
"""
