"""
User entitlement persistence SQL definitions.

This module stores named SQL constants for static entitlement queries.
Dynamic query builders remain in the persistence layer for later migration.
"""

CHECK_USER_EXISTS_FOR_ENTITLEMENT_SQL = """
    SELECT 1 FROM users
    WHERE user_id = :user_id
    LIMIT 1
"""

CHECK_PROVIDER_MODEL_EXISTS_FOR_ENTITLEMENT_SQL = """
    SELECT 1 FROM llm_models
    WHERE llm_provider = :llm_provider
      AND llm_model_name = :llm_model_name
    LIMIT 1
"""

CHECK_USER_ENTITLEMENT_EXISTS_SQL = """
    SELECT 1 FROM user_llm_entitlements
    WHERE user_id = :user_id
      AND llm_provider = :llm_provider
      AND llm_model_name = :llm_model_name
      AND api_endpoint_url = :api_endpoint_url
    LIMIT 1
"""

CREATE_USER_ENTITLEMENT_SQL = """
    INSERT INTO user_llm_entitlements (
        user_id, llm_provider, llm_model_name, api_key_value,
        api_endpoint_url, cloud_provider, deployment_name, deployment_region,
        created_at, updated_at, created_by_user_id
    )
    VALUES (
        :user_id, :llm_provider, :llm_model_name, :api_key_value,
        :api_endpoint_url, :cloud_provider, :deployment_name, :deployment_region,
        :created_at, :updated_at, :created_by_user_id
    )
    RETURNING entitlement_id, user_id, llm_provider, llm_model_name,
              api_endpoint_url, cloud_provider, deployment_name, deployment_region,
              created_at, updated_at, created_by_user_id
"""

GET_USER_ENTITLEMENT_BY_ID_SQL = """
    SELECT entitlement_id, user_id, llm_provider, llm_model_name,
           api_endpoint_url, cloud_provider, deployment_name, deployment_region,
           created_at, updated_at, created_by_user_id
    FROM user_llm_entitlements
    WHERE entitlement_id = :entitlement_id
"""

LIST_USER_ENTITLEMENTS_SQL = """
    SELECT entitlement_id, user_id, llm_provider, llm_model_name,
           api_endpoint_url, cloud_provider, deployment_name, deployment_region,
           created_at, updated_at, created_by_user_id
    FROM user_llm_entitlements
    WHERE user_id = :user_id
    ORDER BY created_at DESC
    LIMIT :limit OFFSET :offset
"""

COUNT_USER_ENTITLEMENTS_SQL = """
    SELECT COUNT(*) FROM user_llm_entitlements
    WHERE user_id = :user_id
"""

DELETE_USER_ENTITLEMENT_BY_ID_SQL = """
    DELETE FROM user_llm_entitlements
    WHERE entitlement_id = :entitlement_id
"""
