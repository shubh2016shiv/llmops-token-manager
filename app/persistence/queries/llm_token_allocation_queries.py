"""
LLM token allocation persistence SQL definitions.

This module stores named SQL constants for static token allocation queries.
Dynamic query builders and transition-specific SQL remain in the persistence
layer for later migration.
"""

CREATE_TOKEN_ALLOCATION_SQL = """
    INSERT INTO token_manager (
        token_request_id, user_id, llm_provider, llm_model_name,
        deployment_name, cloud_provider, api_endpoint_url, deployment_region,
        token_count, allocation_status, allocated_at, expires_at,
        request_context, temperature, top_p, seed
    ) VALUES (
        :token_request_id, :user_id, :llm_provider, :llm_model_name,
        :deployment_name, :cloud_provider, :api_endpoint_url, :deployment_region,
        :token_count, :allocation_status, :allocated_at, :expires_at,
        :request_context, :temperature, :top_p, :seed
    )
    RETURNING *
"""

CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL = """
    WITH chosen_deployment AS (
        SELECT *
        FROM llm_models
        WHERE
            llm_provider = :llm_provider
            AND llm_model_name = :llm_model_name
            AND api_endpoint_url = :api_endpoint_url
            AND is_active_status = TRUE
        FOR UPDATE
    ),
    current_load AS (
        SELECT COALESCE(SUM(tm.token_count), 0)::INTEGER AS total_tokens
        FROM token_manager tm
        JOIN chosen_deployment deployment
          ON deployment.llm_provider = tm.llm_provider
         AND deployment.llm_model_name = tm.llm_model_name
         AND deployment.api_endpoint_url = tm.api_endpoint_url
        WHERE
            tm.allocation_status IN ('ACQUIRED', 'PAUSED')
            AND (tm.expires_at IS NULL OR tm.expires_at > NOW())
    ),
    allocation_decision AS (
        SELECT
            deployment.*,
            current_load.total_tokens,
            CASE
                WHEN current_load.total_tokens + CAST(:token_count AS INTEGER)
                    <= deployment.max_tokens
                THEN 'ACQUIRED'
                ELSE 'WAITING'
            END AS decided_status
        FROM chosen_deployment deployment
        CROSS JOIN current_load
        WHERE CAST(:token_count AS INTEGER) <= deployment.max_tokens
    )
    INSERT INTO token_manager (
        token_request_id, user_id, llm_provider, llm_model_name,
        deployment_name, cloud_provider, api_endpoint_url, deployment_region,
        token_count, allocation_status, allocated_at, expires_at,
        request_context, temperature, top_p, seed
    )
    SELECT
        :token_request_id,
        :user_id,
        :llm_provider,
        :llm_model_name,
        COALESCE(:deployment_name, deployment_name),
        COALESCE(:cloud_provider, cloud_provider),
        api_endpoint_url,
        COALESCE(:deployment_region, deployment_region),
        :token_count,
        decided_status,
        :allocated_at,
        :expires_at,
        :request_context,
        COALESCE(:temperature, temperature),
        COALESCE(:top_p, top_p),
        COALESCE(:seed, random_seed)
    FROM allocation_decision
    RETURNING *
"""

GET_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL = """
    SELECT * FROM token_manager
    WHERE token_request_id = :token_request_id
"""

LIST_TOTAL_ALLOCATED_TOKENS_BY_MODEL_SQL = """
    SELECT
        llm_model_name,
        api_endpoint_url,
        deployment_region,
        cloud_provider,
        SUM(token_count) as total_tokens,
        COUNT(*) as allocation_count
    FROM token_manager
    WHERE
        llm_model_name = :llm_model_name
        AND allocation_status = ANY(:included_statuses)
        AND (expires_at IS NULL OR expires_at > NOW())
    GROUP BY llm_model_name, api_endpoint_url, deployment_region, cloud_provider
    ORDER BY total_tokens ASC
"""

GET_TOTAL_ALLOCATED_TOKENS_FOR_ENDPOINT_SQL = """
    SELECT COALESCE(SUM(token_count), 0) as total_tokens
    FROM token_manager
    WHERE
        llm_model_name = :llm_model_name
        AND api_endpoint_url = :api_endpoint_url
        AND allocation_status IN ('ACQUIRED', 'PAUSED')
        AND (expires_at IS NULL OR expires_at > NOW())
"""

LIST_USER_ALLOCATIONS_BY_STATUS_SQL = """
    SELECT * FROM token_manager
    WHERE user_id = :user_id AND allocation_status = ANY(:status_filter)
    ORDER BY allocated_at DESC
    LIMIT :limit
"""

LIST_USER_ALLOCATIONS_SQL = """
    SELECT * FROM token_manager
    WHERE user_id = :user_id
    ORDER BY allocated_at DESC
    LIMIT :limit
"""

COUNT_ACTIVE_ALLOCATIONS_BY_MODEL_SQL = """
    SELECT COUNT(*)
    FROM token_manager
    WHERE
        llm_model_name = :llm_model_name
        AND allocation_status IN ('ACQUIRED', 'PAUSED')
        AND (expires_at IS NULL OR expires_at > NOW())
"""

DELETE_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL = """
    DELETE FROM token_manager
    WHERE token_request_id = :token_request_id
"""

DELETE_EXPIRED_TOKEN_ALLOCATIONS_SQL = """
    DELETE FROM token_manager
    WHERE
        expires_at IS NOT NULL
        AND expires_at < NOW()
        AND allocation_status IN ('ACQUIRED', 'PAUSED', 'WAITING')
"""

DELETE_USER_ALLOCATIONS_BY_STATUS_SQL = """
    DELETE FROM token_manager
    WHERE user_id = :user_id AND allocation_status = :status
"""

DELETE_USER_ALLOCATIONS_SQL = """
    DELETE FROM token_manager
    WHERE user_id = :user_id
"""

CHECK_ACTIVE_PAUSE_ALLOCATION_EXISTS_SQL = """
    SELECT 1 FROM token_manager
    WHERE llm_model_name = :llm_model_name
      AND api_endpoint_url = :api_endpoint_url
      AND allocation_status = 'PAUSED'
      AND expires_at > NOW()
"""

GET_ACTIVE_DEPLOYMENT_BY_MODEL_AND_ENDPOINT_SQL = """
    SELECT *
    FROM llm_models
    WHERE llm_model_name = :llm_model_name AND api_endpoint_url = :api_endpoint_url AND is_active_status = TRUE
"""

LIST_TOKEN_ALLOCATION_SUMMARY_BY_MODEL_SQL = """
    SELECT
        allocation_status,
        COUNT(*) as count,
        SUM(token_count) as total_tokens,
        AVG(token_count) as avg_tokens
    FROM token_manager
    WHERE
        llm_model_name = :llm_model_name
        AND (expires_at IS NULL OR expires_at > NOW())
    GROUP BY allocation_status
"""

GET_USER_TOKEN_USAGE_STATS_SQL = """
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

LIST_ACTIVE_MODEL_DEPLOYMENTS_SQL = """
    SELECT *
    FROM llm_models
    WHERE
        llm_provider = :llm_provider
        AND llm_model_name = :llm_model_name
        AND is_active_status = TRUE
"""

LIST_LEAST_LOADED_ALLOCATIONS_BY_MODEL_SQL = """
    SELECT
        llm_provider,
        llm_model_name,
        api_endpoint_url,
        SUM(token_count) as total_tokens
    FROM token_manager
    WHERE
        llm_provider = :llm_provider
        AND llm_model_name = :llm_model_name
        AND allocation_status IN ('ACQUIRED', 'PAUSED')
        AND (expires_at IS NULL OR expires_at > NOW())
    GROUP BY llm_provider, llm_model_name, api_endpoint_url
    ORDER BY total_tokens ASC
"""

TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL = """
    WITH waiting_allocation AS (
        SELECT *
        FROM token_manager
        WHERE
            token_request_id = :token_request_id
            AND allocation_status = 'WAITING'
        FOR UPDATE
    ),
    chosen_deployment AS (
        SELECT deployment.*
        FROM llm_models deployment
        JOIN waiting_allocation waiting
          ON waiting.llm_provider = deployment.llm_provider
         AND waiting.llm_model_name = deployment.llm_model_name
        WHERE
            deployment.api_endpoint_url = :api_endpoint_url
            AND deployment.is_active_status = TRUE
        FOR UPDATE OF deployment
    ),
    current_load AS (
        SELECT COALESCE(SUM(tm.token_count), 0)::INTEGER AS total_tokens
        FROM token_manager tm
        JOIN chosen_deployment deployment
          ON deployment.llm_provider = tm.llm_provider
         AND deployment.llm_model_name = tm.llm_model_name
         AND deployment.api_endpoint_url = tm.api_endpoint_url
        WHERE
            tm.allocation_status IN ('ACQUIRED', 'PAUSED')
            AND (tm.expires_at IS NULL OR tm.expires_at > NOW())
    ),
    eligible_transition AS (
        SELECT
            waiting.token_request_id,
            deployment.api_endpoint_url,
            deployment.deployment_name,
            deployment.cloud_provider,
            deployment.deployment_region,
            deployment.temperature,
            deployment.top_p,
            deployment.random_seed
        FROM waiting_allocation waiting
        CROSS JOIN chosen_deployment deployment
        CROSS JOIN current_load
        WHERE current_load.total_tokens + waiting.token_count <= deployment.max_tokens
    )
    UPDATE token_manager allocation
    SET
        allocation_status = 'ACQUIRED',
        api_endpoint_url = eligible.api_endpoint_url,
        deployment_name = eligible.deployment_name,
        cloud_provider = eligible.cloud_provider,
        deployment_region = COALESCE(:deployment_region, eligible.deployment_region),
        temperature = eligible.temperature,
        top_p = eligible.top_p,
        seed = eligible.random_seed,
        expires_at = :expires_at
    FROM eligible_transition eligible
    WHERE allocation.token_request_id = eligible.token_request_id
    RETURNING allocation.*
"""
