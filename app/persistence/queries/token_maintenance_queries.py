"""
Token maintenance SQL definitions - raw queries for Layer 4 maintenance data.

This module stores named SQL statements for the token maintenance persistence
layer. It is the canonical source for all token-maintenance query text.

Author: Engineering Team
Last Updated: 2026-05-10
"""

LIST_ACTIVE_DEPLOYMENT_CAPACITY_SNAPSHOTS_SQL = """
    SELECT
        lm.llm_model_name,
        lm.api_endpoint_url,
        COALESCE(SUM(tm.token_count), 0)::int AS allocated_tokens,
        lm.max_tokens::int AS max_tokens
    FROM llm_models lm
    LEFT JOIN token_manager tm
        ON tm.llm_model_name = lm.llm_model_name
        AND tm.api_endpoint_url = lm.api_endpoint_url
        AND tm.allocation_status IN ('ACQUIRED', 'PAUSED')
        AND (tm.expires_at IS NULL OR tm.expires_at > NOW())
    WHERE lm.is_active_status = TRUE
        AND lm.max_tokens IS NOT NULL
    GROUP BY lm.llm_model_name, lm.api_endpoint_url, lm.max_tokens
"""

LIST_STARTUP_COUNTER_SEED_SNAPSHOTS_SQL = """
    SELECT
        lm.llm_model_name,
        lm.api_endpoint_url,
        COALESCE(SUM(tm.token_count), 0)::int AS allocated_tokens,
        lm.max_tokens::int AS max_tokens
    FROM llm_models lm
    LEFT JOIN token_manager tm
        ON tm.llm_model_name = lm.llm_model_name
        AND tm.api_endpoint_url = lm.api_endpoint_url
        AND tm.allocation_status IN ('ACQUIRED', 'PAUSED')
        AND (tm.expires_at IS NULL OR tm.expires_at > NOW())
    WHERE lm.is_active_status = TRUE
        AND lm.max_tokens IS NOT NULL
    GROUP BY lm.llm_model_name, lm.api_endpoint_url, lm.max_tokens
"""

DELETE_EXPIRED_ALLOCATIONS_SQL = """
    DELETE FROM token_manager
    WHERE
        expires_at IS NOT NULL
        AND expires_at < NOW()
        AND allocation_status IN ('ACQUIRED', 'PAUSED', 'WAITING')
"""

LIST_INVALID_ACTIVE_MODELS_WITHOUT_CAPACITY_SQL = """
    SELECT
        llm_provider,
        llm_model_name,
        api_endpoint_url,
        deployment_name,
        deployment_region
    FROM llm_models
    WHERE is_active_status = TRUE
        AND max_tokens IS NULL
    ORDER BY llm_provider, llm_model_name, api_endpoint_url
"""
