"""
Token maintenance SQL — reconciliation, startup seeding, and expiry cleanup.

Reads deployment capacity from the llm_services-owned
``token_manager_deployment_capacity`` view and current load from the token
manager's own ``llm_token_allocations`` table. Redis fast-path counters are keyed
by (model_name, api_endpoint_url), so capacity and load are aggregated to that
grain here.
"""

# Capacity (from the read view) LEFT JOINed with current active load (from our
# allocations), aggregated to the Redis counter grain (model_name, endpoint).
_CAPACITY_SNAPSHOT_SQL = """
    SELECT
        cap.model_name AS llm_model_name,
        cap.api_endpoint_url AS api_endpoint_url,
        COALESCE(load.allocated_tokens, 0)::int AS allocated_tokens,
        cap.max_tokens::int AS max_tokens
    FROM (
        SELECT model_name, api_endpoint_url,
               SUM(token_capacity_limit) AS max_tokens
        FROM token_manager_deployment_capacity
        GROUP BY model_name, api_endpoint_url
    ) cap
    LEFT JOIN (
        SELECT model_name, api_endpoint_url,
               SUM(token_count) AS allocated_tokens
        FROM llm_token_allocations
        WHERE allocation_status IN ('ACQUIRED', 'PAUSED')
            AND (expires_at IS NULL OR expires_at > NOW())
        GROUP BY model_name, api_endpoint_url
    ) load
        ON load.model_name = cap.model_name
        AND load.api_endpoint_url = cap.api_endpoint_url
"""

LIST_ACTIVE_DEPLOYMENT_CAPACITY_SNAPSHOTS_SQL = _CAPACITY_SNAPSHOT_SQL
LIST_STARTUP_COUNTER_SEED_SNAPSHOTS_SQL = _CAPACITY_SNAPSHOT_SQL

DELETE_EXPIRED_ALLOCATIONS_SQL = """
    DELETE FROM llm_token_allocations
    WHERE
        expires_at IS NOT NULL
        AND expires_at < NOW()
        AND allocation_status IN ('ACQUIRED', 'PAUSED', 'WAITING')
"""

# In the real schema ``token_capacity_limit`` is NOT NULL (CHECK > 0), so an
# active deployment can never lack capacity — this query is a defensive invariant
# check that returns nothing under normal operation.
LIST_INVALID_ACTIVE_MODELS_WITHOUT_CAPACITY_SQL = """
    SELECT
        provider_name AS llm_provider,
        model_name AS llm_model_name,
        api_endpoint_url AS api_endpoint_url,
        deployment_name AS deployment_name,
        cloud_region AS deployment_region
    FROM token_manager_deployment_capacity
    WHERE token_capacity_limit IS NULL
    ORDER BY provider_name, model_name, api_endpoint_url
"""
