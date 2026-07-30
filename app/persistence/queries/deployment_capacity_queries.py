"""
SQL for reading deployment capacity — the read contract llm_services exposes to
the token manager.

These are read-only against llm_services-owned data: the
``token_manager_deployment_capacity`` view and (for the pause path only) a
row-locked read of ``tenant_deployments``. The token manager never writes here.
The current-load aggregate reads the token manager's own allocations table to
rank candidate deployments.
"""

# Active deployment capacity for a tenant/provider/model, via the
# llm_services-owned read view (never tenant_deployments directly, except the
# locked pause path below).
LIST_ACTIVE_MODEL_DEPLOYMENTS_SQL = """
    SELECT *
    FROM token_manager_deployment_capacity
    WHERE
        tenant_id = :tenant_id
        AND provider_name = :llm_provider
        AND model_name = :llm_model_name
"""

# Current active load per deployment, used to pick the least-loaded candidate.
LIST_LEAST_LOADED_ALLOCATIONS_BY_MODEL_SQL = """
    SELECT
        deployment_id,
        provider_name,
        model_name,
        api_endpoint_url,
        SUM(token_count) as total_tokens
    FROM llm_token_allocations
    WHERE
        tenant_id = :tenant_id
        AND provider_name = :llm_provider
        AND model_name = :llm_model_name
        AND allocation_status IN ('ACQUIRED', 'PAUSED')
        AND (expires_at IS NULL OR expires_at > NOW())
    GROUP BY deployment_id, provider_name, model_name, api_endpoint_url
    ORDER BY total_tokens ASC
"""

# Used by pause_deployment() to serialize concurrent pause requests per
# deployment. FOR UPDATE OF td locks only the tenant_deployments row (not the
# joined catalog rows) for the duration of the calling transaction, so a
# second concurrent caller blocks here until the first commits/rolls back.
GET_ACTIVE_DEPLOYMENT_BY_MODEL_AND_ENDPOINT_LOCKED_SQL = """
    SELECT
        td.deployment_id, td.deployment_key, td.deployment_name,
        td.provider_deployment_name, pc.provider_name, mc.model_name,
        td.api_endpoint_url, td.cloud_provider, td.cloud_region,
        td.token_capacity_limit, td.token_lock_duration_seconds,
        td.default_temperature, td.default_top_p, td.default_max_output_tokens
    FROM tenant_deployments td
    JOIN provider_catalog pc ON pc.provider_id = td.provider_id
    JOIN model_catalog mc ON mc.model_id = td.model_id
    WHERE td.tenant_id = :tenant_id
      AND pc.provider_name = :llm_provider
      AND mc.model_name = :llm_model_name
      AND td.api_endpoint_url = :api_endpoint_url
      AND td.status = 'active'
    FOR UPDATE OF td
"""
