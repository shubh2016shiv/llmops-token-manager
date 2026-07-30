"""
SQL for the token manager's one owned table: ``llm_token_allocations``.

Grouped by CRUD operation. Deployment-capacity reads (the read contract exposed
by llm_services) live in ``deployment_capacity_queries`` — this module only ever
touches the allocations table the token manager owns.
"""

# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------

CREATE_TOKEN_ALLOCATION_SQL = """
    INSERT INTO llm_token_allocations (
        token_request_id, tenant_id, user_id, deployment_id, provider_name, model_name,
        deployment_key, deployment_name, provider_deployment_name, api_endpoint_url,
        cloud_provider, cloud_region, token_count, allocation_status, allocated_at,
        expires_at, request_context, temperature, top_p, seed
    ) VALUES (
        :token_request_id, :tenant_id, :user_id, :deployment_id, :provider_name, :model_name,
        :deployment_key, :deployment_name, :provider_deployment_name, :api_endpoint_url,
        :cloud_provider, :cloud_region, :token_count, :allocation_status, :allocated_at,
        :expires_at, :request_context, :temperature, :top_p, :seed
    )
    RETURNING *
"""

# Atomic DB-fallback allocation primitive. Locks the target tenant_deployments
# row, recomputes current active load from llm_token_allocations, and decides
# ACQUIRED vs WAITING in the same transaction as the insert.
CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL = """
    WITH chosen_deployment AS (
        SELECT *
        FROM tenant_deployments
        WHERE deployment_id = :deployment_id AND status = 'active'
        FOR UPDATE
    ),
    current_load AS (
        SELECT COALESCE(SUM(a.token_count), 0)::INTEGER AS total_tokens
        FROM llm_token_allocations a
        JOIN chosen_deployment deployment ON deployment.deployment_id = a.deployment_id
        WHERE
            a.allocation_status IN ('ACQUIRED', 'PAUSED')
            AND (a.expires_at IS NULL OR a.expires_at > NOW())
    ),
    allocation_decision AS (
        SELECT
            deployment.*,
            current_load.total_tokens,
            CASE
                WHEN current_load.total_tokens + CAST(:token_count AS INTEGER)
                    <= deployment.token_capacity_limit
                THEN 'ACQUIRED'
                ELSE 'WAITING'
            END AS decided_status
        FROM chosen_deployment deployment
        CROSS JOIN current_load
        WHERE CAST(:token_count AS INTEGER) <= deployment.token_capacity_limit
    )
    INSERT INTO llm_token_allocations (
        token_request_id, tenant_id, user_id, deployment_id, provider_name, model_name,
        deployment_key, deployment_name, provider_deployment_name, api_endpoint_url,
        cloud_provider, cloud_region, token_count, allocation_status, allocated_at,
        expires_at, request_context, temperature, top_p, seed
    )
    SELECT
        :token_request_id,
        :tenant_id,
        :user_id,
        deployment.deployment_id,
        :provider_name,
        :model_name,
        deployment.deployment_key,
        deployment.deployment_name,
        deployment.provider_deployment_name,
        deployment.api_endpoint_url,
        deployment.cloud_provider,
        deployment.cloud_region,
        :token_count,
        decided_status,
        :allocated_at,
        :expires_at,
        :request_context,
        COALESCE(:temperature, deployment.default_temperature),
        COALESCE(:top_p, deployment.default_top_p),
        :seed
    FROM allocation_decision deployment
    RETURNING *
"""

# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

GET_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL = """
    SELECT * FROM llm_token_allocations
    WHERE token_request_id = :token_request_id
"""

# Existence guard for pause_deployment, evaluated under the deployment row lock.
# Scoped by deployment_id (the stable FK), not by name/URL.
CHECK_ACTIVE_PAUSE_ALLOCATION_EXISTS_SQL = """
    SELECT 1 FROM llm_token_allocations
    WHERE deployment_id = :deployment_id
      AND allocation_status = 'PAUSED'
      AND expires_at > NOW()
"""

# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------

# Atomically transition a WAITING allocation to ACQUIRED against an already
# chosen deployment, re-checking capacity under lock.
TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL = """
    WITH waiting_allocation AS (
        SELECT *
        FROM llm_token_allocations
        WHERE
            token_request_id = :token_request_id
            AND allocation_status = 'WAITING'
        FOR UPDATE
    ),
    chosen_deployment AS (
        SELECT *
        FROM tenant_deployments
        WHERE deployment_id = :deployment_id AND status = 'active'
        FOR UPDATE
    ),
    current_load AS (
        SELECT COALESCE(SUM(a.token_count), 0)::INTEGER AS total_tokens
        FROM llm_token_allocations a
        JOIN chosen_deployment deployment ON deployment.deployment_id = a.deployment_id
        WHERE
            a.allocation_status IN ('ACQUIRED', 'PAUSED')
            AND (a.expires_at IS NULL OR a.expires_at > NOW())
    ),
    eligible_transition AS (
        SELECT
            waiting.token_request_id,
            deployment.deployment_id,
            deployment.api_endpoint_url,
            deployment.deployment_key,
            deployment.deployment_name,
            deployment.provider_deployment_name,
            deployment.cloud_provider,
            deployment.cloud_region,
            deployment.default_temperature,
            deployment.default_top_p
        FROM waiting_allocation waiting
        CROSS JOIN chosen_deployment deployment
        CROSS JOIN current_load
        WHERE current_load.total_tokens + waiting.token_count
            <= deployment.token_capacity_limit
    )
    UPDATE llm_token_allocations allocation
    SET
        allocation_status = 'ACQUIRED',
        deployment_id = eligible.deployment_id,
        api_endpoint_url = eligible.api_endpoint_url,
        deployment_key = eligible.deployment_key,
        deployment_name = eligible.deployment_name,
        provider_deployment_name = eligible.provider_deployment_name,
        cloud_provider = eligible.cloud_provider,
        cloud_region = eligible.cloud_region,
        temperature = eligible.default_temperature,
        top_p = eligible.default_top_p,
        expires_at = :expires_at
    FROM eligible_transition eligible
    WHERE allocation.token_request_id = eligible.token_request_id
    RETURNING allocation.*
"""

# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

DELETE_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL = """
    DELETE FROM llm_token_allocations
    WHERE token_request_id = :token_request_id
"""
