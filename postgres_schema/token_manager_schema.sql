-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS token_manager CASCADE;
-- ============================================================================
-- TOKEN MANAGER TABLE - MANAGES TOKEN ALLOCATION REQUESTS
-- Central gateway for LLM token allocation requests, ensuring fair usage, cost control, and deployment resilience.
-- ============================================================================
CREATE TABLE IF NOT EXISTS token_manager (
    -- Request Identity: Unique identifier for the token allocation request
    token_request_id TEXT PRIMARY KEY,
    -- User Reference: Links to the user requesting tokens
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    -- Model & Deployment Configuration: Specifies the target LLM and deployment
    llm_model_name TEXT NOT NULL,
    llm_provider TEXT NOT NULL
    CHECK (llm_provider IN (
        'openai',
        'gemini',
        'anthropic',
        'cohere',
        'mistral',
        'deepseek',
        'meta',
        'hugging_face',
        'together_ai',
        'fireworks_ai',
        'replicate',
        'xai',
        'deepinfra',
        'novita',
        'on_premise'
    )),
    deployment_name TEXT,
    cloud_provider TEXT
    CHECK (cloud_provider IN (
        'Azure',
        'Google Cloud Platform',
        'Amazon Web Services',
        'IBM Watsonx',
        'Oracle',
        'On Premise')),
    api_endpoint_url TEXT,
    deployment_region TEXT,
    -- Token Allocation Management: Tracks allocated tokens and their status
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    allocation_status TEXT NOT NULL DEFAULT 'ACQUIRED'
        CHECK (allocation_status IN ('ACQUIRED', 'WAITING', 'RELEASED', 'EXPIRED', 'PAUSED', 'FAILED')),
    -- Timing & Expiration: Manages allocation lifecycle
    allocated_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ,
    request_context JSONB,
    temperature FLOAT,
    top_p FLOAT,
    seed INTEGER,

    -- PAUSED allocations must have an expiration to prevent permanent blocking
    CONSTRAINT chk_pause_must_expire CHECK (allocation_status != 'PAUSED' OR expires_at IS NOT NULL)
);
COMMENT ON TABLE token_manager IS 'Central gateway for token allocations, ensuring fair usage, cost control, and regional resilience';
COMMENT ON COLUMN token_manager.token_request_id IS 'Unique identifier for the token allocation request';
COMMENT ON COLUMN token_manager.llm_model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN token_manager.llm_provider IS 'Provider of the LLM model (e.g., openai, anthropic, azure_openai)';
COMMENT ON COLUMN token_manager.deployment_name IS 'Specific deployment of the model, if applicable';
COMMENT ON COLUMN token_manager.cloud_provider IS 'Cloud provider hosting the LLM (e.g., Azure, AWS), if applicable';
COMMENT ON COLUMN token_manager.api_endpoint_url IS 'API endpoint for the selected LLM instance, if applicable';
COMMENT ON COLUMN token_manager.deployment_region IS 'Region where the model is deployed on the cloud provider, if applicable';
COMMENT ON COLUMN token_manager.token_count IS 'Number of tokens allocated for this request';
COMMENT ON COLUMN token_manager.allocation_status IS 'Current status: ACQUIRED, RELEASED, EXPIRED, PAUSED, or FAILED';
COMMENT ON COLUMN token_manager.allocated_at IS 'Timestamp when tokens were allocated';
COMMENT ON COLUMN token_manager.expires_at IS 'Optional expiration time for the allocation lock';
COMMENT ON COLUMN token_manager.request_context IS 'Additional context data in JSON format (e.g., team, application)';
COMMENT ON COLUMN token_manager.temperature IS 'Temperature for the LLM';
COMMENT ON COLUMN token_manager.top_p IS 'Top P for the LLM';
COMMENT ON COLUMN token_manager.seed IS 'Seed value for reproducible LLM outputs';
-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Supports efficient querying for token allocation lifecycle management.
-- ============================================================================
CREATE INDEX idx_token_expiry_status_model ON token_manager(expires_at, allocation_status, llm_model_name, api_endpoint_url);
CREATE INDEX idx_token_model ON token_manager(llm_model_name, api_endpoint_url);

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

/*
===============================================================================
USAGE EXAMPLES FOR TOKEN_MANAGER TABLE
===============================================================================

-- Example 1: Direct OpenAI API allocation (no cloud provider)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    api_endpoint_url, token_count, allocation_status, allocated_at
) VALUES (
    'req_abc123def456',
    '550e8400-e29b-41d4-a716-446655440000',
    'gpt-4o',
    'openai',
    'https://api.openai.com/v1',
    1500,
    'ACQUIRED',
    CURRENT_TIMESTAMP
);

-- Example 2: Direct Anthropic API allocation with LLM parameters
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    token_count, allocation_status, allocated_at,
    temperature, top_p, seed
) VALUES (
    'req_xyz789ghi012',
    '550e8400-e29b-41d4-a716-446655440000',
    'claude-3-5-sonnet-20240620',
    'anthropic',
    2000,
    'ACQUIRED',
    CURRENT_TIMESTAMP,
    0.7,
    0.9,
    42
);

-- Example 3: Azure OpenAI deployment allocation (cloud provider)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    cloud_provider, deployment_name, api_endpoint_url, deployment_region,
    token_count, allocation_status, allocated_at, request_context
) VALUES (
    'req_azure456jkl789',
    '550e8400-e29b-41d4-a716-446655440000',
    'gpt-4o',
    'openai',
    'azure_openai',
    'gpt4o-eastus-prod',
    'https://my-resource.openai.azure.com/',
    'eastus',
    2500,
    'ACQUIRED',
    CURRENT_TIMESTAMP,
    '{"application": "customer_support_chatbot", "environment": "production"}'
);

-- Example 4: AWS Bedrock allocation (different cloud provider)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    cloud_provider, deployment_region,
    token_count, allocation_status, allocated_at
) VALUES (
    'req_bedrock123mno456',
    '550e8400-e29b-41d4-a716-446655440000',
    'claude-3-5-sonnet-20240620',
    'anthropic',
    'aws_bedrock',
    'us-west-2',
    1800,
    'ACQUIRED',
    CURRENT_TIMESTAMP
);

-- Example 5: Google Vertex AI allocation (Gemini model)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    cloud_provider, deployment_region,
    token_count, allocation_status, allocated_at, temperature
) VALUES (
    'req_vertex789pqr012',
    '550e8400-e29b-41d4-a716-446655440000',
    'gemini-pro',
    'gemini',
    'google_vertex',
    'us-central1',
    1200,
    'ACQUIRED',
    CURRENT_TIMESTAMP,
    0.8
);

-- Example 6: PAUSED allocation (requires expires_at)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    api_endpoint_url, token_count, allocation_status,
    allocated_at, expires_at
) VALUES (
    'req_paused345stu678',
    '550e8400-e29b-41d4-a716-446655440000',
    'gpt-4o',
    'openai',
    'https://api.openai.com/v1',
    1000,
    'PAUSED',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP + INTERVAL '5 minutes'
);

-- Example 7: WAITING allocation (queue scenario)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    cloud_provider, deployment_name, deployment_region,
    token_count, allocation_status, allocated_at
) VALUES (
    'req_waiting901vwx234',
    '550e8400-e29b-41d4-a716-446655440000',
    'gpt-4o',
    'openai',
    'azure_openai',
    'gpt4o-eastus-prod',
    'eastus',
    3000,
    'WAITING',
    CURRENT_TIMESTAMP
);

-- Example 8: Token release after usage
UPDATE token_manager
SET allocation_status = 'RELEASED'
WHERE token_request_id = 'req_abc123def456';

-- Example 9: Expired allocation cleanup
UPDATE token_manager
SET allocation_status = 'EXPIRED'
WHERE expires_at < CURRENT_TIMESTAMP
  AND allocation_status IN ('ACQUIRED', 'PAUSED');

-- Example 10: Failed allocation (API error scenario)
INSERT INTO token_manager (
    token_request_id, user_id, llm_model_name, llm_provider,
    api_endpoint_url, token_count, allocation_status, allocated_at
) VALUES (
    'req_failed567yza890',
    '550e8400-e29b-41d4-a716-446655440000',
    'gpt-4o',
    'openai',
    'https://api.openai.com/v1',
    1500,
    'FAILED',
    CURRENT_TIMESTAMP
);

-- Example 11: Query active allocations for a specific user
SELECT
    token_request_id,
    llm_model_name,
    llm_provider,
    cloud_provider,
    deployment_name,
    token_count,
    allocated_at,
    expires_at
FROM token_manager
WHERE user_id = '550e8400-e29b-41d4-a716-446655440000'
  AND allocation_status = 'ACQUIRED'
ORDER BY allocated_at DESC;

-- Example 12: Find all allocations by model and provider
SELECT
    tm.token_request_id,
    u.username,
    tm.llm_model_name,
    tm.llm_provider,
    tm.token_count,
    tm.allocation_status,
    tm.allocated_at
FROM token_manager tm
JOIN users u ON tm.user_id = u.user_id
WHERE tm.llm_model_name = 'gpt-4o'
  AND tm.llm_provider = 'openai'
ORDER BY tm.allocated_at DESC;

-- Example 13: Find PAUSED allocations expiring soon (for management)
SELECT
    token_request_id,
    user_id,
    llm_model_name,
    token_count,
    allocated_at,
    expires_at,
    (expires_at - CURRENT_TIMESTAMP) as time_remaining
FROM token_manager
WHERE allocation_status = 'PAUSED'
  AND expires_at BETWEEN CURRENT_TIMESTAMP AND CURRENT_TIMESTAMP + INTERVAL '10 minutes'
ORDER BY expires_at;

-- Example 14: Find allocations by cloud provider and region
SELECT
    tm.token_request_id,
    u.username,
    tm.llm_model_name,
    tm.deployment_name,
    tm.deployment_region,
    tm.token_count,
    tm.allocation_status
FROM token_manager tm
JOIN users u ON tm.user_id = u.user_id
WHERE tm.cloud_provider = 'azure_openai'
  AND tm.deployment_region = 'eastus'
ORDER BY tm.allocated_at DESC;

-- Example 15: Get allocation summary by status (for monitoring)
SELECT
    allocation_status,
    COUNT(*) as allocation_count,
    SUM(token_count) as total_tokens,
    AVG(token_count) as avg_tokens_per_allocation
FROM token_manager
GROUP BY allocation_status
ORDER BY allocation_status;

-- Example 16: Find high-usage users (token consumption analysis)
SELECT
    u.username,
    u.email,
    COUNT(tm.token_request_id) as total_allocations,
    SUM(tm.token_count) as total_tokens_used,
    AVG(tm.token_count) as avg_tokens_per_request
FROM token_manager tm
JOIN users u ON tm.user_id = u.user_id
WHERE tm.allocation_status IN ('RELEASED', 'EXPIRED')
  AND tm.allocated_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY u.user_id, u.username, u.email
ORDER BY total_tokens_used DESC;

-- Example 17: Find allocations with specific LLM parameters
SELECT
    token_request_id,
    user_id,
    llm_model_name,
    temperature,
    top_p,
    seed,
    token_count,
    allocated_at
FROM token_manager
WHERE temperature IS NOT NULL
  AND top_p IS NOT NULL
ORDER BY allocated_at DESC;

-- Example 18: Find allocations expiring in next hour (for proactive management)
SELECT
    token_request_id,
    llm_model_name,
    llm_provider,
    token_count,
    allocated_at,
    expires_at,
    (expires_at - CURRENT_TIMESTAMP) as minutes_until_expiry
FROM token_manager
WHERE allocation_status IN ('ACQUIRED', 'PAUSED')
  AND expires_at BETWEEN CURRENT_TIMESTAMP AND CURRENT_TIMESTAMP + INTERVAL '1 hour'
ORDER BY expires_at;

-- Example 19: Bulk cleanup of expired allocations
UPDATE token_manager
SET allocation_status = 'EXPIRED'
WHERE allocation_status IN ('ACQUIRED', 'PAUSED')
  AND expires_at < CURRENT_TIMESTAMP;

-- Example 20: Find allocation patterns by request context
SELECT
    request_context->>'application' as application,
    request_context->>'environment' as environment,
    COUNT(*) as allocation_count,
    SUM(token_count) as total_tokens,
    AVG(token_count) as avg_tokens
FROM token_manager
WHERE request_context IS NOT NULL
  AND allocation_status = 'RELEASED'
GROUP BY request_context->>'application', request_context->>'environment'
ORDER BY total_tokens DESC;

===============================================================================
*/
