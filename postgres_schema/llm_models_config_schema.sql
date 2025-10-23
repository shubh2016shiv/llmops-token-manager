-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS llm_models CASCADE;
-- ============================================================================
-- LLM MODELS TABLE - CATALOG OF AVAILABLE LLM MODELS
-- Represents available LLM models with their configurations and usage metrics.
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_models (
    llm_provider TEXT NOT NULL DEFAULT 'openai'
        CHECK (llm_provider IN (
        'azure_openai', 'google_vertex', 'aws_bedrock', 'ibm_watsonx', 'oracle',
        'openai', 'gemini', 'anthropic', 'cohere', 'mistral', 'deepseek', 'meta', 'hugging_face', 'together_ai',
        'fireworks_ai', 'replicate', 'xai', 'deepinfra', 'novita', 'on_premise'
    )),
    llm_model_name TEXT NOT NULL,
    deployment_name TEXT,
    cloud_provider TEXT,
    api_key_variable_name TEXT,
    api_endpoint_url TEXT,
    llm_model_version TEXT,
    -- Model Specifications: Defines model capabilities and limits
    max_tokens INTEGER,
    tokens_per_minute_limit INTEGER,
    requests_per_minute_limit INTEGER,
    -- Configuration: Indicates if the model is available for use
    is_active_status BOOLEAN NOT NULL DEFAULT true,
    temperature FLOAT,
    random_seed INTEGER,
    deployment_region TEXT,
    -- Audit Trail: Tracks model creation, updates, and last usage
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Composite Primary Key
    PRIMARY KEY (llm_provider, llm_model_name)
);
COMMENT ON TABLE llm_models IS 'Catalog of LLM models, tracking configurations and usage metrics';
COMMENT ON COLUMN llm_models.llm_provider IS 'LLM provider name (e.g., openai, gemini, anthropic)';
COMMENT ON COLUMN llm_models.llm_model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN llm_models.deployment_name IS 'Name of the LLM deployment (e.g., gpt-4o)';
COMMENT ON COLUMN llm_models.cloud_provider IS 'Cloud provider hosting the LLM (e.g., Azure, AWS)';
COMMENT ON COLUMN llm_models.api_key_variable_name IS 'Variable Name of LLM API Key (e.g., OPENAI_API_KEY_GPT4O)';
COMMENT ON COLUMN llm_models.api_endpoint_url IS 'API endpoint for the selected LLM instance, if applicable';
COMMENT ON COLUMN llm_models.llm_model_version IS 'Specific version of the model';
COMMENT ON COLUMN llm_models.max_tokens IS 'Maximum tokens the model can process in a single request';
COMMENT ON COLUMN llm_models.tokens_per_minute_limit IS 'Token rate limit per minute';
COMMENT ON COLUMN llm_models.requests_per_minute_limit IS 'Request rate limit per minute';
COMMENT ON COLUMN llm_models.is_active_status IS 'Indicates if the model is available for use';
COMMENT ON COLUMN llm_models.temperature IS 'Temperature for the LLM model';
COMMENT ON COLUMN llm_models.random_seed IS 'Random seed for the LLM model';
COMMENT ON COLUMN llm_models.deployment_region IS 'Deployment region of the LLM instance (e.g., eastus2, westus2)';

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

/*
===============================================================================
USAGE EXAMPLES FOR LLM_MODELS TABLE
===============================================================================

-- Example 1: Direct OpenAI API model (no cloud provider, minimal configuration)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    api_key_variable_name, api_endpoint_url,
    max_tokens, tokens_per_minute_limit, requests_per_minute_limit,
    is_active_status
) VALUES (
    'openai',
    'gpt-4o',
    'OPENAI_API_KEY_GPT4O',
    'https://api.openai.com/v1',
    8192,
    100000,
    1000,
    true
);

-- Example 2: Direct Anthropic API model (with default parameters)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    api_key_variable_name,
    max_tokens, tokens_per_minute_limit,
    is_active_status, temperature, random_seed
) VALUES (
    'anthropic',
    'claude-3-5-sonnet-20240620',
    'ANTHROPIC_API_KEY',
    4096,
    50000,
    true,
    0.7,
    42
);

-- Example 3: Azure OpenAI deployment (cloud provider with full deployment details)
INSERT INTO llm_models (
    llm_provider, llm_model_name, deployment_name,
    cloud_provider, api_key_variable_name, api_endpoint_url,
    deployment_region, max_tokens, tokens_per_minute_limit, requests_per_minute_limit,
    is_active_status, temperature
) VALUES (
    'openai',
    'gpt-4o',
    'gpt4o-eastus-prod',
    'azure_openai',
    'AZURE_OPENAI_API_KEY_GPT4O',
    'https://my-resource.openai.azure.com/',
    'eastus',
    8192,
    80000,
    500,
    true,
    0.7
);

-- Example 4: AWS Bedrock model (different cloud provider)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    cloud_provider, deployment_region,
    max_tokens, tokens_per_minute_limit, requests_per_minute_limit,
    is_active_status
) VALUES (
    'anthropic',
    'claude-3-5-sonnet-20240620',
    'aws_bedrock',
    'us-west-2',
    4096,
    40000,
    200,
    true
);

-- Example 5: Google Vertex AI model (Gemini Pro)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    cloud_provider, deployment_region,
    api_key_variable_name,
    max_tokens, tokens_per_minute_limit,
    is_active_status, temperature
) VALUES (
    'gemini',
    'gemini-pro',
    'google_vertex',
    'us-central1',
    'GOOGLE_CLOUD_API_KEY',
    30720,
    60000,
    true,
    0.8
);

-- Example 6: Model without rate limits (unlimited usage scenario)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    api_key_variable_name, api_endpoint_url,
    max_tokens,
    is_active_status
) VALUES (
    'openai',
    'gpt-3.5-turbo',
    'OPENAI_API_KEY_GPT35',
    'https://api.openai.com/v1',
    4096,
    true
);

-- Example 7: High-capacity model configuration (for heavy workloads)
INSERT INTO llm_models (
    llm_provider, llm_model_name, deployment_name,
    cloud_provider, api_key_variable_name, api_endpoint_url,
    deployment_region,
    max_tokens, tokens_per_minute_limit, requests_per_minute_limit,
    is_active_status
) VALUES (
    'openai',
    'gpt-4',
    'gpt4-global-high-capacity',
    'azure_openai',
    'AZURE_OPENAI_API_KEY_GPT4',
    'https://high-capacity-resource.openai.azure.com/',
    'eastus2',
    32768,
    500000,
    2000,
    true
);

-- Example 8: Inactive model (for maintenance or deprecation)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    api_key_variable_name, api_endpoint_url,
    max_tokens, tokens_per_minute_limit,
    is_active_status
) VALUES (
    'openai',
    'gpt-3.5-turbo-0613',
    'OPENAI_API_KEY_GPT35_LEGACY',
    'https://api.openai.com/v1',
    4096,
    10000,
    false
);

-- Example 9: On-premise model configuration (local deployment)
INSERT INTO llm_models (
    llm_provider, llm_model_name,
    api_endpoint_url, deployment_region,
    max_tokens, tokens_per_minute_limit, requests_per_minute_limit,
    is_active_status
) VALUES (
    'on_premise',
    'llama-2-7b',
    'http://localhost:8000/v1',
    'local',
    2048,
    10000,
    100,
    true
);

-- Example 10: Model with specific version tracking
INSERT INTO llm_models (
    llm_provider, llm_model_name, llm_model_version,
    deployment_name, cloud_provider,
    api_key_variable_name, api_endpoint_url, deployment_region,
    max_tokens, is_active_status
) VALUES (
    'openai',
    'gpt-4o',
    '2024-08-06',
    'gpt4o-versioned-eastus',
    'azure_openai',
    'AZURE_OPENAI_API_KEY_GPT4O_V1',
    'https://versioned-resource.openai.azure.com/',
    'eastus',
    16384,
    true
);

-- Example 11: Query all active models by provider
SELECT
    llm_provider,
    llm_model_name,
    deployment_name,
    cloud_provider,
    deployment_region,
    max_tokens,
    tokens_per_minute_limit,
    requests_per_minute_limit,
    temperature,
    created_at
FROM llm_models
WHERE is_active_status = true
ORDER BY llm_provider, llm_model_name;

-- Example 12: Find direct provider models (no cloud provider)
SELECT
    llm_provider,
    llm_model_name,
    api_key_variable_name,
    api_endpoint_url,
    max_tokens,
    tokens_per_minute_limit,
    temperature
FROM llm_models
WHERE cloud_provider IS NULL
  AND is_active_status = true
ORDER BY llm_provider, llm_model_name;

-- Example 13: Find models by cloud provider and region
SELECT
    llm_provider,
    llm_model_name,
    deployment_name,
    deployment_region,
    max_tokens,
    tokens_per_minute_limit,
    requests_per_minute_limit
FROM llm_models
WHERE cloud_provider = 'azure_openai'
  AND deployment_region = 'eastus'
  AND is_active_status = true
ORDER BY max_tokens DESC;

-- Example 14: Find high-capacity models (for load balancing)
SELECT
    llm_provider,
    llm_model_name,
    cloud_provider,
    deployment_name,
    deployment_region,
    tokens_per_minute_limit,
    requests_per_minute_limit,
    max_tokens
FROM llm_models
WHERE tokens_per_minute_limit >= 100000
  AND is_active_status = true
ORDER BY tokens_per_minute_limit DESC;

-- Example 15: Get model capacity summary by provider
SELECT
    llm_provider,
    cloud_provider,
    COUNT(*) as total_models,
    COUNT(CASE WHEN is_active_status THEN 1 END) as active_models,
    AVG(max_tokens) as avg_max_tokens,
    SUM(tokens_per_minute_limit) as total_tokens_per_minute,
    SUM(requests_per_minute_limit) as total_requests_per_minute
FROM llm_models
GROUP BY llm_provider, cloud_provider
ORDER BY total_tokens_per_minute DESC;

-- Example 16: Find models with specific capabilities (temperature, seed)
SELECT
    llm_provider,
    llm_model_name,
    cloud_provider,
    temperature,
    random_seed,
    max_tokens,
    is_active_status
FROM llm_models
WHERE temperature IS NOT NULL
  AND random_seed IS NOT NULL
  AND is_active_status = true
ORDER BY temperature;

-- Example 17: Find models without rate limits (unlimited usage)
SELECT
    llm_provider,
    llm_model_name,
    cloud_provider,
    deployment_name,
    max_tokens,
    is_active_status
FROM llm_models
WHERE tokens_per_minute_limit IS NULL
  AND requests_per_minute_limit IS NULL
  AND is_active_status = true
ORDER BY llm_provider, llm_model_name;

-- Example 18: Update model status (enable/disable for maintenance)
UPDATE llm_models
SET
    is_active_status = false,
    updated_at = CURRENT_TIMESTAMP
WHERE llm_provider = 'openai'
  AND llm_model_name = 'gpt-3.5-turbo-0613';

-- Example 19: Update model rate limits (capacity adjustment)
UPDATE llm_models
SET
    tokens_per_minute_limit = 200000,
    requests_per_minute_limit = 1500,
    updated_at = CURRENT_TIMESTAMP
WHERE llm_provider = 'openai'
  AND llm_model_name = 'gpt-4o'
  AND deployment_name = 'gpt4o-eastus-prod';

-- Example 20: Reactivate models after maintenance
UPDATE llm_models
SET
    is_active_status = true,
    updated_at = CURRENT_TIMESTAMP
WHERE cloud_provider = 'azure_openai'
  AND deployment_region = 'eastus'
  AND is_active_status = false;

-- Example 21: Find models by deployment region (for geographic routing)
SELECT
    llm_provider,
    llm_model_name,
    deployment_name,
    cloud_provider,
    deployment_region,
    tokens_per_minute_limit,
    requests_per_minute_limit,
    is_active_status
FROM llm_models
WHERE deployment_region = 'eastus'
  AND is_active_status = true
ORDER BY tokens_per_minute_limit DESC;

-- Example 22: Get model configuration for specific provider-model combination
SELECT
    lm.*,
    CASE
        WHEN cloud_provider IS NOT NULL THEN 'cloud'
        ELSE 'direct'
    END as provider_type
FROM llm_models lm
WHERE llm_provider = 'openai'
  AND llm_model_name = 'gpt-4o';

-- Example 23: Find recently added models (for monitoring new deployments)
SELECT
    llm_provider,
    llm_model_name,
    deployment_name,
    cloud_provider,
    deployment_region,
    created_at,
    is_active_status
FROM llm_models
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Example 24: Compare model capabilities across providers
SELECT
    'OpenAI' as provider,
    llm_model_name,
    max_tokens,
    tokens_per_minute_limit,
    requests_per_minute_limit,
    temperature
FROM llm_models
WHERE llm_provider = 'openai'
  AND is_active_status = true

UNION ALL

SELECT
    'Anthropic' as provider,
    llm_model_name,
    max_tokens,
    tokens_per_minute_limit,
    requests_per_minute_limit,
    temperature
FROM llm_models
WHERE llm_provider = 'anthropic'
  AND is_active_status = true

ORDER BY provider, max_tokens DESC;

-- Example 25: Find models suitable for different use cases
SELECT
    llm_provider,
    llm_model_name,
    cloud_provider,
    deployment_region,
    max_tokens,
    tokens_per_minute_limit,
    CASE
        WHEN max_tokens >= 32000 THEN 'long_context'
        WHEN tokens_per_minute_limit >= 100000 THEN 'high_throughput'
        WHEN temperature IS NOT NULL THEN 'configurable'
        ELSE 'standard'
    END as use_case_category
FROM llm_models
WHERE is_active_status = true
ORDER BY use_case_category, tokens_per_minute_limit DESC;

===============================================================================
*/
