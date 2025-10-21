-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS user_llm_entitlements CASCADE;

-- ============================================================================
-- USER LLM ENTITLEMENTS TABLE - MANAGES USER ACCESS TO LLM MODELS
-- Maps users to their entitled LLM configurations with API key storage.
-- Supports both direct providers (OpenAI, Anthropic) and cloud providers (Azure OpenAI, Google Vertex, AWS Bedrock).
-- Provider validation is handled at the application level for centralized control.
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_llm_entitlements (
    -- Identifiers: Core keys for uniqueness and referencing
    entitlement_id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Core References: Specifies the entitled provider and model
    llm_provider TEXT NOT NULL CHECK (llm_provider IN (
        'azure_openai', 'google_vertex', 'aws_bedrock', 'ibm_watsonx', 'oracle',
        'openai', 'gemini', 'anthropic', 'cohere', 'mistral', 'deepseek', 'meta', 'hugging_face', 'together_ai',
        'fireworks_ai', 'replicate', 'xai', 'deepinfra', 'novita', 'on_premise'
    )),  -- All ProviderType enum values (application validates)
    llm_model_name TEXT NOT NULL,

    -- Configurations: API and deployment details for client init
    api_key TEXT NOT NULL,  -- Encrypted API key value (use pgcrypto for encryption)
    api_endpoint_url TEXT,  -- Specific endpoint URL (nullable for some providers)
    cloud_provider TEXT,    -- e.g., 'azure_openai', 'google_vertex', 'aws_bedrock' (nullable for direct)
    deployment_name TEXT,   -- Physical deployment identifier (e.g., 'gpt4o-eastus-prod')
    region TEXT,            -- Geographic region (e.g., 'eastus', 'us-west-2')

    -- Audit Trail: Tracks creation, updates, and admin actions
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,

    -- Foreign Key: Ensures the LLM model exists in the catalog (optional - remove if not needed)
    -- Note: This FK can be removed since we're using application-level validation
    FOREIGN KEY (llm_provider, llm_model_name)
        REFERENCES llm_models(llm_provider, llm_model_name)
        ON DELETE CASCADE,

    -- Unique Constraint: Ensures unique entitlements per user/provider/model/endpoint
    UNIQUE (user_id, llm_provider, llm_model_name, api_endpoint_url)
);

-- ============================================================================
-- COMMENTS - DOCUMENTATION FOR EACH COLUMN
-- ============================================================================
COMMENT ON TABLE user_llm_entitlements IS 'Maps users to their entitled LLM configurations with API key storage, supporting both direct and cloud providers';
COMMENT ON COLUMN user_llm_entitlements.entitlement_id IS 'Unique identifier for the entitlement record';
COMMENT ON COLUMN user_llm_entitlements.user_id IS 'User who has the entitlement (references users.user_id)';
COMMENT ON COLUMN user_llm_entitlements.llm_provider IS 'LLM provider type (e.g., openai, anthropic, azure_openai) - all ProviderType enum values supported';
COMMENT ON COLUMN user_llm_entitlements.llm_model_name IS 'Logical model name (e.g., gpt-4o, claude-3-5-sonnet)';
COMMENT ON COLUMN user_llm_entitlements.api_key IS 'Encrypted API key for the LLM provider (use pgcrypto for at-rest encryption)';
COMMENT ON COLUMN user_llm_entitlements.api_endpoint_url IS 'Specific API endpoint URL (nullable for some providers)';
COMMENT ON COLUMN user_llm_entitlements.cloud_provider IS 'Cloud provider hosting the LLM (e.g., azure_openai, google_vertex, aws_bedrock)';
COMMENT ON COLUMN user_llm_entitlements.deployment_name IS 'Physical deployment identifier for cloud providers';
COMMENT ON COLUMN user_llm_entitlements.region IS 'Geographic region where the model is deployed';
COMMENT ON COLUMN user_llm_entitlements.created_at IS 'When the entitlement was created';
COMMENT ON COLUMN user_llm_entitlements.updated_at IS 'When the entitlement was last updated';
COMMENT ON COLUMN user_llm_entitlements.created_by_user_id IS 'Admin user who created this entitlement';

-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Supports efficient querying for entitlement management and client resolution.
-- ============================================================================

-- Index for user-based lookups (most common query pattern)
CREATE INDEX idx_entitlements_user_id ON user_llm_entitlements(user_id);

-- Index for provider/model-based queries (useful for admin operations)
CREATE INDEX idx_entitlements_provider_model ON user_llm_entitlements(llm_provider, llm_model_name);

-- Composite index for efficient user/provider/model lookups
CREATE INDEX idx_entitlements_composite ON user_llm_entitlements(user_id, llm_provider, llm_model_name);

-- Index for cloud provider queries (useful for deployment management)
CREATE INDEX idx_entitlements_cloud_provider ON user_llm_entitlements(cloud_provider) WHERE cloud_provider IS NOT NULL;

-- Index for region-based queries (useful for geographic load balancing)
CREATE INDEX idx_entitlements_region ON user_llm_entitlements(region) WHERE region IS NOT NULL;

-- ============================================================================
-- TRIGGER - AUTO-UPDATE updated_at TIMESTAMP
-- Automatically updates the updated_at column on any row modification
-- ============================================================================
CREATE OR REPLACE FUNCTION update_entitlements_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_user_llm_entitlements_updated_at
    BEFORE UPDATE ON user_llm_entitlements
    FOR EACH ROW
    EXECUTE FUNCTION update_entitlements_updated_at_column();

COMMENT ON FUNCTION update_entitlements_updated_at_column() IS 'Automatically updates updated_at timestamp on entitlement modifications';

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

/*
-- Example 1: Direct provider entitlement (OpenAI)
INSERT INTO user_llm_entitlements
(user_id, llm_provider, llm_model_name, api_key, api_endpoint_url, created_by_user_id)
VALUES
('550e8400-e29b-41d4-a716-446655440000', 'openai', 'gpt-4o', 'encrypted_sk-...', 'https://api.openai.com/v1', 'admin-uuid-here');

-- Example 2: Cloud provider entitlement (Azure OpenAI)
INSERT INTO user_llm_entitlements
(user_id, llm_provider, llm_model_name, api_key, cloud_provider, deployment_name, api_endpoint_url, region, created_by_user_id)
VALUES
('550e8400-e29b-41d4-a716-446655440000', 'openai', 'gpt-4o', 'encrypted_azure_key', 'azure_openai', 'gpt4o-eastus-prod', 'https://my-resource.openai.azure.com/', 'eastus', 'admin-uuid-here');

-- Example 3: AWS Bedrock entitlement
INSERT INTO user_llm_entitlements
(user_id, llm_provider, llm_model_name, api_key, cloud_provider, region, created_by_user_id)
VALUES
('550e8400-e29b-41d4-a716-446655440000', 'anthropic', 'claude-3-5-sonnet-20240620', 'encrypted_aws_key', 'aws_bedrock', 'us-west-2', 'admin-uuid-here');

-- Example 4: Query user entitlements
SELECT
    u.username,
    ule.llm_provider,
    ule.llm_model_name,
    ule.cloud_provider,
    ule.deployment_name,
    ule.region
FROM user_llm_entitlements ule
JOIN users u ON ule.user_id = u.user_id
WHERE ule.user_id = '550e8400-e29b-41d4-a716-446655440000';

-- Example 5: Query entitlements by provider (all supported providers)
SELECT
    u.username,
    ule.llm_model_name,
    ule.api_endpoint_url
FROM user_llm_entitlements ule
JOIN users u ON ule.user_id = u.user_id
WHERE ule.llm_provider = 'openai'
  AND ule.cloud_provider IS NULL;  -- Direct provider only
*/
