-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS llm_models CASCADE;
-- ============================================================================
-- LLM MODELS TABLE - CATALOG OF AVAILABLE LLM MODELS
-- Represents available LLM models with their configurations and usage metrics.
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_models (
    provider_name TEXT NOT NULL DEFAULT 'openai'
        CHECK (provider IN ('openai', 'gemini', 'anthropic', 'mistral', 'cohere', 'xai', 'deepseek', 'meta')),
    llm_model_name TEXT NOT NULL,
    deployment_name TEXT,
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
    PRIMARY KEY (provider_name, llm_model_name, llm_model_version)
);
COMMENT ON TABLE llm_models IS 'Catalog of LLM models, tracking configurations and usage metrics';
COMMENT ON COLUMN llm_models.provider_name IS 'LLM provider name (e.g., openai, gemini, anthropic)';
COMMENT ON COLUMN llm_models.llm_model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN llm_models.deployment_name IS 'Name of the LLM deployment (e.g., gpt-4o)';
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
