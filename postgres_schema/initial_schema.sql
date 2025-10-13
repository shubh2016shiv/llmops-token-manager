-- ########################################################
-- USERS TABLE
-- ########################################################

-- Enable the uuid-ossp extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS users CASCADE;
-- ============================================================================
-- USERS TABLE - MANAGES USER IDENTITIES AND QUOTA TRACKING
-- Represents users, tracking their identities, authentication, roles, and token quotas for LLM invoke requests.
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    -- Identity: Unique identifier for each user
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL UNIQUE,
    -- Authorization & Access Control: Defines user roles and status
    role TEXT NOT NULL DEFAULT 'developer'
        CHECK (role IN ('owner', 'admin', 'developer', 'viewer')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'suspended', 'inactive')),
    -- Audit Trail: Tracks user creation and updates
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE users IS 'Manages users with authentication and roles for LLM token allocation';
COMMENT ON COLUMN users.user_id IS 'Unique identifier for the user';
COMMENT ON COLUMN users.email IS 'Unique email address for user identification';
COMMENT ON COLUMN users.role IS 'User role: owner, admin, developer, or viewer';
COMMENT ON COLUMN users.status IS 'User status: active, suspended, or inactive';
COMMENT ON COLUMN users.created_at IS 'Timestamp of the user''s creation';
COMMENT ON COLUMN users.updated_at IS 'Timestamp of the user''s last update';
-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Enhances query efficiency for common access patterns in user management.
-- ============================================================================
CREATE INDEX idx_users_user_id_created_at ON users(user_id, created_at);
CREATE INDEX idx_users_email ON users(email);

-- ########################################################
-- LLM MODELS TABLE
-- ########################################################
-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS llm_models CASCADE;
-- ============================================================================
-- LLM MODELS TABLE - CATALOG OF AVAILABLE LLM MODELS
-- Represents available LLM models with their configurations and usage metrics.
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_models (
    -- Identity: Unique identifier for each LLM model
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider TEXT NOT NULL DEFAULT 'openai'
        CHECK (provider IN ('openai', 'gemini', 'anthropic')),
    model_name TEXT NOT NULL,
    deployment_name TEXT,
    api_key_vault_id TEXT,
    api_endpoint TEXT,
    model_version TEXT,
    -- Model Specifications: Defines model capabilities and limits
    max_tokens INTEGER,
    tokens_per_minute_limit INTEGER,
    requests_per_minute_limit INTEGER,
    -- Configuration: Indicates if the model is available for use
    is_active BOOLEAN NOT NULL DEFAULT true,
    temperature FLOAT,
    seed INTEGER,
    region TEXT,
    -- Performance Metrics: Tracks usage statistics for the model
    total_requests BIGINT NOT NULL DEFAULT 0,
    total_tokens_processed BIGINT NOT NULL DEFAULT 0,
    -- Audit Trail: Tracks model creation, updates, and last usage
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ,
    UNIQUE(provider, model_name, model_version)
);
COMMENT ON TABLE llm_models IS 'Catalog of LLM models, tracking configurations and usage metrics';
COMMENT ON COLUMN llm_models.model_id IS 'Unique identifier for the LLM model';
COMMENT ON COLUMN llm_models.provider IS 'LLM provider';
COMMENT ON COLUMN llm_models.model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN llm_models.deployment_name IS 'Name of the LLM deployment (e.g., gpt-4o)';
COMMENT ON COLUMN llm_models.api_key_vault_id IS 'Reference to the API key vault entry';
COMMENT ON COLUMN llm_models.api_endpoint IS 'API endpoint for the selected LLM instance, if applicable';
COMMENT ON COLUMN llm_models.model_version IS 'Specific version of the model';
COMMENT ON COLUMN llm_models.max_tokens IS 'Maximum tokens the model can process in a single request';
COMMENT ON COLUMN llm_models.tokens_per_minute_limit IS 'Token rate limit per minute';
COMMENT ON COLUMN llm_models.requests_per_minute_limit IS 'Request rate limit per minute';
COMMENT ON COLUMN llm_models.is_active IS 'Indicates if the model is available for use';
COMMENT ON COLUMN llm_models.temperature IS 'Temperature for the LLM model';
COMMENT ON COLUMN llm_models.seed IS 'Seed for the LLM model';
COMMENT ON COLUMN llm_models.region IS 'Geographic region of the LLM instance (e.g., eastus2, westus2)';
-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Supports efficient querying for model configurations and usage.
-- ============================================================================
-- Add index for faster lookups by model name + api_endpoint
CREATE INDEX IF NOT EXISTS idx_models_name_api_base ON llm_models(model_name, api_endpoint);

-- ########################################################
-- TOKEN MANAGER TABLE
-- ########################################################
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
    model_name TEXT NOT NULL,
    model_id UUID REFERENCES llm_models(model_id) ON DELETE SET NULL,
    deployment_name TEXT,
    cloud_provider TEXT,
    api_endpoint TEXT,
    region TEXT,
    -- Token Allocation Management: Tracks allocated tokens and their status
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    allocation_status TEXT NOT NULL DEFAULT 'ACQUIRED'
        CHECK (allocation_status IN ('ACQUIRED', 'RELEASED', 'EXPIRED', 'PAUSED', 'FAILED')),
    -- Timing & Expiration: Manages allocation lifecycle
    allocated_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ,
    request_context JSONB,
    temperature FLOAT,
    top_p FLOAT,
    seed FLOAT
);
COMMENT ON TABLE token_manager IS 'Central gateway for token allocations, ensuring fair usage, cost control, and regional resilience';
COMMENT ON COLUMN token_manager.token_request_id IS 'Unique identifier for the token allocation request';
COMMENT ON COLUMN token_manager.model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN token_manager.model_id IS 'Reference to the specific LLM model in llm_models';
COMMENT ON COLUMN token_manager.deployment_name IS 'Specific deployment of the model, if applicable';
COMMENT ON COLUMN token_manager.cloud_provider IS 'Cloud provider hosting the LLM (e.g., Azure, AWS), if applicable';
COMMENT ON COLUMN token_manager.api_endpoint IS 'API endpoint for the selected LLM instance, if applicable';
COMMENT ON COLUMN token_manager.region IS 'Geographic region of the LLM instance (e.g., eastus2, westus2), if applicable';
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
CREATE INDEX idx_token_expiry_status_model ON token_manager(expires_at, allocation_status, model_name);
CREATE INDEX idx_token_expiry_status_model_endpoint ON token_manager(expires_at, allocation_status, model_name, api_endpoint);
CREATE INDEX idx_token_model ON token_manager(model_name);
CREATE INDEX idx_token_model_endpoint ON token_manager(model_name, api_endpoint);
