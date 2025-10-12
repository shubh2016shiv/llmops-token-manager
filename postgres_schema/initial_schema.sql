-- ============================================================================
-- Token Manager Service Database Schema
-- Acts as the central "Front Desk Manager" for LLM token allocations, ensuring fair
-- resource distribution, cost control, and regional resilience, akin to an arcade
-- ticket system where tokens are allocated for game access.
-- ============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop tables if they exist (for clean slate)
DROP TABLE IF EXISTS token_manager CASCADE;
DROP TABLE IF EXISTS llm_models CASCADE;
DROP TABLE IF EXISTS llm_providers CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ============================================================================
-- LLM PROVIDERS TABLE - CATALOG OF LLM PROVIDERS
-- Represents arcade game manufacturers (OpenAI, Anthropic, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_providers (
    -- Identity: Unique identifier for each provider
    provider_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider_name TEXT NOT NULL UNIQUE,
    
    -- Provider details
    provider_type TEXT NOT NULL CHECK (provider_type IN ('openai', 'azure_openai', 'anthropic', 'google', 'custom')),
    base_url TEXT,
    
    -- Configuration
    is_active BOOLEAN NOT NULL DEFAULT true,
    default_api_version TEXT,
    
    -- Authentication
    auth_type TEXT NOT NULL DEFAULT 'api_key' CHECK (auth_type IN ('api_key', 'oauth', 'custom')),
    auth_config JSONB,
    
    -- Rate limiting
    rate_limit_requests INTEGER,
    rate_limit_tokens INTEGER,
    rate_limit_window_seconds INTEGER DEFAULT 60,
    
    -- Audit trail
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE llm_providers IS 'Catalog of LLM providers (arcade game manufacturers) with their configurations';
COMMENT ON COLUMN llm_providers.provider_id IS 'Unique identifier for the provider';
COMMENT ON COLUMN llm_providers.provider_name IS 'Display name of the provider (e.g., OpenAI, Anthropic)';
COMMENT ON COLUMN llm_providers.provider_type IS 'Type of provider for integration logic';
COMMENT ON COLUMN llm_providers.base_url IS 'Base URL for API requests';
COMMENT ON COLUMN llm_providers.is_active IS 'Whether the provider is currently active';
COMMENT ON COLUMN llm_providers.default_api_version IS 'Default API version to use';
COMMENT ON COLUMN llm_providers.auth_type IS 'Authentication method used by the provider';
COMMENT ON COLUMN llm_providers.auth_config IS 'JSON configuration for authentication';
COMMENT ON COLUMN llm_providers.rate_limit_requests IS 'Maximum requests per window';
COMMENT ON COLUMN llm_providers.rate_limit_tokens IS 'Maximum tokens per window';

-- ============================================================================
-- USERS TABLE - MANAGES USER IDENTITIES AND QUOTA TRACKING
-- Represents arcade players, tracking their identities, authentication, and token quotas.
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    -- Identity: Unique identifier for each player (user)
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,

    -- Authentication: How players verify their identity
    auth_provider TEXT NOT NULL,
    external_user_id TEXT,
    password_hash TEXT,

    -- Authorization & Access Control: Defines player roles and status
    role TEXT NOT NULL DEFAULT 'developer'
        CHECK (role IN ('owner', 'admin', 'developer', 'viewer')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'suspended', 'inactive')),

    -- Subscription & Billing: Tracks subscription plans for access levels
    subscription_plan TEXT NOT NULL DEFAULT 'enterprise'
        CHECK (subscription_plan IN ('free', 'professional', 'enterprise')),

    -- Token Quota Management: Tracks daily and monthly token usage limits
    daily_token_limit INTEGER NOT NULL DEFAULT 100000 CHECK (daily_token_limit >= 0),
    monthly_token_limit INTEGER NOT NULL DEFAULT 3000000 CHECK (monthly_token_limit >= 0),
    daily_tokens_used INTEGER NOT NULL DEFAULT 0,
    monthly_tokens_used INTEGER NOT NULL DEFAULT 0,
    total_tokens_used BIGINT NOT NULL DEFAULT 0,

    -- Request Statistics: Tracks total API requests made by the user
    total_requests INTEGER NOT NULL DEFAULT 0,

    -- Quota Reset Tracking: Manages automatic reset of token quotas
    daily_quota_reset_at TIMESTAMPTZ NOT NULL DEFAULT (CURRENT_TIMESTAMP + INTERVAL '1 day'),
    monthly_quota_reset_at TIMESTAMPTZ NOT NULL DEFAULT (CURRENT_TIMESTAMP + INTERVAL '1 month'),

    -- Audit Trail: Tracks user creation, updates, and login activity
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ,

    UNIQUE(auth_provider, external_user_id)
);

COMMENT ON TABLE users IS 'Manages arcade players (users) with authentication, roles, and token quota tracking to prevent cost overruns';
COMMENT ON COLUMN users.user_id IS 'Unique identifier for the user (player)';
COMMENT ON COLUMN users.email IS 'Unique email address for user identification';
COMMENT ON COLUMN users.display_name IS 'User-friendly name for display purposes';
COMMENT ON COLUMN users.auth_provider IS 'Authentication provider (e.g., OAuth, internal)';
COMMENT ON COLUMN users.role IS 'User role: owner, admin, developer, or viewer';
COMMENT ON COLUMN users.status IS 'User status: active, suspended, or inactive';
COMMENT ON COLUMN users.subscription_plan IS 'Subscription tier: free, professional, or enterprise';
COMMENT ON COLUMN users.daily_token_limit IS 'Maximum tokens the user can consume per day, enforced by the Token Manager';
COMMENT ON COLUMN users.daily_tokens_used IS 'Tokens consumed today, reset at daily_quota_reset_at';
COMMENT ON COLUMN users.monthly_token_limit IS 'Maximum tokens the user can consume per month';
COMMENT ON COLUMN users.monthly_tokens_used IS 'Tokens consumed this month, reset at monthly_quota_reset_at';
COMMENT ON COLUMN users.total_tokens_used IS 'Lifetime tokens consumed by the user';
COMMENT ON COLUMN users.total_requests IS 'Lifetime API request count, incremented on token allocation';
COMMENT ON COLUMN users.daily_quota_reset_at IS 'Timestamp for next daily quota reset';
COMMENT ON COLUMN users.monthly_quota_reset_at IS 'Timestamp for next monthly quota reset';
COMMENT ON COLUMN users.last_login_at IS 'Timestamp of the user''s last login';

-- ============================================================================
-- LLM MODELS TABLE - CATALOG OF AVAILABLE LLM MODELS
-- Represents arcade games (LLMs) with their configurations and performance metrics.
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_models (
    -- Identity: Unique identifier for each LLM model
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider_id UUID NOT NULL REFERENCES llm_providers(provider_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT,

    -- Model Specifications: Defines model capabilities and limits
    max_tokens INTEGER,
    tokens_per_minute_limit INTEGER,
    requests_per_minute_limit INTEGER,

    -- Configuration: Indicates if the model is available for use
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Performance Metrics: Tracks usage statistics for the model
    total_requests BIGINT NOT NULL DEFAULT 0,
    total_tokens_processed BIGINT NOT NULL DEFAULT 0,

    -- Audit Trail: Tracks model creation, updates, and last usage
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ,

    UNIQUE(provider_id, model_name, model_version)
);

COMMENT ON TABLE llm_models IS 'Catalog of LLM models (arcade games), tracking configurations and usage metrics';
COMMENT ON COLUMN llm_models.model_id IS 'Unique identifier for the LLM model';
COMMENT ON COLUMN llm_models.provider_id IS 'Reference to the LLM provider';
COMMENT ON COLUMN llm_models.model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN llm_models.model_version IS 'Specific version of the model';
COMMENT ON COLUMN llm_models.max_tokens IS 'Maximum tokens the model can process in a single request';
COMMENT ON COLUMN llm_models.tokens_per_minute_limit IS 'Token rate limit per minute';
COMMENT ON COLUMN llm_models.requests_per_minute_limit IS 'Request rate limit per minute';
COMMENT ON COLUMN llm_models.is_active IS 'Indicates if the model is available for use';
COMMENT ON COLUMN llm_models.total_requests IS 'Total requests processed by the model';
COMMENT ON COLUMN llm_models.total_tokens_processed IS 'Total tokens processed by the model';
COMMENT ON COLUMN llm_models.last_used_at IS 'Timestamp of the last request to this model';

-- ============================================================================
-- TOKEN MANAGER TABLE - MANAGES TOKEN ALLOCATION REQUESTS
-- Acts as the arcade's front desk, issuing reservation stubs (token_request_id)
-- and directing players to optimal machines (regions/endpoints).
-- ============================================================================
CREATE TABLE IF NOT EXISTS token_manager (
    -- Request Identity: Unique reservation stub for the token allocation
    token_request_id TEXT PRIMARY KEY,

    -- User Reference: Links to the player (user) requesting tokens
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
    completed_at TIMESTAMPTZ,

    -- Performance Tracking: Measures request performance
    latency_ms INTEGER,

    -- Request Context: Stores additional metadata (e.g., team, application)
    request_context JSONB,

    -- Audit Trail: Tracks allocation creation and updates
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE token_manager IS 'Central gateway for token allocations, ensuring fair usage, cost control, and regional resilience';
COMMENT ON COLUMN token_manager.token_request_id IS 'Unique identifier for the token allocation request (reservation stub)';
COMMENT ON COLUMN token_manager.user_id IS 'User who requested the allocation, used for quota tracking and billing';
COMMENT ON COLUMN token_manager.model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN token_manager.model_id IS 'Reference to the specific LLM model in llm_models';
COMMENT ON COLUMN token_manager.deployment_name IS 'Specific deployment of the model, if applicable';
COMMENT ON COLUMN token_manager.cloud_provider IS 'Cloud provider hosting the LLM (e.g., Azure, AWS)';
COMMENT ON COLUMN token_manager.api_endpoint IS 'API endpoint for the selected LLM instance';
COMMENT ON COLUMN token_manager.region IS 'Geographic region of the LLM instance (e.g., eastus2, westus2)';
COMMENT ON COLUMN token_manager.token_count IS 'Number of tokens allocated for this request';
COMMENT ON COLUMN token_manager.allocation_status IS 'Current status: ACQUIRED, RELEASED, EXPIRED, PAUSED, or FAILED';
COMMENT ON COLUMN token_manager.allocated_at IS 'Timestamp when tokens were allocated';
COMMENT ON COLUMN token_manager.expires_at IS 'Optional expiration time for the allocation lock';
COMMENT ON COLUMN token_manager.completed_at IS 'Timestamp when the request was completed';
COMMENT ON COLUMN token_manager.latency_ms IS 'Request duration in milliseconds (completed_at - allocated_at)';
COMMENT ON COLUMN token_manager.request_context IS 'Additional context data in JSON format (e.g., team, application)';

-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Enhances query efficiency for common access patterns in the Token Manager Service.
-- ============================================================================

-- Users: Indexes for quota tracking and user activity
CREATE INDEX idx_users_daily_usage ON users(daily_tokens_used) WHERE status = 'active';
CREATE INDEX idx_users_quota_exceeded ON users(user_id)
    WHERE daily_tokens_used >= daily_token_limit OR monthly_tokens_used >= monthly_token_limit;
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_auth ON users(auth_provider, external_user_id);
CREATE INDEX idx_users_status ON users(status);

-- LLM Providers: Indexes for provider lookup
CREATE INDEX idx_providers_name ON llm_providers(provider_name);
CREATE INDEX idx_providers_type ON llm_providers(provider_type) WHERE is_active = true;

-- LLM Models: Indexes for active models and usage tracking
CREATE INDEX idx_models_provider ON llm_models(provider_id);
CREATE INDEX idx_models_name_version ON llm_models(model_name, model_version);
CREATE INDEX idx_models_last_used ON llm_models(last_used_at DESC) WHERE is_active = true;

-- Token Manager: Indexes for user-based queries and allocation status
CREATE INDEX idx_token_manager_by_user ON token_manager(user_id);
CREATE INDEX idx_token_manager_user_status ON token_manager(user_id, allocation_status);
CREATE INDEX idx_token_manager_by_model ON token_manager(model_name);
CREATE INDEX idx_token_manager_by_status ON token_manager(allocation_status);
CREATE INDEX idx_token_manager_by_time ON token_manager(allocated_at);
CREATE INDEX idx_active_allocations_by_model ON token_manager(model_name, allocation_status, token_count)
    WHERE allocation_status IN ('ACQUIRED', 'PAUSED');
CREATE INDEX idx_active_allocations_by_deployment ON token_manager(deployment_name, allocation_status, token_count)
    WHERE allocation_status IN ('ACQUIRED', 'PAUSED') AND deployment_name IS NOT NULL;
CREATE INDEX idx_expiring_allocations ON token_manager(expires_at)
    WHERE expires_at IS NOT NULL AND allocation_status IN ('ACQUIRED', 'PAUSED');
CREATE INDEX idx_token_manager_by_provider_region ON token_manager(cloud_provider, region, model_name, allocation_status)
    WHERE cloud_provider IS NOT NULL AND region IS NOT NULL AND allocation_status IN ('ACQUIRED', 'PAUSED');

-- ============================================================================
-- TRIGGERS - AUTOMATE TIMESTAMP UPDATES
-- Ensures updated_at is automatically set on record modifications.
-- ============================================================================
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_timestamp_trigger
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_providers_timestamp_trigger
BEFORE UPDATE ON llm_providers
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_models_timestamp_trigger
BEFORE UPDATE ON llm_models
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_token_manager_timestamp_trigger
BEFORE UPDATE ON token_manager
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- ============================================================================
-- HELPER FUNCTION: Check User Quota Availability
-- Verifies if a user has sufficient daily and monthly tokens for a request.
-- ============================================================================
CREATE OR REPLACE FUNCTION check_user_quota(
    p_user_id UUID,
    p_token_count INTEGER
) RETURNS BOOLEAN AS $$
DECLARE
    v_daily_available INTEGER;
    v_monthly_available INTEGER;
BEGIN
    SELECT daily_token_limit - daily_tokens_used,
           monthly_token_limit - monthly_tokens_used
    INTO v_daily_available, v_monthly_available
    FROM users
    WHERE user_id = p_user_id AND status = 'active';

    RETURN (v_daily_available >= p_token_count AND v_monthly_available >= p_token_count);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- HELPER FUNCTION: Reset Daily Quotas
-- Resets daily token usage for users when the reset timestamp is reached.
-- ============================================================================
CREATE OR REPLACE FUNCTION reset_daily_quotas()
RETURNS void AS $$
BEGIN
    UPDATE users
    SET daily_tokens_used = 0,
        daily_quota_reset_at = CURRENT_TIMESTAMP + INTERVAL '1 day'
    WHERE daily_quota_reset_at <= CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- HELPER FUNCTION: Reset Monthly Quotas
-- Resets monthly token usage for users when the reset timestamp is reached.
-- ============================================================================
CREATE OR REPLACE FUNCTION reset_monthly_quotas()
RETURNS void AS $$
BEGIN
    UPDATE users
    SET monthly_tokens_used = 0,
        monthly_quota_reset_at = CURRENT_TIMESTAMP + INTERVAL '1 month'
    WHERE monthly_quota_reset_at <= CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA - INITIAL RECORDS FOR TESTING
-- Provides basic data for development and testing purposes.
-- ============================================================================

-- Sample LLM Providers
INSERT INTO llm_providers (provider_name, provider_type, base_url, auth_type, auth_config, rate_limit_requests, rate_limit_tokens)
VALUES
    ('OpenAI', 'openai', 'https://api.openai.com/v1', 'api_key', '{"key_name": "api_key"}', 3000, 250000),
    ('Azure OpenAI', 'azure_openai', NULL, 'api_key', '{"key_name": "api-key", "requires_endpoint": true}', 5000, 500000),
    ('Anthropic', 'anthropic', 'https://api.anthropic.com', 'api_key', '{"key_name": "x-api-key"}', 2000, 200000),
    ('Google Gemini', 'google', 'https://generativelanguage.googleapis.com', 'api_key', '{"key_name": "key"}', 4000, 300000);

-- Sample Users
INSERT INTO users (email, display_name, auth_provider, role, subscription_plan, daily_token_limit, monthly_token_limit)
VALUES
    ('admin@example.com', 'Admin User', 'internal', 'admin', 'enterprise', 1000000, 30000000),
    ('developer@example.com', 'Developer User', 'internal', 'developer', 'professional', 500000, 15000000),
    ('viewer@example.com', 'Viewer User', 'internal', 'viewer', 'free', 100000, 3000000);

-- Sample LLM Models (after providers are created)
INSERT INTO llm_models (provider_id, model_name, model_version, max_tokens, tokens_per_minute_limit, requests_per_minute_limit)
VALUES
    ((SELECT provider_id FROM llm_providers WHERE provider_name = 'OpenAI'), 'gpt-4', 'turbo', 128000, 100000, 500),
    ((SELECT provider_id FROM llm_providers WHERE provider_name = 'OpenAI'), 'gpt-3.5-turbo', '0125', 16385, 200000, 1000),
    ((SELECT provider_id FROM llm_providers WHERE provider_name = 'Azure OpenAI'), 'gpt-4', 'turbo', 128000, 100000, 500),
    ((SELECT provider_id FROM llm_providers WHERE provider_name = 'Anthropic'), 'claude-3', 'opus', 200000, 80000, 400),
    ((SELECT provider_id FROM llm_providers WHERE provider_name = 'Google Gemini'), 'gemini-1.5', 'pro', 1000000, 150000, 600);

-- Sample Token Allocations
INSERT INTO token_manager (
    token_request_id,
    user_id,
    model_name,
    model_id,
    deployment_name,
    cloud_provider,
    api_endpoint,
    region,
    token_count,
    allocation_status,
    allocated_at,
    expires_at,
    request_context
) VALUES 
(
    'req_01HQWXYZ123456789',
    (SELECT user_id FROM users WHERE email = 'developer@example.com'),
    'gpt-4',
    (SELECT model_id FROM llm_models WHERE model_name = 'gpt-4' AND model_version = 'turbo' LIMIT 1),
    'gpt4-prod-01',
    'azure',
    'https://eastus2.api.cognitive.microsoft.com/',
    'eastus2',
    1000,
    'ACQUIRED',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP + INTERVAL '1 hour',
    '{"application": "customer-service-bot", "team": "support", "priority": "high"}'
),
(
    'req_01HQWXYZ987654321',
    (SELECT user_id FROM users WHERE email = 'viewer@example.com'),
    'gpt-3.5-turbo',
    (SELECT model_id FROM llm_models WHERE model_name = 'gpt-3.5-turbo' LIMIT 1),
    'gpt35-prod-02',
    'azure',
    'https://westus2.api.cognitive.microsoft.com/',
    'westus2',
    500,
    'RELEASED',
    CURRENT_TIMESTAMP - INTERVAL '30 minutes',
    NULL,
    '{"application": "content-generator", "team": "marketing", "priority": "medium"}'
),
(
    'req_01HQWXYZ456789123',
    (SELECT user_id FROM users WHERE email = 'admin@example.com'),
    'claude-3',
    (SELECT model_id FROM llm_models WHERE model_name = 'claude-3' LIMIT 1),
    NULL,
    'anthropic',
    'https://api.anthropic.com/v1',
    NULL,
    2000,
    'ACQUIRED',
    CURRENT_TIMESTAMP - INTERVAL '5 minutes',
    CURRENT_TIMESTAMP + INTERVAL '2 hours',
    '{"application": "research-assistant", "team": "data-science", "priority": "high"}'
);