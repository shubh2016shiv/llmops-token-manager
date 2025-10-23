----      Central identity management      ----


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

    -- User Information
    username VARCHAR(50) NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,

    -- Authentication
    password_hash VARCHAR(255) NOT NULL,

    -- Authorization & Access Control: Defines user roles and status
    role TEXT NOT NULL DEFAULT 'developer'
        CHECK (role IN ('owner', 'admin', 'developer', 'viewer', 'user', 'operator')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'suspended', 'inactive')),

    -- Audit Trail: Tracks user creation and updates
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE users IS 'Manages users with authentication and roles for LLM token allocation';
COMMENT ON COLUMN users.user_id IS 'Unique identifier for the user';
COMMENT ON COLUMN users.username IS 'Unique username for login';
COMMENT ON COLUMN users.email IS 'Unique email address for user identification';
COMMENT ON COLUMN users.first_name IS 'User''s first name';
COMMENT ON COLUMN users.last_name IS 'User''s last name';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hashed password';
COMMENT ON COLUMN users.role IS 'User role: owner, admin, developer, viewer, user, or operator';
COMMENT ON COLUMN users.status IS 'User status: active, suspended, or inactive';
COMMENT ON COLUMN users.created_at IS 'Timestamp of the user''s creation';
COMMENT ON COLUMN users.updated_at IS 'Timestamp of the user''s last update';

-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Enhances query efficiency for common access patterns in user management.
-- ============================================================================
CREATE INDEX idx_users_user_id_created_at ON users(user_id, created_at);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_role ON users(role);

-- ============================================================================
-- TRIGGER - AUTO-UPDATE updated_at TIMESTAMP
-- Automatically updates the updated_at column on any row modification
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

COMMENT ON FUNCTION update_updated_at_column() IS 'Automatically updates updated_at timestamp on row modifications';


----      LLM Model catalog with Configurations      ----


-- Drop table if it exists (for clean slate)
DROP TABLE IF EXISTS llm_models CASCADE;
-- ============================================================================
-- LLM MODELS TABLE - CATALOG OF AVAILABLE LLM MODELS
-- Represents available LLM models with their configurations and usage metrics.
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_models (
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
    llm_model_name TEXT NOT NULL,
    deployment_name TEXT,
    cloud_provider TEXT CHECK (
        cloud_provider IN (
            'Azure',
            'Google Cloud Platform',
            'Amazon Web Services',
            'IBM Watsonx',
            'Oracle',
            'On Premise')),
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

-- ============================================================================
-- TRIGGER - AUTO-UPDATE updated_at TIMESTAMP
-- Automatically updates the updated_at column on any row modification
-- ============================================================================
CREATE OR REPLACE FUNCTION update_llm_models_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_llm_models_updated_at
BEFORE UPDATE ON llm_models
FOR EACH ROW
EXECUTE FUNCTION update_llm_models_updated_at_column();

COMMENT ON FUNCTION update_llm_models_updated_at_column() IS 'Automatically updates updated_at timestamp on LLM model modifications';
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


----      User LLM Entitlements for LLM Token Allocation and Release     ----


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
    llm_model_name TEXT NOT NULL,

    -- Configurations: API and deployment details for client init
    api_key_variable_name TEXT,  -- Environment variable name for API key
    api_key_value TEXT NOT NULL,  -- Encrypted API key value (use pgcrypto for encryption)
    api_endpoint_url TEXT,  -- Specific endpoint URL (nullable for some providers)
    cloud_provider TEXT CHECK (
        cloud_provider IN (
            'Azure',
            'Google Cloud Platform',
            'Amazon Web Services',
            'IBM Watsonx',
            'Oracle',
            'On Premise')),
    deployment_name TEXT,   -- Physical deployment identifier (e.g., 'gpt4o-eastus-prod')
    deployment_region TEXT,            -- Geographic deployment_region (e.g., 'eastus', 'us-west-2')

    -- Audit Trail: Tracks creation, updates, and admin actions
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Foreign Key: Ensures the LLM model exists in the catalog (optional - remove if not needed)
    -- Note: This FK can be removed since we're using application-level validation
    FOREIGN KEY (llm_provider, llm_model_name)
        REFERENCES llm_models(llm_provider, llm_model_name)
        ON DELETE CASCADE,

    -- Unique Constraint: Ensures unique entitlements per user/cloud_provider/provider/llm_model/endpoint
    UNIQUE (user_id, cloud_provider, llm_provider, llm_model_name, api_endpoint_url, api_key_variable_name)
);

-- ============================================================================
-- COMMENTS - DOCUMENTATION FOR EACH COLUMN
-- ============================================================================
COMMENT ON TABLE user_llm_entitlements IS 'Maps users to their entitled LLM configurations with API key storage, supporting both direct and cloud providers';
COMMENT ON COLUMN user_llm_entitlements.entitlement_id IS 'Unique identifier for the entitlement record';
COMMENT ON COLUMN user_llm_entitlements.user_id IS 'User who has the entitlement (references users.user_id)';
COMMENT ON COLUMN user_llm_entitlements.llm_provider IS 'LLM provider type (e.g., openai, anthropic, gemini) - all ProviderType enum values supported';
COMMENT ON COLUMN user_llm_entitlements.llm_model_name IS 'Logical model name (e.g., gpt-4o, claude-3-5-sonnet)';
COMMENT ON COLUMN user_llm_entitlements.api_key_variable_name IS 'Environment variable name for API key';
COMMENT ON COLUMN user_llm_entitlements.api_key_value IS 'Encrypted API key for the LLM provider (use pgcrypto for at-rest encryption)';
COMMENT ON COLUMN user_llm_entitlements.api_endpoint_url IS 'Specific API endpoint URL (nullable for some providers)';
COMMENT ON COLUMN user_llm_entitlements.cloud_provider IS 'Cloud provider hosting the LLM (e.g., Azure, Google Cloud Platform, Amazon Web Services)';
COMMENT ON COLUMN user_llm_entitlements.deployment_name IS 'Physical deployment identifier for cloud providers';
COMMENT ON COLUMN user_llm_entitlements.deployment_region IS 'Region where the model is deployed on the cloud provider';
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
CREATE INDEX idx_entitlements_region ON user_llm_entitlements(deployment_region) WHERE deployment_region IS NOT NULL;

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


----      Token Manager for LLM Token Allocation and Release     ----


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
    CONSTRAINT chk_pause_must_expire CHECK (allocation_status != 'PAUSED' OR expires_at IS NOT NULL),
    CONSTRAINT fk_token_manager_llm_models
    FOREIGN KEY (llm_provider, llm_model_name)
    REFERENCES llm_models(llm_provider, llm_model_name)
    ON DELETE CASCADE
);
COMMENT ON TABLE token_manager IS 'Central gateway for token allocations, ensuring fair usage, cost control, and regional resilience';
COMMENT ON COLUMN token_manager.token_request_id IS 'Unique identifier for the token allocation request';
COMMENT ON COLUMN token_manager.llm_model_name IS 'Name of the LLM model (e.g., GPT-4)';
COMMENT ON COLUMN token_manager.llm_provider IS 'Provider of the LLM model (e.g., openai, anthropic, gemini)';
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
