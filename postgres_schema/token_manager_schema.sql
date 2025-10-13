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
COMMENT ON COLUMN token_manager.seed IS 'Frequency penalty for the LLM';
-- ============================================================================
-- INDEXES - OPTIMIZED FOR PERFORMANCE
-- Supports efficient querying for token allocation lifecycle management.
-- ============================================================================
CREATE INDEX idx_token_expiry_status_model ON token_manager(expires_at, allocation_status, model_name);
CREATE INDEX idx_token_expiry_status_model_endpoint ON token_manager(expires_at, allocation_status, model_name, api_endpoint);
CREATE INDEX idx_token_model ON token_manager(model_name);
CREATE INDEX idx_token_model_endpoint ON token_manager(model_name, api_endpoint);
