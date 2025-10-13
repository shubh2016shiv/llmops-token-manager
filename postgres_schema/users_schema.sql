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
