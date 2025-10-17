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
