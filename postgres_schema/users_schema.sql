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

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

/*
===============================================================================
USAGE EXAMPLES FOR USERS TABLE
===============================================================================

-- Example 1: Create a basic developer user (default role and status)
INSERT INTO users (
    username, email, first_name, last_name, password_hash
) VALUES (
    'johndoe',
    'john.doe@example.com',
    'John',
    'Doe',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da'
);

-- Example 2: Create an admin user with explicit role
INSERT INTO users (
    username, email, first_name, last_name,
    password_hash, role
) VALUES (
    'admin.smith',
    'admin.smith@company.com',
    'Admin',
    'Smith',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da',
    'admin'
);

-- Example 3: Create an owner user (highest privilege level)
INSERT INTO users (
    username, email, first_name, last_name,
    password_hash, role, status
) VALUES (
    'ceo.johnson',
    'ceo@company.com',
    'Sarah',
    'Johnson',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da',
    'owner',
    'active'
);

-- Example 4: Create a suspended user (for security or policy reasons)
INSERT INTO users (
    username, email, first_name, last_name,
    password_hash, role, status
) VALUES (
    'suspended.user',
    'suspended.user@company.com',
    'Suspended',
    'User',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da',
    'developer',
    'suspended'
);

-- Example 5: Create an operator user (for system operations)
INSERT INTO users (
    username, email, first_name, last_name,
    password_hash, role
) VALUES (
    'ops.wilson',
    'ops.wilson@company.com',
    'Michael',
    'Wilson',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da',
    'operator'
);

-- Example 6: Create multiple users in batch for team onboarding
INSERT INTO users (username, email, first_name, last_name, password_hash, role) VALUES
    ('alice.dev', 'alice@team.com', 'Alice', 'Developer', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da', 'developer'),
    ('bob.analyst', 'bob@team.com', 'Bob', 'Analyst', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da', 'viewer'),
    ('charlie.lead', 'charlie@team.com', 'Charlie', 'Lead', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fEtTT2/Da', 'admin');

-- Example 7: Query user by email address (common login lookup)
SELECT
    user_id,
    username,
    first_name,
    last_name,
    role,
    status,
    created_at
FROM users
WHERE email = 'john.doe@example.com';

-- Example 8: Query user by username (alternative login lookup)
SELECT
    user_id,
    email,
    first_name,
    last_name,
    role,
    status,
    updated_at
FROM users
WHERE username = 'johndoe';

-- Example 9: Find all active admin users (for authorization checks)
SELECT
    user_id,
    username,
    email,
    first_name,
    last_name,
    created_at
FROM users
WHERE role = 'admin'
  AND status = 'active'
ORDER BY created_at;

-- Example 10: Find all users by role (for role-based access control)
SELECT
    username,
    email,
    first_name,
    last_name,
    status,
    created_at
FROM users
WHERE role = 'developer'
ORDER BY last_name, first_name;

-- Example 11: Find suspended users (for security monitoring)
SELECT
    user_id,
    username,
    email,
    role,
    created_at,
    updated_at
FROM users
WHERE status = 'suspended'
ORDER BY updated_at DESC;

-- Example 12: Get user count by role (for reporting and analytics)
SELECT
    role,
    status,
    COUNT(*) as user_count
FROM users
GROUP BY role, status
ORDER BY role, status;

-- Example 13: Find recently created users (for onboarding tracking)
SELECT
    username,
    email,
    first_name,
    last_name,
    role,
    created_at
FROM users
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY created_at DESC;

-- Example 14: Update user role (promotion scenario)
UPDATE users
SET
    role = 'admin',
    updated_at = CURRENT_TIMESTAMP
WHERE username = 'alice.dev';

-- Example 15: Update user status (account suspension)
UPDATE users
SET
    status = 'suspended',
    updated_at = CURRENT_TIMESTAMP
WHERE email = 'suspended.user@company.com';

-- Example 16: Reactivate suspended user
UPDATE users
SET
    status = 'active',
    updated_at = CURRENT_TIMESTAMP
WHERE user_id = '550e8400-e29b-41d4-a716-446655440000';

-- Example 17: Bulk update user roles (team restructuring)
UPDATE users
SET
    role = 'operator',
    updated_at = CURRENT_TIMESTAMP
WHERE role = 'viewer'
  AND status = 'active';

-- Example 18: Find users by name pattern (for search functionality)
SELECT
    user_id,
    username,
    email,
    first_name,
    last_name,
    role,
    status
FROM users
WHERE first_name ILIKE '%john%'
   OR last_name ILIKE '%john%'
   OR username ILIKE '%john%'
ORDER BY last_name, first_name;

-- Example 19: Get user activity summary (for dashboard reporting)
SELECT
    status,
    role,
    COUNT(*) as count,
    MIN(created_at) as first_created,
    MAX(updated_at) as last_updated
FROM users
GROUP BY status, role
ORDER BY status, role;

-- Example 20: Find inactive users for cleanup (older than 1 year)
SELECT
    user_id,
    username,
    email,
    role,
    status,
    created_at,
    updated_at
FROM users
WHERE status = 'inactive'
  AND updated_at < CURRENT_DATE - INTERVAL '1 year'
ORDER BY updated_at;

===============================================================================
*/
