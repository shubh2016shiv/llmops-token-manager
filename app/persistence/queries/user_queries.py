"""
User persistence SQL definitions.

This module stores named SQL constants for static user-management queries.
Dynamic query builders remain in the persistence layer for later migration.
"""

CHECK_USER_EMAIL_EXISTS_SQL = """
    SELECT 1 FROM users WHERE email = :email LIMIT 1
"""

CHECK_USERNAME_EXISTS_SQL = """
    SELECT 1 FROM users WHERE username = :username LIMIT 1
"""

CREATE_USER_SQL = """
    INSERT INTO users (
        user_id, username, email, first_name, last_name,
        password_hash, role, status, created_at, updated_at
    )
    VALUES (
        :user_id, :username, :email, :first_name, :last_name,
        :password_hash, :role, :status, :created_at, :updated_at
    )
    RETURNING user_id, username, email, first_name, last_name,
              role, status, created_at, updated_at
"""

GET_USER_BY_ID_SQL = """
    SELECT * FROM users
    WHERE user_id = :user_id
"""

GET_USER_BY_EMAIL_SQL = """
    SELECT * FROM users
    WHERE email = :email
"""

GET_USER_BY_USERNAME_SQL = """
    SELECT * FROM users
    WHERE username = :username
"""

COUNT_USERS_BY_STATUS_SQL = """
    SELECT COUNT(*) FROM users
    WHERE status = :status
"""

DELETE_USER_BY_ID_SQL = """
    DELETE FROM users
    WHERE user_id = :user_id
"""

DELETE_USER_BY_EMAIL_SQL = """
    DELETE FROM users
    WHERE email = :email
"""
