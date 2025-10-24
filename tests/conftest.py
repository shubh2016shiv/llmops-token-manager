"""
Pytest configuration for LLM Token Manager tests.
Sets up the Python path and common test fixtures.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
import pytest

# Add the project root to Python path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up test environment variables
os.environ.setdefault("DATABASE_HOST", "localhost")
os.environ.setdefault("DATABASE_PORT", "5432")
os.environ.setdefault("DATABASE_NAME", "mydb")
os.environ.setdefault("DATABASE_USER", "myuser")
os.environ.setdefault("DATABASE_PASSWORD", "mypassword")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")

# ============================================================================
# AUTHENTICATION FIXTURES FOR TESTING
# ============================================================================


@pytest.fixture
def mock_developer_user():
    """
    Create mock developer user token payload.

    NOTE: Returns TokenPayload directly, not an actual JWT token.
    This is used with app.dependency_overrides to bypass real JWT validation.
    Best Practice: Testing secured endpoints should mock the dependency,
    not generate real tokens, for speed and isolation.
    """
    from app.models.auth_models import AuthTokenPayload

    return AuthTokenPayload(
        user_id=uuid4(),
        role="developer",
        expire_at_time=datetime.utcnow() + timedelta(hours=24),
        issued_at_time=datetime.utcnow(),
        type="access",
    )


@pytest.fixture
def mock_operator_user():
    """Mock operator user token payload for testing."""
    from app.models.auth_models import AuthTokenPayload

    return AuthTokenPayload(
        user_id=uuid4(),
        role="operator",
        expire_at_time=datetime.utcnow() + timedelta(hours=24),
        issued_at_time=datetime.utcnow(),
        type="access",
    )


@pytest.fixture
def mock_admin_user():
    """Mock admin user token payload for testing."""
    from app.models.auth_models import AuthTokenPayload

    return AuthTokenPayload(
        user_id=uuid4(),
        role="admin",
        expire_at_time=datetime.utcnow() + timedelta(hours=24),
        issued_at_time=datetime.utcnow(),
        type="access",
    )


@pytest.fixture
def mock_owner_user():
    """Mock owner user token payload for testing."""
    from app.models.auth_models import AuthTokenPayload

    return AuthTokenPayload(
        user_id=uuid4(),
        role="owner",
        expire_at_time=datetime.utcnow() + timedelta(hours=24),
        issued_at_time=datetime.utcnow(),
        type="access",
    )


@pytest.fixture
def override_get_current_user(app):
    """
    Factory fixture to override get_current_user dependency.

    Usage in tests:
        override_get_current_user(app, mock_developer_user)

    NOTE: This uses FastAPI's dependency_overrides feature.
    Best Practice: Override dependencies at the app level rather than
    mocking internal functions. This tests the actual dependency injection
    flow while controlling the authentication result.
    """
    from app.auth.auth_dependencies import get_current_user

    def _override(app, user_payload):
        app.dependency_overrides[get_current_user] = lambda: user_payload
        return app

    return _override
