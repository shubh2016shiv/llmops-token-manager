"""
Rate Limiter Tests
==================
Unit tests for the rate limiter integration using in-memory storage.

These tests avoid Redis and validate:
- under-limit requests are allowed
- over-limit requests return 429 + Retry-After
- login keying differentiates by username (same IP)
"""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from app.core.rate_limiter import (
    auth_login_rate_limiter,
    get_rate_limit_storage,
    get_rate_limiter,
    register_rate_limit_exception_handler,
)


@pytest.fixture(autouse=True)
def _use_memory_storage(monkeypatch):
    # Force in-memory rate limiting for deterministic tests.
    monkeypatch.setenv("RATE_LIMIT_STORAGE", "memory")
    # Clear caches between tests to avoid cross-test interference.
    get_rate_limit_storage.cache_clear()
    get_rate_limiter.cache_clear()
    yield
    get_rate_limit_storage.cache_clear()
    get_rate_limiter.cache_clear()


@pytest.fixture
def app():
    app = FastAPI()
    register_rate_limit_exception_handler(app)

    @app.post("/login-protected", dependencies=[Depends(auth_login_rate_limiter())])
    async def login_protected(payload: dict):
        return {"ok": True, "username": payload.get("username")}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_under_limit_allows_requests(client):
    # Default settings: 10/minute, so 2 should be allowed.
    for _ in range(2):
        r = client.post(
            "/login-protected",
            json={"username": "alice", "password": "x"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        )
        assert r.status_code == 200


def test_over_limit_returns_429_and_retry_after(client):
    # Hit > 10 times for same IP+username.
    for _ in range(12):
        r = client.post(
            "/login-protected",
            json={"username": "alice", "password": "x"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        )
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert r.headers["Retry-After"].isdigit()
    body = r.json()
    assert body["error"] == "RATE_LIMITED"


def test_same_ip_different_username_has_separate_bucket(client):
    # Exhaust for alice
    for _ in range(11):
        client.post(
            "/login-protected",
            json={"username": "alice", "password": "x"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        )

    # Bob should still be allowed from same IP (different key)
    r = client.post(
        "/login-protected",
        json={"username": "bob", "password": "x"},
        headers={"X-Forwarded-For": "1.2.3.4"},
    )
    assert r.status_code == 200
