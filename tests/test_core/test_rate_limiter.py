"""
Rate Limiter Tests
==================
Unit tests for app.core.redis_rate_limiter using in-memory storage.

These tests avoid Redis and validate:
- under-limit requests are allowed
- over-limit requests return 429 + Retry-After
- different client IPs get separate buckets (IP keying)
- RateLimiterManager lifecycle (fail-loud before init, idempotent init/close)
- get_client_ip's X-Forwarded-For trust-hop boundary
- untrusted service headers cannot mint fresh buckets
- transient Redis outages fail open; unexpected bugs raise
- the DSN builder's password URL-encoding (regression coverage for the
  SecretStr migration — a plain str slipping back in here would silently
  break auth against a Redis password containing '@', ':', or '/')
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from limits.aio.storage import RedisStorage
from pydantic import SecretStr
import pytest

from app.core.config import settings
from app.core.exceptions import RateLimitExceededError
from app.core.redis_rate_limiter import (
    auth_token_generate_rate_limiter,
    rate_limiter_manager,
    register_rate_limit_exception_handler,
)
from app.core.redis_rate_limiter import rate_limit_enforcement as enforcement_module
from app.core.redis_rate_limiter.moving_window_limiter import (
    RateLimiterManager,
    _redis_dsn,
)
from app.core.redis_rate_limiter.rate_limit_enforcement import (
    _retry_after_seconds,
    rate_limit_dependency,
)
from app.core.redis_rate_limiter.rate_limit_keys import get_client_ip, ip_only_key
from app.models.redis_rate_limit_models import RateLimitRule

# The IP-keyed limiter under test and its configured per-minute budget.
_LIMIT_PER_MINUTE = settings.rate_limit_token_generate_per_minute


@pytest.fixture(autouse=True)
def _use_memory_storage(monkeypatch):
    # Force in-memory rate limiting for deterministic tests.
    monkeypatch.setenv("RATE_LIMIT_STORAGE", "memory")
    # Re-initialize the lifecycle-managed limiter with a fresh in-memory
    # store per test (production wires this via the app lifespan instead).
    asyncio.run(rate_limiter_manager.close())
    rate_limiter_manager.initialize()
    yield
    asyncio.run(rate_limiter_manager.close())


@pytest.fixture
def app():
    app = FastAPI()
    register_rate_limit_exception_handler(app)

    @app.post("/protected", dependencies=[Depends(auth_token_generate_rate_limiter())])
    async def protected(payload: dict):
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_under_limit_allows_requests(client):
    # Two requests are well under the per-minute budget.
    for _ in range(2):
        r = client.post(
            "/protected",
            json={"data": "x"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        )
        assert r.status_code == 200


def test_over_limit_returns_429_and_retry_after(client):
    # Exceed the budget from a single IP.
    for _ in range(_LIMIT_PER_MINUTE + 2):
        r = client.post(
            "/protected",
            json={"data": "x"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        )
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert r.headers["Retry-After"].isdigit()
    body = r.json()
    assert body["error"] == "RATE_LIMITED"


def test_different_ip_has_separate_bucket(app):
    first_client = TestClient(app, client=("1.2.3.4", 50000))
    second_client = TestClient(app, client=("5.6.7.8", 50000))
    # Exhaust the budget for the first IP.
    for _ in range(_LIMIT_PER_MINUTE + 1):
        first_client.post(
            "/protected",
            json={"data": "x"},
            headers={"X-Forwarded-For": "1.2.3.4"},
        )

    # A different IP has its own bucket and is still allowed.
    r = second_client.post(
        "/protected",
        json={"data": "x"},
        headers={"X-Forwarded-For": "5.6.7.8"},
    )
    assert r.status_code == 200


def _make_request(
    headers: dict[str, str] | None = None, client_host: str | None = "10.0.0.9"
) -> Request:
    """
    Build a bare Starlette Request from an ASGI scope — no TestClient/app
    needed. get_client_ip and ip_only_key only read `.headers` and
    `.client`, so this is enough to unit test them directly.
    """
    raw_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    scope: dict[str, object] = {
        "type": "http",
        "headers": raw_headers,
        "client": (client_host, 12345) if client_host is not None else None,
    }
    return Request(scope)


class TestRateLimiterManagerLifecycle:
    """
    Lifecycle contract: fail loud before init, idempotent init/close.

    This is the regression surface for the bug the manager class replaced
    (`@lru_cache` binding a coredis pool to a dead event loop) — the actual
    event-loop-rebinding scenario needs a live Redis to reproduce, but the
    *contract* the manager promises (RuntimeError pre-init, safe repeated
    init/close) is fully testable without one.
    """

    def test_limiter_property_raises_before_initialize(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_STORAGE", "memory")
        manager = RateLimiterManager()

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.limiter

    def test_initialize_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_STORAGE", "memory")
        manager = RateLimiterManager()

        manager.initialize()
        first_limiter = manager.limiter
        manager.initialize()  # second call must be a no-op, not a rebuild

        assert manager.limiter is first_limiter

    def test_close_is_idempotent_and_resets_to_uninitialized(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_STORAGE", "memory")
        manager = RateLimiterManager()
        manager.initialize()

        asyncio.run(manager.close())
        asyncio.run(manager.close())  # second close must not raise

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.limiter

    def test_close_before_initialize_does_not_raise(self):
        manager = RateLimiterManager()

        asyncio.run(manager.close())  # must be a no-op, not an AttributeError

    def test_close_disconnects_redis_pool(self):
        storage = object.__new__(RedisStorage)
        pool = MagicMock()
        storage.bridge = SimpleNamespace(
            get_connection=lambda: SimpleNamespace(connection_pool=pool)
        )
        manager = RateLimiterManager()
        manager._storage = storage
        manager._limiter = object()

        asyncio.run(manager.close())

        pool.disconnect.assert_called_once_with()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.limiter


class TestRedisDsnPasswordEncoding:
    """
    Regression coverage for the SecretStr migration: `_redis_dsn` must keep
    unwrapping `settings.redis_password` with `.get_secret_value()` and
    URL-encoding the result. A password containing '@', ':', or '/' would
    otherwise be misparsed as DSN delimiters and point the limiter at the
    wrong Redis address (or the wrong db).
    """

    def test_password_with_special_characters_is_url_encoded(self, monkeypatch):
        # settings.redis_password is a SecretStr, not a plain str — assigning
        # a bare string here (as monkeypatch.setattr would do on a pydantic
        # model with no validate_assignment) would silently replace it with
        # an object that has no .get_secret_value(), masking exactly the kind
        # of regression this test exists to catch.
        monkeypatch.setattr(settings, "redis_password", SecretStr("p@ss:word/1"))
        monkeypatch.setattr(settings, "redis_host", "cache-host")
        monkeypatch.setattr(settings, "redis_port", 6379)
        monkeypatch.setattr(settings, "redis_db", 2)

        dsn = _redis_dsn()

        assert dsn == "async+redis://:p%40ss%3Aword%2F1@cache-host:6379/2"

    def test_no_password_omits_auth_section(self, monkeypatch):
        monkeypatch.setattr(settings, "redis_password", None)
        monkeypatch.setattr(settings, "redis_host", "cache-host")
        monkeypatch.setattr(settings, "redis_port", 6379)
        monkeypatch.setattr(settings, "redis_db", 0)

        dsn = _redis_dsn()

        assert dsn == "async+redis://cache-host:6379/0"
        assert "@" not in dsn


class TestGetClientIp:
    """Forwarding headers are accepted only from explicitly trusted peers."""

    def test_zero_trusted_hops_ignores_forwarded_header(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 0)
        request = _make_request(
            headers={"X-Forwarded-For": "1.2.3.4"}, client_host="10.0.0.9"
        )

        assert get_client_ip(request) == "10.0.0.9"

    def test_one_trusted_hop_reads_rightmost_entry(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 1)
        monkeypatch.setattr(
            settings, "rate_limit_trusted_proxy_networks", ["10.0.0.0/8"]
        )
        # Leftmost entry (1.2.3.4) is client-supplied and untrusted; the
        # rightmost (10.0.0.1) is what our own proxy appended.
        request = _make_request(headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1"})

        assert get_client_ip(request) == "10.0.0.1"

    def test_two_trusted_hops_reads_second_from_right(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 2)
        monkeypatch.setattr(
            settings, "rate_limit_trusted_proxy_networks", ["10.0.0.0/8"]
        )
        request = _make_request(
            headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1, 10.0.0.2"}
        )

        assert get_client_ip(request) == "10.0.0.1"

    def test_header_shorter_than_configured_hops_falls_back_to_tcp_peer(
        self, monkeypatch
    ):
        # Configured for 2 hops but the header only has 1 entry — an
        # incomplete/malformed forwarding chain. Falling back to the raw TCP
        # peer (never client-controlled) is the fail-safe, not a bypass.
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 2)
        request = _make_request(
            headers={"X-Forwarded-For": "1.2.3.4"}, client_host="10.0.0.9"
        )

        assert get_client_ip(request) == "10.0.0.9"

    def test_missing_header_falls_back_to_tcp_peer(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 1)
        request = _make_request(headers={}, client_host="10.0.0.9")

        assert get_client_ip(request) == "10.0.0.9"

    def test_untrusted_peer_cannot_spoof_forwarded_address(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 1)
        monkeypatch.setattr(
            settings, "rate_limit_trusted_proxy_networks", ["10.0.0.0/8"]
        )
        request = _make_request(
            headers={"X-Forwarded-For": "1.2.3.4"}, client_host="198.51.100.7"
        )
        assert get_client_ip(request) == "198.51.100.7"

    def test_no_tcp_peer_returns_unknown(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_trusted_proxy_hops", 1)
        request = _make_request(headers={}, client_host=None)

        assert get_client_ip(request) == "unknown"


class TestUnverifiedServiceHeader:
    """A caller-supplied service header does not change the bucket identity."""

    @pytest.mark.asyncio
    async def test_known_service_id_is_ignored(self):
        request = _make_request(
            headers={"X-Service-Id": "ms-llm-gateway"}, client_host="10.0.1.5"
        )

        assert await ip_only_key(request) == "10.0.1.5"

    @pytest.mark.asyncio
    async def test_missing_service_id_uses_peer_address(self):
        request = _make_request(headers={}, client_host="10.0.1.5")

        assert await ip_only_key(request) == "10.0.1.5"

    @pytest.mark.asyncio
    async def test_blank_service_id_uses_peer_address(self):
        request = _make_request(headers={"X-Service-Id": "   "}, client_host="10.0.1.5")

        assert await ip_only_key(request) == "10.0.1.5"


class TestRetryAfterSeconds:
    """Retry-After must always be a positive integer, per HTTP semantics."""

    def test_future_reset_rounds_up_to_whole_seconds(self):
        reset_at = __import__("time").time() + 10.9

        assert _retry_after_seconds(reset_at) == 11

    def test_already_elapsed_window_floors_at_minimum(self):
        reset_at = __import__("time").time() - 5  # reset time already passed

        assert _retry_after_seconds(reset_at) == 1


class TestFailOpenClassification:
    """Transient Redis failures fail open; implementation errors propagate."""

    def _dependency_with_fake_limiter(self, monkeypatch, hit_side_effect):
        fake_limiter = SimpleNamespace(hit=AsyncMock(side_effect=hit_side_effect))
        fake_manager = SimpleNamespace(limiter=fake_limiter)
        monkeypatch.setattr(enforcement_module, "rate_limiter_manager", fake_manager)

        rule = RateLimitRule(
            name="test_rule", limit="5/minute", key_namespace="test_rule"
        )
        return rate_limit_dependency(
            rule=rule, key_fn=lambda request: _fixed_key(request)
        )

    @pytest.mark.asyncio
    async def test_transient_storage_error_fails_open(self, monkeypatch, caplog):
        dependency = self._dependency_with_fake_limiter(
            monkeypatch, hit_side_effect=ConnectionError("redis unreachable")
        )
        request = _make_request()

        # Fails open: returns None instead of raising.
        assert await dependency(request) is None

    @pytest.mark.asyncio
    async def test_unexpected_error_raises(self, monkeypatch):
        dependency = self._dependency_with_fake_limiter(
            monkeypatch, hit_side_effect=RuntimeError("unexpected driver bug")
        )
        request = _make_request()

        # Still fails open (availability over strict enforcement), but this
        # class of error is logged at ERROR with a traceback elsewhere so
        # it pages on-call rather than degrading silently.
        with pytest.raises(RuntimeError, match="unexpected driver bug"):
            await dependency(request)

    @pytest.mark.asyncio
    async def test_over_limit_raises_rate_limit_exceeded(self, monkeypatch):
        fake_limiter = SimpleNamespace(
            hit=AsyncMock(return_value=False),
            get_window_stats=AsyncMock(
                return_value=(__import__("time").time() + 30, 0)
            ),
        )
        fake_manager = SimpleNamespace(limiter=fake_limiter)
        monkeypatch.setattr(enforcement_module, "rate_limiter_manager", fake_manager)
        rule = RateLimitRule(
            name="test_rule", limit="5/minute", key_namespace="test_rule"
        )
        dependency = rate_limit_dependency(
            rule=rule, key_fn=lambda request: _fixed_key(request)
        )
        request = _make_request()

        with pytest.raises(RateLimitExceededError) as exc_info:
            await dependency(request)

        assert exc_info.value.retry_after >= 1


async def _fixed_key(request: Request) -> str:
    """A trivial key_fn stand-in — these tests exercise error handling, not keying."""
    return "fixed-key"
