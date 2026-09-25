"""
Unit tests for the composed app's middleware stack (app/app.py).

Covers two regressions fixed in this pass:
1. CORS credentials: allow_origins=["*"] + allow_credentials=True is a
   well-known antipattern (browsers require Starlette to echo the request's
   own Origin back, so a wildcard allowlist + credentials effectively trusts
   every origin for credentialed requests). Nothing in this codebase uses
   cookie/session auth, so allow_credentials should be False.
2. Middleware ordering: CORSMiddleware short-circuits an OPTIONS preflight
   before it ever reaches an inner middleware. correlation_id_middleware
   must be registered AFTER CORSMiddleware (making it the outermost layer)
   for every response — including a preflight — to carry X-Correlation-Id.
   Getting the registration order backwards silently drops correlation ids
   from preflight responses; there's no error, just missing data, which is
   why this needs its own regression test rather than relying on someone
   noticing a header is missing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.app import app
from app.core.request_tracing import CORRELATION_ID_HEADER


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestCorsConfiguration:
    def test_preflight_does_not_grant_credentialed_cross_origin_access(
        self, client: TestClient
    ) -> None:
        response = client.options(
            "/",
            headers={
                "Origin": "https://an-arbitrary-untrusted-origin.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        # allow_credentials=False means Starlette never emits this header at
        # all — its presence (regardless of value) is what a browser uses to
        # decide whether to expose the response to credentialed JS callers.
        assert "access-control-allow-credentials" not in {
            k.lower() for k in response.headers
        }

    def test_wildcard_origin_is_still_allowed_for_non_credentialed_requests(
        self, client: TestClient
    ) -> None:
        """Disabling credentials must not break ordinary (non-credentialed) CORS."""
        response = client.options(
            "/",
            headers={
                "Origin": "https://any-origin.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"


class TestCorrelationIdMiddlewareOrdering:
    def test_correlation_id_present_on_ordinary_response(
        self, client: TestClient
    ) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert CORRELATION_ID_HEADER.lower() in {k.lower() for k in response.headers}

    def test_correlation_id_present_on_cors_preflight_response(
        self, client: TestClient
    ) -> None:
        """
        Regression test: CORSMiddleware answers an OPTIONS preflight itself,
        without calling the wrapped app — so this header only appears if
        correlation_id_middleware is registered AFTER (and therefore wraps
        OUTSIDE) CORSMiddleware. Registered in the opposite order (as this
        file used to), this assertion fails with the header silently absent
        rather than any explicit error.
        """
        response = client.options(
            "/",
            headers={
                "Origin": "https://any-origin.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        assert CORRELATION_ID_HEADER.lower() in {k.lower() for k in response.headers}

    def test_correlation_id_echoed_when_caller_supplies_one(
        self, client: TestClient
    ) -> None:
        response = client.get(
            "/", headers={CORRELATION_ID_HEADER: "caller-supplied-id"}
        )

        assert response.headers[CORRELATION_ID_HEADER] == "caller-supplied-id"
