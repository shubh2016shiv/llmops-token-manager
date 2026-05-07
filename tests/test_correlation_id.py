"""
Correlation ID Middleware Tests
===============================
Focused tests for correlation id propagation and response header behavior.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.core.correlation_id import CORRELATION_ID_HEADER, correlation_id_middleware


@pytest.fixture
def app():
    app = FastAPI()
    app.middleware("http")(correlation_id_middleware)

    # Minimal endpoint for middleware verification.
    # Avoid importing the full application routers to keep this test focused and
    # independent of optional infrastructure dependencies (e.g., Redis).
    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_correlation_id_generated_when_missing(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert CORRELATION_ID_HEADER in response.headers
    assert response.headers[CORRELATION_ID_HEADER].strip() != ""


def test_correlation_id_echoed_when_provided(client):
    response = client.get(
        "/ping",
        headers={CORRELATION_ID_HEADER: "abc123"},
    )
    assert response.status_code == 200
    assert response.headers[CORRELATION_ID_HEADER] == "abc123"


def test_correlation_id_blank_treated_as_missing(client):
    response = client.get(
        "/ping",
        headers={CORRELATION_ID_HEADER: "   "},
    )
    assert response.status_code == 200
    assert response.headers[CORRELATION_ID_HEADER].strip() != ""
    assert response.headers[CORRELATION_ID_HEADER] != "   "
