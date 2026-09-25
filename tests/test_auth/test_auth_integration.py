"""HTTP authentication behavior against the active API."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.app import app
from app.auth.jwt_auth_token_service import create_access_token


def test_token_validate_round_trip():
    user_id, tenant_id = uuid4(), uuid4()
    token = create_access_token(user_id, "developer", tenant_id)
    response = TestClient(app).get(
        "/api/v1/auth/token/validate", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == str(user_id)
    assert response.json()["tenant_id"] == str(tenant_id)


def test_token_validate_requires_valid_access_token():
    client = TestClient(app)
    assert client.get("/api/v1/auth/token/validate").status_code == 401
    assert (
        client.get(
            "/api/v1/auth/token/validate", headers={"Authorization": "Bearer invalid"}
        ).status_code
        == 401
    )


def test_config_endpoint_reports_public_fields():
    response = TestClient(app).get("/api/v1/auth/config")
    assert response.status_code == 200
    assert "jwt_secret_key" not in response.text
    assert "jwt_algorithm" in response.json()
