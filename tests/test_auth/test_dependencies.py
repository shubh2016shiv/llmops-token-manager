"""The FastAPI authentication boundary maps invalid credentials to HTTP 401."""

from uuid import uuid4

from fastapi import HTTPException
from jose import JWTError
import pytest

from app.auth.auth_dependencies import get_current_user
from app.auth.jwt_auth_token_service import create_access_token, create_refresh_token
from app.core.config import settings


@pytest.mark.asyncio
async def test_valid_access_token():
    user_id, tenant_id = uuid4(), uuid4()
    token = create_access_token(user_id, "developer", tenant_id)
    payload = await get_current_user(token)
    assert (payload.user_id, payload.tenant_id) == (user_id, tenant_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "", "invalid.token.format"])
async def test_missing_or_invalid_token_returns_401(token: str | None):
    with pytest.raises(HTTPException) as caught:
        await get_current_user(token)
    assert caught.value.status_code == 401
    assert caught.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_refresh_token_cannot_authenticate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "jwt_refresh_enabled", True)
    token = create_refresh_token(uuid4(), "developer", uuid4())
    with pytest.raises(HTTPException) as caught:
        await get_current_user(token)
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_jwt_error_does_not_reach_http_response(monkeypatch: pytest.MonkeyPatch):
    def fail(_token: str) -> None:
        raise JWTError("secret internal detail")

    monkeypatch.setattr("app.auth.auth_dependencies.decode_token", fail)
    with pytest.raises(HTTPException) as caught:
        await get_current_user("token")
    assert "secret internal detail" not in caught.value.detail
