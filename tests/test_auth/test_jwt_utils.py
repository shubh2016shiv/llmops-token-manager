"""JWT signing and validation contracts."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import JWTError, jwt
import pytest

from app.auth.jwt_auth_token_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiration_seconds,
    is_refresh_enabled,
    verify_token_type,
)
from app.core.config import ApplicationSettings, settings


def _claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    claims: dict[str, object] = {
        "user_id": str(uuid4()),
        "role": "developer",
        "tenant_id": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
        "type": "access",
    }
    return claims | overrides


def _signed(**overrides: object) -> str:
    return jwt.encode(
        _claims(**overrides),
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def test_access_token_round_trip():
    user_id, tenant_id = uuid4(), uuid4()
    payload = decode_token(create_access_token(user_id, "custom-role", tenant_id))
    assert (payload.user_id, payload.tenant_id, payload.role, payload.type) == (
        user_id,
        tenant_id,
        "custom-role",
        "access",
    )
    assert payload.expire_at_time.tzinfo is not None
    assert payload.issued_at_time.tzinfo is not None
    verify_token_type(payload, "access")
    with pytest.raises(ValueError, match="Token type mismatch"):
        verify_token_type(payload, "refresh")


def test_refresh_token_setting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "jwt_refresh_enabled", False)
    assert not is_refresh_enabled()
    with pytest.raises(ValueError, match="disabled"):
        create_refresh_token(uuid4(), "developer", uuid4())
    monkeypatch.setattr(settings, "jwt_refresh_enabled", True)
    assert (
        decode_token(create_refresh_token(uuid4(), "developer", uuid4())).type
        == "refresh"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": "not-a-uuid"},
        {"role": ""},
        {"role": 123},
        {"type": "other"},
        {"iat": "bad"},
        {"iat": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())},
        {"exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())},
    ],
)
def test_invalid_claims_rejected(changes: dict[str, object]):
    with pytest.raises((ValueError, JWTError)):
        decode_token(_signed(**changes))


def test_missing_claims_and_oversized_tokens_rejected():
    claims = _claims()
    del claims["tenant_id"]
    token = jwt.encode(
        claims,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(ValueError):
        decode_token(token)
    with pytest.raises(ValueError):
        decode_token("x" * 8193)


def test_bad_signature_and_expiration():
    with pytest.raises(JWTError):
        decode_token("invalid.token.format")
    with pytest.raises(JWTError):
        decode_token(
            jwt.encode(
                _claims(), "a-different-secret", algorithm=settings.jwt_algorithm
            )
        )
    assert (
        get_token_expiration_seconds() == settings.jwt_access_token_expire_hours * 3600
    )


def test_production_rejects_placeholder_secret():
    with pytest.raises(ValueError, match="unique JWT secret"):
        ApplicationSettings(
            **(
                settings.model_dump()
                | {
                    "app_environment": "production",
                    "jwt_secret_key": "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET",
                }
            )
        )
