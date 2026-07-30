"""
JWT token operations — signing, decoding, and type verification.

----
What this module owns
---------------------
- Signing a JWT payload into a token string (dev only).
- Decoding a token string back into a verified payload (production path).
- Checking that a token is the right type (access vs refresh).

----
What this module does NOT own
------------------------------
- Role definitions.  The token manager is role-agnostic: it embeds whatever
  role string the caller provides into the JWT, but never validates that
  string against a known list.  Role taxonomy is owned by ``llm_services``,
  the upstream microservice that authenticates users and mints tokens.
  Hardcoding a role list here would couple two services that should evolve
  independently — every new role in llm_services would require a
  coordinated deploy of the token manager.

- User identity.  This service never looks up users from a database or
  verifies credentials.  It trusts the JWT signature: if the token was
  signed with the shared secret, the payload is accepted as-is.

----
Token type enforcement
----------------------
Every JWT carries a ``"type"`` claim: ``"access"`` or ``"refresh"``.
``verify_token_type()`` checks this claim on every validation call so that
a refresh token cannot be used to call an endpoint that expects an access
token.  This is a defense against "token confusion" attacks — a standard
JWT best practice (RFC 8725).

----
Production path vs dev path
---------------------------
    Production:  llm_services signs the JWT → token_manager validates it.
                 ``decode_token()`` and ``verify_token_type()`` are the only
                 functions used on the hot path.

    Dev only:    ``create_access_token()`` and ``create_refresh_token()``
                 are called exclusively by ``/token/generate`` — a dev-only
                 endpoint that returns 403 in production.  They exist so
                 developers can generate test tokens without standing up
                 the full auth infrastructure.
"""

from datetime import datetime, timedelta
from uuid import UUID

from jose import JWTError, jwt
from loguru import logger

from app.core.config import settings
from app.models.auth_models import AuthTokenPayload

# ===========================================================================
# Token creation (development/testing only — disabled in production)
# ===========================================================================


def create_access_token(user_id: UUID, role: str, tenant_id: UUID) -> str:
    """
    Sign a JWT access token with the given user identity.

    This function is role-agnostic: it accepts any role string and embeds
    it verbatim into the token payload.  Role validation is the upstream
    service's responsibility — the token manager just trusts that a valid
    role was provided by a caller who already passed authorization checks.

    Args:
        user_id:  User's unique identifier.
        role:     Arbitrary role string (set by the upstream auth service).
        tenant_id: Tenant the user is acting within.

    Returns:
        A signed JWT access token string.

    Raises:
        JWTError: If the cryptographic signing operation fails.
    """
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_access_token_expire_hours)

    # The JWT payload is a simple dictionary.  Claims follow the standard
    # JWT naming conventions:
    #   exp  — expiration time (UNIX timestamp)
    #   iat  — issued-at time
    #   type — "access" or "refresh" (prevents token confusion attacks)
    payload = {
        "user_id": str(user_id),
        "role": role,
        "tenant_id": str(tenant_id),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    try:
        token: str = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        logger.debug(f"Access token created for user {user_id} with role {role}")
        return token
    except JWTError as e:
        logger.error(f"Failed to create access token: {e}")
        raise JWTError(f"Token creation failed: {str(e)}") from e


def create_refresh_token(user_id: UUID, role: str, tenant_id: UUID) -> str:
    """
    Sign a JWT refresh token with the given user identity.

    Only usable when ``jwt_refresh_enabled=True`` in configuration.
    Like ``create_access_token``, this is role-agnostic — it embeds
    whatever role string is provided without validating it.

    Refresh tokens have a longer lifetime than access tokens (configured
    via ``jwt_refresh_token_expire_days``) and carry ``"type": "refresh"``
    to prevent them from being used as access tokens.

    Args:
        user_id:  User's unique identifier.
        role:     Arbitrary role string.
        tenant_id: Tenant the user is acting within.

    Returns:
        A signed JWT refresh token string.

    Raises:
        ValueError: If refresh tokens are disabled in configuration.
        JWTError:   If the cryptographic signing operation fails.
    """
    if not settings.jwt_refresh_enabled:
        raise ValueError("Refresh tokens are disabled in configuration")

    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)

    payload = {
        "user_id": str(user_id),
        "role": role,
        "tenant_id": str(tenant_id),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }

    try:
        token: str = jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        logger.debug(f"Refresh token created for user {user_id} with role {role}")
        return token
    except JWTError as e:
        logger.error(f"Failed to create refresh token: {e}")
        raise JWTError(f"Refresh token creation failed: {str(e)}") from e


# ===========================================================================
# Token validation (production hot path — runs on every authenticated request)
# ===========================================================================


def decode_token(token: str) -> AuthTokenPayload:
    """
    Decode a JWT token string into a validated ``AuthTokenPayload``.

    This is the production validation path.  It performs three checks:

    1. **Cryptographic**: the HMAC signature is verified against the
       shared secret key.  A tampered token fails here.
    2. **Expiration**: the ``exp`` claim is checked.  An expired token
       fails here (``jose`` handles this automatically).
    3. **Structural**: the payload must contain all required fields
       (user_id, role, tenant_id, exp, iat, type).  A malformed payload
       fails here.

    No database queries.  No role validation.  Pure cryptography.

    Args:
        token: Raw JWT string from the Authorization header.

    Returns:
        ``AuthTokenPayload`` with the decoded claims.

    Raises:
        JWTError:  Signature invalid, token expired, or algorithm mismatch.
        ValueError: Required claims missing from the payload.
    """
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )

        # Structural validation: every required claim must be present.
        # We don't validate the VALUES of these claims (e.g., whether
        # "role" is a known role) — that's the upstream service's job.
        required = ("user_id", "role", "tenant_id", "exp", "iat", "type")
        for field in required:
            if field not in payload:
                raise ValueError(f"Token missing '{field}'")

        exp_datetime = datetime.utcfromtimestamp(payload["exp"])
        iat_datetime = datetime.utcfromtimestamp(payload["iat"])

        token_payload = AuthTokenPayload(
            user_id=UUID(payload["user_id"]),
            role=payload["role"],
            tenant_id=UUID(payload["tenant_id"]),
            expire_at_time=exp_datetime,
            issued_at_time=iat_datetime,
            type=payload["type"],
        )

        logger.debug(f"Token decoded successfully for user {token_payload.user_id}")
        return token_payload

    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        raise JWTError(f"Invalid token: {str(e)}") from e
    except (ValueError, TypeError) as e:
        logger.warning(f"Token payload validation failed: {e}")
        raise ValueError(f"Invalid token payload: {str(e)}") from e


def verify_token_type(payload: AuthTokenPayload, expected_type: str) -> None:
    """
    Check that a token's ``type`` claim matches the expected value.

    This prevents "token confusion" attacks: using a refresh token where
    an access token is required, or vice versa.  Both token types are
    valid JWTs signed with the same secret — without this check, they'd
    be interchangeable.

    Args:
        payload:       Decoded token payload.
        expected_type: Either ``"access"`` or ``"refresh"``.

    Raises:
        ValueError: If ``payload.type != expected_type``.
    """
    if payload.type != expected_type:
        raise ValueError(
            f"Token type mismatch. Expected '{expected_type}', got '{payload.type}'"
        )
    logger.debug(f"Token type verified: {payload.type}")


# ===========================================================================
# Configuration helpers
# ===========================================================================


def get_token_expiration_seconds() -> int:
    """Return the access token lifetime in seconds (from settings)."""
    return settings.jwt_access_token_expire_hours * 3600


def is_refresh_enabled() -> bool:
    """Return whether refresh tokens are enabled (from settings)."""
    return settings.jwt_refresh_enabled
