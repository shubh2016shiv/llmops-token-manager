"""
FastAPI Authentication Dependency — enterprise JWT validation, no authorization.

----
What this module does (and does NOT do)
----------------------------------------
This module answers exactly ONE question about every protected request:

    "Is the Bearer token in the Authorization header real?"

It verifies the JWT's cryptographic signature and checks that it hasn't expired.
That's authentication — confirming WHO you are.

It does NOT check whether the caller is allowed to perform a specific action.
That's authorization — and in this architecture, authorization is owned by
`llm_services`, the upstream microservice that issues the tokens. The token
manager is a resource server: it trusts that if you have a valid JWT, you
were already authorized by the service that minted it.

----
The enterprise pattern: how FastAPI authentication actually works
-----------------------------------------------------------------
Authentication in FastAPI follows a 3-step pipeline built entirely from
standard library primitives (no middleware, no monkey-patching):

    STEP 1 — Extract the token from the HTTP request
    ─────────────────────────────────────────────────
    ``OAuth2PasswordBearer`` is a FastAPI utility that reads the
    ``Authorization: Bearer <token>`` header from every incoming request.
    It returns the raw token string, or None if the header is missing.

        Client sends:  Authorization: Bearer eyJhbGciOi...
                       ─────────────── ───────────────────
                       scheme           token (what we get)

    STEP 2 — Validate the token cryptographically
    ──────────────────────────────────────────────
    ``get_current_user()`` takes the raw token, decodes it using the
    shared secret key, and verifies:
      - The signature is valid (the token wasn't tampered with).
      - The expiration time hasn't passed.
      - The token type is "access" (not "refresh" — prevents confusion).
    If any check fails → HTTP 401 Unauthorized. The route handler never runs.

    STEP 3 — Inject the validated payload into the route handler
    ─────────────────────────────────────────────────────────────
    If the token passes, ``get_current_user()`` returns an
    ``AuthTokenPayload`` object (user_id, role, tenant_id, etc.).
    FastAPI injects this into the route handler's parameters — your
    handler code receives a fully-validated user without writing a
    single line of auth logic.

----
How to protect an endpoint (one line)
--------------------------------------
    from app.auth import CurrentUser

    @router.get("/protected")
    async def my_endpoint(current_user: CurrentUser):
        # `current_user` is guaranteed to be a valid, non-expired JWT payload.
        # If the token was missing or invalid, this line never runs —
        # FastAPI already returned a 401.
        return {"user_id": str(current_user.user_id)}

That's the entire pattern. No role checks, no database queries, no
middleware — just annotate the parameter with ``CurrentUser``.

----
Why no role-based authorization here?
--------------------------------------
This service is the LLM Token Manager. Its job is to allocate and track
token capacity, not to decide who gets to use which model. The upstream
``llm_services`` microservice already handles:

    - User login (username/password → JWT)
    - Role assignment (is this user an admin? a developer?)
    - Model access policy (can this tenant use GPT-4?)

By the time a request reaches the token manager, the caller has already
been authenticated AND authorized by llm_services. Double-checking roles
here would be redundant — and, worse, it would couple the token manager
to a role model that it doesn't own. If llm_services adds a new role
("viewer"), every downstream service would need updating. That's the
anti-pattern: authorization should live in ONE place.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from loguru import logger

from app.auth.jwt_auth_token_service import decode_token, verify_token_type

# Runtime import, deliberately NOT under `if TYPE_CHECKING:`.
# The `CurrentUser` alias at the bottom of this module is a plain assignment,
# so `Annotated[AuthTokenPayload, ...]` is evaluated the moment this module is
# imported. A TYPE_CHECKING-only import would leave the name undefined at
# runtime and raise NameError. (Function *annotations* can stay lazy thanks to
# `from __future__ import annotations`; a module-level assignment cannot.)
from app.models.auth_models import AuthTokenPayload

# ---------------------------------------------------------------------------
# Step 1: the token extractor
# ---------------------------------------------------------------------------
# ``OAuth2PasswordBearer`` is FastAPI's built-in parser for the standard
# ``Authorization: Bearer <token>`` HTTP header.  You configure it with:
#
#   tokenUrl  — the URL shown in Swagger's "Authorize" dialog (UI only).
#   auto_error — False means "return None if the header is missing" instead
#                of auto-raising 401.  We handle the missing case ourselves
#                in get_current_user() so we control the error message.
#
# The ``tokenUrl`` points at our dev-only /token/generate endpoint so
# developers can click "Authorize" in Swagger, paste credentials, and get
# a token for testing.  In production this endpoint is disabled.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token/generate",
    auto_error=False,
)


# ---------------------------------------------------------------------------
# Step 2: the validator — turns a raw token string into a trusted payload
# ---------------------------------------------------------------------------
async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> AuthTokenPayload:
    """
    Validate the JWT from the Authorization header and return its payload.

    This is the ONE authentication dependency for the entire service.  Every
    protected endpoint chains through this function.  The validation is
    purely cryptographic — no database query, no network call, just a
    signature check against the shared secret key.

    How it's used (FastAPI wires this automatically):
        @router.get("/protected")
        async def endpoint(current_user: CurrentUser):
            # current_user is an AuthTokenPayload — guaranteed valid

    The flow through this function:
        1. Extract the Bearer token from the header (via oauth2_scheme).
        2. If missing → 401 immediately.
        3. Decode the JWT signature using the shared secret.
        4. Check the token type is "access" (not "refresh").
        5. Return the payload → FastAPI injects it into the route handler.

    Returns:
        AuthTokenPayload with user_id, role, tenant_id, and expiration info.

    Raises:
        HTTPException 401: Token missing, expired, or signature invalid.
    """
    if not token:
        # No Authorization header at all.  The caller didn't even try.
        logger.warning("Missing authorization token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # ---- Cryptographic validation ----
        # decode_token() verifies the JWT's HMAC signature against the
        # shared secret and checks the "exp" (expiration) claim.  If either
        # fails, it raises JWTError — which we catch below and convert to 401.
        payload = decode_token(token)

        # ---- Token type guard ----
        # A refresh token is also a valid JWT — same secret, same algorithm.
        # Without this check, someone could use a refresh token to call
        # endpoints that expect an access token.  verify_token_type() blocks
        # that by checking the "type" claim in the payload.
        verify_token_type(payload, "access")

        logger.debug(
            f"Token validated for user {payload.user_id} with role {payload.role}"
        )
        return payload

    except JWTError as e:
        # The token's signature is invalid, or it's expired, or the
        # algorithm doesn't match.  Either way — 401.
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except ValueError as e:
        # The token decoded successfully but its payload has an unexpected
        # shape (e.g., missing required fields, wrong token type).  401.
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ---------------------------------------------------------------------------
# Step 3: the reusable dependency every protected endpoint declares
# ---------------------------------------------------------------------------
# HOW `Depends` WORKS - the mental model worth memorising
# --------------------------------------------------------
# `Depends(fn)` does not call `fn`, and does not return a function. It returns
# an inert marker object. FastAPI scans every route signature, and wherever it
# finds one of these markers it:
#
#     1. reads the callable stored inside the marker,
#     2. recursively resolves that callable's own dependencies,
#     3. calls it once per request,
#     4. injects the return value into the handler.
#
# The argument to `Depends()` is therefore always a CALLABLE - something
# FastAPI can invoke. That is the whole contract.
#
# WHY THIS MODULE EXPOSES A TYPE ALIAS
# -------------------------------------
# `CurrentUser` binds two things into one reusable name: the type a protected
# endpoint receives, and the dependency that produces it. Endpoints declare it
# as a parameter TYPE:
#
#     async def endpoint(current_user: CurrentUser):
#
# That shape is correct for three concrete reasons:
#
#   Single source of wiring. How an authenticated user is produced is stated
#   once, here. An endpoint declares only the type it needs, so the wiring
#   cannot drift or be restated inconsistently across dozens of routes.
#
#   Signature ergonomics. `CurrentUser` annotates the parameter instead of
#   supplying a default value. Parameters declared after it are therefore free
#   to remain required, and route signatures keep their natural order.
#
#   Tooling. Type checkers resolve `CurrentUser` to `AuthTokenPayload`, so
#   editors autocomplete `.user_id`, `.role`, and `.tenant_id`, and a mistyped
#   attribute is caught statically rather than at request time.
#
# Markers still appear directly in one other place in this codebase: a route's
# `dependencies=[...]` list, where a dependency runs for its side effect and
# its return value is discarded. The rate limiters use that form, because a
# limiter enforces a rule rather than producing a value the handler consumes.

CurrentUser = Annotated[AuthTokenPayload, Depends(get_current_user)]
"""
The single authentication dependency for this service.

Declare it as a parameter type:

    from app.auth import CurrentUser

    @router.get("/protected")
    async def endpoint(current_user: CurrentUser):
        return {"user_id": str(current_user.user_id)}

By the time the handler body runs, `current_user` is guaranteed to be a valid,
non-expired access-token payload. If the token was missing, malformed, or
expired, FastAPI already returned 401 and the body never executed.

This establishes WHO the caller is. It does not decide what they may do -
authorization is owned upstream by llm_services, which issued the token.
"""
