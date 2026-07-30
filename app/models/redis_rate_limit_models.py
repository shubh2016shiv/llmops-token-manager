"""
Pydantic models for Redis-backed rate limiting.

These models define the configuration (rules) and the structured
response payload returned when a rate limit is exceeded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RateLimitRule(BaseModel):
    """
    Rate limit rule definition.

    Immutable configuration object that defines a single rate-limit bucket.

    Attributes:
        name: Human-readable rule identifier (e.g. "auth_login", "token_acquire").
        limit: Limit string in `limits`-library format (e.g. "10/minute", "500/hour").
        key_namespace: Namespace used to partition keys in the storage backend.
            Prevents collisions between different rules that might share key values.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Human-readable rule name.")
    limit: str = Field(
        ...,
        description="Limit expression (e.g. '10/minute', '500/hour').",
        examples=["10/minute", "500/minute", "1000/hour"],
    )
    key_namespace: str = Field(
        ...,
        description="Storage namespace to isolate keys between rules.",
        examples=["auth_login", "token_acquire"],
    )


class RateLimitedErrorDetail(BaseModel):
    """Nested detail block inside a rate-limited error response."""

    rule: str = Field(..., description="Name of the rate limit rule that was hit.")
    retry_after_seconds: int = Field(
        ..., description="Seconds until the caller should retry.", ge=1
    )
    remaining: int = Field(
        ...,
        description="Remaining requests in the current window (always 0 when limited).",
        ge=0,
    )


class RateLimitedResponse(BaseModel):
    """
    Structured 429 response payload returned when a rate limit is exceeded.

    This model serves as the JSON body contract for every rate-limited response,
    ensuring callers always receive a consistent, machine-readable payload.
    """

    error: str = Field(
        default="RATE_LIMITED",
        description="Stable error code for programmatic handling.",
    )
    message: str = Field(
        default="Too many requests. Please retry later.",
        description="Human-readable explanation.",
    )
    details: RateLimitedErrorDetail = Field(
        ..., description="Machine-readable detail about which limit was hit."
    )

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSONResponse content."""
        return self.model_dump()
