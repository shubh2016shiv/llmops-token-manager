"""
Resilience Models - Typed contracts for queue, worker, and hot-path coordination.

Architecture:
-------------
    ┌─────────────────────────┐     ┌─────────────────────────┐
    │ token_manager_endpoints │────▶│ TokenAllocationPublisher│
    └─────────────────────────┘     └────────────┬────────────┘
                                                 │ serializes
                                                 ▼
                                      ┌─────────────────────────┐
                                      │ TokenAllocationPersist  │
                                      │ DlqPayload              │
                                      └────────────┬────────────┘
                                                   │ validates
                                                   ▼
                                      ┌─────────────────────────┐
                                      │ token_maintenance /     │
                                      │ Redis reconciliation    │
                                      └─────────────────────────┘

Dependencies:
    - app.models.token_manager_models.py - shared allocation status enum

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any, Literal
from uuid import UUID  # noqa: TC003

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.models.token_manager_models import AllocationStatus


class TokenAllocationPersistPayload(BaseModel):
    """Message contract for async token allocation persistence."""

    model_config = ConfigDict(populate_by_name=True)

    token_request_id: str = Field(
        ..., min_length=1, description="Unique token allocation request identifier"
    )
    user_id: UUID = Field(..., description="User that owns the token allocation")
    llm_provider: str = Field(..., min_length=1, description="LLM provider name")
    llm_model_name: str = Field(..., min_length=1, description="Target LLM model name")
    token_count: int = Field(..., gt=0, description="Reserved token count")
    api_endpoint_url: str = Field(
        default="", description="Resolved API endpoint for the chosen deployment"
    )
    allocation_status: AllocationStatus = Field(
        default=AllocationStatus.ACQUIRED,
        description="Allocation state to persist",
    )
    deployment_name: str | None = Field(
        default=None, description="Chosen deployment name"
    )
    cloud_provider: str | None = Field(
        default=None, description="Cloud provider that hosts the deployment"
    )
    deployment_region: str | None = Field(
        default=None, description="Region of the chosen deployment"
    )
    request_context: dict[str, Any] | None = Field(
        default=None, description="Request metadata persisted alongside the allocation"
    )
    temperature: float | None = Field(
        default=None, description="Resolved generation temperature"
    )
    top_p: float | None = Field(
        default=None, description="Resolved nucleus sampling value"
    )
    seed: int | None = Field(default=None, description="Resolved generation seed")
    expires_at: datetime | None = Field(
        default=None, description="Allocation expiration timestamp"
    )
    message_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("message_id", "_message_id"),
        description="Transport-level message identifier",
    )

    @field_validator(
        "token_request_id",
        "llm_provider",
        "llm_model_name",
        "api_endpoint_url",
        "deployment_name",
        "cloud_provider",
        "deployment_region",
        "message_id",
    )
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        """Trim string fields without turning missing optional values into text."""
        if value is None:
            return value
        return value.strip()

    @field_validator("token_request_id", "llm_provider", "llm_model_name")
    @classmethod
    def validate_required_non_empty_strings(cls, value: str) -> str:
        """Ensure required string fields remain non-empty after normalization."""
        if not value:
            raise ValueError("Required payload string field must not be blank")
        return value


class DlqPayload(TokenAllocationPersistPayload):
    """Message contract for explicit dead-letter queue routing."""

    dlq_reason: str = Field(..., min_length=1, description="Reason for DLQ routing")
    dlq_routed_by: Literal["explicit"] = Field(
        default="explicit",
        description="Marks messages deliberately routed to the DLQ by application logic",
    )
    retry_attempts: int = Field(
        default=0,
        ge=0,
        description="Number of retry attempts performed before DLQ routing",
    )

    @field_validator("dlq_reason")
    @classmethod
    def validate_dlq_reason(cls, value: str) -> str:
        """Ensure the DLQ reason is not only whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("dlq_reason must not be blank")
        return normalized


class CounterSeedRecord(BaseModel):
    """Represents one Redis counter seed or reconciliation record."""

    llm_model_name: str = Field(..., min_length=1, description="LLM model name")
    api_endpoint_url: str = Field(..., min_length=1, description="Deployment endpoint")
    allocated_tokens: int = Field(..., ge=0, description="Current allocated token sum")
    max_tokens: int = Field(..., gt=0, description="Configured token capacity limit")


class InvalidActiveDeploymentRecord(BaseModel):
    """Represents one invalid active deployment missing configured capacity."""

    llm_provider: str = Field(..., min_length=1, description="LLM provider name")
    llm_model_name: str = Field(..., min_length=1, description="LLM model name")
    api_endpoint_url: str = Field(..., min_length=1, description="Deployment endpoint")
    deployment_name: str | None = Field(
        default=None,
        description="Optional deployment identifier",
    )
    deployment_region: str | None = Field(
        default=None,
        description="Optional deployment region",
    )


class DeploymentCapacitySnapshot(BaseModel):
    """Represents one deployment capacity view used by the hot path."""

    llm_model_name: str = Field(..., min_length=1, description="LLM model name")
    api_endpoint_url: str = Field(..., min_length=1, description="Deployment endpoint")
    current_allocated_tokens: int = Field(
        ..., ge=0, description="Current allocated tokens for this deployment"
    )
    max_tokens: int = Field(..., gt=0, description="Deployment token capacity limit")
    available_tokens: int = Field(
        ..., ge=0, description="Remaining token capacity available for reservation"
    )
    deployment_name: str | None = Field(
        default=None, description="Deployment identifier"
    )
    cloud_provider: str | None = Field(
        default=None, description="Cloud provider that hosts this deployment"
    )
    deployment_region: str | None = Field(default=None, description="Deployment region")

    @model_validator(mode="after")
    def validate_capacity_bounds(self) -> DeploymentCapacitySnapshot:
        """Ensure available capacity does not exceed the configured maximum."""
        if self.available_tokens > self.max_tokens:
            raise ValueError("available_tokens cannot exceed max_tokens")
        return self


class CircuitBreakerSnapshot(BaseModel):
    """Structured view of circuit breaker state for health and diagnostics."""

    name: str = Field(..., min_length=1, description="Circuit breaker name")
    state: str = Field(..., min_length=1, description="Current circuit breaker state")
    failure_count: int = Field(
        default=0, ge=0, description="Consecutive failure count recorded by the breaker"
    )
    recovery_timeout_seconds: int = Field(
        ..., gt=0, description="Configured recovery timeout for the breaker"
    )
    opened_at: datetime | None = Field(
        default=None, description="Timestamp when the breaker most recently opened"
    )


class BackpressureDecision(BaseModel):
    """Decision contract returned by backpressure evaluation logic."""

    should_reject_request: bool = Field(
        ..., description="Whether the current request should be rejected"
    )
    reason: str | None = Field(
        default=None, description="Primary reason for rejecting the request"
    )
    retry_after_seconds: int | None = Field(
        default=None, ge=1, description="Suggested Retry-After duration in seconds"
    )
    queue_depth: int | None = Field(
        default=None,
        ge=0,
        description="Observed queue depth when the decision was made",
    )
    pool_utilization_pct: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Observed database pool utilization percentage",
    )
    circuit_breaker_name: str | None = Field(
        default=None, description="Circuit breaker that triggered the decision"
    )

    @model_validator(mode="after")
    def validate_rejection_details(self) -> BackpressureDecision:
        """Require actionable context when rejecting a request."""
        if self.should_reject_request and self.retry_after_seconds is None:
            raise ValueError(
                "retry_after_seconds is required when should_reject_request is True"
            )
        return self
