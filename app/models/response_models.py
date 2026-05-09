"""
Response Models
--------------
Pydantic models for API response validation.
Simple, focused schemas for returning data to clients.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Valid LLM providers from database schema
VALID_LLM_PROVIDERS = [
    "openai",
    "gemini",
    "anthropic",
    "cohere",
    "mistral",
    "deepseek",
    "meta",
    "hugging_face",
    "together_ai",
    "fireworks_ai",
    "replicate",
    "xai",
    "deepinfra",
    "novita",
    "on_premise",
]

# Valid cloud providers from database schema
VALID_CLOUD_PROVIDERS = [
    "Azure",
    "Google Cloud Platform",
    "Amazon Web Services",
    "IBM Watsonx",
    "Oracle",
    "On Premise",
]


class AllocationStatus(str, Enum):
    ACQUIRED = "ACQUIRED"
    WAITING = "WAITING"
    PAUSED = "PAUSED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class AuthConfigResponse(BaseModel):
    """Response schema for JWT authentication configuration (non-sensitive)."""

    jwt_algorithm: str = Field(..., description="JWT algorithm used to sign tokens")
    access_token_expire_hours: int = Field(
        ..., description="Access token expiration duration in hours"
    )
    refresh_enabled: bool = Field(..., description="Whether refresh tokens are enabled")
    refresh_token_expire_days: int | None = Field(
        default=None,
        description="Refresh token expiration duration in days (null when refresh disabled)",
    )
    token_type: Literal["bearer"] = Field(
        ..., description="Token type for Authorization header"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "jwt_algorithm": "HS256",
                "access_token_expire_hours": 24,
                "refresh_enabled": False,
                "refresh_token_expire_days": None,
                "token_type": "bearer",
            }
        }


# ============================================================================
# USER RESPONSE MODEL
# ============================================================================
class UserResponse(BaseModel):
    """Response schema for user data - no sensitive info"""

    user_id: UUID
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "johndoe",
                "email": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "developer",
                "status": "active",
                "created_at": "2025-10-18T08:00:00Z",
                "updated_at": "2025-10-18T08:00:00Z",
            }
        }


# ============================================================================
# TOKEN ALLOCATION RESPONSE MODEL
# ============================================================================
class TokenAllocationResponse(BaseModel):
    """
    Response schema for successful token allocation.
    """

    # -- Request Identity: Unique identifier for the token allocation request
    token_request_id: str = Field(
        ..., description="Unique identifier for this allocation"
    )
    user_id: UUID = Field(..., description="User who received the allocation")
    llm_provider: str = Field(
        ..., description="LLM provider (e.g., openai, anthropic, gemini)"
    )
    # -- Model & Deployment Configuration: Specifies the target LLM and deployment
    llm_model_name: str = Field(..., description="Name of the LLM model (e.g., GPT-4)")
    deployment_name: str | None = Field(
        default=None, description="Specific deployment of the model, if applicable"
    )
    cloud_provider: str | None = Field(
        default=None, description="Cloud provider hosting the LLM (e.g., Azure, AWS)"
    )
    api_endpoint_url: str | None = Field(
        default=None, description="API endpoint URL to use"
    )
    deployment_region: str | None = Field(
        default=None, description="Deployment region where model is deployed"
    )
    # -- Token Allocation Management: Tracks allocated tokens and their status
    token_count: int = Field(..., description="Number of tokens allocated")
    allocation_status: str = Field(
        ...,
        description="Current allocation status: ACQUIRED, RELEASED, EXPIRED, PAUSED, or FAILED",
    )
    # -- Timing & Expiration: Manages allocation lifecycle
    allocated_at: datetime = Field(..., description="When tokens were allocated")
    expires_at: datetime | None = Field(
        default=None, description="When allocation expires (if applicable)"
    )
    # -- Additional context: Additional context data in JSON format (e.g., team, application)
    request_context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context data in JSON format (e.g., team, application)",
    )
    temperature: float | None = Field(
        default=None, description="Temperature setting for this specific request"
    )
    top_p: float | None = Field(
        default=None, description="Top P (nucleus sampling) parameter for this request"
    )
    seed: int | None = Field(
        default=None, description="Seed value for reproducible LLM outputs"
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider matches database schema."""
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(f"Invalid LLM provider: {v}")
        return v

    @field_validator("cloud_provider")
    @classmethod
    def validate_cloud_provider(cls, v: str | None) -> str | None:
        """Validate cloud provider matches database schema."""
        if v is not None and v not in VALID_CLOUD_PROVIDERS:
            raise ValueError(f"Invalid cloud provider: {v}")
        return v

    @field_validator("allocation_status")
    @classmethod
    def validate_allocation_status(cls, v: str) -> str:
        """Validate allocation status matches database CHECK constraint."""
        allowed_statuses = [allowed_status.value for allowed_status in AllocationStatus]
        if v not in allowed_statuses:
            raise ValueError(
                f"Allocation status must be one of: {', '.join(allowed_statuses)}"
            )
        return v

    class Config:
        # Allow usage of field aliases for serialization and deserialization
        populate_by_name = True
        # Show multiple example responses in OpenAPI schema
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Direct OpenAI API with GPT-4o",
                    "description": "Accessing GPT-4o directly via OpenAI's API, no deployment name required.",
                    "value": {
                        "token_request_id": "req_pqr789stu",
                        "user_id": "a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6",
                        "llm_provider": "openai",
                        "llm_model_name": "gpt-4o",
                        "deployment_name": None,
                        "cloud_provider": None,
                        "api_endpoint_url": "https://api.openai.com/v1",
                        "deployment_region": "us-east-1",
                        "token_count": 1500,
                        "allocation_status": "ACQUIRED",  # Changed from "ACTIVE" to "ACQUIRED"
                        "allocated_at": "2025-10-19T17:19:00Z",  # 10:49 PM IST = 5:19 PM UTC
                        "expires_at": "2025-10-19T17:24:00Z",  # 5-minute token duration
                        "request_context": {
                            "application": "customer_support_chatbot",
                            "task_type": "text_generation",
                            "environment": "production",
                            "request_purpose": "user_query_response",
                        },
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "seed": 42,
                    },
                },
                {
                    "summary": "Azure OpenAI Deployment with GPT-4o",
                    "description": "Accessing GPT-4o via Azure OpenAI with a custom deployment name.",
                    "value": {
                        "token_request_id": "req_xyz987qwe",
                        "user_id": "a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c8",
                        "llm_provider": "openai",
                        "llm_model_name": "gpt-4o",
                        "deployment_name": "gpt4o-eastus-prod",
                        "cloud_provider": "Azure",
                        "api_endpoint_url": "https://my-resource.openai.azure.com/openai/deployments/gpt4o-eastus-prod",
                        "deployment_region": "eastus",
                        "token_count": 2500,
                        "allocation_status": "ACQUIRED",  # Changed from "ACTIVE" to "ACQUIRED"
                        "allocated_at": "2025-10-19T17:19:00Z",  # Same UTC time
                        "expires_at": "2025-10-19T17:24:00Z",  # Same 5-minute duration
                        "request_context": {
                            "application": "document_analyzer",
                            "task_type": "summarization",
                            "environment": "staging",
                            "request_purpose": "batch_document_processing",
                        },
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "seed": 123,
                    },
                },
            ]
        }


class PauseDeploymentAlreadyPausedResponse(BaseModel):
    """Response schema when a deployment is already paused."""

    alloc_status: Literal["ALREADY_PAUSED"] = Field(
        ..., description="Pause operation status"
    )
    llm_model_name: str = Field(..., description="Model name for the paused deployment")
    api_endpoint_url: str = Field(..., description="API endpoint URL that is paused")
    reason: str = Field(..., description="Reason explaining why it is already paused")


class PauseDeploymentAllocationResponse(BaseModel):
    """
    Response schema for a successful pause allocation record.

    This mirrors the allocation dict returned from the token allocation service.
    """

    model_config = ConfigDict(extra="ignore")

    token_request_id: str = Field(
        ..., description="Token request ID for pause allocation"
    )
    user_id: UUID = Field(..., description="User who requested the pause")
    llm_provider: str = Field(..., description="LLM provider")
    llm_model_name: str = Field(..., description="Model name paused")
    deployment_name: str | None = Field(default=None, description="Deployment name")
    cloud_provider: str | None = Field(default=None, description="Cloud provider")
    api_endpoint_url: str = Field(..., description="Paused API endpoint URL")
    deployment_region: str | None = Field(default=None, description="Deployment region")
    token_count: int = Field(
        ..., description="Token count consumed by pause allocation"
    )
    allocation_status: str = Field(..., description="Allocation status (PAUSED)")
    allocated_at: datetime = Field(..., description="When pause allocation was created")
    expires_at: datetime | None = Field(default=None, description="When pause expires")
    request_context: dict[str, Any] | str | None = Field(
        default=None, description="Request context metadata (dict or JSON string)"
    )
    temperature: float | None = Field(default=None, description="Temperature")
    top_p: float | None = Field(default=None, description="Top-p")
    seed: int | None = Field(default=None, description="Seed")


PauseDeploymentResponse: TypeAlias = (
    PauseDeploymentAlreadyPausedResponse | PauseDeploymentAllocationResponse
)


# ============================================================================
# TOKEN RELEASE RESPONSE MODEL
# ============================================================================
class TokenReleaseResponse(BaseModel):
    """
    Response schema for token release confirmation.
    """

    token_request_id: str = Field(..., description="ID of the released allocation")
    allocation_status: str = Field(..., description="Updated allocation status")
    message: str = Field(..., description="Confirmation message")

    class Config:
        json_schema_extra = {
            "example": {
                "token_request_id": "req_abc123xyz",
                "allocation_status": "RELEASED",
                "message": "Tokens released successfully",
            }
        }


class LLMModelResponse(BaseModel):
    """
    Response schema for LLM model data.

    Based on the llm_models table schema with composite primary key
    (provider_name, llm_model_name, llm_model_version).
    """

    llm_provider: str = Field(..., description="LLM provider name")
    llm_model_name: str = Field(..., description="Name of the LLM model")
    cloud_provider: str | None = Field(
        default=None,
        description="Cloud provider hosting the LLM (e.g., Azure, Google Cloud Platform)",
    )
    deployment_name: str | None = Field(default=None, description="Deployment name")
    api_key_variable_name: str | None = Field(
        default=None, description="Environment variable name for API key"
    )
    api_endpoint_url: str | None = Field(default=None, description="API endpoint URL")
    llm_model_version: str | None = Field(default=None, description="Model version")
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens per request"
    )
    tokens_per_minute_limit: int | None = Field(
        default=None, description="Token rate limit per minute"
    )
    requests_per_minute_limit: int | None = Field(
        default=None, description="Request rate limit per minute"
    )
    is_active_status: bool = Field(..., description="Whether model is currently active")
    temperature: float = Field(..., description="Default temperature setting")
    top_p: float = Field(..., description="Default top_p (nucleus sampling) setting")
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducible results"
    )
    deployment_region: str | None = Field(
        default=None, description="Geographic deployment region"
    )
    created_at: datetime = Field(..., description="When model was registered")
    updated_at: datetime = Field(..., description="When model was last updated")

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider matches database schema."""
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(f"Invalid LLM provider: {v}")
        return v

    @field_validator("cloud_provider")
    @classmethod
    def validate_cloud_provider(cls, v: str | None) -> str | None:
        """Validate cloud provider matches database schema."""
        if v is not None and v not in VALID_CLOUD_PROVIDERS:
            raise ValueError(f"Invalid cloud provider: {v}")
        return v

    class Config:
        # Enable ORM mode for SQLAlchemy compatibility
        from_attributes = True
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "cloud_provider": "Azure",
                "deployment_name": "gpt-4o-eastus",
                "api_key_variable_name": "OPENAI_API_KEY_GPT4O",
                "api_endpoint_url": "https://api.openai.com/v1",
                "llm_model_version": "2024-08",
                "max_tokens": 8192,
                "tokens_per_minute_limit": 100000,
                "requests_per_minute_limit": 1000,
                "is_active_status": True,
                "temperature": 0.7,
                "top_p": 1.0,
                "random_seed": 42,
                "deployment_region": "eastus2",
                "created_at": "2025-09-01T08:00:00Z",
                "updated_at": "2025-10-13T10:30:00Z",
            }
        }


class LLMModelListResponse(BaseModel):
    """
    Response schema for listing LLM models with pagination.

    Used by: GET /api/v1/llm-models/provider/{provider} endpoint
    Purpose: Returns a paginated list of LLM model configurations for a specific provider
    Includes: Array of model objects, total count, pagination metadata, and navigation flags

    When to use:
    - When retrieving multiple LLM model configurations
    - When implementing pagination for large result sets
    - When providing navigation metadata (next/previous page availability)
    - When filtering models by provider with optional active-only filtering
    """

    models: list[LLMModelResponse] = Field(
        ..., description="List of LLM model configurations"
    )
    total_count: int = Field(..., description="Total number of models available")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Items per page")
    has_next: bool = Field(..., description="Whether there are more results available")
    has_previous: bool = Field(
        ..., description="Whether there are previous results available"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "llm_provider": "openai",
                        "llm_model_name": "gpt-4o",
                        "deployment_name": "gpt-4o-eastus",
                        "api_key_variable_name": "OPENAI_API_KEY_GPT4O",
                        "api_endpoint_url": "https://api.openai.com/v1",
                        "llm_model_version": "2024-08",
                        "max_tokens": 8192,
                        "tokens_per_minute_limit": 100000,
                        "requests_per_minute_limit": 1000,
                        "is_active_status": True,
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "random_seed": 42,
                        "deployment_region": "eastus2",
                        "created_at": "2025-09-01T08:00:00Z",
                        "updated_at": "2025-10-13T10:30:00Z",
                    }
                ],
                "total_count": 25,
                "page": 1,
                "page_size": 50,
                "has_next": False,
                "has_previous": False,
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    """

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When error occurred"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "INSUFFICIENT_TOKENS",
                "message": "Not enough tokens available for this request",
                "details": {"requested": 5000, "available": 3000},
                "timestamp": "2025-10-13T10:30:00Z",
            }
        }


class RateLimitErrorResponse(BaseModel):
    """
    Standard rate limit error response schema.

    Returned with HTTP 429 and typically accompanied by `Retry-After` header.
    """

    error: Literal["RATE_LIMITED"] = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        ...,
        description="Rate limit rule details (rule name, retry_after_seconds, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "RATE_LIMITED",
                "message": "Too many requests. Please retry later.",
                "details": {"rule": "auth_login", "retry_after_seconds": 12},
            }
        }


class AllocationListResponse(BaseModel):
    """
    Response schema for listing token allocations.
    """

    allocations: list[TokenAllocationResponse] = Field(
        ..., description="List of token allocations"
    )
    total_count: int = Field(..., description="Total number of allocations")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Items per page")

    class Config:
        json_schema_extra = {
            "example": {
                "allocations": [
                    {
                        "token_request_id": "req_abc123",
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "llm_model_name": "gpt-4",
                        "token_count": 2000,
                        "allocation_status": "ACQUIRED",
                        "allocated_at": "2025-10-13T10:30:00Z",
                    }
                ],
                "total_count": 125,
                "page": 1,
                "page_size": 50,
            }
        }


# ============================================================================
# DEPENDENCY HEALTH RESPONSE MODEL
# ============================================================================
class DependencyHealth(BaseModel):
    """
    Dependency health response model.

    Provides detailed health status for each infrastructure component:
    - PostgreSQL database
    - Redis cache
    - RabbitMQ message broker
    - Celery worker readiness
    - Token-maintenance runtime readiness

    Each component is represented as a boolean indicating if it's operational.
    """

    postgresql: bool = Field(..., description="PostgreSQL database health status")
    redis: bool = Field(..., description="Redis cache health status")
    rabbitmq: bool = Field(..., description="RabbitMQ message broker health status")
    celery_worker: bool = Field(..., description="Celery worker readiness status")
    token_maintenance: bool = Field(
        ..., description="Token-maintenance runtime readiness status"
    )
    status: str = Field(
        ..., description="Overall health status: 'healthy' or 'unhealthy'"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "postgresql": True,
                "redis": True,
                "rabbitmq": True,
                "celery_worker": True,
                "token_maintenance": True,
                "status": "healthy",
                "timestamp": "2025-10-13T10:30:00Z",
            }
        }


# class LLMRequestResponse(BaseModel):
#     """
#     Response schema for LLM request submission.
#     """

#     request_id: str = Field(..., description="Unique request identifier")
#     status: str = Field(..., description="Request status")
#     message: str = Field(..., description="Status message")
#     estimated_completion: Optional[datetime] = Field(
#         default=None, description="Estimated completion time"
#     )

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "request_id": "req_abc123xyz",
#                 "status": "queued",
#                 "message": "Request queued for processing",
#                 "estimated_completion": "2025-10-13T10:35:00Z",
#             }
#         }


# class LLMTaskStatus(BaseModel):
#     """
#     LLM task status response model.
#     """

#     request_id: str = Field(..., description="Request identifier")
#     status: str = Field(
#         ..., description="Task status: pending, processing, completed, failed"
#     )
#     progress: Optional[float] = Field(
#         default=None, description="Progress percentage (0-100)"
#     )
#     created_at: datetime = Field(..., description="When task was created")
#     started_at: Optional[datetime] = Field(
#         default=None, description="When processing started"
#     )
#     completed_at: Optional[datetime] = Field(
#         default=None, description="When processing completed"
#     )

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "request_id": "req_abc123xyz",
#                 "status": "processing",
#                 "progress": 75.0,
#                 "created_at": "2025-10-13T10:30:00Z",
#                 "started_at": "2025-10-13T10:30:05Z",
#                 "completed_at": None,
#             }
#         }


# class LLMTaskResult(BaseModel):
#     """
#     LLM task result response model.
#     """

#     request_id: str = Field(..., description="Request identifier")
#     status: str = Field(..., description="Final task status")
#     result: Optional[str] = Field(default=None, description="Generated response text")
#     tokens_used: Optional[int] = Field(
#         default=None, description="Number of tokens used"
#     )
#     processing_time: Optional[float] = Field(
#         default=None, description="Processing time in seconds"
#     )
#     error_message: Optional[str] = Field(
#         default=None, description="Error message if failed"
#     )
#     completed_at: datetime = Field(..., description="When task completed")

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "request_id": "req_abc123xyz",
#                 "status": "completed",
#                 "result": "Quantum computing is a revolutionary approach...",
#                 "tokens_used": 150,
#                 "processing_time": 2.5,
#                 "error_message": None,
#                 "completed_at": "2025-10-13T10:30:10Z",
#             }
#         }


# ============================================================================
# HEALTH RESPONSE MODEL
# ============================================================================


class Health(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class HealthStatus(BaseModel):
    """
    Health check response schema.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-13T10:30:00Z",
                "version": "1.0.0",
            }
        }
    )

    status: str = Field(..., description="Token allocation service health status")
    version: str | None = Field(default=None, description="Application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value > datetime.now(timezone.utc):
            raise ValueError("Timestamp cannot be in the future")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        valid_values = [member.value for member in Health]
        if value not in valid_values:
            raise ValueError(f"Status must be one of: {', '.join(valid_values)}")
        return value


# ============================================================================
# USER LLM ENTITLEMENTS RESPONSE MODELS
# ============================================================================


class UserEntitlementResponse(BaseModel):
    """
    Response schema for user LLM entitlement.

    Security: API key is intentionally excluded from response for security.
    """

    entitlement_id: int = Field(..., description="Unique entitlement identifier")
    user_id: UUID = Field(..., description="User who has the entitlement")
    llm_provider: str = Field(..., description="LLM provider type")
    llm_model_name: str = Field(..., description="Logical model name")
    api_key_variable_name: str | None = Field(
        None, description="Environment variable name for the API key"
    )
    api_endpoint_url: str | None = Field(None, description="Specific API endpoint URL")
    cloud_provider: str | None = Field(
        None, description="Cloud provider hosting the LLM"
    )
    deployment_name: str | None = Field(
        None, description="Physical deployment identifier"
    )
    deployment_region: str | None = Field(
        None, description="Geographic deployment region"
    )
    created_at: datetime = Field(..., description="When entitlement was created")
    updated_at: datetime = Field(..., description="When entitlement was last updated")
    created_by_user_id: UUID = Field(
        ..., description="Admin who created this entitlement"
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider matches database schema."""
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(f"Invalid LLM provider: {v}")
        return v

    @field_validator("cloud_provider")
    @classmethod
    def validate_cloud_provider(cls, v: str | None) -> str | None:
        """Validate cloud provider matches database schema."""
        if v is not None and v not in VALID_CLOUD_PROVIDERS:
            raise ValueError(f"Invalid cloud provider: {v}")
        return v

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "entitlement_id": 1,
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_key_variable_name": "OPENAI_API_KEY",
                "api_endpoint_url": "https://api.openai.com/v1",
                "cloud_provider": None,
                "deployment_name": None,
                "deployment_region": None,
                "created_at": "2025-10-21T10:00:00Z",
                "updated_at": "2025-10-21T10:00:00Z",
                "created_by_user_id": "a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6",
            }
        }


class UserEntitlementListResponse(BaseModel):
    """
    Response schema for paginated list of user LLM entitlements.
    """

    entitlements: list[UserEntitlementResponse] = Field(
        ..., description="List of user entitlements"
    )
    total_count: int = Field(..., description="Total number of entitlements")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")

    class Config:
        json_schema_extra = {
            "example": {
                "entitlements": [
                    {
                        "entitlement_id": 1,
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "llm_provider": "openai",
                        "llm_model_name": "gpt-4o",
                        "api_endpoint_url": "https://api.openai.com/v1",
                        "cloud_provider": None,
                        "deployment_name": None,
                        "deployment_region": "us-east-1",
                        "created_at": "2025-10-21T10:00:00Z",
                        "updated_at": "2025-10-21T10:00:00Z",
                        "created_by_user_id": "a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6",
                    }
                ],
                "total_count": 1,
                "page": 1,
                "page_size": 50,
            }
        }


class EntitlementDeleteUserDetails(BaseModel):
    """User details included in delete entitlement response."""

    user_id: UUID = Field(..., description="Target user ID")
    username: str = Field(..., description="Target username")
    email: str = Field(..., description="Target user email")


class EntitlementDeleteEntitlementDetails(BaseModel):
    """Entitlement details included in delete entitlement response."""

    llm_provider: str = Field(..., description="LLM provider type")
    llm_model_name: str = Field(..., description="Logical model name")
    cloud_provider: str | None = Field(
        default=None, description="Cloud provider hosting the LLM"
    )
    api_endpoint_url: str | None = Field(
        default=None, description="Specific API endpoint URL"
    )


class EntitlementDeleteActor(BaseModel):
    """Actor metadata for delete entitlement response."""

    admin_user_id: UUID = Field(..., description="Admin user ID who performed deletion")
    admin_username: str = Field(
        ..., description="Admin username who performed deletion"
    )


class EntitlementDeleteResponse(BaseModel):
    """
    Response schema for successful entitlement deletion.

    Note: This models only the 200-success response payload.
    """

    deletion_status: Literal["success"] = Field(
        ..., description="Deletion status (success)"
    )
    entitlement_id: int = Field(..., description="Deleted entitlement identifier")
    user_details: EntitlementDeleteUserDetails = Field(
        ..., description="Target user details"
    )
    entitlement_details: EntitlementDeleteEntitlementDetails = Field(
        ..., description="Deleted entitlement details"
    )
    deleted_by: EntitlementDeleteActor = Field(
        ..., description="Admin/owner who performed deletion"
    )
    message: str = Field(..., description="Human-friendly deletion summary")
