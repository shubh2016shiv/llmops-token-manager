"""
Response Models
--------------
Pydantic models for API response validation.
Simple, focused schemas for returning data to clients.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


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
    created_at: datetime
    updated_at: datetime

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

    IMPORTANT: Field names with 'model_' prefix have been renamed to 'llm_*' to avoid
    conflicts with Pydantic's protected namespaces.
    """

    token_request_id: str = Field(
        ..., description="Unique identifier for this allocation"
    )
    user_id: UUID = Field(..., description="User who received the allocation")
    llm_model_name: str = Field(
        ...,
        description="Model name for this allocation",
        alias="model_name",  # Maps to database column 'model_name'
    )
    token_count: int = Field(..., description="Number of tokens allocated")
    allocation_status: str = Field(..., description="Current allocation status")
    allocated_at: datetime = Field(..., description="When tokens were allocated")
    expires_at: Optional[datetime] = Field(
        default=None, description="When allocation expires (if applicable)"
    )
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint to use")
    region: Optional[str] = Field(
        default=None, description="Region where model is deployed"
    )

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "token_request_id": "req_abc123xyz",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "llm_model_name": "gpt-4",
                "token_count": 2000,
                "allocation_status": "ACQUIRED",
                "allocated_at": "2025-10-13T10:30:00Z",
                "expires_at": "2025-10-13T10:35:00Z",
                "api_endpoint": "https://api.openai.com/v1",
                "region": "eastus2",
            }
        }


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

    provider_name: str = Field(..., description="LLM provider name")
    llm_model_name: str = Field(..., description="Name of the LLM model")
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    api_key_variable_name: Optional[str] = Field(
        default=None, description="Environment variable name for API key"
    )
    api_endpoint_url: Optional[str] = Field(
        default=None, description="API endpoint URL"
    )
    llm_model_version: Optional[str] = Field(default=None, description="Model version")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens per request"
    )
    tokens_per_minute_limit: Optional[int] = Field(
        default=None, description="Token rate limit per minute"
    )
    requests_per_minute_limit: Optional[int] = Field(
        default=None, description="Request rate limit per minute"
    )
    is_active_status: bool = Field(..., description="Whether model is currently active")
    temperature: Optional[float] = Field(
        default=None, description="Default temperature setting"
    )
    random_seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible results"
    )
    deployment_region: Optional[str] = Field(
        default=None, description="Geographic deployment region"
    )
    created_at: datetime = Field(..., description="When model was registered")
    updated_at: datetime = Field(..., description="When model was last updated")

    class Config:
        # Enable ORM mode for SQLAlchemy compatibility
        from_attributes = True
        json_schema_extra = {
            "example": {
                "provider_name": "openai",
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

    models: List[LLMModelResponse] = Field(
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
                        "provider_name": "openai",
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
    details: Optional[Dict[str, Any]] = Field(
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


class AllocationListResponse(BaseModel):
    """
    Response schema for listing token allocations.
    """

    allocations: List[TokenAllocationResponse] = Field(
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

    Each component is represented as a boolean indicating if it's operational.
    """

    postgresql: bool = Field(..., description="PostgreSQL database health status")
    redis: bool = Field(..., description="Redis cache health status")
    rabbitmq: bool = Field(..., description="RabbitMQ message broker health status")
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
    version: Optional[str] = Field(default=None, description="Application version")
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
