"""
Response Models
--------------
Pydantic models for API response validation.
Simple, focused schemas for returning data to clients.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field
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
    # Renamed from model_name to llm_name
    llm_name: str = Field(
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
                "llm_name": "gpt-4",  # Updated field name
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

    IMPORTANT: Field names with 'model_' prefix have been renamed to 'llm_*' to avoid
    conflicts with Pydantic's protected namespaces.
    """

    # Renamed from model_id to llm_id
    llm_id: UUID = Field(
        ...,
        description="Model's unique identifier",
        alias="model_id",  # Maps to database column 'model_id'
    )
    provider: str = Field(..., description="LLM provider")
    # Renamed from model_name to llm_name
    llm_name: str = Field(
        ...,
        description="Model name",
        alias="model_name",  # Maps to database column 'model_name'
    )
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens per request"
    )
    is_active: bool = Field(..., description="Whether model is currently active")
    region: Optional[str] = Field(default=None, description="Geographic region")
    total_requests: int = Field(..., description="Total requests processed")
    total_tokens_processed: int = Field(..., description="Total tokens processed")
    created_at: datetime = Field(..., description="When model was registered")
    last_used_at: Optional[datetime] = Field(
        default=None, description="Last usage timestamp"
    )

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "llm_id": "650e8400-e29b-41d4-a716-446655440000",  # Updated field name
                "provider": "openai",
                "llm_name": "gpt-4",  # Updated field name
                "deployment_name": "gpt-4-turbo",
                "api_endpoint": "https://api.openai.com/v1",
                "max_tokens": 8192,
                "is_active": True,
                "region": "eastus2",
                "total_requests": 1500,
                "total_tokens_processed": 3000000,
                "created_at": "2025-09-01T08:00:00Z",
                "last_used_at": "2025-10-13T10:30:00Z",
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
                        "llm_name": "gpt-4",  # Updated field name
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


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """

    status: str = Field(..., description="Token allocation service health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )
    version: Optional[str] = Field(default=None, description="Application version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": Health.HEALTHY,
                "timestamp": "2025-10-13T10:30:00Z",
                "version": "1.0.0",
            }
        }

        @Field.validate("status")
        @classmethod
        def validate_status(cls, value: str) -> str:
            if value not in Health._value2member_map_:
                raise ValueError(
                    f"Status must be one of: {', '.join(Health._value2member_map_)}"
                )
            return value

        @Field.validate("timestamp")
        @classmethod
        def validate_timestamp(cls, value: datetime) -> datetime:
            if value > datetime.utcnow():
                raise ValueError("Timestamp cannot be in the future")
            return value
