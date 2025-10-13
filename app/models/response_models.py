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


class TokenAllocationResponse(BaseModel):
    """
    Response schema for successful token allocation.
    """

    token_request_id: str = Field(
        ..., description="Unique identifier for this allocation"
    )
    user_id: UUID = Field(..., description="User who received the allocation")
    model_name: str = Field(..., description="Model name for this allocation")
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
        json_schema_extra = {
            "example": {
                "token_request_id": "req_abc123xyz",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "model_name": "gpt-4",
                "token_count": 2000,
                "allocation_status": "ACQUIRED",
                "allocated_at": "2025-10-13T10:30:00Z",
                "expires_at": "2025-10-13T10:35:00Z",
                "api_endpoint": "https://api.openai.com/v1",
                "region": "eastus2",
            }
        }


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


class UserResponse(BaseModel):
    """
    Response schema for user data.
    """

    user_id: UUID = Field(..., description="User's unique identifier")
    email: str = Field(..., description="User's email address")
    role: str = Field(..., description="User's role")
    status: str = Field(..., description="User's account status")
    created_at: datetime = Field(..., description="When user was created")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "role": "developer",
                "status": "active",
                "created_at": "2025-10-01T08:00:00Z",
                "updated_at": "2025-10-13T10:30:00Z",
            }
        }


class LLMModelResponse(BaseModel):
    """
    Response schema for LLM model data.
    """

    model_id: UUID = Field(..., description="Model's unique identifier")
    provider: str = Field(..., description="LLM provider")
    model_name: str = Field(..., description="Model name")
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
        json_schema_extra = {
            "example": {
                "model_id": "650e8400-e29b-41d4-a716-446655440000",
                "provider": "openai",
                "model_name": "gpt-4",
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
                        "model_name": "gpt-4",
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


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """

    status: str = Field(..., description="Overall system health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )
    database: bool = Field(..., description="Database connection status")
    version: Optional[str] = Field(default=None, description="Application version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-13T10:30:00Z",
                "database": True,
                "version": "1.0.0",
            }
        }
