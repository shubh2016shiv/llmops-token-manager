"""
Request Models
-------------
Pydantic models for API request validation.
Simple, focused schemas aligned with the token allocation system.
"""

from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class TokenAllocationRequest(BaseModel):
    """
    Request schema for allocating LLM tokens.
    Maps to the core fields needed for token_manager table.
    """

    user_id: UUID = Field(..., description="User requesting token allocation")
    model_name: str = Field(
        ...,
        description="Name of the LLM model (e.g., gpt-4, claude-3-opus)",
        min_length=1,
        max_length=100,
    )
    token_count: int = Field(
        ..., description="Number of tokens to allocate", gt=0, le=1000000
    )
    deployment_name: Optional[str] = Field(
        default=None, description="Specific deployment name if applicable"
    )
    region: Optional[str] = Field(
        default=None, description="Preferred geographic region (e.g., eastus2, westus2)"
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature parameter for generation", ge=0.0, le=2.0
    )
    top_p: Optional[float] = Field(
        default=None, description="Top-p (nucleus sampling) parameter", ge=0.0, le=1.0
    )
    seed: Optional[float] = Field(
        default=None, description="Seed for reproducible outputs"
    )
    request_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context metadata"
    )

    @field_validator("token_count")
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Token count must be positive")
        if v > 1000000:
            raise ValueError("Token count exceeds maximum limit")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "model_name": "gpt-4",
                "token_count": 2000,
                "deployment_name": "gpt-4-turbo",
                "region": "eastus2",
                "temperature": 0.7,
                "request_context": {"application": "chatbot", "team": "ml-research"},
            }
        }


class TokenReleaseRequest(BaseModel):
    """
    Request schema for releasing allocated tokens.
    """

    token_request_id: str = Field(
        ..., description="ID of the token allocation to release", min_length=1
    )
    user_id: UUID = Field(..., description="User releasing the tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "token_request_id": "req_abc123xyz",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


class UserCreateRequest(BaseModel):
    """
    Request schema for creating a new user.
    """

    email: str = Field(
        ..., description="User's email address", min_length=3, max_length=255
    )
    role: str = Field(
        default="developer", description="User role: owner, admin, developer, or viewer"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = ["owner", "admin", "developer", "viewer"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower().strip()

    class Config:
        json_schema_extra = {
            "example": {"email": "user@example.com", "role": "developer"}
        }


class LLMModelCreateRequest(BaseModel):
    """
    Request schema for registering a new LLM model.
    """

    provider: str = Field(..., description="LLM provider: openai, gemini, or anthropic")
    model_name: str = Field(
        ..., description="Name of the model", min_length=1, max_length=100
    )
    deployment_name: Optional[str] = Field(default=None, description="Deployment name")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint URL")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens per request", gt=0
    )
    tokens_per_minute_limit: Optional[int] = Field(
        default=None, description="Rate limit: tokens per minute", gt=0
    )
    requests_per_minute_limit: Optional[int] = Field(
        default=None, description="Rate limit: requests per minute", gt=0
    )
    region: Optional[str] = Field(default=None, description="Geographic region")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed_providers = ["openai", "gemini", "anthropic"]
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {', '.join(allowed_providers)}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model_name": "gpt-4",
                "deployment_name": "gpt-4-turbo",
                "max_tokens": 8192,
                "tokens_per_minute_limit": 90000,
                "region": "eastus2",
            }
        }
