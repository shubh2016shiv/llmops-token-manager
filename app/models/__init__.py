"""
Database Models Package
---------------------
Pydantic models that exactly match the PostgreSQL database schema.

This package provides validated data models for:
- User management (users table)
- LLM model configurations (llm_models table) - using response models
- Token allocations (token_manager table)
- API request/response models
- Token estimation models

All models include field validation matching database CHECK constraints.
"""

# Core database models
from app.models.users_model import User
from app.models.token_manager_models import TokenAllocation, TokenEstimation, InputType

# Request models
from app.models.request_models import (
    # Enums
    ProviderType,
    UserRole,
    UserStatus,
    # User management requests
    UserCreateRequest,
    UserUpdateRequest,
    # Token allocation requests
    TokenAllocationRequest,
    TokenReleaseRequest,
    # Deployment management requests
    PauseDeploymentRequest,
    ResumeDeploymentRequest,
    DeploymentConfigCreate,
    DeploymentConfigUpdate,
)

# Response models
from app.models.response_models import (
    # Enums
    AllocationStatus,
    Health,
    # User responses
    UserResponse,
    # Token allocation responses
    TokenAllocationResponse,
    TokenReleaseResponse,
    AllocationListResponse,
    # LLM model responses
    LLMModelResponse,
    LLMModelListResponse,
    # Health and error responses
    HealthStatus,
    DependencyHealth,
    ErrorResponse,
)

__all__ = [
    # Core database models
    "User",
    "TokenAllocation",
    "TokenEstimation",
    "InputType",
    # Enums
    "AllocationStatus",
    "ProviderType",
    "UserRole",
    "UserStatus",
    "Health",
    # User management
    "UserCreateRequest",
    "UserUpdateRequest",
    "UserResponse",
    # Token allocation
    "TokenAllocationRequest",
    "TokenReleaseRequest",
    "TokenAllocationResponse",
    "TokenReleaseResponse",
    "AllocationListResponse",
    # Deployment management
    "PauseDeploymentRequest",
    "ResumeDeploymentRequest",
    "DeploymentConfigCreate",
    "DeploymentConfigUpdate",
    # LLM models
    "LLMModelResponse",
    "LLMModelListResponse",
    # Health and errors
    "HealthStatus",
    "DependencyHealth",
    "ErrorResponse",
]
