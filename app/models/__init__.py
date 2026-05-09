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
# Request models
from app.models.request_models import (
    CloudProvider,
    DeploymentConfigCreate,
    DeploymentConfigUpdate,
    # Enums
    LLMProvider,
    # Deployment management requests
    PauseDeploymentRequest,
    ResumeDeploymentRequest,
    # Token allocation requests
    TokenAllocationRequest,
    TokenReleaseRequest,
    # User management requests
    UserCreateRequest,
    UserRole,
    UserStatus,
    UserUpdateRequest,
)

# Response models
from app.models.resilience_models import (
    BackpressureDecision,
    CircuitBreakerSnapshot,
    CounterSeedRecord,
    DeploymentCapacitySnapshot,
    DlqPayload,
    TokenAllocationPersistPayload,
)
from app.models.response_models import (
    AllocationListResponse,
    # Enums
    AllocationStatus,
    DependencyHealth,
    ErrorResponse,
    Health,
    # Health and error responses
    HealthStatus,
    LLMModelListResponse,
    # LLM model responses
    LLMModelResponse,
    # Token allocation responses
    TokenAllocationResponse,
    TokenReleaseResponse,
    # User responses
    UserResponse,
)
from app.models.token_manager_models import InputType, TokenAllocation, TokenEstimation
from app.models.users_models import User

__all__ = [
    # Core database models
    "User",
    "TokenAllocation",
    "TokenEstimation",
    "InputType",
    "TokenAllocationPersistPayload",
    "DlqPayload",
    "CounterSeedRecord",
    "DeploymentCapacitySnapshot",
    "CircuitBreakerSnapshot",
    "BackpressureDecision",
    # Enums
    "AllocationStatus",
    "LLMProvider",
    "CloudProvider",
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
