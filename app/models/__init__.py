"""
Database Models Package
---------------------
Pydantic models that exactly match the PostgreSQL database schema.

This package provides validated data models for:
- User management (users table)
- LLM model configurations (llm_models table)
- Token allocations (token_manager table)
- API request/response models

All models include field validation matching database CHECK constraints.
"""

from app.models.users import User
from app.models.llm_models import LLMModel
from app.models.token_allocation import TokenAllocation
from app.models.request_models import (
    LLMRequestCreate,
    LLMRequestResponse,
    LLMTaskStatus,
    LLMTaskResult,
    HealthStatus,
    DependencyHealth,
)

__all__ = [
    # Database models
    "User",
    "LLMModel",
    "TokenAllocation",
    # Request/Response models
    "LLMRequestCreate",
    "LLMRequestResponse",
    "LLMTaskStatus",
    "LLMTaskResult",
    "HealthStatus",
    "DependencyHealth",
]
