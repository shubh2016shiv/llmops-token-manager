"""
Database Services Package
-------------------------
Production-ready database services for LLM token management system.

This package provides:
- Base service class with optimized connection pooling and transaction management
- User management service (CRUD operations for users)
- LLM model management service (CRUD operations for model configurations)
- Token allocation service (token allocation, tracking, and lifecycle management)

All services are optimized for high-concurrency environments (10,000+ concurrent users)
with proper error handling, validation, and logging.
"""

from app.psql_db_services.base_service import BaseDatabaseService
from app.psql_db_services.users_service import UsersService
from app.psql_db_services.llm_models_service import LLMModelsService
from app.psql_db_services.token_allocation_manager import TokenAllocationService

__all__ = [
    "BaseDatabaseService",
    "UsersService",
    "LLMModelsService",
    "TokenAllocationService",
]
