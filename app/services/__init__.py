"""
Services Package — Application-layer use-case orchestration.

============================================================

This package contains the business logic layer of the LLM Token Manager.
Each service owns exactly one use case and coordinates across persistence
adapters, Redis, and the RabbitMQ publisher without importing HTTP or
framework specifics.

Architecture:
-------------
    ┌─────────────────────────────────────────────────────────┐
    │  app/api/  (Interface Layer)                            │
    │  FastAPI routers — routing, auth, HTTP serialisation    │
    └───────────────────────┬─────────────────────────────────┘
                            │  FastAPI Depends()
    ┌───────────────────────▼─────────────────────────────────┐
    │  app/services/  (Application Layer — this package)      │
    │                                                         │
    │  TokenAcquisitionService  — acquire_tokens use case     │
    │  TokenReleaseService      — release_tokens use case     │
    │  TokenRetryService        — retry_acquire use case      │
    └──────────┬─────────────────────────┬───────────────────┘
               │                         │
    ┌──────────▼──────────┐   ┌──────────▼──────────────────┐
    │  app/persistence/   │   │  app/resilience/             │
    │  (Adapter Layer)    │   │  Redis + RMQ adapters        │
    └─────────────────────┘   └──────────────────────────────┘

Layer contract:
    - Services MUST NOT import from app/api/ or reference HTTP concepts.
    - Services MUST NOT call os.environ or instantiate infrastructure clients.
    - Services receive all dependencies via constructor injection.
    - Errors are expressed as domain exceptions from app/core/exceptions.py.
    - Endpoints catch domain exceptions and map them to HTTP status codes.

Author: Engineering Team
Last Updated: 2026-05-10
"""

from app.services.token_acquisition_service import TokenAcquisitionService
from app.services.token_release_service import TokenReleaseService
from app.services.token_retry_service import TokenRetryService

__all__ = [
    "TokenAcquisitionService",
    "TokenReleaseService",
    "TokenRetryService",
]
