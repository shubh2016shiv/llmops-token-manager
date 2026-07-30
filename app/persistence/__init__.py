"""
Persistence Package
-------------------
PostgreSQL-backed persistence for the LLM token management system.

Token manager owns exactly one table: llm_token_allocations. Tenant, user,
and deployment-capacity data are owned by llm_services — token manager only
reads deployment capacity through the token_manager_deployment_capacity view
(see LLMTokenAllocationPersistence / TokenMaintenancePersistence). It never
queries users, tenants, or entitlement tables directly.

Themes:
- allocations/              — the token manager's owned table, one file per
  operation (acquire/release/retry/pause)
- deployed_llm_endpoints    — read-only reads of deployed LLM endpoints via
  llm_services' read view (no decision logic)
- token_maintenance         — periodic reconciliation/cleanup

This package provides:
- A shared base persistence class with session and validation utilities
- Token allocation persistence (the owned table)
- Deployed LLM endpoint reads (read-only reference data)
"""

from app.persistence.allocations import LLMTokenAllocationPersistence
from app.persistence.base import BasePersistence
from app.persistence.deployed_llm_endpoints import DeployedLLMReadPersistence
from app.persistence.token_maintenance import TokenMaintenancePersistence

__all__ = [
    "BasePersistence",
    "LLMTokenAllocationPersistence",
    "DeployedLLMReadPersistence",
    "TokenMaintenancePersistence",
]
