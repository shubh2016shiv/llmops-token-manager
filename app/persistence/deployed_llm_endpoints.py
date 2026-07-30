"""
Deployed LLM endpoints — the read contract llm_services exposes to the token manager.

Pure data access, no decision. A "deployed LLM endpoint" is one concrete place a
token request can be routed to: a tenant's provider + model + cloud_provider +
cloud_region + endpoint URL (one row in llm_services' ``tenant_deployments``,
surfaced through the ``token_manager_deployment_capacity`` view).

Two reads that together let the service layer choose where to place a request:
- ``list_active_endpoints``                — candidate endpoints for a
  tenant/provider/model (from the read view).
- ``get_active_token_load_per_endpoint``   — currently reserved tokens per
  endpoint (from the token manager's own allocations table).

Ranking these by available capacity and picking one is a load-balancing
decision — that lives in ``app/services/deployment_load_balancer.py``, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import text

from app.persistence.base import BasePersistence
from app.persistence.queries.deployment_capacity_queries import (
    LIST_ACTIVE_MODEL_DEPLOYMENTS_SQL,
    LIST_LEAST_LOADED_ALLOCATIONS_BY_MODEL_SQL,
)

if TYPE_CHECKING:
    from uuid import UUID


class DeployedLLMReadPersistence(BasePersistence):
    """Read-only access to deployed LLM endpoints and their current token load."""

    async def list_active_endpoints(
        self, tenant_id: UUID, provider_name: str, model_name: str
    ) -> list[dict[str, Any]]:
        """Return active deployed endpoints for a tenant/provider/model."""
        self.validate_uuid(tenant_id, "tenant_id")
        self.validate_string_not_empty(provider_name, "provider_name")
        self.validate_string_not_empty(model_name, "model_name")
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(LIST_ACTIVE_MODEL_DEPLOYMENTS_SQL),
                    {
                        "tenant_id": tenant_id,
                        "llm_provider": provider_name,
                        "llm_model_name": model_name,
                    },
                )
                return [dict(row) for row in result.mappings().all()]
        except Exception as e:
            logger.error(
                f"Error listing endpoints for {provider_name}/{model_name}: {e}"
            )
            raise

    async def get_active_token_load_per_endpoint(
        self, tenant_id: UUID, provider_name: str, model_name: str
    ) -> dict[Any, int]:
        """Return ``{deployment_id: reserved_tokens}`` for active allocations."""
        self.validate_uuid(tenant_id, "tenant_id")
        self.validate_string_not_empty(provider_name, "provider_name")
        self.validate_string_not_empty(model_name, "model_name")
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(LIST_LEAST_LOADED_ALLOCATIONS_BY_MODEL_SQL),
                    {
                        "tenant_id": tenant_id,
                        "llm_provider": provider_name,
                        "llm_model_name": model_name,
                    },
                )
                return {
                    row["deployment_id"]: int(row["total_tokens"] or 0)
                    for row in result.mappings().all()
                }
        except Exception as e:
            logger.error(
                f"Error loading endpoint token load for {provider_name}/{model_name}: {e}"
            )
            raise
