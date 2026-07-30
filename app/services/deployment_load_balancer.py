"""
DeploymentLoadBalancer — chooses which deployment a token request routes to.

This is a decision, not data access. It reads candidate deployments and their
current load from the persistence layer, then ranks them least-loaded first so
the caller can place a request on the emptiest endpoint and fail over to the
next one when the top pick is full.

    persistence reads  →  load balancer ranks  →  acquisition service reserves

Example: a tenant runs gpt-5 on three endpoints (azure-east, azure-west,
openai-direct). If azure-east is busiest, it ranks last; the request lands on
whichever endpoint currently holds the fewest reserved tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.exceptions import DeploymentConfigurationError

if TYPE_CHECKING:
    from uuid import UUID

    from app.persistence.deployed_llm_endpoints import DeployedLLMReadPersistence


class DeploymentLoadBalancer:
    """Ranks a tenant's active deployed endpoints for a model by available capacity."""

    def __init__(self, endpoint_reads: DeployedLLMReadPersistence) -> None:
        """Initialise with the read-only deployed-endpoint persistence adapter."""
        self._endpoint_reads = endpoint_reads

    async def rank_by_available_capacity(
        self, tenant_id: UUID, provider_name: str, model_name: str
    ) -> list[dict[str, Any]]:
        """
        Return the tenant's active deployments for provider/model, least-loaded first.

        Ordered in failover order. Each returned deployment dict carries a
        ``current_token_load`` key so the caller knows how full it already is.
        Ties break by higher ``routing_priority``.

        Raises:
            DeploymentConfigurationError: If the tenant has no active deployment
                for this provider/model.
        """
        candidates = await self._endpoint_reads.list_active_endpoints(
            tenant_id, provider_name, model_name
        )
        if not candidates:
            raise DeploymentConfigurationError(
                f"{provider_name}/{model_name}", "active deployment"
            )

        load_by_deployment = (
            await self._endpoint_reads.get_active_token_load_per_endpoint(
                tenant_id, provider_name, model_name
            )
        )
        for deployment in candidates:
            load = load_by_deployment.get(deployment["deployment_id"], 0)
            deployment["current_token_load"] = load
            deployment["available_token_capacity"] = (
                deployment["token_capacity_limit"] - load
            )

        # Most available headroom first (correct across differing capacity
        # limits); ties broken by higher routing_priority.
        candidates.sort(
            key=lambda d: (
                -d["available_token_capacity"],
                -(d.get("routing_priority") or 0),
            )
        )
        return candidates

    async def choose_least_loaded(
        self, tenant_id: UUID, provider_name: str, model_name: str
    ) -> dict[str, Any]:
        """
        Return the single least-loaded active deployment (the top of the ranking).

        Raises:
            DeploymentConfigurationError: If no active deployment exists.
        """
        ranked = await self.rank_by_available_capacity(
            tenant_id, provider_name, model_name
        )
        return ranked[0]
