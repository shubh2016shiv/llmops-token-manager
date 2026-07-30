"""
ACQUIRE — insert a new token allocation with an atomic, per-deployment capacity check.

The caller (service layer) has already picked the deployment (via the load
balancer). This just reserves against that deployment_id: it locks the
tenant_deployments row, recomputes current load, decides ACQUIRED vs WAITING,
and inserts — all in one transaction so concurrent requests can't over-allocate.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import text

from app.persistence.allocations._base import AllocationPersistenceBase
from app.persistence.queries.allocation_queries import (
    CREATE_TOKEN_ALLOCATION_SQL,
    CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL,
)

if TYPE_CHECKING:
    from uuid import UUID


class AllocationAcquireMixin(AllocationPersistenceBase):
    """The acquire write path against ``llm_token_allocations``."""

    async def create_reserved_allocation(
        self,
        token_request_identifier: str,
        tenant_id: UUID,
        user_id: UUID,
        deployment_id: UUID,
        provider_name: str,
        model_name: str,
        deployment_key: str,
        api_endpoint_url: str,
        token_count: int,
        allocation_status: str = "ACQUIRED",
        deployment_name: str | None = None,
        provider_deployment_name: str | None = None,
        cloud_provider: str | None = None,
        cloud_region: str | None = None,
        expiration_timestamp: datetime | None = None,
        request_metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Persist an allocation whose capacity was ALREADY reserved on the Redis
        fast path — a plain insert with no capacity re-check.

        Used by the async RabbitMQ consumer to durably record a fast-path
        allocation after the caller already received ACQUIRED. Idempotent at the
        DB level via the token_request_id primary key.
        """
        self.validate_string_not_empty(
            token_request_identifier, "token_request_identifier"
        )
        self.validate_uuid(tenant_id, "tenant_id")
        self.validate_uuid(user_id, "user_id")
        self.validate_uuid(deployment_id, "deployment_id")
        self.validate_allocation_status(allocation_status)
        request_context_json = self._validate_and_serialize_json(
            request_metadata, "request_metadata"
        )

        params = {
            "token_request_id": token_request_identifier,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "deployment_id": deployment_id,
            "provider_name": provider_name,
            "model_name": model_name,
            "deployment_key": deployment_key,
            "deployment_name": deployment_name,
            "provider_deployment_name": provider_deployment_name,
            "api_endpoint_url": api_endpoint_url,
            "cloud_provider": cloud_provider,
            "cloud_region": cloud_region,
            "token_count": token_count,
            "allocation_status": allocation_status,
            "allocated_at": datetime.now(),
            "expires_at": expiration_timestamp,
            "request_context": request_context_json or "{}",
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(CREATE_TOKEN_ALLOCATION_SQL), params
                )
                created = result.mappings().one_or_none()
                if not created:
                    raise RuntimeError("Failed to persist reserved allocation")
                return dict(created)
        except Exception as e:
            logger.error(
                f"Error persisting reserved allocation {token_request_identifier}: {e}"
            )
            raise

    async def create_allocation_with_capacity_check(
        self,
        token_request_identifier: str,
        tenant_id: UUID,
        user_id: UUID,
        deployment_id: UUID,
        provider_name: str,
        model_name: str,
        token_count: int,
        expiration_timestamp: datetime | None = None,
        request_metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Reserve tokens against an already-chosen deployment, atomically.

        Returns the created allocation record (status ACQUIRED or WAITING).

        Raises:
            ValueError: If the deployment cannot fit a single request of this size.
        """
        self.validate_string_not_empty(
            token_request_identifier, "token_request_identifier"
        )
        self.validate_uuid(tenant_id, "tenant_id")
        self.validate_uuid(user_id, "user_id")
        self.validate_uuid(deployment_id, "deployment_id")
        self.validate_string_not_empty(provider_name, "provider_name")
        self.validate_llm_provider(provider_name)
        self.validate_string_not_empty(model_name, "model_name")
        self.validate_positive_integer(token_count, "token_count")
        request_context_json = self._validate_and_serialize_json(
            request_metadata, "request_metadata"
        )

        try:
            async with self.get_session() as session:
                params = {
                    "token_request_id": token_request_identifier,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "deployment_id": deployment_id,
                    "provider_name": provider_name,
                    "model_name": model_name,
                    "token_count": token_count,
                    "allocated_at": datetime.now(),
                    "expires_at": expiration_timestamp,
                    "request_context": request_context_json or "{}",
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                }
                result = await session.execute(
                    text(CREATE_TOKEN_ALLOCATION_WITH_CAPACITY_CHECK_SQL), params
                )
                created = result.mappings().one_or_none()
                if not created:
                    raise ValueError(
                        "No active deployment with enough single-request capacity "
                        f"for {provider_name}/{model_name} (deployment {deployment_id})"
                    )

                allocation = dict(created)
                self.log_operation(
                    "ACQUIRE",
                    token_request_identifier,
                    success=True,
                    additional_context=(
                        f"{token_count} tokens for {model_name} "
                        f"as {allocation.get('allocation_status')}"
                    ),
                )
                return allocation
        except Exception as e:
            logger.error(f"Error acquiring allocation {token_request_identifier}: {e}")
            raise
