"""
PAUSE — take a deployment out of rotation by inserting a PAUSED full-capacity
blocker allocation.

A PAUSED row that consumes the deployment's whole ``token_capacity_limit`` makes
the load balancer see the endpoint as full, so new traffic routes elsewhere.
Concurrent pause requests are serialized by a ``FOR UPDATE`` lock on the target
``tenant_deployments`` row: the lock, the already-paused check, and the insert
all run in one transaction, so no cross-session TOCTOU window exists.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Any
import uuid
from uuid import UUID

from loguru import logger
from sqlalchemy import text

from app.persistence.allocations._base import AllocationPersistenceBase
from app.persistence.queries.allocation_queries import (
    CHECK_ACTIVE_PAUSE_ALLOCATION_EXISTS_SQL,
    CREATE_TOKEN_ALLOCATION_SQL,
)
from app.persistence.queries.deployment_capacity_queries import (
    GET_ACTIVE_DEPLOYMENT_BY_MODEL_AND_ENDPOINT_LOCKED_SQL,
)


class AllocationPauseMixin(AllocationPersistenceBase):
    """The pause write path (PAUSED blocker) against ``llm_token_allocations``."""

    async def pause_deployment(
        self,
        tenant_id: UUID,
        user_id: UUID,
        provider_name: str,
        model_name: str,
        api_endpoint: str,
        pause_reason: str = "",
        pause_duration_minutes: int = 30,
    ) -> dict[str, Any]:
        """
        Pause a deployment by inserting a PAUSED full-capacity blocker.

        Returns the created allocation dict on success, or a sentinel dict with
        ``alloc_status`` = NOT_FOUND / ALREADY_PAUSED.
        """
        if pause_duration_minutes <= 0:
            raise ValueError(
                f"Pause duration must be positive, got {pause_duration_minutes}"
            )
        self.validate_uuid(tenant_id, "tenant_id")
        self.validate_uuid(user_id, "user_id")
        self.validate_string_not_empty(provider_name, "provider_name")
        self.validate_string_not_empty(api_endpoint, "api_endpoint")

        try:
            async with self.get_session() as session:
                # Step 1 — Lock the deployment row (serializes concurrent pauses).
                result = await session.execute(
                    text(GET_ACTIVE_DEPLOYMENT_BY_MODEL_AND_ENDPOINT_LOCKED_SQL),
                    {
                        "tenant_id": tenant_id,
                        "llm_provider": provider_name,
                        "llm_model_name": model_name,
                        "api_endpoint_url": api_endpoint,
                    },
                )
                row = result.mappings().one_or_none()
                if not row:
                    logger.warning(
                        f"Deployment not found: {model_name} at {api_endpoint}"
                    )
                    return {
                        "alloc_status": "NOT_FOUND",
                        "model_name": model_name,
                        "api_endpoint_url": api_endpoint,
                        "reason": "Deployment not found",
                    }

                deployment: dict[str, Any] = dict(row)

                # Step 2 — Check for an existing active pause under the lock.
                existing = await session.execute(
                    text(CHECK_ACTIVE_PAUSE_ALLOCATION_EXISTS_SQL),
                    {"deployment_id": deployment["deployment_id"]},
                )
                if existing.scalar_one_or_none():
                    logger.warning(
                        f"Deployment {model_name} at {api_endpoint} already paused."
                    )
                    return {
                        "alloc_status": "ALREADY_PAUSED",
                        "model_name": model_name,
                        "api_endpoint_url": api_endpoint,
                        "reason": "Deployment is already in a paused state.",
                    }

                # Step 3 — Insert the PAUSED blocker in the same transaction.
                context: dict[str, Any] = {"operation": "pause_deployment"}
                if pause_reason:
                    context["reason"] = pause_reason

                params = {
                    "token_request_id": f"pause_{uuid.uuid4().hex}",
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "deployment_id": deployment["deployment_id"],
                    "provider_name": deployment["provider_name"],
                    "model_name": deployment["model_name"],
                    "deployment_key": deployment["deployment_key"],
                    "deployment_name": deployment.get("deployment_name"),
                    "provider_deployment_name": deployment.get(
                        "provider_deployment_name"
                    ),
                    "api_endpoint_url": deployment["api_endpoint_url"],
                    "cloud_provider": deployment.get("cloud_provider"),
                    "cloud_region": deployment.get("cloud_region"),
                    "token_count": deployment["token_capacity_limit"],
                    "allocation_status": "PAUSED",
                    "allocated_at": datetime.now(),
                    "expires_at": datetime.now()
                    + timedelta(minutes=pause_duration_minutes),
                    "request_context": json.dumps(context),
                    "temperature": None,
                    "top_p": None,
                    "seed": None,
                }
                insert_result = await session.execute(
                    text(CREATE_TOKEN_ALLOCATION_SQL), params
                )
                created = insert_result.mappings().one_or_none()
                if not created:
                    raise RuntimeError("Failed to create pause allocation record")

                logger.info(
                    f"Paused {model_name} at {api_endpoint} for {pause_duration_minutes}m"
                )
                return dict(created)

        except ValueError as e:
            logger.error(f"Value error in pause_deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error in pause_deployment: {e}")
            raise
