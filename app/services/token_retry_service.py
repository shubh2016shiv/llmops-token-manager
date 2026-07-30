"""
TokenRetryService — WAITING allocation retry use-case orchestration.

Retries allocations that could not be fulfilled immediately (status = WAITING):
picks the least-loaded deployment via the load balancer and, if there is now
room, atomically transitions the allocation from WAITING to ACQUIRED.

Architecture:
-------------
    retry_acquire_tokens API (app/api/)
            │  Depends()
            ▼
    TokenRetryService ──▶ DeploymentLoadBalancer          (which endpoint?)
            │         ──▶ LLMTokenAllocationPersistence    (fetch + transition)

Retry decision logic:
    1. Fetch the WAITING allocation (carries its own tenant_id).
    2. Ask the load balancer for the least-loaded deployment.
    3. If the request fits the deployment's available capacity → transition.
    4. Otherwise → return the allocation unchanged with WAITING status.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from app.core.exceptions import AllocationNotFoundError, AllocationStateError
from app.models.response_models import TokenAllocationResponse

if TYPE_CHECKING:
    from uuid import UUID

    from app.persistence.allocations import LLMTokenAllocationPersistence
    from app.services.deployment_load_balancer import DeploymentLoadBalancer

logger = logging.getLogger(__name__)

_REQUIRED_STATUS = "WAITING"
_DEFAULT_LOCK_SECONDS = 70


class TokenRetryService:
    """Orchestrates the WAITING → ACQUIRED retry use case."""

    def __init__(
        self,
        allocation_persistence: LLMTokenAllocationPersistence,
        load_balancer: DeploymentLoadBalancer,
    ) -> None:
        """Initialise with injected dependencies."""
        self._allocation_persistence = allocation_persistence
        self._load_balancer = load_balancer

    async def retry_acquire(self, token_request_id: str) -> TokenAllocationResponse:
        """
        Attempt to promote a WAITING allocation to ACQUIRED.

        Returns ACQUIRED if capacity was available, else WAITING.

        Raises:
            AllocationNotFoundError: If no record matches token_request_id.
            AllocationStateError: If the allocation is not in WAITING status.
            DeploymentConfigurationError: If no active deployment exists.
        """
        allocation = await self._fetch_waiting_allocation(token_request_id)
        tenant_id: UUID = allocation["tenant_id"]
        provider_name: str = allocation["provider_name"]
        model_name: str = allocation["model_name"]
        token_count: int = allocation["token_count"]

        deployment = await self._load_balancer.choose_least_loaded(
            tenant_id, provider_name, model_name
        )

        if token_count > deployment["available_token_capacity"]:
            logger.debug(
                "[retry] Capacity still exhausted — keeping WAITING",
                extra={"token_request_id": token_request_id, "model": model_name},
            )
            return _build_waiting_response(allocation)

        return await self._transition_to_acquired(token_request_id, deployment)

    async def _fetch_waiting_allocation(self, token_request_id: str) -> dict[str, Any]:
        """Return the allocation if it exists and is WAITING, else raise."""
        allocation = await self._allocation_persistence.get_allocation_by_request_id(
            token_request_id
        )
        if allocation is None:
            raise AllocationNotFoundError(token_request_id)

        current_status: str = allocation.get("allocation_status", "")
        if current_status != _REQUIRED_STATUS:
            raise AllocationStateError(
                token_request_id, current_status, _REQUIRED_STATUS
            )
        return allocation

    async def _transition_to_acquired(
        self,
        token_request_id: str,
        deployment: dict[str, Any],
    ) -> TokenAllocationResponse:
        """
        Atomically flip WAITING → ACQUIRED against the chosen deployment.

        Raises:
            AllocationNotFoundError: If the record vanished (already transitioned).
        """
        lock_secs = (
            deployment.get("token_lock_duration_seconds") or _DEFAULT_LOCK_SECONDS
        )
        expires_at = datetime.now() + timedelta(seconds=lock_secs)

        updated = await self._allocation_persistence.transition_waiting_to_acquired(
            token_request_id=token_request_id,
            deployment_id=deployment["deployment_id"],
            expires_at=expires_at,
        )
        if updated is None:
            latest = await self._allocation_persistence.get_allocation_by_request_id(
                token_request_id
            )
            if (
                latest is not None
                and latest.get("allocation_status") == _REQUIRED_STATUS
            ):
                logger.debug(
                    "[retry] Atomic transition found no capacity; keeping WAITING",
                    extra={"token_request_id": token_request_id},
                )
                return _build_waiting_response(latest)
            raise AllocationNotFoundError(token_request_id)

        logger.info(
            "[retry] Allocation transitioned to ACQUIRED",
            extra={"token_request_id": token_request_id},
        )
        return TokenAllocationResponse(**updated)


def _build_waiting_response(allocation: dict[str, Any]) -> TokenAllocationResponse:
    """Build a TokenAllocationResponse preserving the WAITING status."""
    return TokenAllocationResponse(**{**allocation, "allocation_status": "WAITING"})
