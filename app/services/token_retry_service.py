"""
TokenRetryService — WAITING allocation retry use-case orchestration.

====================================================================

Orchestrates the retry flow for allocations that could not be fulfilled
immediately (status = WAITING).  Checks current capacity and, if space is
available, atomically transitions the allocation from WAITING to ACQUIRED.

Architecture:
-------------
    ┌────────────────────────────┐
    │  retry_acquire_tokens API  │
    │  (app/api/)                │
    └──────────┬─────────────────┘
               │  Depends()
    ┌──────────▼─────────────────┐     ┌──────────────────────────────┐
    │ TokenRetryService          │────▶│ LLMTokenAllocationPersistence│
    │ (app/services/)            │     │  get_allocation_by_request_id│
    │                            │     │  get_least_loaded_deployment  │
    └────────────────────────────┘     │  transition_waiting_to_acquired│
                                       └──────────────────────────────┘

Retry decision logic:
    1. Fetch the WAITING allocation record.
    2. Look up the least-loaded deployment for the model.
    3. If current load + request tokens <= max_tokens → transition to ACQUIRED.
    4. Otherwise → return the allocation unchanged with WAITING status.

Dependencies:
    - app/persistence/llm_token_allocations.py — DB operations
    - app/core/exceptions.py                   — domain exceptions

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from app.core.exceptions import (
    AllocationNotFoundError,
    AllocationStateError,
    DeploymentConfigurationError,
)
from app.models.response_models import TokenAllocationResponse

if TYPE_CHECKING:
    from app.persistence.llm_token_allocations import LLMTokenAllocationPersistence

logger = logging.getLogger(__name__)

_REQUIRED_STATUS = "WAITING"


class TokenRetryService:
    """
    Orchestrates the WAITING → ACQUIRED retry use case.

    Fetches the WAITING allocation, checks current deployment capacity, and
    either promotes the allocation to ACQUIRED or returns it in WAITING status.

    Does not use a circuit breaker because the retry path is already a
    fallback; callers should handle DB exceptions at the endpoint level.

    Example:
        >>> service = TokenRetryService(
        ...     allocation_persistence=LLMTokenAllocationPersistence(),
        ... )
        >>> response = await service.retry_acquire(token_request_id)
        >>> if response.allocation_status == "WAITING":
        ...     # still no capacity; tell client to retry later
    """

    def __init__(
        self,
        allocation_persistence: LLMTokenAllocationPersistence,
    ) -> None:
        """
        Initialise with injected dependencies.

        Args:
            allocation_persistence: Persistence adapter for token allocations.
        """
        self._allocation_persistence = allocation_persistence

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def retry_acquire(self, token_request_id: str) -> TokenAllocationResponse:
        """
        Attempt to promote a WAITING allocation to ACQUIRED.

        Args:
            token_request_id: The WAITING allocation to retry.

        Returns:
            TokenAllocationResponse with status ACQUIRED if capacity was
            available, or WAITING if capacity is still exhausted.

        Raises:
            AllocationNotFoundError: If no record matches token_request_id.
            AllocationStateError: If the allocation is not in WAITING status.
            DeploymentConfigurationError: If the deployment is missing max_tokens.
        """
        allocation = await self._fetch_waiting_allocation(token_request_id)
        provider_name: str = allocation["llm_provider"]
        model_name: str = allocation["llm_model_name"]
        token_count: int = allocation["token_count"]

        (
            total_allocated,
            chosen_config,
        ) = await self._allocation_persistence.get_least_loaded_deployment(
            provider_name, model_name
        )
        max_token_limit = _require_max_token_limit(chosen_config, model_name)

        if total_allocated + token_count > max_token_limit:
            logger.debug(
                "[retry] Capacity still exhausted — keeping WAITING",
                extra={"token_request_id": token_request_id, "model": model_name},
            )
            return _build_waiting_response(allocation)

        return await self._transition_to_acquired(token_request_id, chosen_config)

    # ------------------------------------------------------------------
    # Fetch + state validation
    # ------------------------------------------------------------------

    async def _fetch_waiting_allocation(self, token_request_id: str) -> dict[str, Any]:
        """
        Return the allocation if it exists and is in WAITING status.

        Args:
            token_request_id: The allocation identifier to look up.

        Returns:
            Allocation record dict.

        Raises:
            AllocationNotFoundError: If no matching record exists.
            AllocationStateError: If the allocation is not in WAITING status.
        """
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

    # ------------------------------------------------------------------
    # WAITING → ACQUIRED transition
    # ------------------------------------------------------------------

    async def _transition_to_acquired(
        self,
        token_request_id: str,
        chosen_config: dict[str, Any],
    ) -> TokenAllocationResponse:
        """
        Atomically update the allocation status from WAITING to ACQUIRED.

        Args:
            token_request_id: The allocation to promote.
            chosen_config: Deployment config with endpoint and timing.

        Returns:
            TokenAllocationResponse with ACQUIRED status.

        Raises:
            AllocationNotFoundError: If the atomic update finds no matching record
                (race condition — another caller already transitioned it).
        """
        max_lock_secs: int = chosen_config.get("max_token_lock_time_secs", 70)
        expires_at = datetime.now() + timedelta(seconds=max_lock_secs)
        api_endpoint: str = chosen_config.get("api_endpoint_url", "")
        deployment_region: str = chosen_config.get("deployment_region", "")

        updated = await self._allocation_persistence.transition_waiting_to_acquired(
            token_request_id=token_request_id,
            api_endpoint=api_endpoint,
            deployment_region=deployment_region,
            expires_at=expires_at,
        )
        if updated is None:
            latest_allocation = (
                await self._allocation_persistence.get_allocation_by_request_id(
                    token_request_id
                )
            )
            if (
                latest_allocation is not None
                and latest_allocation.get("allocation_status") == _REQUIRED_STATUS
            ):
                logger.debug(
                    "[retry] Atomic transition found no capacity; keeping WAITING",
                    extra={"token_request_id": token_request_id},
                )
                return _build_waiting_response(latest_allocation)
            raise AllocationNotFoundError(token_request_id)

        updated["temperature"] = chosen_config.get("temperature", 0.0)
        updated["seed"] = chosen_config.get("seed", 42)
        updated["api_version"] = chosen_config.get("api_version", "")

        logger.info(
            "[retry] Allocation transitioned to ACQUIRED",
            extra={"token_request_id": token_request_id},
        )
        return TokenAllocationResponse(**updated)


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _require_max_token_limit(chosen_config: dict[str, Any], model_name: str) -> int:
    """
    Extract max_tokens from deployment config; raise if absent.

    Args:
        chosen_config: Deployment configuration dict.
        model_name: Used in the error message when max_tokens is missing.

    Returns:
        Maximum token allocation as a positive integer.

    Raises:
        DeploymentConfigurationError: If max_tokens is None or absent.
    """
    max_token_limit: int | None = chosen_config.get("max_tokens")
    if max_token_limit is None:
        raise DeploymentConfigurationError(model_name, "max_tokens")
    return max_token_limit


def _build_waiting_response(allocation: dict[str, Any]) -> TokenAllocationResponse:
    """
    Build a TokenAllocationResponse preserving the WAITING status.

    Args:
        allocation: The unmodified WAITING allocation record dict.

    Returns:
        TokenAllocationResponse with allocation_status = 'WAITING'.
    """
    return TokenAllocationResponse(
        token_request_id=allocation["token_request_id"],
        user_id=allocation["user_id"],
        llm_provider=allocation["llm_provider"],
        llm_model_name=allocation["llm_model_name"],
        token_count=allocation["token_count"],
        allocation_status="WAITING",
        allocated_at=allocation["allocated_at"],
        expires_at=allocation.get("expires_at"),
        deployment_name=allocation.get("deployment_name"),
        cloud_provider=allocation.get("cloud_provider"),
        api_endpoint_url=allocation.get("api_endpoint_url"),
        deployment_region=allocation.get("deployment_region"),
        request_context=allocation.get("request_context"),
        temperature=allocation.get("temperature"),
        seed=allocation.get("seed"),
    )
