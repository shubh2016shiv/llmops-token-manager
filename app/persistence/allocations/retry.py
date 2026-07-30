"""
RETRY — promote a WAITING allocation to ACQUIRED when capacity frees up.

The service fetches the allocation (``get_allocation_by_request_id``, shared on
the base) and picks a deployment via the load balancer, then calls the atomic
transition here: it re-checks capacity under lock and flips WAITING -> ACQUIRED
only if it still fits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import text

from app.persistence.allocations._base import AllocationPersistenceBase
from app.persistence.queries.allocation_queries import (
    TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL,
)

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


class AllocationRetryMixin(AllocationPersistenceBase):
    """The retry (WAITING -> ACQUIRED) write path against ``llm_token_allocations``."""

    async def transition_waiting_to_acquired(
        self,
        token_request_id: str,
        deployment_id: UUID,
        expires_at: datetime,
    ) -> dict[str, Any] | None:
        """
        Atomically flip a WAITING allocation to ACQUIRED against the given
        deployment, re-checking capacity under lock.

        Returns the updated record, or None if it did not apply (no longer
        WAITING, or the deployment has no room).
        """
        self.validate_string_not_empty(token_request_id, "token_request_id")
        self.validate_uuid(deployment_id, "deployment_id")
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(TRANSITION_WAITING_TO_ACQUIRED_WITH_CAPACITY_CHECK_SQL),
                    {
                        "token_request_id": token_request_id,
                        "deployment_id": deployment_id,
                        "expires_at": expires_at,
                    },
                )
                updated = result.mappings().one_or_none()
                if updated:
                    logger.info(f"Retried {token_request_id}: WAITING -> ACQUIRED")
                    return dict(updated)
                logger.debug(
                    f"Retry did not apply for {token_request_id} "
                    "(not WAITING or insufficient capacity)"
                )
                return None
        except Exception as e:
            logger.error(f"Error retrying allocation {token_request_id}: {e}")
            raise
