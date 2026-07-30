"""
RELEASE — delete an allocation, freeing its reserved tokens.

The service layer fetches the allocation first (``get_allocation_by_request_id``,
shared on the base) to run object-level authorization, then calls delete here.
Bulk expiry cleanup is a maintenance concern and lives in ``token_maintenance``.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text

from app.persistence.allocations._base import AllocationPersistenceBase
from app.persistence.queries.allocation_queries import (
    DELETE_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL,
)


class AllocationReleaseMixin(AllocationPersistenceBase):
    """The release (delete) write path against ``llm_token_allocations``."""

    async def delete_allocation(self, token_request_id: str) -> bool:
        """
        Delete a token allocation (release tokens permanently).

        Returns True if a row was deleted, False if none matched.
        """
        self.validate_string_not_empty(token_request_id, "token_request_id")
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(DELETE_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL),
                    {"token_request_id": token_request_id},
                )
                deleted = getattr(result, "rowcount", 0) > 0
                if deleted:
                    logger.info(f"Released allocation: {token_request_id}")
                else:
                    logger.debug(
                        f"Allocation not found for release: {token_request_id}"
                    )
                return bool(deleted)
        except Exception as e:
            logger.error(f"Error releasing allocation {token_request_id}: {e}")
            raise
