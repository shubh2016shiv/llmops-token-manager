"""
Token allocation persistence — the token manager's one owned table.

Split by the operations the token manager actually performs, so each is
findable by name (not buried under generic CRUD):

    acquire.py  → AllocationAcquireMixin   (reserve tokens on a chosen deployment)
    release.py  → AllocationReleaseMixin   (delete an allocation)
    retry.py    → AllocationRetryMixin     (WAITING -> ACQUIRED)
    pause.py    → AllocationPauseMixin     (PAUSED capacity blocker)

The shared fetch (``get_allocation_by_request_id``) lives on ``_base`` since
release and retry both use it. ``LLMTokenAllocationPersistence`` composes the
operation mixins into one repository so callers keep a single injection point.

Deciding *which* deployment to reserve against is NOT here — that is a
load-balancing decision owned by the service layer
(``app/services/deployment_load_balancer.py``), fed by the read-only
``app/persistence/deployed_llm_endpoints.py`` reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.persistence.allocations.acquire import AllocationAcquireMixin
from app.persistence.allocations.pause import AllocationPauseMixin
from app.persistence.allocations.release import AllocationReleaseMixin
from app.persistence.allocations.retry import AllocationRetryMixin

if TYPE_CHECKING:
    from app.core.database import DatabaseSessionManager


class LLMTokenAllocationPersistence(
    AllocationAcquireMixin,
    AllocationReleaseMixin,
    AllocationRetryMixin,
    AllocationPauseMixin,
):
    """Cohesive repository over ``llm_token_allocations`` (one mixin per operation)."""


def get_token_allocation_repository(
    db_manager: DatabaseSessionManager | None = None,
) -> LLMTokenAllocationPersistence:
    """Factory for LLMTokenAllocationPersistence (uses the singleton if omitted)."""
    return LLMTokenAllocationPersistence(db_manager)


__all__ = [
    "LLMTokenAllocationPersistence",
    "get_token_allocation_repository",
]
