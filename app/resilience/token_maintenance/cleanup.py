"""
Cleanup job — delete expired token-allocation rows from PostgreSQL.

Housekeeping. Expired allocations accumulate in PostgreSQL forever unless something
deletes them; this job runs on a timer and removes them, keeping the table bounded.
It is the least critical of the three maintenance jobs (it could even be pushed
down into the database), but unbounded table growth is a slow-motion outage.

Returns the number of rows deleted so the scheduler/logs can report it.

Author: Engineering Team
"""

from __future__ import annotations

from app.persistence.token_maintenance import TokenMaintenancePersistence


async def cleanup_expired_allocations() -> int:
    """Delete expired allocations and return how many rows were removed."""
    # All the SQL lives in the persistence layer; this job just invokes it. Keeping
    # the job thin means "what to delete" stays in one place (persistence) and "when
    # to delete" stays in another (the scheduler).
    persistence = TokenMaintenancePersistence()
    return await persistence.delete_expired_allocations()
