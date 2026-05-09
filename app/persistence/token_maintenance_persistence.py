"""
Token maintenance persistence - PostgreSQL access for Layer 4 maintenance flows.

Architecture:
-------------
    ┌────────────────────────────────────┐     ┌────────────────────────────────────┐
    │ token_maintenance/                 │────▶│ TokenMaintenancePersistence        │
    │ reconciliation + startup seeding   │     │ persistence-owned SQL access       │
    └────────────────────────────────────┘     └────────────────┬───────────────────┘
                                                                │
                                                                ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ token_maintenance_queries.py + PostgreSQL                                    │
    └──────────────────────────────────────────────────────────────────────────────┘

Dependencies:
    - app/models/resilience_models.py - typed maintenance records
    - app/persistence/base.py - session management and shared validation
    - app/persistence/token_maintenance_queries.py - raw SQL text

Author: Engineering Team
Last Updated: 2026-05-10
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text

from app.models.resilience_models import (
    CounterSeedRecord,
    InvalidActiveDeploymentRecord,
)
from app.persistence.base import BasePersistence
from app.persistence.token_maintenance_queries import (
    DELETE_EXPIRED_ALLOCATIONS_SQL,
    LIST_ACTIVE_DEPLOYMENT_CAPACITY_SNAPSHOTS_SQL,
    LIST_INVALID_ACTIVE_MODELS_WITHOUT_CAPACITY_SQL,
    LIST_STARTUP_COUNTER_SEED_SNAPSHOTS_SQL,
)


class TokenMaintenancePersistence(BasePersistence):
    """Persistence service for resilience maintenance tasks and startup seeding."""

    async def list_active_deployment_capacity_snapshots(
        self,
    ) -> list[CounterSeedRecord]:
        """Return the authoritative active deployment capacity snapshot."""
        return await self._load_counter_seed_records(
            LIST_ACTIVE_DEPLOYMENT_CAPACITY_SNAPSHOTS_SQL,
        )

    async def list_startup_counter_seed_snapshots(self) -> list[CounterSeedRecord]:
        """Return active deployment counters to seed during application startup."""
        return await self._load_counter_seed_records(
            LIST_STARTUP_COUNTER_SEED_SNAPSHOTS_SQL,
        )

    async def delete_expired_allocations(self) -> int:
        """Delete expired token allocation rows and return the deleted count."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text(DELETE_EXPIRED_ALLOCATIONS_SQL))
                deleted_count = int(getattr(result, "rowcount", 0))
                if deleted_count > 0:
                    logger.info(
                        "Token maintenance deleted expired allocations",
                        deleted_count=deleted_count,
                    )
                else:
                    logger.debug("Token maintenance found no expired allocations")
                return deleted_count
        except Exception as error:
            logger.error(
                "Token maintenance failed to delete expired allocations",
                error=str(error),
            )
            raise

    async def list_invalid_active_models_without_capacity(
        self,
    ) -> list[InvalidActiveDeploymentRecord]:
        """Return active deployment rows that violate the max-token invariant."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(LIST_INVALID_ACTIVE_MODELS_WITHOUT_CAPACITY_SQL)
                )
                return [
                    InvalidActiveDeploymentRecord.model_validate(dict(row))
                    for row in result.mappings().all()
                ]
        except Exception as error:
            logger.error(
                "Token maintenance failed to load invalid active deployment rows",
                error=str(error),
            )
            raise

    async def _load_counter_seed_records(
        self,
        sql_query: str,
    ) -> list[CounterSeedRecord]:
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql_query))
                return [
                    CounterSeedRecord.model_validate(dict(row))
                    for row in result.mappings().all()
                ]
        except Exception as error:
            logger.error(
                "Token maintenance failed to load counter seed records",
                error=str(error),
            )
            raise
