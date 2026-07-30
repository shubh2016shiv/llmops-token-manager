"""
Shared base for the token-allocation operation modules.

Holds constants, validators, and the one shared primitive read
(``get_allocation_by_request_id``) reused by the release and retry operations,
so each operation file stays focused on its own write path. The public facade
(``LLMTokenAllocationPersistence``) composes the operation mixins; every mixin
derives from this base so shared helpers resolve through a single MRO entry.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text

from app.models.response_models import VALID_CLOUD_PROVIDERS, VALID_LLM_PROVIDERS
from app.persistence.base import BasePersistence
from app.persistence.queries.allocation_queries import (
    GET_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL,
)


class AllocationPersistenceBase(BasePersistence):
    """Constants, validators, and the shared fetch used by the operations."""

    VALID_ALLOCATION_STATUSES = [
        "ACQUIRED",
        "WAITING",
        "PAUSED",
        "RELEASED",
        "EXPIRED",
        "FAILED",
    ]

    DEFAULT_ALLOCATION_STATUS = "ACQUIRED"

    def validate_allocation_status(self, allocation_status: str) -> None:
        """Validate an allocation status against the DB CHECK constraint set."""
        self.validate_enum_value(
            allocation_status, self.VALID_ALLOCATION_STATUSES, "allocation status"
        )

    def validate_llm_provider(self, provider_name: str) -> None:
        """Validate provider name against the canonical provider set."""
        self.validate_enum_value(provider_name, VALID_LLM_PROVIDERS, "provider_name")

    def validate_cloud_provider(self, cloud_provider: str | None) -> None:
        """Validate an optional cloud provider (None = direct/on-prem, always ok)."""
        if cloud_provider is not None:
            self.validate_enum_value(
                cloud_provider, VALID_CLOUD_PROVIDERS, "cloud_provider"
            )

    async def get_allocation_by_request_id(
        self, token_request_identifier: str
    ) -> dict[str, Any] | None:
        """
        Fetch an allocation by its request id (shared by release + retry).

        Returns the allocation record dict, or None if not found.
        """
        self.validate_string_not_empty(
            token_request_identifier, "token_request_identifier"
        )
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(GET_TOKEN_ALLOCATION_BY_REQUEST_ID_SQL),
                    {"token_request_id": token_request_identifier},
                )
                record = result.mappings().one_or_none()
                return dict(record) if record else None
        except Exception as e:
            logger.error(f"Error fetching allocation {token_request_identifier}: {e}")
            raise
