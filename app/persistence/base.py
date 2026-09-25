"""Shared session ownership and input validation for active repositories."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import json
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import DatabaseSessionManager


class BasePersistence:
    """Provide one transaction-scoped session and common validation primitives."""

    def __init__(self, database_manager: DatabaseSessionManager | None = None) -> None:
        self.database_manager = database_manager or DatabaseSessionManager()
        self._service_name = type(self).__name__

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Use the database manager's commit/rollback/close session lifecycle."""
        async with self.database_manager.get_session() as session:
            yield session

    @staticmethod
    def validate_uuid(uuid_value: UUID | str, parameter_name: str = "UUID") -> None:
        """Reject values that are not valid UUIDs."""
        if isinstance(uuid_value, UUID):
            return
        if isinstance(uuid_value, str):
            try:
                UUID(uuid_value)
            except ValueError as exc:
                raise ValueError(
                    f"{parameter_name} must be a valid UUID string"
                ) from exc
            return
        raise ValueError(f"{parameter_name} must be a valid UUID")

    @staticmethod
    def validate_positive_integer(
        integer_value: int,
        parameter_name: str = "value",
        allow_zero: bool = False,
    ) -> None:
        """Reject booleans, non-integers, and values below the chosen minimum."""
        if isinstance(integer_value, bool) or not isinstance(integer_value, int):
            raise ValueError(f"{parameter_name} must be an integer")
        minimum = 0 if allow_zero else 1
        if integer_value < minimum:
            kind = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{parameter_name} must be {kind}, got {integer_value}")

    @staticmethod
    def validate_string_not_empty(
        string_value: str, parameter_name: str = "string"
    ) -> None:
        """Reject non-string or blank values before opening a DB session."""
        if not isinstance(string_value, str) or not string_value.strip():
            raise ValueError(f"{parameter_name} must be a non-empty string")

    @staticmethod
    def validate_enum_value(
        enum_value: str,
        valid_values: list[str],
        parameter_name: str = "value",
    ) -> None:
        """Validate an enum value against the repository's accepted values."""
        if enum_value not in valid_values:
            raise ValueError(
                f"Invalid {parameter_name}: '{enum_value}'. "
                f"Must be one of: {', '.join(valid_values)}"
            )

    @staticmethod
    def _validate_and_serialize_json(
        data: dict[str, Any] | None,
        param_name: str = "metadata",
    ) -> str | None:
        """Serialize strict JSON, rejecting non-finite numbers and invalid values."""
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError(f"{param_name} must be a JSON object")
        try:
            return json.dumps(data, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{param_name} must contain valid JSON values") from exc

    def log_operation(
        self,
        operation_type: str,
        entity_identifier: Any,
        success: bool = True,
        additional_context: str | None = None,
    ) -> None:
        """Log a repository operation without recording query parameters."""
        logger.log(
            "INFO" if success else "ERROR",
            "{} {} {} for entity={} context={}",
            self._service_name,
            operation_type,
            "succeeded" if success else "failed",
            entity_identifier,
            additional_context,
        )
