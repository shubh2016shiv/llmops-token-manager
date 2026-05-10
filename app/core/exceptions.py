"""
Domain Exceptions — LLM Token Manager.

======================================

All service-layer exceptions for the token allocation system.
Endpoints catch these and convert them to appropriate HTTP responses.

Architecture:
-------------
    ┌────────────────────────┐     ┌────────────────────────┐
    │  app/services/         │────▶│  app/core/exceptions.py│
    │  (raises these)        │     │  (domain errors)       │
    └────────────────────────┘     └────────────┬───────────┘
                                                │
                                   ┌────────────▼───────────┐
                                   │  app/api/              │
                                   │  (converts to HTTP)    │
                                   └────────────────────────┘

Author: Engineering Team
Last Updated: 2026-05-10

"""

from __future__ import annotations


class TokenManagerError(Exception):
    """Base exception for all LLM token manager domain errors."""


class UserNotFoundError(TokenManagerError):
    """
    Raised when a user cannot be found by their ID.

    Args:
        user_id: The UUID string that could not be resolved to a user record.

    Example:
        >>> raise UserNotFoundError("550e8400-e29b-41d4-a716-446655440000")
    """

    def __init__(self, user_id: str) -> None:
        super().__init__(f"User not found: user_id={user_id!r}")
        self.user_id = user_id


class UserInactiveError(TokenManagerError):
    """
    Raised when a user exists but is not in 'active' status.

    Args:
        user_id: The user's UUID string.
        status: The user's current status (e.g. 'suspended', 'inactive').

    Example:
        >>> raise UserInactiveError("abc123", "suspended")
    """

    def __init__(self, user_id: str, status: str) -> None:
        super().__init__(
            f"User is not active: user_id={user_id!r} current_status={status!r}"
        )
        self.user_id = user_id
        self.status = status


class AllocationNotFoundError(TokenManagerError):
    """
    Raised when a token allocation record cannot be found by its request ID.

    Args:
        token_request_id: The allocation identifier that was not found.

    Example:
        >>> raise AllocationNotFoundError("req_abc123def456")
    """

    def __init__(self, token_request_id: str) -> None:
        super().__init__(
            f"Token allocation not found: token_request_id={token_request_id!r}"
        )
        self.token_request_id = token_request_id


class AllocationStateError(TokenManagerError):
    """
    Raised when an allocation is in an unexpected state for the requested operation.

    Args:
        token_request_id: The allocation's identifier.
        current_status: The actual allocation status found.
        required_status: The status required for the operation to proceed.

    Example:
        >>> raise AllocationStateError("req_abc123", "ACQUIRED", "WAITING")
    """

    def __init__(
        self, token_request_id: str, current_status: str, required_status: str
    ) -> None:
        super().__init__(
            f"Allocation {token_request_id!r} is in {current_status!r} state; "
            f"required: {required_status!r}"
        )
        self.token_request_id = token_request_id
        self.current_status = current_status
        self.required_status = required_status


class DatabaseUnavailableError(TokenManagerError):
    """
    Raised when the database circuit breaker is open and cannot serve requests.

    Example:
        >>> raise DatabaseUnavailableError("DB circuit breaker open")
    """


class TokenLimitExceededError(TokenManagerError):
    """
    Raised when a single request's token count exceeds the deployment's max limit.

    Args:
        token_count: The requested number of tokens.
        max_limit: The deployment's configured maximum token allocation.
        model_name: The logical model name.

    Example:
        >>> raise TokenLimitExceededError(50000, 20000, "gpt-4o")
    """

    def __init__(self, token_count: int, max_limit: int, model_name: str) -> None:
        super().__init__(
            f"Token count {token_count} exceeds max limit {max_limit} "
            f"for model {model_name!r}"
        )
        self.token_count = token_count
        self.max_limit = max_limit
        self.model_name = model_name


class DeploymentConfigurationError(TokenManagerError):
    """
    Raised when an active deployment is missing required configuration fields.

    Args:
        model_name: The logical model name.
        missing_field: The name of the missing configuration field.

    Example:
        >>> raise DeploymentConfigurationError("gpt-4o", "max_tokens")
    """

    def __init__(self, model_name: str, missing_field: str) -> None:
        super().__init__(
            f"Active deployment for model {model_name!r} is missing "
            f"required field {missing_field!r}"
        )
        self.model_name = model_name
        self.missing_field = missing_field
