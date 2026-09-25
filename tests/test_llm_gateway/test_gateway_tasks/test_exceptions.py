"""
Unit tests for gateway_tasks/exceptions.py.

Coverage targets:
    - GatewayTaskError is a subclass of TokenManagerError
    - TaskInputValidationError carries field + reason attributes
    - ProviderCircuitOpenError carries provider_name attribute
    - ProviderExecutionError carries provider_name, error_class, original_exc
    - All exceptions are raise-able and produce correct str() messages
"""

from __future__ import annotations

import pytest

from app.core.exceptions import TokenManagerError
from app.llm_gateway.gateway_tasks.exceptions import (
    GatewayTaskError,
    ProviderCircuitOpenError,
    ProviderExecutionError,
    TaskInputValidationError,
)


class TestGatewayTaskError:
    def test_is_token_manager_error_subclass(self) -> None:
        assert issubclass(GatewayTaskError, TokenManagerError)

    def test_is_exception_subclass(self) -> None:
        assert issubclass(GatewayTaskError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(GatewayTaskError):
            raise GatewayTaskError("test")


class TestTaskInputValidationError:
    def test_is_gateway_task_error_subclass(self) -> None:
        assert issubclass(TaskInputValidationError, GatewayTaskError)

    def test_carries_field_attribute(self) -> None:
        exc = TaskInputValidationError("user_id", "invalid UUID")
        assert exc.field == "user_id"

    def test_carries_reason_attribute(self) -> None:
        exc = TaskInputValidationError("user_id", "invalid UUID")
        assert exc.reason == "invalid UUID"

    def test_message_contains_field_and_reason(self) -> None:
        exc = TaskInputValidationError("user_prompt", "must not be blank")
        assert "user_prompt" in str(exc)
        assert "must not be blank" in str(exc)

    def test_can_be_raised(self) -> None:
        with pytest.raises(TaskInputValidationError):
            raise TaskInputValidationError("llm_provider", "missing")


class TestProviderCircuitOpenError:
    def test_is_gateway_task_error_subclass(self) -> None:
        assert issubclass(ProviderCircuitOpenError, GatewayTaskError)

    def test_carries_provider_name(self) -> None:
        exc = ProviderCircuitOpenError("openai")
        assert exc.provider_name == "openai"

    def test_message_contains_provider(self) -> None:
        exc = ProviderCircuitOpenError("aws_bedrock")
        assert "aws_bedrock" in str(exc)

    def test_can_be_raised(self) -> None:
        with pytest.raises(ProviderCircuitOpenError):
            raise ProviderCircuitOpenError("anthropic")


class TestProviderExecutionError:
    def test_is_gateway_task_error_subclass(self) -> None:
        assert issubclass(ProviderExecutionError, GatewayTaskError)

    def test_carries_provider_name(self) -> None:
        orig = RuntimeError("sdk error")
        exc = ProviderExecutionError("openai", "TRANSIENT", orig)
        assert exc.provider_name == "openai"

    def test_carries_error_class(self) -> None:
        orig = RuntimeError("sdk error")
        exc = ProviderExecutionError("openai", "TRANSIENT", orig)
        assert exc.error_class == "TRANSIENT"

    def test_carries_original_exc(self) -> None:
        orig = ValueError("bad value")
        exc = ProviderExecutionError("anthropic", "AUTH_STALE", orig)
        assert exc.original_exc is orig

    def test_message_contains_all_context(self) -> None:
        orig = OSError("timeout")
        exc = ProviderExecutionError("mistral", "CONTEXT_EXCEEDED", orig)
        msg = str(exc)
        assert "mistral" in msg
        assert "CONTEXT_EXCEEDED" in msg

    def test_can_be_chained_with_cause(self) -> None:
        orig = RuntimeError("root cause")
        try:
            raise ProviderExecutionError("openai", "TRANSIENT", orig) from orig
        except ProviderExecutionError as caught:
            assert caught.__cause__ is orig
