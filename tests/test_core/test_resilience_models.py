"""Unit tests for resilience model contracts."""

from uuid import UUID

from pydantic import ValidationError
import pytest

from app.models.resilience_models import (
    BackpressureDecision,
    CounterSeedRecord,
    DeploymentCapacitySnapshot,
    DlqPayload,
    TokenAllocationPersistPayload,
)


def _base_persist_fields(**overrides: object) -> dict[str, object]:
    """
    Every required field TokenAllocationPersistPayload needs to validate.

    Centralized so each test only spells out the field(s) it's actually
    exercising instead of repeating (and risking drifting) a full payload —
    the fix for this file: every test below used to use field names
    (llm_provider/llm_model_name) that don't exist on this model at all
    (that's InvalidActiveDeploymentRecord/CounterSeedRecord's shape, not
    this one) and omitted several required fields (tenant_id, deployment_id,
    deployment_key). Two of those tests still "passed" — pytest.raises
    (ValidationError) is satisfied by ANY validation error, including
    "field required", so they weren't verifying the behavior their
    docstrings claimed. TokenAllocationPersistPayload.model_validate() was
    called directly to confirm the corrected fixture below actually fails
    for the reason each test intends, not merely fails.
    """
    fields: dict[str, object] = {
        "token_request_id": "req_123",
        "tenant_id": "6f2c9e1a-6b8e-4f0a-9b7e-2f7a9c1d5e3a",
        "user_id": "89e0d113-912f-4272-ba13-6b3b6d9677c4",
        "deployment_id": "a1b2c3d4-e5f6-4789-9abc-def012345678",
        "provider_name": "openai",
        "model_name": "gpt-4o",
        "deployment_key": "tenant-a:openai:gpt-4o",
        "token_count": 150,
    }
    fields.update(overrides)
    return fields


class TestTokenAllocationPersistPayload:
    """Validate queue payload parsing and normalization."""

    def test_parses_legacy_message_id_alias(self) -> None:
        """The payload should accept the legacy `_message_id` field."""
        payload = TokenAllocationPersistPayload.model_validate(
            _base_persist_fields(
                token_request_id=" req_123 ",
                provider_name=" openai ",
                model_name=" gpt-4o ",
                _message_id="msg_123",
            )
        )

        assert payload.token_request_id == "req_123"
        assert payload.user_id == UUID("89e0d113-912f-4272-ba13-6b3b6d9677c4")
        assert payload.provider_name == "openai"
        assert payload.model_name == "gpt-4o"
        assert payload.message_id == "msg_123"

    def test_rejects_blank_required_string_fields(self) -> None:
        """Blank required string values should fail validation."""
        with pytest.raises(ValidationError, match="must not be blank"):
            TokenAllocationPersistPayload.model_validate(
                _base_persist_fields(token_request_id="   ")
            )

    def test_missing_required_field_fails_for_that_reason(self) -> None:
        """Regression guard: a truly incomplete payload must fail on 'missing'."""
        incomplete = _base_persist_fields()
        del incomplete["tenant_id"]

        with pytest.raises(ValidationError, match="tenant_id"):
            TokenAllocationPersistPayload.model_validate(incomplete)

    def test_accepts_a_complete_valid_payload(self) -> None:
        """Happy path: a fully-populated payload validates without error."""
        payload = TokenAllocationPersistPayload.model_validate(_base_persist_fields())

        assert payload.provider_name == "openai"
        assert payload.model_name == "gpt-4o"
        assert payload.token_count == 150


class TestDlqPayload:
    """Validate dead-letter queue payload behavior."""

    def test_rejects_blank_dlq_reason(self) -> None:
        """DLQ payloads require an actionable reason."""
        with pytest.raises(ValidationError, match="dlq_reason must not be blank"):
            DlqPayload.model_validate(_base_persist_fields(dlq_reason="   "))

    def test_accepts_a_complete_valid_dlq_payload(self) -> None:
        """Happy path: a fully-populated DLQ payload validates without error."""
        payload = DlqPayload.model_validate(
            _base_persist_fields(dlq_reason="db down", retry_attempts=3)
        )

        assert payload.dlq_reason == "db down"
        assert payload.dlq_routed_by == "explicit"
        assert payload.retry_attempts == 3


class TestCounterSeedRecord:
    """Validate reconciliation seed records."""

    def test_accepts_valid_seed_record(self) -> None:
        """Valid seed records should parse cleanly."""
        record = CounterSeedRecord(
            llm_model_name="gpt-4o",
            api_endpoint_url="https://api.example.com/v1",
            allocated_tokens=500,
            max_tokens=1000,
        )

        assert record.allocated_tokens == 500
        assert record.max_tokens == 1000


class TestDeploymentCapacitySnapshot:
    """Validate capacity snapshot invariants."""

    def test_rejects_available_tokens_above_max(self) -> None:
        """Available tokens cannot exceed the configured maximum."""
        with pytest.raises(ValidationError):
            DeploymentCapacitySnapshot(
                llm_model_name="gpt-4o",
                api_endpoint_url="https://api.example.com/v1",
                current_allocated_tokens=100,
                max_tokens=1000,
                available_tokens=1500,
            )


class TestBackpressureDecision:
    """Validate backpressure decision requirements."""

    def test_requires_retry_after_when_rejecting(self) -> None:
        """Rejected requests must include a retry-after value."""
        with pytest.raises(ValidationError):
            BackpressureDecision(
                should_reject_request=True,
                reason="queue_depth_exceeded",
            )

    def test_accepts_complete_rejection_decision(self) -> None:
        """A complete rejection decision should validate successfully."""
        decision = BackpressureDecision(
            should_reject_request=True,
            reason="queue_depth_exceeded",
            retry_after_seconds=15,
            queue_depth=15000,
            pool_utilization_pct=95,
            circuit_breaker_name="postgres",
        )

        assert decision.retry_after_seconds == 15
        assert decision.queue_depth == 15000
