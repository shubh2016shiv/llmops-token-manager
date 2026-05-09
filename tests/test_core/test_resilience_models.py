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


class TestTokenAllocationPersistPayload:
    """Validate queue payload parsing and normalization."""

    def test_parses_legacy_message_id_alias(self) -> None:
        """The payload should accept the legacy `_message_id` field."""
        payload = TokenAllocationPersistPayload.model_validate(
            {
                "token_request_id": " req_123 ",
                "user_id": "89e0d113-912f-4272-ba13-6b3b6d9677c4",
                "llm_provider": " openai ",
                "llm_model_name": " gpt-4o ",
                "token_count": 150,
                "_message_id": "msg_123",
            }
        )

        assert payload.token_request_id == "req_123"
        assert payload.user_id == UUID("89e0d113-912f-4272-ba13-6b3b6d9677c4")
        assert payload.llm_provider == "openai"
        assert payload.llm_model_name == "gpt-4o"
        assert payload.message_id == "msg_123"

    def test_rejects_blank_required_string_fields(self) -> None:
        """Blank required string values should fail validation."""
        with pytest.raises(ValidationError):
            TokenAllocationPersistPayload.model_validate(
                {
                    "token_request_id": "   ",
                    "user_id": "89e0d113-912f-4272-ba13-6b3b6d9677c4",
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "token_count": 150,
                }
            )


class TestDlqPayload:
    """Validate dead-letter queue payload behavior."""

    def test_rejects_blank_dlq_reason(self) -> None:
        """DLQ payloads require an actionable reason."""
        with pytest.raises(ValidationError):
            DlqPayload.model_validate(
                {
                    "token_request_id": "req_123",
                    "user_id": "89e0d113-912f-4272-ba13-6b3b6d9677c4",
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "token_count": 150,
                    "dlq_reason": "   ",
                }
            )


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
