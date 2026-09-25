"""
Unit tests for app.resilience.token_queue.handlers.

Regression note: the original version of test_process_dlq_alert_
releases_reserved_tokens used field names (llm_provider, llm_model_name)
that don't exist on DlqPayload/TokenAllocationPersistPayload at all (that's
CounterSeedRecord/InvalidActiveDeploymentRecord's shape — see
app/models/resilience_models.py) and omitted several required fields
(tenant_id, deployment_id, deployment_key), so process_dlq_alert() raised a
pydantic ValidationError before ever reaching the counter release it meant
to test. Fixed here with a payload matching the actual schema, plus explicit
argument assertions on release_tokens (assert_awaited_once() alone doesn't
prove the *right* model/endpoint/count were released) and coverage for the
broker-dead-lettered normalization path this module documents but the
original file never exercised.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

from app.resilience.token_queue.handlers import process_dlq_alert


def _explicit_dlq_payload(**overrides: object) -> dict[str, object]:
    """
    A complete payload as process_dlq_alert() would receive when WE
    published it explicitly to the DLQ (already has `dlq_reason`).
    """
    fields: dict[str, object] = {
        "token_request_id": "req_123",
        "tenant_id": str(uuid4()),
        "user_id": str(uuid4()),
        "deployment_id": str(uuid4()),
        "provider_name": "openai",
        "model_name": "gpt-4o",
        "deployment_key": "tenant-a:openai:gpt-4o",
        "token_count": 50,
        "api_endpoint_url": "https://example.test/v1",
        "allocation_status": "ACQUIRED",
        "dlq_reason": "db down",
    }
    fields.update(overrides)
    return fields


def test_process_dlq_alert_releases_reserved_tokens(monkeypatch) -> None:
    fake_counter = type("Counter", (), {})()
    fake_counter.release_tokens = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.resilience.token_queue.handlers.get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )

    result = process_dlq_alert(
        _explicit_dlq_payload(),
        headers={"x-token-retry-attempt": 5},
    )

    # The exact model/endpoint/count released must match the payload — a
    # generic assert_awaited_once() would pass even if the wrong deployment's
    # tokens were released.
    fake_counter.release_tokens.assert_awaited_once_with(
        "gpt-4o", "https://example.test/v1", 50
    )
    assert result.dlq_reason == "db down"
    assert result.token_request_id == "req_123"


def test_process_dlq_alert_normalizes_broker_dead_lettered_payload(
    monkeypatch,
) -> None:
    """
    A message the BROKER dead-lettered (not us) lacks our `dlq_reason`
    enrichment entirely — process_dlq_alert must still normalize it into a
    valid DlqPayload using the retry headers, per _normalize_dlq_payload's
    documented fallback path.
    """
    fake_counter = type("Counter", (), {})()
    fake_counter.release_tokens = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.resilience.token_queue.handlers.get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )
    broker_payload = _explicit_dlq_payload()
    del broker_payload["dlq_reason"]  # broker-routed messages have no enrichment

    result = process_dlq_alert(
        broker_payload,
        headers={
            "x-token-retry-attempt": 3,
            "x-token-retry-reason": "quorum_delivery_limit_exceeded",
        },
    )

    assert result.dlq_reason == "quorum_delivery_limit_exceeded"
    assert result.retry_attempts == 3
    fake_counter.release_tokens.assert_awaited_once_with(
        "gpt-4o", "https://example.test/v1", 50
    )


def test_process_dlq_alert_logs_when_redis_release_could_not_be_applied(
    monkeypatch,
) -> None:
    """
    release_tokens() returning None means the release couldn't be applied
    (e.g. Redis circuit breaker open) — process_dlq_alert must not raise;
    the compensation failure is logged for follow-up, not retried inline.
    """
    fake_counter = type("Counter", (), {})()
    fake_counter.release_tokens = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "app.resilience.token_queue.handlers.get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )

    result = process_dlq_alert(_explicit_dlq_payload(), headers=None)

    fake_counter.release_tokens.assert_awaited_once()
    assert result.token_request_id == "req_123"
