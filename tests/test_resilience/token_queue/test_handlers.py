from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

from app.resilience.token_queue.handlers import process_dlq_alert


def test_process_dlq_alert_releases_reserved_tokens(monkeypatch) -> None:
    fake_counter = type("Counter", (), {})()
    fake_counter.release_tokens = AsyncMock(return_value=True)

    monkeypatch.setattr(
        "app.resilience.token_queue.handlers.get_shared_redis_token_counter_service",
        lambda: fake_counter,
    )

    process_dlq_alert(
        {
            "token_request_id": "req_123",
            "user_id": str(uuid4()),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "token_count": 50,
            "api_endpoint_url": "https://example.test/v1",
            "allocation_status": "ACQUIRED",
            "dlq_reason": "db down",
        },
        headers={"x-token-retry-attempt": 5},
    )

    fake_counter.release_tokens.assert_awaited_once()
