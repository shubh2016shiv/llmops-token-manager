from __future__ import annotations

from uuid import uuid4

import pybreaker

from app.models.resilience_models import TokenAllocationPersistPayload
from app.resilience.token_queue.consumer import TokenQueueConsumerService
from app.resilience.token_queue.topology import TOKEN_RETRY_ATTEMPT_HEADER


class _FakeMessage:
    def __init__(self, headers: dict[str, object] | None = None) -> None:
        self.headers = headers or {}
        self.acked = False
        self.rejected_with_requeue: bool | None = None

    def ack(self) -> None:
        self.acked = True

    def reject(self, *, requeue: bool) -> None:
        self.rejected_with_requeue = requeue


def _validated_payload() -> TokenAllocationPersistPayload:
    return TokenAllocationPersistPayload.model_validate(
        {
            "token_request_id": "req_123",
            "user_id": str(uuid4()),
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "token_count": 50,
            "api_endpoint_url": "https://example.test/v1",
            "allocation_status": "ACQUIRED",
        }
    )


def test_work_message_successfully_persists_and_acks(monkeypatch) -> None:
    service = TokenQueueConsumerService()
    message = _FakeMessage()

    monkeypatch.setattr(
        "app.resilience.token_queue.consumer.persist_allocation_message",
        lambda body: _validated_payload(),
    )

    service._on_work_message({"token_request_id": "req_123"}, message)

    assert message.acked is True
    assert message.rejected_with_requeue is None


def test_work_message_failure_schedules_retry_and_acks(monkeypatch) -> None:
    service = TokenQueueConsumerService()
    message = _FakeMessage()
    captured: dict[str, object] = {}

    def _raise(_body: dict[str, object]) -> TokenAllocationPersistPayload:
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "app.resilience.token_queue.consumer.persist_allocation_message",
        _raise,
    )
    monkeypatch.setattr(
        service._publisher,
        "publish_retry_request",
        lambda body, attempt, reason: captured.update(
            {"body": body, "attempt": attempt, "reason": reason}
        ),
    )

    service._on_work_message({"token_request_id": "req_123"}, message)

    assert message.acked is True
    assert captured["attempt"] == 1
    assert captured["reason"] == "db down"


def test_work_message_failure_after_final_retry_routes_to_dlq(monkeypatch) -> None:
    service = TokenQueueConsumerService()
    message = _FakeMessage(headers={TOKEN_RETRY_ATTEMPT_HEADER: 5})
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "app.resilience.token_queue.consumer.persist_allocation_message",
        lambda _body: (_ for _ in ()).throw(RuntimeError("db still down")),
    )
    monkeypatch.setattr(
        service._publisher,
        "publish_dlq_notification",
        lambda body, reason, retry_attempts: captured.update(
            {
                "body": body,
                "reason": reason,
                "retry_attempts": retry_attempts,
            }
        ),
    )

    service._on_work_message({"token_request_id": "req_123"}, message)

    assert message.acked is True
    assert captured["retry_attempts"] == 5


def test_work_message_retry_publish_breaker_open_requeues_after_backoff(
    monkeypatch,
) -> None:
    service = TokenQueueConsumerService()
    message = _FakeMessage()
    slept: dict[str, object] = {}

    monkeypatch.setattr(
        "app.resilience.token_queue.consumer.persist_allocation_message",
        lambda _body: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    monkeypatch.setattr(
        service._publisher,
        "publish_retry_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pybreaker.CircuitBreakerError("open")
        ),
    )
    monkeypatch.setattr(
        "app.resilience.token_queue.consumer.time.sleep",
        lambda seconds: slept.update({"seconds": seconds}),
    )

    service._on_work_message({"token_request_id": "req_123"}, message)

    assert slept["seconds"] == 1
    assert message.rejected_with_requeue is True


def test_get_consumers_uses_configurable_prefetch_count() -> None:
    service = TokenQueueConsumerService(prefetch_count=7)
    captured: list[dict[str, object]] = []

    def _fake_consumer_cls(channel: object, **kwargs: object) -> dict[str, object]:
        consumer_kwargs = {"channel": channel, **kwargs}
        captured.append(consumer_kwargs)
        return consumer_kwargs

    consumers = service.get_consumers(_fake_consumer_cls, channel=object())

    assert len(consumers) == 2
    assert captured[0]["prefetch_count"] == 7
    assert captured[1]["prefetch_count"] == 1


def test_dlq_message_invokes_alert_handler_and_acks(monkeypatch) -> None:
    service = TokenQueueConsumerService()
    message = _FakeMessage(headers={TOKEN_RETRY_ATTEMPT_HEADER: 5})
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "app.resilience.token_queue.consumer.process_dlq_alert",
        lambda body, headers: captured.update({"body": body, "headers": headers}),
    )

    service._on_dlq_message({"token_request_id": "req_123"}, message)

    assert message.acked is True
    assert captured["headers"] == {TOKEN_RETRY_ATTEMPT_HEADER: 5}
