from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import aiobreaker

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
    """
    Regression fix: this previously used field names (llm_provider,
    llm_model_name) that don't exist on TokenAllocationPersistPayload and
    omitted required fields (tenant_id, deployment_id, deployment_key), so
    model_validate() raised here — inside the lambda standing in for
    persist_allocation_message() — which then fell through
    _on_work_message's real exception handler into a real (unmocked)
    publish_retry_request() call. That call needs live RabbitMQ, which is
    what actually made this test fail in this environment; the stack trace's
    "Failed to build storage" noise was a red herring one level removed from
    the real bug.
    """
    return TokenAllocationPersistPayload.model_validate(
        {
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
            aiobreaker.CircuitBreakerError(
                "circuit open", datetime.now() + timedelta(seconds=30)
            )
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
    """
    Regression fix: the original fake required `channel` as an explicit
    positional argument to consumer_cls(...). Real kombu never calls
    get_consumers()'s consumer_cls that way — per ConsumerMixin.consume()
    (kombu/mixins.py), it pre-binds the channel via
    `functools.partial(Consumer, channel, ...)` *before* calling
    get_consumers(), which is exactly why consumer.py's own docstring says
    "consumer_cls is already bound to the channel... must NOT be passed
    again." The old fake enforced the opposite of what the real contract is,
    so this test failed on a TypeError that had nothing to do with the
    prefetch-count behavior it was meant to verify.
    """
    service = TokenQueueConsumerService(prefetch_count=7)
    captured: list[dict[str, object]] = []

    class _FakeConsumer:
        """
        Stands in for a real kombu Consumer: get_consumers() calls
        `.qos(prefetch_count=...)` on whatever consumer_cls(...) returns, so
        the fake must support that (a plain dict, as the original fixture
        returned, does not — AttributeError: 'dict' object has no attribute
        'qos', a second bug the channel-argument fix alone didn't surface).
        """

        def __init__(self, **kwargs: object) -> None:
            self.init_kwargs = kwargs

        def qos(self, *, prefetch_count: int) -> None:
            self.init_kwargs["prefetch_count"] = prefetch_count
            captured.append(self.init_kwargs)

    consumers = service.get_consumers(_FakeConsumer, channel=object())

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
