from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import aiobreaker
from amqp.exceptions import RecoverableConnectionError
import pytest

from app.resilience.token_queue.publisher import (
    TokenAllocationPublisher,
    TokenPublishError,
)
from app.resilience.token_queue.topology import TOKEN_ALLOCATION_QUEUE


def _payload_dict() -> dict[str, object]:
    return {
        "token_request_id": "req_123",
        "tenant_id": str(uuid4()),
        "user_id": str(uuid4()),
        "deployment_id": str(uuid4()),
        "provider_name": "openai",
        "model_name": "gpt-4o",
        "deployment_key": "openai-deployment",
        "token_count": 50,
        "api_endpoint_url": "https://example.test/v1",
        "allocation_status": "ACQUIRED",
    }


def test_publish_allocation_request_returns_message_id(monkeypatch) -> None:
    publisher = TokenAllocationPublisher()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        TokenAllocationPublisher,
        "_publish_sync",
        staticmethod(
            lambda payload, message_id, **kwargs: captured.update(
                {
                    "payload": payload,
                    "message_id": message_id,
                    "headers": kwargs["headers"],
                }
            )
        ),
    )
    monkeypatch.setattr(
        publisher._rmq_cb,
        "call",
        lambda func, *args, **kwargs: func(*args, **kwargs),
    )

    message_id = publisher.publish_allocation_request(_payload_dict())

    assert message_id == "req_123"
    assert captured["message_id"] == "req_123"
    assert captured["headers"] == {"x-token-retry-attempt": 0}


def test_publish_allocation_request_raises_when_breaker_open(monkeypatch) -> None:
    publisher = TokenAllocationPublisher()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise aiobreaker.CircuitBreakerError(
            "circuit open", datetime.now() + timedelta(seconds=30)
        )

    monkeypatch.setattr(publisher._rmq_cb, "call", _raise)

    with pytest.raises(aiobreaker.CircuitBreakerError):
        publisher.publish_allocation_request(_payload_dict())


def test_publish_allocation_request_translates_transport_failure(monkeypatch) -> None:
    publisher = TokenAllocationPublisher()

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise RecoverableConnectionError("broker disconnected")

    monkeypatch.setattr(publisher._rmq_cb, "call", _fail)

    with pytest.raises(TokenPublishError, match="RabbitMQ allocation publish failed"):
        publisher.publish_allocation_request(_payload_dict())


def test_publish_retry_request_raises_bare_when_breaker_open(monkeypatch) -> None:
    """
    publish_retry_request must NOT wrap CircuitBreakerError — the current
    caller (consumer._on_work_message) matches on this exact exception type
    to choose pause-and-requeue over log-and-requeue, so wrapping it would
    silently break that branch.
    """
    publisher = TokenAllocationPublisher()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise aiobreaker.CircuitBreakerError(
            "circuit open", datetime.now() + timedelta(seconds=30)
        )

    monkeypatch.setattr(publisher._rmq_cb, "call", _raise)

    with pytest.raises(aiobreaker.CircuitBreakerError):
        publisher.publish_retry_request(_payload_dict(), attempt=1, reason="db down")


def test_publish_retry_request_logs_and_reraises_transport_failure(
    monkeypatch,
) -> None:
    """
    Regression test: publish_retry_request previously let transport failures
    (AMQPError/KombuError/ConnectionError/...) propagate completely
    unlabeled, unlike its two sibling methods on the same class. It must now
    log with attempt/msg_id context and re-raise the SAME exception type
    (not a translated TokenPublishError) so consumer.py's existing bare
    `except Exception` fallback still catches it.
    """
    publisher = TokenAllocationPublisher()

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise RecoverableConnectionError("broker disconnected")

    monkeypatch.setattr(publisher._rmq_cb, "call", _fail)

    with pytest.raises(RecoverableConnectionError):
        publisher.publish_retry_request(_payload_dict(), attempt=1, reason="db down")


def test_publish_sync_uses_pooled_connection(monkeypatch) -> None:
    published: dict[str, object] = {}

    class _FakeChannel:
        def __enter__(self) -> _FakeChannel:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class _FakeConnection:
        def ensure_connection(self, *, max_retries: int) -> None:
            assert max_retries == 1
            published["reconnected"] = True

        def channel(self) -> _FakeChannel:
            assert published["reconnected"] is True
            return _FakeChannel()

    class _AcquireContext:
        def __enter__(self) -> _FakeConnection:
            return _FakeConnection()

        def __exit__(self, *_args: object) -> None:
            return None

    class _FakeConnectionPoolMap:
        def __init__(self) -> None:
            self.used = False

        def __getitem__(self, _key: object) -> _FakeConnectionPoolMap:
            self.used = True
            return self

        def acquire(self, *, block: bool) -> _AcquireContext:
            assert block is True
            return _AcquireContext()

    class _FakePools:
        def __init__(self) -> None:
            self.connections = _FakeConnectionPoolMap()

    class _FakeProducer:
        def __init__(self, _channel: object) -> None:
            pass

        def publish(self, payload: dict[str, object], **kwargs: object) -> None:
            published["payload"] = payload
            published["kwargs"] = kwargs

    fake_pools = _FakePools()
    monkeypatch.setattr(
        "app.resilience.token_queue.publisher.pools",
        fake_pools,
    )
    monkeypatch.setattr(
        "app.resilience.token_queue.publisher.Producer",
        _FakeProducer,
    )

    TokenAllocationPublisher._publish_sync(
        {"token_request_id": "req_123"},
        "req_123",
        queue=TOKEN_ALLOCATION_QUEUE,
        routing_key="token.allocate",
        exchange="exchange",
        headers={"x-token-retry-attempt": 0},
    )

    assert fake_pools.connections.used is True
    assert published["reconnected"] is True
    assert published["payload"] == {"token_request_id": "req_123"}
