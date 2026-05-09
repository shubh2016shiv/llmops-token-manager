from __future__ import annotations

from uuid import uuid4

import pybreaker
import pytest

from app.resilience.token_queue.publisher import TokenAllocationPublisher
from app.resilience.token_queue.topology import TOKEN_ALLOCATION_QUEUE


def _payload_dict() -> dict[str, object]:
    return {
        "token_request_id": "req_123",
        "user_id": str(uuid4()),
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
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
        raise pybreaker.CircuitBreakerError("open")

    monkeypatch.setattr(publisher._rmq_cb, "call", _raise)

    with pytest.raises(pybreaker.CircuitBreakerError):
        publisher.publish_allocation_request(_payload_dict())


def test_publish_sync_uses_pooled_connection(monkeypatch) -> None:
    published: dict[str, object] = {}

    class _FakeChannel:
        def __enter__(self) -> _FakeChannel:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class _FakeConnection:
        def channel(self) -> _FakeChannel:
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
    assert published["payload"] == {"token_request_id": "req_123"}
