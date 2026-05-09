from __future__ import annotations

from app.resilience.token_queue import topology


def test_retry_stages_follow_configured_schedule() -> None:
    delays = [stage.delay_seconds for stage in topology.TOKEN_RETRY_STAGES]
    queue_names = [stage.queue.name for stage in topology.TOKEN_RETRY_STAGES]

    assert tuple(delays) == topology.settings.token_queue_retry_schedule_seconds
    assert queue_names == [
        "token.allocation.retry.5s",
        "token.allocation.retry.10s",
        "token.allocation.retry.20s",
        "token.allocation.retry.40s",
        "token.allocation.retry.60s",
    ]


def test_declare_token_queues_declares_exchanges_before_queues(monkeypatch) -> None:
    declare_order: list[str] = []
    channel_closed = {"value": False}

    class _FakeChannel:
        def close(self) -> None:
            channel_closed["value"] = True

    class _FakeConnection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _FakeConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def channel(self) -> _FakeChannel:
            return _FakeChannel()

    monkeypatch.setattr(topology, "Connection", _FakeConnection)
    monkeypatch.setattr(
        topology.TOKEN_EXCHANGE,
        "declare",
        lambda channel: declare_order.append("exchange"),
    )
    monkeypatch.setattr(
        topology.TOKEN_DLX,
        "declare",
        lambda channel: declare_order.append("dlx"),
    )

    for queue in topology.ALL_TOKEN_QUEUES:
        monkeypatch.setattr(
            queue,
            "declare",
            lambda channel, queue_name=queue.name: declare_order.append(queue_name),
        )

    topology.declare_token_queues()

    assert declare_order[0:2] == ["exchange", "dlx"]
    assert declare_order[2] == topology.TOKEN_ALLOCATION_QUEUE.name
    assert channel_closed["value"] is True
