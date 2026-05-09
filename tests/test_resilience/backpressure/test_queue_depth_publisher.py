from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.resilience.backpressure import (
    queue_depth_publisher as queue_depth_publisher_module,
)
from app.resilience.backpressure.constants import (
    QUEUE_DEPTH_PUBLISH_SCHEDULE_NAME,
    QUEUE_DEPTH_PUBLISH_TASK_NAME,
)
from app.resilience.token_maintenance import (
    schedule_registry as schedule_registry_module,
)


class _FakeRedisClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    async def set(self, key: str, value: int, ex: int) -> None:
        self.calls.append((key, value, ex))


class _RedisManagerWithClient:
    def __init__(self, client: object) -> None:
        self.client = client


class _FakeChannel:
    def __enter__(self) -> _FakeChannel:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeConnection:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def channel(self) -> _FakeChannel:
        return _FakeChannel()


class _QueueWithDepth:
    def __init__(self, _channel: object, _queue_name: str) -> None:
        return None

    def qsize(self) -> int:
        return 42


class _ExplodingQueue:
    def __init__(self, _channel: object, _queue_name: str) -> None:
        raise RuntimeError("rabbitmq unavailable")


def test_queue_depth_publisher_writes_depth_with_expected_ttl(monkeypatch) -> None:
    fake_redis_client = _FakeRedisClient()
    monkeypatch.setattr(
        queue_depth_publisher_module,
        "Connection",
        _FakeConnection,
    )
    monkeypatch.setattr(
        queue_depth_publisher_module,
        "SimpleQueue",
        _QueueWithDepth,
    )
    monkeypatch.setattr(
        queue_depth_publisher_module,
        "redis_manager",
        _RedisManagerWithClient(fake_redis_client),
    )

    asyncio.run(queue_depth_publisher_module.publish_queue_depth_snapshot())

    assert fake_redis_client.calls == [
        (
            "token_alloc:queue_depth",
            42,
            queue_depth_publisher_module.settings.bp_queue_depth_publish_interval_secs
            * 3,
        )
    ]


def test_queue_depth_publisher_skips_redis_write_when_sampling_fails(
    monkeypatch,
) -> None:
    fake_redis_client = _FakeRedisClient()
    monkeypatch.setattr(
        queue_depth_publisher_module,
        "Connection",
        _FakeConnection,
    )
    monkeypatch.setattr(
        queue_depth_publisher_module,
        "SimpleQueue",
        _ExplodingQueue,
    )
    monkeypatch.setattr(
        queue_depth_publisher_module,
        "redis_manager",
        _RedisManagerWithClient(fake_redis_client),
    )

    asyncio.run(queue_depth_publisher_module.publish_queue_depth_snapshot())

    assert fake_redis_client.calls == []


def test_publish_task_is_registered_in_beat_schedule(monkeypatch) -> None:
    fake_celery_app = SimpleNamespace(conf=SimpleNamespace(beat_schedule={}))
    monkeypatch.setattr(schedule_registry_module, "_beat_registered", False)

    schedule_registry_module.register_beat_schedule(fake_celery_app)

    assert (
        fake_celery_app.conf.beat_schedule[QUEUE_DEPTH_PUBLISH_SCHEDULE_NAME]["task"]
        == QUEUE_DEPTH_PUBLISH_TASK_NAME
    )
    assert (
        schedule_registry_module.MAINTENANCE_TASK_ROUTES[QUEUE_DEPTH_PUBLISH_TASK_NAME][
            "queue"
        ]
        == schedule_registry_module.settings.celery_token_maintenance_queue_name
    )
