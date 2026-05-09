from __future__ import annotations

import asyncio

from app.resilience.redis_token_counter import RedisTokenCounterService
from app.resilience.redis_token_counter import service_registry as registry_module


class _ScriptRedis:
    def __init__(self) -> None:
        self.close_count = 0

    def register_script(self, _script_text):
        async def _runner(*_args, **_kwargs):
            return 1

        return _runner

    async def close(self):
        self.close_count += 1


def test_shared_service_registry_reuses_singleton_and_resets_after_close(
    monkeypatch,
) -> None:
    service_one = RedisTokenCounterService(_ScriptRedis())
    service_two = RedisTokenCounterService(_ScriptRedis())

    build_sequence = [service_one, service_two]
    monkeypatch.setattr(
        registry_module,
        "create_redis_token_counter_service",
        lambda: build_sequence.pop(0),
    )
    registry_module._shared_redis_token_counter_service = None

    first = registry_module.get_shared_redis_token_counter_service()
    second = registry_module.get_shared_redis_token_counter_service()
    asyncio.run(registry_module.close_shared_redis_token_counter_service())
    third = registry_module.get_shared_redis_token_counter_service()

    assert first is second
    assert third is not first
    asyncio.run(registry_module.close_shared_redis_token_counter_service())
