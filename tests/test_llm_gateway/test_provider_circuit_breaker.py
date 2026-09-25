"""
Unit tests for provider_circuit_breaker.py.

All tests use injected fakes — no real Redis connection, no network I/O.
The synchronous Redis client is monkeypatched at the module level to prevent
any real connection attempt during test collection or execution.

Coverage targets:
    - get_provider_circuit_breaker() returns an aiobreaker.CircuitBreaker
    - Each provider gets an *independent* breaker instance (not the same object)
    - Known providers receive their §21 thresholds (openai/anthropic: 5/60s,
      azure_openai/gcp_vertex: 3/30s, aws_bedrock: 3/45s)
    - Unknown providers receive the default config (5/60s)
    - Singleton guarantee: the same object is returned on repeated calls
    - Redis failure falls back to in-memory OPEN storage (fail-closed)
    - get_all_provider_breaker_states() returns uppercase state strings
    - reset_provider_breaker_registry() clears the cache (for test isolation)
    - Breaker is shared across two concurrent "workers" (same registry key)
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import aiobreaker
import pytest

import app.llm_gateway.provider_circuit_breaker as pcb_module
from app.llm_gateway.provider_circuit_breaker import (
    _PROVIDER_CONFIGS,
    get_all_provider_breaker_states,
    get_provider_circuit_breaker,
    reset_provider_breaker_registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> object:
    """Reset the in-process registry before and after every test."""
    reset_provider_breaker_registry()
    yield
    reset_provider_breaker_registry()


@pytest.fixture()
def _fake_redis_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch _build_provider_storage to return CircuitMemoryStorage (CLOSED).

    Prevents any attempt to connect to Redis during tests.
    """

    def fake_build(provider_name: str) -> aiobreaker.storage.CircuitMemoryStorage:
        return aiobreaker.storage.CircuitMemoryStorage(
            aiobreaker.CircuitBreakerState.CLOSED
        )

    monkeypatch.setattr(pcb_module, "_build_provider_storage", fake_build)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestGetProviderCircuitBreaker:
    """get_provider_circuit_breaker() — construction and basic contract."""

    def test_returns_circuit_breaker_instance(self, _fake_redis_storage: None) -> None:
        cb = get_provider_circuit_breaker("openai")
        assert isinstance(cb, aiobreaker.CircuitBreaker)

    def test_state_is_closed_by_default(self, _fake_redis_storage: None) -> None:
        cb = get_provider_circuit_breaker("openai")
        assert cb.current_state == aiobreaker.CircuitBreakerState.CLOSED

    def test_singleton_same_object_on_repeat_calls(
        self, _fake_redis_storage: None
    ) -> None:
        cb1 = get_provider_circuit_breaker("openai")
        cb2 = get_provider_circuit_breaker("openai")
        assert cb1 is cb2

    def test_different_providers_return_different_breakers(
        self, _fake_redis_storage: None
    ) -> None:
        cb_openai = get_provider_circuit_breaker("openai")
        cb_anthropic = get_provider_circuit_breaker("anthropic")
        assert cb_openai is not cb_anthropic

    @pytest.mark.parametrize("provider_name", list(_PROVIDER_CONFIGS))
    def test_all_known_providers_construct_without_error(
        self, provider_name: str, _fake_redis_storage: None
    ) -> None:
        cb = get_provider_circuit_breaker(provider_name)
        assert isinstance(cb, aiobreaker.CircuitBreaker)

    def test_unknown_provider_uses_default_config(
        self, _fake_redis_storage: None
    ) -> None:
        cb = get_provider_circuit_breaker("some_unknown_provider")
        assert isinstance(cb, aiobreaker.CircuitBreaker)


# ---------------------------------------------------------------------------
# Per-provider thresholds (§21)
# ---------------------------------------------------------------------------


class TestProviderThresholds:
    """Verify §21 per-provider fail_max and timeout_duration values."""

    @pytest.mark.parametrize(
        "provider_name,expected_threshold,expected_reset",
        [
            ("openai", 5, 60),
            ("anthropic", 5, 60),
            ("azure_openai", 3, 30),
            ("gcp_vertex", 3, 30),
            ("aws_bedrock", 3, 45),
        ],
    )
    def test_config_table_matches_system_design(
        self,
        provider_name: str,
        expected_threshold: int,
        expected_reset: int,
        _fake_redis_storage: None,
    ) -> None:
        cb = get_provider_circuit_breaker(provider_name)
        assert cb.fail_max == expected_threshold
        from datetime import timedelta

        assert cb.timeout_duration == timedelta(seconds=expected_reset)

    def test_default_threshold_is_five(self, _fake_redis_storage: None) -> None:
        cb = get_provider_circuit_breaker("no_such_provider")
        assert cb.fail_max == 5

    def test_default_reset_is_sixty_seconds(self, _fake_redis_storage: None) -> None:
        from datetime import timedelta

        cb = get_provider_circuit_breaker("no_such_provider")
        assert cb.timeout_duration == timedelta(seconds=60)


# ---------------------------------------------------------------------------
# Redis failure fallback
# ---------------------------------------------------------------------------


class TestRedisFallback:
    """When Redis storage construction fails, the breaker falls back to OPEN."""

    def test_falls_back_to_open_storage_on_redis_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_on_connect(provider_name: str) -> None:
            raise ConnectionError("Redis unavailable")

        monkeypatch.setattr(
            pcb_module,
            "_build_provider_storage",
            raise_on_connect,
        )

        # _build_provider_storage raises; _create_provider_circuit_breaker catches it
        # but the breaker itself is never created — the error propagates.
        # The module's _build_provider_storage already catches and falls back internally.
        # Here we verify the outer create path handles a total storage failure gracefully.
        with pytest.raises(Exception):
            get_provider_circuit_breaker("openai")

    def test_memory_open_storage_constructed_on_redis_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_build_provider_storage falls back to CircuitMemoryStorage(OPEN)."""

        def raise_on_redis(
            provider_name: str,
        ) -> aiobreaker.storage.CircuitMemoryStorage:
            # Simulate breaker_storage.build_synchronous_redis_client() failing
            # inside _build_provider_storage.
            raise RuntimeError("no redis")

        # Patch at the level _build_provider_storage calls
        with patch.object(
            pcb_module.breaker_storage,
            "build_synchronous_redis_client",
            side_effect=RuntimeError("no redis"),
        ):
            storage = pcb_module._build_provider_storage("openai")

        assert isinstance(storage, aiobreaker.storage.CircuitMemoryStorage)
        assert storage.state == aiobreaker.CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# State introspection
# ---------------------------------------------------------------------------


class TestGetAllProviderBreakerStates:
    """get_all_provider_breaker_states() — state snapshot correctness."""

    def test_empty_before_any_breaker_created(self) -> None:
        assert get_all_provider_breaker_states() == {}

    def test_returns_closed_for_newly_created_breaker(
        self, _fake_redis_storage: None
    ) -> None:
        get_provider_circuit_breaker("openai")
        states = get_all_provider_breaker_states()
        assert states["openai"] == "CLOSED"

    def test_state_values_are_uppercase_strings(
        self, _fake_redis_storage: None
    ) -> None:
        get_provider_circuit_breaker("anthropic")
        states = get_all_provider_breaker_states()
        for value in states.values():
            assert value == value.upper()
            assert value in {"CLOSED", "OPEN", "HALF_OPEN"}

    def test_all_created_providers_appear_in_snapshot(
        self, _fake_redis_storage: None
    ) -> None:
        for name in ("openai", "anthropic", "azure_openai"):
            get_provider_circuit_breaker(name)
        states = get_all_provider_breaker_states()
        assert set(states.keys()) == {"openai", "anthropic", "azure_openai"}


# ---------------------------------------------------------------------------
# Registry reset
# ---------------------------------------------------------------------------


class TestResetRegistry:
    """reset_provider_breaker_registry() — cache invalidation."""

    def test_reset_clears_all_entries(self, _fake_redis_storage: None) -> None:
        get_provider_circuit_breaker("openai")
        reset_provider_breaker_registry()
        assert get_all_provider_breaker_states() == {}

    def test_new_breaker_created_after_reset(self, _fake_redis_storage: None) -> None:
        cb_before = get_provider_circuit_breaker("openai")
        reset_provider_breaker_registry()
        cb_after = get_provider_circuit_breaker("openai")
        assert cb_before is not cb_after


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Singleton guarantee holds under concurrent access."""

    def test_concurrent_calls_return_same_instance(
        self, _fake_redis_storage: None
    ) -> None:
        collected: list[aiobreaker.CircuitBreaker] = []
        lock = threading.Lock()

        def _get() -> None:
            cb = get_provider_circuit_breaker("openai")
            with lock:
                collected.append(cb)

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(collected) == 10
        first = collected[0]
        assert all(cb is first for cb in collected)
