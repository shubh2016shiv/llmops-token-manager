from __future__ import annotations

from datetime import datetime, timezone

from app.resilience.backpressure import (
    circuit_state_probe as circuit_state_probe_module,
)


class _FakeStateStorage:
    def __init__(self, opened_at: object) -> None:
        self.opened_at = opened_at


class _FakeCircuitBreaker:
    def __init__(self, opened_at: object) -> None:
        self.name = "postgres"
        self.current_state = "open"
        self.fail_counter = 5
        self.reset_timeout = 30
        self._state_storage = _FakeStateStorage(opened_at)


def test_circuit_state_probe_coerces_unix_timestamp_to_datetime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        circuit_state_probe_module,
        "get_db_circuit_breaker",
        lambda: _FakeCircuitBreaker(1715270400.123),
    )

    snapshot = circuit_state_probe_module.read_db_circuit_breaker_snapshot()

    assert snapshot.opened_at == datetime.fromtimestamp(
        1715270400.123,
        tz=timezone.utc,
    )


def test_circuit_state_probe_preserves_datetime_value(monkeypatch) -> None:
    opened_at = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        circuit_state_probe_module,
        "get_db_circuit_breaker",
        lambda: _FakeCircuitBreaker(opened_at),
    )

    snapshot = circuit_state_probe_module.read_db_circuit_breaker_snapshot()

    assert snapshot.opened_at == opened_at
