from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.resilience.backpressure.probes import (
    circuit_state as circuit_state_probe_module,
)


class _FakeStateStorage:
    def __init__(self, opened_at: object) -> None:
        self.opened_at = opened_at


class _FakeCircuitBreaker:
    """
    Stands in for the real aiobreaker breaker the probe introspects.

    Mirrors the attributes the probe actually reads:
      - current_state.name  (an object with a `.name`, e.g. "OPEN")
      - timeout_duration    (a timedelta — recovery window)
      - fail_counter, name, _state_storage.opened_at
    """

    def __init__(self, opened_at: object) -> None:
        self.name = "postgres"
        self.current_state = SimpleNamespace(name="OPEN")
        self.fail_counter = 5
        self.timeout_duration = timedelta(seconds=30)
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
