from __future__ import annotations

import asyncio

from app.models.resilience_models import CircuitBreakerSnapshot
from app.resilience.backpressure import evaluator as evaluator_module


def test_evaluator_accepts_when_all_signals_are_healthy(monkeypatch) -> None:
    async def _read_queue_depth() -> int | None:
        return 100

    monkeypatch.setattr(evaluator_module, "read_queue_depth", _read_queue_depth)
    monkeypatch.setattr(
        evaluator_module,
        "read_db_pool_utilization_pct",
        lambda: 50,
    )
    monkeypatch.setattr(
        evaluator_module,
        "read_db_circuit_breaker_snapshot",
        lambda: CircuitBreakerSnapshot(
            name="postgres",
            state="closed",
            failure_count=0,
            recovery_timeout_seconds=30,
            opened_at=None,
        ),
    )

    decision = asyncio.run(evaluator_module.evaluate_backpressure())

    assert decision.should_reject_request is False
    assert decision.reason is None


def test_evaluator_rejects_when_queue_depth_exceeds_threshold(monkeypatch) -> None:
    async def _read_queue_depth() -> int | None:
        return evaluator_module.settings.bp_max_queue_depth + 1

    monkeypatch.setattr(evaluator_module, "read_queue_depth", _read_queue_depth)
    monkeypatch.setattr(
        evaluator_module,
        "read_db_pool_utilization_pct",
        lambda: (_ for _ in ()).throw(AssertionError("pool probe should not run")),
    )
    monkeypatch.setattr(
        evaluator_module,
        "read_db_circuit_breaker_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("breaker probe should not run")),
    )

    decision = asyncio.run(evaluator_module.evaluate_backpressure())

    assert decision.should_reject_request is True
    assert decision.reason == "queue_depth_exceeded"
    assert decision.queue_depth == evaluator_module.settings.bp_max_queue_depth + 1
    assert decision.retry_after_seconds is not None


def test_evaluator_rejects_when_db_pool_is_saturated(monkeypatch) -> None:
    async def _read_queue_depth() -> int | None:
        return evaluator_module.settings.bp_max_queue_depth

    monkeypatch.setattr(evaluator_module, "read_queue_depth", _read_queue_depth)
    monkeypatch.setattr(
        evaluator_module,
        "read_db_pool_utilization_pct",
        lambda: evaluator_module.settings.bp_db_pool_saturation_pct,
    )
    monkeypatch.setattr(
        evaluator_module,
        "read_db_circuit_breaker_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("breaker probe should not run")),
    )

    decision = asyncio.run(evaluator_module.evaluate_backpressure())

    assert decision.should_reject_request is True
    assert decision.reason == "db_pool_saturated"
    assert (
        decision.retry_after_seconds
        == evaluator_module.settings.bp_db_pool_retry_after_seconds
    )


def test_evaluator_rejects_when_db_breaker_is_open(monkeypatch) -> None:
    async def _read_queue_depth() -> int | None:
        return None

    monkeypatch.setattr(evaluator_module, "read_queue_depth", _read_queue_depth)
    monkeypatch.setattr(
        evaluator_module,
        "read_db_pool_utilization_pct",
        lambda: None,
    )
    monkeypatch.setattr(
        evaluator_module,
        "read_db_circuit_breaker_snapshot",
        lambda: CircuitBreakerSnapshot(
            name="postgres",
            state="open",
            failure_count=5,
            recovery_timeout_seconds=30,
            opened_at=None,
        ),
    )

    decision = asyncio.run(evaluator_module.evaluate_backpressure())

    assert decision.should_reject_request is True
    assert decision.reason == "db_circuit_breaker_open"
    assert decision.circuit_breaker_name == "postgres"
    assert decision.retry_after_seconds == 30


def test_evaluator_checks_signals_in_layer1_order(monkeypatch) -> None:
    call_order: list[str] = []

    async def _read_queue_depth() -> int | None:
        call_order.append("queue")
        return None

    def _read_db_pool_utilization_pct() -> int | None:
        call_order.append("pool")
        return None

    def _read_db_circuit_breaker_snapshot() -> CircuitBreakerSnapshot:
        call_order.append("breaker")
        return CircuitBreakerSnapshot(
            name="postgres",
            state="closed",
            failure_count=0,
            recovery_timeout_seconds=30,
            opened_at=None,
        )

    monkeypatch.setattr(evaluator_module, "read_queue_depth", _read_queue_depth)
    monkeypatch.setattr(
        evaluator_module,
        "read_db_pool_utilization_pct",
        _read_db_pool_utilization_pct,
    )
    monkeypatch.setattr(
        evaluator_module,
        "read_db_circuit_breaker_snapshot",
        _read_db_circuit_breaker_snapshot,
    )

    asyncio.run(evaluator_module.evaluate_backpressure())

    assert call_order == ["queue", "pool", "breaker"]
