"""
Regression test: token estimation must not block the event loop.

TokenEstimator.estimate() (app/utils/token_count_estimation.py) is a
synchronous, CPU-bound call into litellm's tokenizer. acquire_tokens()
previously called it directly — `self._estimate_token_count(request)` with
no `await`, no `asyncio.to_thread` — which blocks the ENTIRE event loop for
the call's duration, freezing every other concurrent request this worker
process is serving, not just the slow one. Fixed by offloading it via
`asyncio.to_thread` in token_acquisition_service.py.

This test proves the fix the only way that actually matters: by showing a
concurrently-scheduled coroutine keeps making progress *while* a
deliberately slow (blocking `time.sleep`) estimation call is in flight. If
the estimation call were still running inline on the loop, the concurrent
task's ticks would be starved until estimation finished, and the recorded
tick times would show a gap instead of a steady cadence.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.request_models import TokenAllocationClientRequest
from app.resilience.redis_token_counter import TokenReservationResult
import app.services.token_acquisition_service as token_acquisition_service_module
from app.services.token_acquisition_service import TokenAcquisitionService

_SLOW_ESTIMATION_SECONDS = 0.3
_TICK_INTERVAL_SECONDS = 0.02


def _blocking_slow_estimate_tokens(*_args: object, **_kwargs: object) -> object:
    """Stand in for TokenEstimator hitting a slow/uncached tokenizer path."""
    time.sleep(_SLOW_ESTIMATION_SECONDS)
    return MagicMock(total_tokens=10)


@pytest.mark.asyncio
async def test_slow_token_estimation_does_not_starve_concurrent_coroutines(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        token_acquisition_service_module,
        "estimate_tokens",
        _blocking_slow_estimate_tokens,
    )

    redis_counter = MagicMock()
    redis_counter.reserve_tokens = AsyncMock(
        return_value=TokenReservationResult.COUNTER_MISS
    )
    service = TokenAcquisitionService(
        allocation_persistence=MagicMock(),
        load_balancer=MagicMock(),
        redis_counter=redis_counter,
        publisher=MagicMock(),
        db_circuit_breaker=MagicMock(),
    )
    service._choose_deployment = AsyncMock(
        return_value={
            "api_endpoint_url": "https://example.test/v1",
            "deployment_id": uuid4(),
            "deployment_key": "openai-deployment",
        }
    )
    db_response = object()
    service._create_db_allocation = AsyncMock(return_value=db_response)
    request = TokenAllocationClientRequest(
        llm_provider="openai",
        model_name="gpt-4o",
        input_data="hello",
        requested_completion_tokens=10,
    )

    tick_gaps: list[float] = []

    async def _ticker(start_time: float) -> None:
        """
        Record the wall-clock gap between scheduler ticks while acquire runs.

        `start_time` is captured by the caller BEFORE this task is created —
        not on first entry to this coroutine body. A still-blocking
        acquire_tokens() would starve this task before its body ever runs a
        single line (asyncio.create_task only *schedules* it; nothing yields
        control to it until the blocking call finally returns control to the
        loop), so timing relative to a start captured inside the coroutine
        would silently miss exactly the stall this test exists to catch.
        """
        previous = start_time
        for _ in range(20):
            await asyncio.sleep(_TICK_INTERVAL_SECONDS)
            now = time.monotonic()
            tick_gaps.append(now - previous)
            previous = now

    loop_start = time.monotonic()
    ticker_task = asyncio.create_task(_ticker(loop_start))
    # Yield once so the ticker task actually starts before acquire_tokens
    # runs — otherwise asyncio may run acquire_tokens to its first await
    # point first purely due to scheduling order, independent of blocking.
    await asyncio.sleep(0)

    result = await service.acquire_tokens(uuid4(), uuid4(), request)

    await ticker_task

    assert result is db_response
    # A blocked event loop would show at least one tick gap close to
    # _SLOW_ESTIMATION_SECONDS (the ticker starved for the full sleep). An
    # unblocked loop keeps ticking close to the requested interval the whole
    # time. The threshold is well below the slow call's duration and well
    # above normal scheduler jitter, so it discriminates cleanly between the
    # two without being flaky on a loaded CI box.
    worst_gap = max(tick_gaps)
    assert worst_gap < _SLOW_ESTIMATION_SECONDS / 2, (
        f"event loop appears blocked: worst tick gap {worst_gap:.3f}s "
        f"(slow estimation call takes {_SLOW_ESTIMATION_SECONDS}s)"
    )
