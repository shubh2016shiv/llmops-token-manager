"""
Unit tests for gateway_tasks/_task_runner.py.

All tests use injected fakes — no Celery broker, no real Redis, no real provider.

Coverage targets:
    - TaskPayload.from_dict() validates required fields and rejects bad UUIDs
    - TaskPayload.to_execution_request() forwards fields correctly
    - _check_circuit_breaker() raises ProviderCircuitOpenError when CB is OPEN
    - _check_circuit_breaker() passes through when CB is CLOSED or HALF_OPEN
    - run_llm_task() happy path returns result dict and writes SUCCESS metadata
    - run_llm_task() TRANSIENT error triggers task.retry() with jittered countdown
    - run_llm_task() QUOTA_CALIBRATION error triggers task.retry()
    - run_llm_task() AUTH_STALE raises ProviderExecutionError immediately (no retry)
    - run_llm_task() CONTEXT_EXCEEDED raises immediately (no retry)
    - run_llm_task() CONTENT_POLICY raises immediately (no retry)
    - run_llm_task() exhausted retry budget raises ProviderExecutionError
    - run_llm_task() bad payload raises TaskInputValidationError (no retry)
    - run_llm_task() CB OPEN raises ProviderCircuitOpenError (passed through)
    - STARTED metadata is written before execution
    - FAILURE metadata is written before re-raise on non-retriable error
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import aiobreaker
import pytest

from app.llm_gateway.error_classifier import GatewayErrorClass
from app.llm_gateway.gateway_tasks._task_runner import (
    TaskPayload,
    _check_circuit_breaker,
    run_llm_task,
)
from app.llm_gateway.gateway_tasks.exceptions import (
    ProviderCircuitOpenError,
    ProviderExecutionError,
    TaskInputValidationError,
)
from app.llm_gateway.gateway_tasks.job_meta import (
    JobMetadata,
    JobStatus,
)
from app.llm_gateway.providers.provider_types import ProviderExecutionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = uuid4()
_JOB_ID = "test-job-id-001"
_TOKEN_REQUEST_ID = "alloc-test-001"

_FAKE_ALLOCATION: dict[str, object] = {
    "token_request_id": _TOKEN_REQUEST_ID,
    "allocation_status": "ACQUIRED",
    "llm_model_name": "gpt-4o",
    "api_endpoint_url": "https://api.openai.com/v1",
    "token_count": 500,
}


def _valid_payload_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "job_id": _JOB_ID,
        "token_request_id": _TOKEN_REQUEST_ID,
        "user_id": str(_USER_ID),
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "user_prompt": "Hello, world!",
        "submitted_at": "2026-05-11T10:00:00Z",
    }
    base.update(overrides)
    return base


def _make_result() -> ProviderExecutionResult:
    return ProviderExecutionResult(
        completion_text="The answer is 42.",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        llm_provider="openai",
        model_name_reported_by_provider="gpt-4o",
        latency_milliseconds=300.0,
    )


class FakeCeleryTask:
    """Minimal Celery task double — records retry calls."""

    def __init__(self, retries: int = 0, max_retries: int = 3) -> None:
        self.request = MagicMock()
        self.request.retries = retries
        self.request.id = _JOB_ID
        self.retry_calls: list[dict] = []

    def retry(self, exc: Exception, countdown: float, max_retries: int) -> Exception:
        self.retry_calls.append(
            {"exc": exc, "countdown": countdown, "max_retries": max_retries}
        )
        # Celery's retry() raises; we replicate that so run_llm_task propagates it
        raise exc


class FakeGateway:
    """Gateway double that returns a configurable result or raises."""

    def __init__(
        self,
        result: ProviderExecutionResult | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[dict] = []

    async def execute(
        self, user_id: UUID, llm_provider: str, llm_model_name: str, request: object
    ) -> ProviderExecutionResult:
        self.calls.append(
            {"user_id": user_id, "provider": llm_provider, "model": llm_model_name}
        )
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


class FakeMetaWriter:
    """JobMetadataWriter double that records write() calls."""

    def __init__(self) -> None:
        self.writes: list[JobMetadata] = []

    async def write(self, meta: JobMetadata) -> None:
        self.writes.append(meta)


# ---------------------------------------------------------------------------
# TaskPayload.from_dict()
# ---------------------------------------------------------------------------


class TestTaskPayloadFromDict:
    def test_valid_dict_returns_payload(self) -> None:
        p = TaskPayload.from_dict(_valid_payload_dict())
        assert p.job_id == _JOB_ID
        assert p.llm_provider == "openai"

    def test_user_id_parsed_to_uuid(self) -> None:
        p = TaskPayload.from_dict(_valid_payload_dict())
        assert isinstance(p.user_id, UUID)
        assert p.user_id == _USER_ID

    @pytest.mark.parametrize(
        "missing_field",
        [
            "job_id",
            "token_request_id",
            "user_id",
            "llm_provider",
            "llm_model_name",
            "user_prompt",
            "submitted_at",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        raw = _valid_payload_dict()
        del raw[missing_field]
        with pytest.raises(TaskInputValidationError) as exc_info:
            TaskPayload.from_dict(raw)
        assert exc_info.value.field == missing_field

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(TaskInputValidationError) as exc_info:
            TaskPayload.from_dict(_valid_payload_dict(user_id="not-a-uuid"))
        assert exc_info.value.field == "user_id"

    def test_optional_fields_default_to_none(self) -> None:
        p = TaskPayload.from_dict(_valid_payload_dict())
        assert p.system_message is None
        assert p.max_tokens is None
        assert p.temperature is None

    def test_optional_fields_passed_through(self) -> None:
        p = TaskPayload.from_dict(
            _valid_payload_dict(
                system_message="You are helpful.",
                max_tokens=512,
                temperature=0.7,
            )
        )
        assert p.system_message == "You are helpful."
        assert p.max_tokens == 512
        assert p.temperature == 0.7


class TestTaskPayloadToExecutionRequest:
    def test_user_prompt_forwarded(self) -> None:
        p = TaskPayload.from_dict(_valid_payload_dict())
        req = p.to_execution_request()
        assert req.user_prompt == "Hello, world!"

    def test_optional_fields_forwarded(self) -> None:
        p = TaskPayload.from_dict(
            _valid_payload_dict(
                system_message="Be concise.",
                max_tokens=256,
                temperature=0.5,
            )
        )
        req = p.to_execution_request()
        assert req.system_message == "Be concise."
        assert req.max_tokens_to_generate == 256
        assert req.temperature == 0.5


# ---------------------------------------------------------------------------
# _check_circuit_breaker()
# ---------------------------------------------------------------------------


class TestCheckCircuitBreaker:
    def _patch_breaker(self, state: aiobreaker.CircuitBreakerState) -> MagicMock:
        breaker = MagicMock()
        breaker.current_state = state
        return breaker

    def test_closed_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        breaker = self._patch_breaker(aiobreaker.CircuitBreakerState.CLOSED)
        monkeypatch.setattr(
            runner_mod, "get_provider_circuit_breaker", lambda _: breaker
        )
        _check_circuit_breaker("openai")  # must not raise

    def test_open_raises_circuit_open_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        breaker = self._patch_breaker(aiobreaker.CircuitBreakerState.OPEN)
        monkeypatch.setattr(
            runner_mod, "get_provider_circuit_breaker", lambda _: breaker
        )
        with pytest.raises(ProviderCircuitOpenError) as exc_info:
            _check_circuit_breaker("openai")
        assert exc_info.value.provider_name == "openai"

    def test_half_open_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        breaker = self._patch_breaker(aiobreaker.CircuitBreakerState.HALF_OPEN)
        monkeypatch.setattr(
            runner_mod, "get_provider_circuit_breaker", lambda _: breaker
        )
        _check_circuit_breaker("anthropic")  # must not raise


# ---------------------------------------------------------------------------
# run_llm_task() — happy path
# ---------------------------------------------------------------------------


def _patch_guardrails(
    monkeypatch: pytest.MonkeyPatch,
    allocation: dict[str, object] | None = None,
) -> None:
    """
    Patch _verify_allocation_is_acquired and _release_tokens* to avoid DB/Redis I/O.

    Both functions perform real async DB/Redis calls which have no place in unit
    tests. We replace them with coroutine stubs so the rest of run_llm_task()
    executes unchanged.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        allocation:  The allocation dict returned by the verify stub.
                     Defaults to _FAKE_ALLOCATION.
    """
    import app.llm_gateway.gateway_tasks._task_runner as runner_mod

    resolved_allocation = (
        allocation if allocation is not None else dict(_FAKE_ALLOCATION)
    )

    async def _fake_verify(_token_request_id: str) -> dict[str, object]:
        return resolved_allocation

    async def _fake_release(_token_request_id: str, _alloc: dict[str, object]) -> None:
        return

    def _fake_release_sync(_token_request_id: str) -> None:
        return

    monkeypatch.setattr(runner_mod, "_verify_allocation_is_acquired", _fake_verify)
    monkeypatch.setattr(runner_mod, "_release_tokens", _fake_release)
    monkeypatch.setattr(runner_mod, "_release_tokens_sync", _fake_release_sync)


class TestRunLlmTaskSuccess:
    def _closed_breaker_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        breaker = MagicMock()
        breaker.current_state = aiobreaker.CircuitBreakerState.CLOSED
        monkeypatch.setattr(
            runner_mod, "get_provider_circuit_breaker", lambda _: breaker
        )

    def test_returns_result_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._closed_breaker_patch(monkeypatch)
        _patch_guardrails(monkeypatch)
        task = FakeCeleryTask()
        gateway = FakeGateway(result=_make_result())
        writer = FakeMetaWriter()
        result = run_llm_task(task, _valid_payload_dict(), gateway, writer)  # type: ignore[arg-type]
        assert result["completion"] == "The answer is 42."
        assert result["total_tokens"] == 15

    def test_result_contains_all_expected_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker_patch(monkeypatch)
        _patch_guardrails(monkeypatch)
        task = FakeCeleryTask()
        result = run_llm_task(  # type: ignore[arg-type]
            task,
            _valid_payload_dict(),
            FakeGateway(result=_make_result()),
            FakeMetaWriter(),
        )
        for key in (
            "completion",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "provider_latency_ms",
            "llm_provider",
            "model_name",
        ):
            assert key in result

    def test_started_metadata_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._closed_breaker_patch(monkeypatch)
        _patch_guardrails(monkeypatch)
        task = FakeCeleryTask()
        writer = FakeMetaWriter()
        run_llm_task(
            task, _valid_payload_dict(), FakeGateway(result=_make_result()), writer
        )  # type: ignore[arg-type]
        statuses = [w.status for w in writer.writes]
        assert JobStatus.STARTED in statuses

    def test_success_metadata_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._closed_breaker_patch(monkeypatch)
        _patch_guardrails(monkeypatch)
        task = FakeCeleryTask()
        writer = FakeMetaWriter()
        run_llm_task(
            task, _valid_payload_dict(), FakeGateway(result=_make_result()), writer
        )  # type: ignore[arg-type]
        statuses = [w.status for w in writer.writes]
        assert JobStatus.SUCCESS in statuses

    def test_gateway_receives_correct_provider_and_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker_patch(monkeypatch)
        _patch_guardrails(monkeypatch)
        task = FakeCeleryTask()
        gateway = FakeGateway(result=_make_result())
        run_llm_task(  # type: ignore[arg-type]
            task,
            _valid_payload_dict(
                llm_provider="anthropic", llm_model_name="claude-3-5-sonnet"
            ),
            gateway,
            FakeMetaWriter(),
        )
        assert gateway.calls[0]["provider"] == "anthropic"
        assert gateway.calls[0]["model"] == "claude-3-5-sonnet"

    def test_tokens_released_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._closed_breaker_patch(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        release_calls: list[str] = []

        async def _fake_verify(_tid: str) -> dict[str, object]:
            return dict(_FAKE_ALLOCATION)

        async def _tracking_release(
            token_request_id: str, _alloc: dict[str, object]
        ) -> None:
            release_calls.append(token_request_id)

        monkeypatch.setattr(runner_mod, "_verify_allocation_is_acquired", _fake_verify)
        monkeypatch.setattr(runner_mod, "_release_tokens", _tracking_release)
        monkeypatch.setattr(runner_mod, "_release_tokens_sync", lambda _: None)

        run_llm_task(  # type: ignore[arg-type]
            FakeCeleryTask(),
            _valid_payload_dict(),
            FakeGateway(result=_make_result()),
            FakeMetaWriter(),
        )
        assert release_calls == [_TOKEN_REQUEST_ID]


# ---------------------------------------------------------------------------
# run_llm_task() — error paths
# ---------------------------------------------------------------------------


class TestRunLlmTaskErrors:
    def _closed_breaker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        breaker = MagicMock()
        breaker.current_state = aiobreaker.CircuitBreakerState.CLOSED
        monkeypatch.setattr(
            runner_mod, "get_provider_circuit_breaker", lambda _: breaker
        )

    def _open_breaker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        breaker = MagicMock()
        breaker.current_state = aiobreaker.CircuitBreakerState.OPEN
        monkeypatch.setattr(
            runner_mod, "get_provider_circuit_breaker", lambda _: breaker
        )

    def test_bad_payload_raises_input_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        task = FakeCeleryTask()
        bad_payload: dict[str, object] = {}
        with pytest.raises(TaskInputValidationError):
            run_llm_task(
                task, bad_payload, FakeGateway(result=_make_result()), FakeMetaWriter()
            )  # type: ignore[arg-type]

    def test_open_cb_raises_circuit_open_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._open_breaker(monkeypatch)
        task = FakeCeleryTask()
        with pytest.raises(ProviderCircuitOpenError):
            run_llm_task(
                task,
                _valid_payload_dict(),
                FakeGateway(result=_make_result()),  # type: ignore[arg-type]
                FakeMetaWriter(),
            )

    @pytest.mark.parametrize(
        "error_class_name",
        [
            "CONTEXT_EXCEEDED",
            "CONTENT_POLICY",
            "PROVIDER_DOWN",
        ],
    )
    def test_non_retriable_raises_immediately(
        self, error_class_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        target_class = GatewayErrorClass(error_class_name)
        monkeypatch.setattr(runner_mod, "classify", lambda _exc: target_class)

        task = FakeCeleryTask()
        gateway = FakeGateway(exc=RuntimeError("provider error"))
        with pytest.raises(ProviderExecutionError) as exc_info:
            run_llm_task(task, _valid_payload_dict(), gateway, FakeMetaWriter())  # type: ignore[arg-type]
        assert exc_info.value.error_class == error_class_name
        assert task.retry_calls == []  # no retry attempted

    def test_auth_stale_raises_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.AUTH_STALE
        )

        task = FakeCeleryTask()
        gateway = FakeGateway(exc=RuntimeError("401"))
        with pytest.raises(ProviderExecutionError) as exc_info:
            run_llm_task(task, _valid_payload_dict(), gateway, FakeMetaWriter())  # type: ignore[arg-type]
        assert exc_info.value.error_class == "AUTH_STALE"
        assert task.retry_calls == []

    def test_transient_triggers_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.TRANSIENT
        )
        monkeypatch.setattr(runner_mod, "compute_backoff_seconds", lambda *_: 5.0)

        task = FakeCeleryTask(retries=0)
        gateway = FakeGateway(exc=RuntimeError("503"))
        with pytest.raises(Exception):
            run_llm_task(
                task, _valid_payload_dict(), gateway, FakeMetaWriter(), max_retries=3
            )  # type: ignore[arg-type]
        assert len(task.retry_calls) == 1
        assert task.retry_calls[0]["countdown"] == 5.0

    def test_quota_calibration_triggers_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.QUOTA_CALIBRATION
        )
        monkeypatch.setattr(runner_mod, "compute_backoff_seconds", lambda *_: 30.0)

        task = FakeCeleryTask(retries=0)
        gateway = FakeGateway(exc=RuntimeError("429"))
        with pytest.raises(Exception):
            run_llm_task(
                task, _valid_payload_dict(), gateway, FakeMetaWriter(), max_retries=3
            )  # type: ignore[arg-type]
        assert len(task.retry_calls) == 1

    def test_exhausted_retry_budget_raises_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.TRANSIENT
        )

        task = FakeCeleryTask(retries=3)  # already at max
        gateway = FakeGateway(exc=RuntimeError("503"))
        with pytest.raises(ProviderExecutionError):
            run_llm_task(
                task, _valid_payload_dict(), gateway, FakeMetaWriter(), max_retries=3
            )  # type: ignore[arg-type]
        assert task.retry_calls == []

    def test_failure_metadata_written_on_non_retriable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.CONTEXT_EXCEEDED
        )

        task = FakeCeleryTask()
        writer = FakeMetaWriter()
        gateway = FakeGateway(exc=RuntimeError("too long"))
        with pytest.raises(ProviderExecutionError):
            run_llm_task(task, _valid_payload_dict(), gateway, writer)  # type: ignore[arg-type]
        statuses = [w.status for w in writer.writes]
        assert JobStatus.FAILURE in statuses

    def test_retrying_metadata_written_on_transient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        _patch_guardrails(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.TRANSIENT
        )
        monkeypatch.setattr(runner_mod, "compute_backoff_seconds", lambda *_: 5.0)

        task = FakeCeleryTask(retries=0)
        writer = FakeMetaWriter()
        gateway = FakeGateway(exc=RuntimeError("503"))
        with pytest.raises(Exception):
            run_llm_task(task, _valid_payload_dict(), gateway, writer, max_retries=3)  # type: ignore[arg-type]
        statuses = [w.status for w in writer.writes]
        assert JobStatus.RETRYING in statuses

    def test_tokens_not_released_during_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        release_calls: list[str] = []

        async def _fake_verify(_tid: str) -> dict[str, object]:
            return dict(_FAKE_ALLOCATION)

        async def _tracking_release(tid: str, _alloc: dict[str, object]) -> None:
            release_calls.append(tid)

        def _tracking_release_sync(tid: str) -> None:
            release_calls.append(tid)

        monkeypatch.setattr(runner_mod, "_verify_allocation_is_acquired", _fake_verify)
        monkeypatch.setattr(runner_mod, "_release_tokens", _tracking_release)
        monkeypatch.setattr(runner_mod, "_release_tokens_sync", _tracking_release_sync)
        monkeypatch.setattr(
            runner_mod, "classify", lambda _: GatewayErrorClass.TRANSIENT
        )
        monkeypatch.setattr(runner_mod, "compute_backoff_seconds", lambda *_: 1.0)

        task = FakeCeleryTask(retries=0)
        gateway = FakeGateway(exc=RuntimeError("503"))
        with pytest.raises(Exception):
            run_llm_task(
                task, _valid_payload_dict(), gateway, FakeMetaWriter(), max_retries=3
            )  # type: ignore[arg-type]
        # Tokens must stay locked while retrying — release must NOT have been called
        assert release_calls == []

    def test_allocation_not_acquired_raises_without_llm_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._closed_breaker(monkeypatch)
        import app.llm_gateway.gateway_tasks._task_runner as runner_mod

        async def _verify_waiting(_tid: str) -> dict[str, object]:
            from app.llm_gateway.gateway_tasks.exceptions import (
                TaskInputValidationError,
            )

            raise TaskInputValidationError(
                "token_request_id",
                "Allocation is in 'WAITING' state, not ACQUIRED.",
            )

        monkeypatch.setattr(
            runner_mod, "_verify_allocation_is_acquired", _verify_waiting
        )
        monkeypatch.setattr(runner_mod, "_release_tokens_sync", lambda _: None)

        gateway = FakeGateway(result=_make_result())
        with pytest.raises(TaskInputValidationError):
            run_llm_task(
                FakeCeleryTask(),
                _valid_payload_dict(),
                gateway,
                FakeMetaWriter(),  # type: ignore[arg-type]
            )
        # Gateway must never be called when allocation guard fails
        assert gateway.calls == []
