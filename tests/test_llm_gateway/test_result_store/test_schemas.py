"""
Unit tests for result_store/schemas.py.

Coverage targets:
    - JobPollResponse validates required fields
    - JobPollResponse serialises JobStatus enum values as strings
    - JobPollResponse optional fields default to None
    - JobNotFoundResponse has fixed error code and message
    - JobNotFoundResponse carries job_id
    - model_dump() produces JSON-serialisable output for both models
"""

from __future__ import annotations

import json

from app.llm_gateway.gateway_tasks.job_meta import JobStatus
from app.llm_gateway.result_store.schemas import JobNotFoundResponse, JobPollResponse


class TestJobPollResponse:
    def _make(self, status: JobStatus = JobStatus.SUCCESS) -> JobPollResponse:
        return JobPollResponse(
            job_id="abc-123",
            status=status,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            submitted_at="2026-05-11T10:00:00Z",
        )

    def test_required_fields_accepted(self) -> None:
        resp = self._make()
        assert resp.job_id == "abc-123"
        assert resp.llm_provider == "openai"
        assert resp.llm_model_name == "gpt-4o"

    def test_status_serialised_as_string(self) -> None:
        resp = self._make(JobStatus.RETRYING)
        dump = resp.model_dump()
        assert dump["status"] == "RETRYING"
        assert isinstance(dump["status"], str)

    def test_optional_fields_default_none(self) -> None:
        resp = self._make()
        assert resp.started_at is None
        assert resp.completed_at is None
        assert resp.completion is None
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None
        assert resp.total_tokens is None
        assert resp.provider_latency_milliseconds is None
        assert resp.error_class is None
        assert resp.error_message is None

    def test_retry_count_defaults_to_zero(self) -> None:
        resp = self._make()
        assert resp.retry_count == 0

    def test_all_statuses_accepted(self) -> None:
        for s in JobStatus:
            resp = self._make(s)
            assert resp.model_dump()["status"] == s.value

    def test_success_fields_populated(self) -> None:
        resp = JobPollResponse(
            job_id="xyz-789",
            status=JobStatus.SUCCESS,
            llm_provider="anthropic",
            llm_model_name="claude-3-5-sonnet",
            submitted_at="2026-05-11T10:00:00Z",
            started_at="2026-05-11T10:00:01Z",
            completed_at="2026-05-11T10:00:02Z",
            completion="Hello, World!",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            provider_latency_milliseconds=420.5,
            retry_count=0,
        )
        dump = resp.model_dump()
        assert dump["completion"] == "Hello, World!"
        assert dump["prompt_tokens"] == 50
        assert dump["total_tokens"] == 60
        assert dump["provider_latency_milliseconds"] == 420.5

    def test_failure_fields_populated(self) -> None:
        resp = JobPollResponse(
            job_id="fail-001",
            status=JobStatus.FAILURE,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            submitted_at="2026-05-11T10:00:00Z",
            error_class="TRANSIENT",
            error_message="upstream timeout",
            retry_count=3,
        )
        dump = resp.model_dump()
        assert dump["error_class"] == "TRANSIENT"
        assert dump["error_message"] == "upstream timeout"
        assert dump["retry_count"] == 3

    def test_model_dump_is_json_serialisable(self) -> None:
        resp = self._make()
        serialised = json.dumps(resp.model_dump())
        assert isinstance(json.loads(serialised), dict)


class TestJobNotFoundResponse:
    def test_error_code_is_fixed(self) -> None:
        resp = JobNotFoundResponse(job_id="missing-id")
        assert resp.error == "job_not_found"

    def test_message_mentions_retention(self) -> None:
        resp = JobNotFoundResponse(job_id="missing-id")
        assert "24 hours" in resp.message

    def test_carries_job_id(self) -> None:
        resp = JobNotFoundResponse(job_id="abc-xyz")
        assert resp.job_id == "abc-xyz"

    def test_model_dump_is_json_serialisable(self) -> None:
        resp = JobNotFoundResponse(job_id="gone-id")
        serialised = json.dumps(resp.model_dump())
        parsed = json.loads(serialised)
        assert parsed["job_id"] == "gone-id"
        assert parsed["error"] == "job_not_found"
