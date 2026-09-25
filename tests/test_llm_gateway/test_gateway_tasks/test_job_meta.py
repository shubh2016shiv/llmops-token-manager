"""
Unit tests for gateway_tasks/job_meta.py.

Coverage targets:
    - JobStatus values match §19 state machine strings
    - JobMetadata.to_json() produces valid JSON with correct status serialisation
    - JobMetadata.utc_now() returns a UTC ISO-8601 string
    - JobMetadataWriter._build_key() produces gw:job:{id}:meta
    - JobMetadataWriter.write() calls redis SET with correct key, payload, TTL
    - JobMetadataWriter.write() swallows Redis errors (non-fatal metadata loss)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.llm_gateway.gateway_tasks.job_meta import (
    JobMetadata,
    JobMetadataWriter,
    JobStatus,
)


class TestJobStatus:
    def test_all_state_machine_values_present(self) -> None:
        values = {s.value for s in JobStatus}
        assert values == {"PENDING", "STARTED", "RETRYING", "SUCCESS", "FAILURE"}

    def test_is_str_enum(self) -> None:
        assert isinstance(JobStatus.SUCCESS, str)
        assert JobStatus.SUCCESS == "SUCCESS"


class TestJobMetadata:
    def _make(self, status: JobStatus = JobStatus.STARTED) -> JobMetadata:
        return JobMetadata(
            job_id="abc-123",
            status=status,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            submitted_at="2026-05-11T10:00:00Z",
        )

    def test_to_json_is_valid_json(self) -> None:
        meta = self._make()
        parsed = json.loads(meta.to_json())
        assert isinstance(parsed, dict)

    def test_to_json_serialises_status_as_string(self) -> None:
        meta = self._make(JobStatus.RETRYING)
        parsed = json.loads(meta.to_json())
        assert parsed["status"] == "RETRYING"
        assert isinstance(parsed["status"], str)

    def test_to_json_contains_required_fields(self) -> None:
        meta = self._make()
        parsed = json.loads(meta.to_json())
        for field in (
            "job_id",
            "status",
            "llm_provider",
            "llm_model_name",
            "submitted_at",
        ):
            assert field in parsed

    def test_to_json_none_fields_serialised_as_null(self) -> None:
        meta = self._make()
        parsed = json.loads(meta.to_json())
        assert parsed["started_at"] is None
        assert parsed["completed_at"] is None
        assert parsed["error_class"] is None

    def test_to_json_populated_fields_round_trip(self) -> None:
        meta = JobMetadata(
            job_id="xyz-789",
            status=JobStatus.SUCCESS,
            llm_provider="anthropic",
            llm_model_name="claude-3-5-sonnet",
            submitted_at="2026-05-11T10:00:00Z",
            started_at="2026-05-11T10:00:01Z",
            completed_at="2026-05-11T10:00:02Z",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            provider_latency_milliseconds=800.0,
            retry_count=1,
        )
        parsed = json.loads(meta.to_json())
        assert parsed["prompt_tokens"] == 100
        assert parsed["total_tokens"] == 150
        assert parsed["provider_latency_milliseconds"] == 800.0
        assert parsed["retry_count"] == 1

    def test_utc_now_is_string(self) -> None:
        result = JobMetadata.utc_now()
        assert isinstance(result, str)

    def test_utc_now_contains_timezone_offset(self) -> None:
        result = JobMetadata.utc_now()
        assert "+" in result or "Z" in result or "00:00" in result


class TestJobMetadataWriter:
    def _make_writer(
        self,
        ttl: int = 86400,
    ) -> tuple[JobMetadataWriter, AsyncMock]:
        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock()
        writer = JobMetadataWriter(redis_client=redis_mock, ttl_seconds=ttl)
        return writer, redis_mock

    def _make_meta(self, status: JobStatus = JobStatus.STARTED) -> JobMetadata:
        return JobMetadata(
            job_id="job-001",
            status=status,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            submitted_at="2026-05-11T10:00:00Z",
        )

    def test_build_key_format(self) -> None:
        writer, _ = self._make_writer()
        assert writer._build_key("abc-123") == "gw:job:abc-123:meta"

    @pytest.mark.asyncio
    async def test_write_calls_redis_set(self) -> None:
        writer, redis_mock = self._make_writer()
        meta = self._make_meta()
        await writer.write(meta)
        redis_mock.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_uses_correct_key(self) -> None:
        writer, redis_mock = self._make_writer()
        meta = self._make_meta()
        await writer.write(meta)
        call_args = redis_mock.set.call_args
        assert call_args[0][0] == "gw:job:job-001:meta"

    @pytest.mark.asyncio
    async def test_write_passes_ttl_as_ex(self) -> None:
        writer, redis_mock = self._make_writer(ttl=3600)
        meta = self._make_meta()
        await writer.write(meta)
        call_kwargs = redis_mock.set.call_args[1]
        assert call_kwargs.get("ex") == 3600

    @pytest.mark.asyncio
    async def test_write_payload_is_valid_json(self) -> None:
        writer, redis_mock = self._make_writer()
        meta = self._make_meta()
        await writer.write(meta)
        payload = redis_mock.set.call_args[0][1]
        parsed = json.loads(payload)
        assert parsed["job_id"] == "job-001"
        assert parsed["status"] == "STARTED"

    @pytest.mark.asyncio
    async def test_write_swallows_redis_errors(self) -> None:
        writer, redis_mock = self._make_writer()
        redis_mock.set.side_effect = ConnectionError("Redis down")
        meta = self._make_meta()
        # Must not raise — metadata loss is non-fatal
        await writer.write(meta)

    @pytest.mark.asyncio
    async def test_write_all_statuses_succeed(self) -> None:
        writer, redis_mock = self._make_writer()
        for status in JobStatus:
            redis_mock.reset_mock()
            meta = self._make_meta(status)
            await writer.write(meta)
            redis_mock.set.assert_awaited_once()
