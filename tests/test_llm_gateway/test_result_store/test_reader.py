"""
Unit tests for result_store/reader.py.

Coverage targets:
    - JobResultReader._build_key() produces gw:job:{id}:meta
    - JobResultReader.read() returns None when Redis returns None (key absent)
    - JobResultReader.read() returns None when Redis payload is invalid JSON
    - JobResultReader.read() calls GET then TTL on the correct key
    - JobResultReader.read() returns JobReadResult with parsed dict
    - JobResultReader.read() computes expires_at from TTL correctly
    - _compute_expires_at() handles ttl=-1 (no expiry → fallback 24h)
    - _compute_expires_at() handles ttl=-2 (absent → fallback 24h)
    - _compute_expires_at() handles ttl=3600 (1h → now + 1h)
    - Redis errors propagate (not swallowed)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock

import pytest

from app.llm_gateway.result_store.reader import (
    JobReadResult,
    JobResultReader,
    _compute_expires_at,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_reader() -> tuple[JobResultReader, AsyncMock]:
    redis_mock = AsyncMock()
    reader = JobResultReader(redis_client=redis_mock)
    return reader, redis_mock


def _valid_payload(job_id: str = "job-001", status: str = "SUCCESS") -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "status": status,
            "llm_provider": "openai",
            "llm_model_name": "gpt-4o",
            "submitted_at": "2026-05-11T10:00:00Z",
        }
    )


# ---------------------------------------------------------------------------
# Key building
# ---------------------------------------------------------------------------


class TestBuildKey:
    def test_format(self) -> None:
        reader, _ = _make_reader()
        assert reader._build_key("abc-123") == "gw:job:abc-123:meta"

    def test_preserves_uuid_format(self) -> None:
        reader, _ = _make_reader()
        uid = "3f2a1b9c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
        key = reader._build_key(uid)
        assert key == f"gw:job:{uid}:meta"


# ---------------------------------------------------------------------------
# read() — key absent
# ---------------------------------------------------------------------------


class TestReadKeyAbsent:
    @pytest.mark.asyncio
    async def test_returns_none_when_key_missing(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = None
        result = await reader.read("missing-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_not_called_when_key_missing(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = None
        await reader.read("missing-id")
        redis_mock.ttl.assert_not_awaited()


# ---------------------------------------------------------------------------
# read() — invalid JSON
# ---------------------------------------------------------------------------


class TestReadInvalidJson:
    @pytest.mark.asyncio
    async def test_returns_none_on_corrupt_payload(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = "not-valid-json{{{}"
        redis_mock.ttl.return_value = 86400
        result = await reader.read("job-001")
        assert result is None


# ---------------------------------------------------------------------------
# read() — happy path
# ---------------------------------------------------------------------------


class TestReadHappyPath:
    @pytest.mark.asyncio
    async def test_returns_job_read_result(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload()
        redis_mock.ttl.return_value = 3600
        result = await reader.read("job-001")
        assert isinstance(result, JobReadResult)

    @pytest.mark.asyncio
    async def test_raw_contains_job_id(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload(job_id="job-007")
        redis_mock.ttl.return_value = 3600
        result = await reader.read("job-007")
        assert result is not None
        assert result.raw["job_id"] == "job-007"

    @pytest.mark.asyncio
    async def test_raw_contains_status(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload(status="RETRYING")
        redis_mock.ttl.return_value = 3600
        result = await reader.read("job-001")
        assert result is not None
        assert result.raw["status"] == "RETRYING"

    @pytest.mark.asyncio
    async def test_get_called_with_correct_key(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload()
        redis_mock.ttl.return_value = 3600
        await reader.read("job-abc")
        redis_mock.get.assert_awaited_once_with("gw:job:job-abc:meta")

    @pytest.mark.asyncio
    async def test_ttl_called_with_correct_key(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload()
        redis_mock.ttl.return_value = 3600
        await reader.read("job-abc")
        redis_mock.ttl.assert_awaited_once_with("gw:job:job-abc:meta")

    @pytest.mark.asyncio
    async def test_expires_at_is_utc_datetime(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload()
        redis_mock.ttl.return_value = 3600
        result = await reader.read("job-001")
        assert result is not None
        assert isinstance(result.expires_at, datetime)
        assert result.expires_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_expires_at_approx_ttl_from_now(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.return_value = _valid_payload()
        redis_mock.ttl.return_value = 7200  # 2h
        before = datetime.now(timezone.utc)
        result = await reader.read("job-001")
        after = datetime.now(timezone.utc)
        assert result is not None
        lower = before + timedelta(seconds=7200)
        upper = after + timedelta(seconds=7200)
        assert lower <= result.expires_at <= upper


# ---------------------------------------------------------------------------
# read() — Redis error propagation
# ---------------------------------------------------------------------------


class TestReadRedisError:
    @pytest.mark.asyncio
    async def test_redis_error_propagates(self) -> None:
        reader, redis_mock = _make_reader()
        redis_mock.get.side_effect = ConnectionError("Redis down")
        with pytest.raises(ConnectionError):
            await reader.read("job-001")


# ---------------------------------------------------------------------------
# _compute_expires_at()
# ---------------------------------------------------------------------------


class TestComputeExpiresAt:
    def test_positive_ttl_adds_seconds(self) -> None:
        before = datetime.now(timezone.utc)
        result = _compute_expires_at(3600)
        after = datetime.now(timezone.utc)
        assert (
            before + timedelta(seconds=3600)
            <= result
            <= after + timedelta(seconds=3600)
        )

    def test_ttl_minus_one_uses_fallback(self) -> None:
        before = datetime.now(timezone.utc)
        result = _compute_expires_at(-1)
        after = datetime.now(timezone.utc)
        assert (
            before + timedelta(seconds=86400)
            <= result
            <= after + timedelta(seconds=86400)
        )

    def test_ttl_minus_two_uses_fallback(self) -> None:
        before = datetime.now(timezone.utc)
        result = _compute_expires_at(-2)
        after = datetime.now(timezone.utc)
        assert (
            before + timedelta(seconds=86400)
            <= result
            <= after + timedelta(seconds=86400)
        )

    def test_zero_ttl_gives_now(self) -> None:
        before = datetime.now(timezone.utc)
        result = _compute_expires_at(0)
        after = datetime.now(timezone.utc)
        assert before <= result <= after

    def test_result_is_utc(self) -> None:
        result = _compute_expires_at(3600)
        assert result.tzinfo == timezone.utc
