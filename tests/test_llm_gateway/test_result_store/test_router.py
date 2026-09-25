"""
Unit tests for result_store/router.py.

Coverage targets:
    - GET /api/v1/llm/jobs/{job_id} returns 200 when job exists
    - GET /api/v1/llm/jobs/{job_id} returns 404 when job absent in Redis
    - 200 response body matches JobPollResponse schema
    - 404 response body matches JobNotFoundResponse schema
    - 200 sets X-Job-Status header to current status string
    - 200 sets X-Job-Expires-At header (non-empty ISO string)
    - 200 sets Cache-Control: no-store
    - 404 sets X-Job-Status: NOT_FOUND
    - 404 sets Cache-Control: no-store
    - Unknown status string in Redis payload falls back to PENDING (no 500)
    - All JobStatus values produce correct X-Job-Status header
    - _parse_status() falls back to PENDING for unknown values
    - _found_response() maps all raw dict fields to response body correctly
    - _not_found_response() carries the job_id in the body
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.llm_gateway.gateway_tasks.job_meta import JobStatus
from app.llm_gateway.result_store.reader import JobReadResult
from app.llm_gateway.result_store.router import (
    _found_response,
    _not_found_response,
    _parse_status,
    router,
)

# ---------------------------------------------------------------------------
# Test app + client
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


_app = _make_app()

_EXPIRES_AT = datetime.now(timezone.utc) + timedelta(hours=24)
_EXPIRES_AT_ISO = _EXPIRES_AT.isoformat()


def _valid_raw(
    job_id: str = "job-001",
    status: str = "SUCCESS",
) -> dict:
    return {
        "job_id": job_id,
        "status": status,
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "submitted_at": "2026-05-11T10:00:00Z",
        "started_at": "2026-05-11T10:00:01Z",
        "completed_at": "2026-05-11T10:00:02Z",
        "retry_count": 0,
        "completion": "Hello!",
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "total_tokens": 60,
        "provider_latency_milliseconds": 420.5,
        "error_class": None,
        "error_message": None,
    }


def _make_job_read_result(raw: dict | None = None) -> JobReadResult:
    return JobReadResult(
        raw=raw or _valid_raw(),
        expires_at=_EXPIRES_AT,
    )


@contextmanager
def _patch_reader(read_return_value):
    """Patch both redis_manager.client (so it doesn't raise) and JobResultReader.read."""
    fake_redis = MagicMock()
    with patch("app.llm_gateway.result_store.router.redis_manager") as mock_mgr:
        mock_mgr.client = fake_redis
        with patch(
            "app.llm_gateway.result_store.router.JobResultReader"
        ) as mock_reader:
            instance = mock_reader.return_value
            instance.read = AsyncMock(return_value=read_return_value)
            yield


# ---------------------------------------------------------------------------
# HTTP 200 — job found
# ---------------------------------------------------------------------------


class TestPollJobStatusFound:
    def test_returns_200(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.status_code == 200

    def test_body_has_job_id(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.json()["job_id"] == "job-001"

    def test_body_has_status(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.json()["status"] == "SUCCESS"

    def test_x_job_status_header(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.headers["X-Job-Status"] == "SUCCESS"

    def test_x_job_expires_at_header_present(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert "X-Job-Expires-At" in resp.headers
        assert len(resp.headers["X-Job-Expires-At"]) > 0

    def test_cache_control_no_store(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.headers["Cache-Control"] == "no-store"

    def test_token_fields_in_body(self) -> None:
        with _patch_reader(_make_job_read_result()):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        body = resp.json()
        assert body["prompt_tokens"] == 50
        assert body["total_tokens"] == 60
        assert body["provider_latency_milliseconds"] == 420.5

    @pytest.mark.parametrize("job_status", list(JobStatus))
    def test_all_statuses_produce_200(self, job_status: JobStatus) -> None:
        raw = _valid_raw(status=job_status.value)
        result = _make_job_read_result(raw)
        with _patch_reader(result):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.status_code == 200
        assert resp.headers["X-Job-Status"] == job_status.value


# ---------------------------------------------------------------------------
# HTTP 404 — job not found
# ---------------------------------------------------------------------------


class TestPollJobStatusNotFound:
    def test_returns_404(self) -> None:
        with _patch_reader(None):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/gone-id")
        assert resp.status_code == 404

    def test_body_error_code(self) -> None:
        with _patch_reader(None):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/gone-id")
        assert resp.json()["error"] == "job_not_found"

    def test_body_contains_job_id(self) -> None:
        with _patch_reader(None):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/gone-id")
        assert resp.json()["job_id"] == "gone-id"

    def test_x_job_status_not_found_header(self) -> None:
        with _patch_reader(None):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/gone-id")
        assert resp.headers["X-Job-Status"] == "NOT_FOUND"

    def test_cache_control_no_store(self) -> None:
        with _patch_reader(None):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/gone-id")
        assert resp.headers["Cache-Control"] == "no-store"

    def test_no_x_job_expires_at_header(self) -> None:
        with _patch_reader(None):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/gone-id")
        assert "X-Job-Expires-At" not in resp.headers


# ---------------------------------------------------------------------------
# Corrupt status string in Redis — must not 500
# ---------------------------------------------------------------------------


class TestCorruptStatusFallback:
    def test_unknown_status_returns_200_with_pending(self) -> None:
        raw = _valid_raw(status="TOTALLY_UNKNOWN_STATE")
        result = _make_job_read_result(raw)
        with _patch_reader(result):
            with TestClient(_app) as client:
                resp = client.get("/api/v1/llm/jobs/job-001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Unit tests for helper functions (no HTTP round-trip needed)
# ---------------------------------------------------------------------------


class TestParseStatus:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("PENDING", JobStatus.PENDING),
            ("STARTED", JobStatus.STARTED),
            ("RETRYING", JobStatus.RETRYING),
            ("SUCCESS", JobStatus.SUCCESS),
            ("FAILURE", JobStatus.FAILURE),
        ],
    )
    def test_valid_values(self, s: str, expected: JobStatus) -> None:
        assert _parse_status(s) == expected

    def test_unknown_value_falls_back_to_pending(self) -> None:
        assert _parse_status("GARBAGE") == JobStatus.PENDING

    def test_empty_string_falls_back_to_pending(self) -> None:
        assert _parse_status("") == JobStatus.PENDING


class TestFoundResponse:
    def test_returns_json_response_with_200(self) -> None:
        resp = _found_response(_valid_raw(), _EXPIRES_AT_ISO)
        assert resp.status_code == 200

    def test_x_job_expires_at_set(self) -> None:
        resp = _found_response(_valid_raw(), _EXPIRES_AT_ISO)
        assert resp.headers["X-Job-Expires-At"] == _EXPIRES_AT_ISO

    def test_cache_control_header(self) -> None:
        resp = _found_response(_valid_raw(), _EXPIRES_AT_ISO)
        assert resp.headers["Cache-Control"] == "no-store"

    def test_body_is_valid_json(self) -> None:
        resp = _found_response(_valid_raw(), _EXPIRES_AT_ISO)
        parsed = json.loads(resp.body)
        assert isinstance(parsed, dict)
        assert parsed["job_id"] == "job-001"


class TestNotFoundResponse:
    def test_returns_404(self) -> None:
        resp = _not_found_response("missing-job")
        assert resp.status_code == 404

    def test_x_job_status_not_found(self) -> None:
        resp = _not_found_response("missing-job")
        assert resp.headers["X-Job-Status"] == "NOT_FOUND"

    def test_body_job_id(self) -> None:
        resp = _not_found_response("missing-job")
        parsed = json.loads(resp.body)
        assert parsed["job_id"] == "missing-job"

    def test_body_error_code(self) -> None:
        resp = _not_found_response("missing-job")
        parsed = json.loads(resp.body)
        assert parsed["error"] == "job_not_found"
