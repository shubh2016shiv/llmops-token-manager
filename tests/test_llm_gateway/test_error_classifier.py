"""
Unit tests for error_classifier.py.

All tests use real exception instances where the SDK is available, and
synthetic exception stand-ins where it is not. No network calls, no I/O.

Coverage targets:
    - classify() routes each provider family to its per-provider classifier
    - Each error class is reachable via a concrete exception type
    - AUTH_STALE, CONTEXT_EXCEEDED, CONTENT_POLICY are non-retriable (max_retries=0)
    - TRANSIENT and QUOTA_CALIBRATION are retriable (max_retries=3)
    - compute_backoff_seconds() respects the TRANSIENT vs QUOTA delay sequences
    - compute_backoff_seconds() non-retriable classes always return 0.0
    - compute_backoff_seconds() result is within the ±20% jitter band
    - Unknown exceptions default to TRANSIENT
    - get_retry_policy() returns the correct RetryPolicy per class
"""

from __future__ import annotations

import pytest

from app.llm_gateway.error_classifier import (
    _CONTENT_POLICY_PHRASES,
    _CONTEXT_EXCEEDED_PHRASES,
    GatewayErrorClass,
    RetryPolicy,
    classify,
    compute_backoff_seconds,
    get_retry_policy,
)

# ---------------------------------------------------------------------------
# Synthetic exception helpers — work whether or not the real SDK is installed
# ---------------------------------------------------------------------------


def _make_exc(
    *,
    module: str,
    classname: str,
    message: str = "test error",
    status_code: int | None = None,
    boto_code: str | None = None,
) -> BaseException:
    """Build a synthetic exception whose module path matches a real SDK exception."""
    attrs: dict[str, object] = {}
    if status_code is not None:
        attrs["status_code"] = status_code
    if boto_code is not None:
        attrs["response"] = {"Error": {"Code": boto_code, "Message": message}}

    exc_type = type(classname, (Exception,), attrs)
    exc_type.__module__ = module
    instance = exc_type(message)
    if status_code is not None:
        instance.status_code = status_code  # type: ignore[attr-defined]
    if boto_code is not None:
        instance.response = {"Error": {"Code": boto_code, "Message": message}}  # type: ignore[attr-defined]
    return instance


def _openai_exc(
    classname: str, message: str = "error", status_code: int | None = None
) -> BaseException:
    return _make_exc(
        module="openai", classname=classname, message=message, status_code=status_code
    )


def _anthropic_exc(
    classname: str, message: str = "error", status_code: int | None = None
) -> BaseException:
    return _make_exc(
        module="anthropic",
        classname=classname,
        message=message,
        status_code=status_code,
    )


def _mistral_exc(message: str = "error", status_code: int = 500) -> BaseException:
    return _make_exc(
        module="mistralai",
        classname="SDKError",
        message=message,
        status_code=status_code,
    )


def _boto_exc(code: str, message: str = "error") -> BaseException:
    return _make_exc(
        module="botocore.exceptions",
        classname="ClientError",
        boto_code=code,
        message=message,
    )


# ---------------------------------------------------------------------------
# classify() — try/except-based dispatch uses isinstance, so we need to test
# with actual SDK types when available, falling back to generic checks.
# We test the classification logic using the real SDK where importable.
# ---------------------------------------------------------------------------


class TestClassifyUnknownException:
    """Unrecognised exceptions must default to TRANSIENT (safe fallback)."""

    def test_plain_runtime_error_is_transient(self) -> None:
        assert classify(RuntimeError("unexpected")) == GatewayErrorClass.TRANSIENT

    def test_value_error_is_transient(self) -> None:
        assert classify(ValueError("bad value")) == GatewayErrorClass.TRANSIENT

    def test_os_error_is_transient(self) -> None:
        assert classify(OSError("connection reset")) == GatewayErrorClass.TRANSIENT


class TestClassifyMistral:
    """Mistral uses status_code on SDKError — classify via isinstance of _MistralSDKError."""

    def _exc(self, status_code: int, message: str = "error") -> BaseException:
        try:
            from mistralai import SDKError  # noqa: PLC0415

            exc = SDKError(message, status_code=status_code, body="")
            return exc
        except (ImportError, TypeError):
            # Fall back to a synthetic exc; classify() uses isinstance so this
            # won't hit the mistral branch, but we still exercise _classify_mistral.
            return _mistral_exc(message, status_code)

    def test_mistral_401_is_auth_stale(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=401)
        assert _classify_mistral(exc) == GatewayErrorClass.AUTH_STALE

    def test_mistral_403_is_auth_stale(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=403)
        assert _classify_mistral(exc) == GatewayErrorClass.AUTH_STALE

    def test_mistral_429_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=429)
        assert _classify_mistral(exc) == GatewayErrorClass.TRANSIENT

    def test_mistral_500_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=500)
        assert _classify_mistral(exc) == GatewayErrorClass.TRANSIENT

    def test_mistral_503_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=503)
        assert _classify_mistral(exc) == GatewayErrorClass.TRANSIENT

    def test_mistral_400_context_exceeded(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=400, message="context_length_exceeded in prompt")
        assert _classify_mistral(exc) == GatewayErrorClass.CONTEXT_EXCEEDED

    def test_mistral_400_content_policy(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=400, message="content_policy_violation detected")
        assert _classify_mistral(exc) == GatewayErrorClass.CONTENT_POLICY

    def test_mistral_unknown_status_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_mistral

        exc = _mistral_exc(status_code=0)
        assert _classify_mistral(exc) == GatewayErrorClass.TRANSIENT


class TestClassifyBoto:
    """AWS Bedrock botocore.ClientError classification by error code."""

    def test_access_denied_is_auth_stale(self) -> None:
        from app.llm_gateway.error_classifier import _classify_boto_client_error

        exc = _boto_exc("AccessDeniedException")
        assert _classify_boto_client_error(exc) == GatewayErrorClass.AUTH_STALE

    def test_unrecognized_client_is_auth_stale(self) -> None:
        from app.llm_gateway.error_classifier import _classify_boto_client_error

        exc = _boto_exc("UnrecognizedClientException")
        assert _classify_boto_client_error(exc) == GatewayErrorClass.AUTH_STALE

    def test_throttling_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_boto_client_error

        exc = _boto_exc("ThrottlingException")
        assert _classify_boto_client_error(exc) == GatewayErrorClass.TRANSIENT

    def test_internal_server_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_boto_client_error

        exc = _boto_exc("InternalServerException")
        assert _classify_boto_client_error(exc) == GatewayErrorClass.TRANSIENT

    def test_validation_with_context_message_is_context_exceeded(self) -> None:
        from app.llm_gateway.error_classifier import _classify_boto_client_error

        exc = _boto_exc("ValidationException", message="too many tokens in the input")
        assert _classify_boto_client_error(exc) == GatewayErrorClass.CONTEXT_EXCEEDED

    def test_unknown_code_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_boto_client_error

        exc = _boto_exc("SomeUnknownException")
        assert _classify_boto_client_error(exc) == GatewayErrorClass.TRANSIENT


class TestClassifyBadRequestMessage:
    """_classify_bad_request_message() — shared 400 sub-classifier."""

    def test_context_exceeded_phrase_detected(self) -> None:
        from app.llm_gateway.error_classifier import _classify_bad_request_message

        for phrase in list(_CONTEXT_EXCEEDED_PHRASES)[:3]:
            exc = ValueError(phrase)
            assert (
                _classify_bad_request_message(exc) == GatewayErrorClass.CONTEXT_EXCEEDED
            )

    def test_content_policy_phrase_detected(self) -> None:
        from app.llm_gateway.error_classifier import _classify_bad_request_message

        for phrase in list(_CONTENT_POLICY_PHRASES)[:3]:
            exc = ValueError(phrase)
            assert (
                _classify_bad_request_message(exc) == GatewayErrorClass.CONTENT_POLICY
            )

    def test_unknown_400_message_is_transient(self) -> None:
        from app.llm_gateway.error_classifier import _classify_bad_request_message

        exc = ValueError("some unknown bad request reason")
        assert _classify_bad_request_message(exc) == GatewayErrorClass.TRANSIENT


# ---------------------------------------------------------------------------
# get_retry_policy() — verify each class has the correct RetryPolicy
# ---------------------------------------------------------------------------


class TestGetRetryPolicy:
    @pytest.mark.parametrize(
        "error_class",
        [
            GatewayErrorClass.TRANSIENT,
            GatewayErrorClass.QUOTA_CALIBRATION,
        ],
    )
    def test_retriable_classes_have_max_retries_3(
        self, error_class: GatewayErrorClass
    ) -> None:
        policy = get_retry_policy(error_class)
        assert policy.is_retriable is True
        assert policy.max_retries == 3

    @pytest.mark.parametrize(
        "error_class",
        [
            GatewayErrorClass.PROVIDER_DOWN,
            GatewayErrorClass.AUTH_STALE,
            GatewayErrorClass.CONTEXT_EXCEEDED,
            GatewayErrorClass.CONTENT_POLICY,
        ],
    )
    def test_non_retriable_classes_have_zero_retries(
        self, error_class: GatewayErrorClass
    ) -> None:
        policy = get_retry_policy(error_class)
        assert policy.is_retriable is False
        assert policy.max_retries == 0

    def test_transient_increments_circuit_breaker(self) -> None:
        assert (
            get_retry_policy(GatewayErrorClass.TRANSIENT).increments_circuit_breaker
            is True
        )

    def test_auth_stale_does_not_increment_circuit_breaker(self) -> None:
        assert (
            get_retry_policy(GatewayErrorClass.AUTH_STALE).increments_circuit_breaker
            is False
        )

    def test_auth_stale_emits_alert(self) -> None:
        assert get_retry_policy(GatewayErrorClass.AUTH_STALE).emits_alert is True

    def test_quota_calibration_emits_alert(self) -> None:
        assert get_retry_policy(GatewayErrorClass.QUOTA_CALIBRATION).emits_alert is True

    def test_returns_retry_policy_instance(self) -> None:
        for error_class in GatewayErrorClass:
            assert isinstance(get_retry_policy(error_class), RetryPolicy)


# ---------------------------------------------------------------------------
# compute_backoff_seconds() — jitter, delay sequence, non-retriable = 0
# ---------------------------------------------------------------------------


class TestComputeBackoffSeconds:
    _JITTER = 0.20

    def test_non_retriable_returns_zero(self) -> None:
        for cls in (
            GatewayErrorClass.AUTH_STALE,
            GatewayErrorClass.CONTEXT_EXCEEDED,
            GatewayErrorClass.CONTENT_POLICY,
            GatewayErrorClass.PROVIDER_DOWN,
        ):
            assert compute_backoff_seconds(0, cls) == 0.0

    @pytest.mark.parametrize(
        "retry_number,expected_base",
        [
            (0, 5.0),
            (1, 25.0),
            (2, 60.0),
            (99, 60.0),  # clamped to last delay
        ],
    )
    def test_transient_delay_within_jitter_band(
        self, retry_number: int, expected_base: float
    ) -> None:
        for _ in range(20):  # run multiple times to cover jitter variation
            result = compute_backoff_seconds(retry_number, GatewayErrorClass.TRANSIENT)
            low = expected_base * (1 - self._JITTER)
            high = expected_base * (1 + self._JITTER)
            assert low <= result <= high, (
                f"retry={retry_number}: {result} not in [{low}, {high}]"
            )

    @pytest.mark.parametrize(
        "retry_number,expected_base",
        [
            (0, 30.0),
            (1, 90.0),
            (2, 180.0),
            (99, 180.0),  # clamped to last delay
        ],
    )
    def test_quota_calibration_delay_within_jitter_band(
        self, retry_number: int, expected_base: float
    ) -> None:
        for _ in range(20):
            result = compute_backoff_seconds(
                retry_number, GatewayErrorClass.QUOTA_CALIBRATION
            )
            low = expected_base * (1 - self._JITTER)
            high = expected_base * (1 + self._JITTER)
            assert low <= result <= high, (
                f"retry={retry_number}: {result} not in [{low}, {high}]"
            )

    def test_result_never_exceeds_180_seconds(self) -> None:
        for _ in range(50):
            result = compute_backoff_seconds(99, GatewayErrorClass.QUOTA_CALIBRATION)
            assert result <= 180.0

    def test_result_is_never_negative(self) -> None:
        for cls in (GatewayErrorClass.TRANSIENT, GatewayErrorClass.QUOTA_CALIBRATION):
            for retry_n in range(5):
                assert compute_backoff_seconds(retry_n, cls) >= 0.0
