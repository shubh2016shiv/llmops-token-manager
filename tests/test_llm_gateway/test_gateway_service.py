"""
Unit tests for GatewayService.

All tests use injected fakes — no database, no Fernet key, no real provider calls.
Each concern is tested independently so failures point to a single layer.

Test structure mirrors the source:
    tests/test_llm_gateway/test_gateway_service.py
    → app/llm_gateway/gateway_service.py

Coverage targets:
    - Happy path: result returned from provider, correct fields forwarded
    - Resolver errors propagate without wrapping
    - Provider errors propagate without wrapping
    - provider_factory receives the config the resolver produced
    - ProviderExecutionRequest fields are forwarded verbatim to provider
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.llm_gateway.credential_resolver.exceptions import (
    EntitlementNotFoundError,
    ModelNotActiveError,
)
from app.llm_gateway.gateway_service import GatewayService
from app.llm_gateway.providers.provider_types import (
    LLMProviderType,
    ProviderExecutionRequest,
    ProviderExecutionResult,
    ProviderInitializationConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures and fakes
# ---------------------------------------------------------------------------

_SAMPLE_USER_ID: UUID = uuid4()
_SAMPLE_PROVIDER = "openai"
_SAMPLE_MODEL = "gpt-4o"


def _make_config(
    llm_provider: str = _SAMPLE_PROVIDER,
    llm_model_name: str = _SAMPLE_MODEL,
) -> ProviderInitializationConfig:
    """Build a minimal valid ProviderInitializationConfig for direct provider."""
    return ProviderInitializationConfig(
        llm_provider=LLMProviderType(llm_provider),
        api_key="sk-test-key",
        api_endpoint_url="https://api.openai.com/v1",
        llm_model_name=llm_model_name,
    )


def _make_result(
    completion_text: str = "Test completion",
    total_tokens: int = 100,
) -> ProviderExecutionResult:
    """Build a minimal ProviderExecutionResult for assertions."""
    return ProviderExecutionResult(
        completion_text=completion_text,
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=total_tokens,
        llm_provider=_SAMPLE_PROVIDER,
        model_name_reported_by_provider=_SAMPLE_MODEL,
        latency_milliseconds=250.0,
    )


def _make_request(user_prompt: str = "Hello, world!") -> ProviderExecutionRequest:
    """Build a minimal ProviderExecutionRequest."""
    return ProviderExecutionRequest(user_prompt=user_prompt)


class FakeResolver:
    """Test double that returns a configurable config without touching the DB."""

    def __init__(self, config: ProviderInitializationConfig) -> None:
        self.config = config
        self.calls: list[tuple[UUID, str, str]] = []

    async def resolve(
        self,
        user_id: UUID,
        llm_provider: str,
        llm_model_name: str,
    ) -> ProviderInitializationConfig:
        self.calls.append((user_id, llm_provider, llm_model_name))
        return self.config


class FailingResolver:
    """Test double that always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def resolve(
        self, user_id: UUID, llm_provider: str, llm_model_name: str
    ) -> None:
        raise self._exc


class FakeProvider:
    """Test double that returns a configurable result without hitting a vendor API."""

    def __init__(self, result: ProviderExecutionResult) -> None:
        self.result = result
        self.execute_calls: list[ProviderExecutionRequest] = []

    async def execute(
        self, request: ProviderExecutionRequest
    ) -> ProviderExecutionResult:
        self.execute_calls.append(request)
        return self.result


class FailingProvider:
    """Test double that always raises RuntimeError on execute."""

    async def execute(self, request: ProviderExecutionRequest) -> None:
        raise RuntimeError("Provider API unavailable")


# ---------------------------------------------------------------------------
# GatewayService tests
# ---------------------------------------------------------------------------


class TestGatewayService:
    """Unit tests for GatewayService — all deps faked, no I/O."""

    def _build_service(
        self,
        config: ProviderInitializationConfig | None = None,
        result: ProviderExecutionResult | None = None,
    ) -> tuple[GatewayService, FakeResolver, FakeProvider]:
        """Build a service wired with configurable fakes; return all three."""
        resolved_config = config or _make_config()
        resolved_result = result or _make_result()
        fake_resolver = FakeResolver(resolved_config)
        fake_provider = FakeProvider(resolved_result)
        service = GatewayService(
            resolver=fake_resolver,  # type: ignore[arg-type]
            provider_factory=lambda _cfg: fake_provider,  # type: ignore[return-value]
        )
        return service, fake_resolver, fake_provider

    @pytest.mark.asyncio
    async def test_execute_returns_provider_result(self) -> None:
        expected = _make_result(completion_text="Paris is the capital of France.")
        service, _, _ = self._build_service(result=expected)

        result = await service.execute(
            user_id=_SAMPLE_USER_ID,
            llm_provider=_SAMPLE_PROVIDER,
            llm_model_name=_SAMPLE_MODEL,
            request=_make_request(),
        )

        assert result is expected

    @pytest.mark.asyncio
    async def test_execute_passes_correct_args_to_resolver(self) -> None:
        service, fake_resolver, _ = self._build_service()

        await service.execute(
            user_id=_SAMPLE_USER_ID,
            llm_provider=_SAMPLE_PROVIDER,
            llm_model_name=_SAMPLE_MODEL,
            request=_make_request(),
        )

        assert len(fake_resolver.calls) == 1
        called_user_id, called_provider, called_model = fake_resolver.calls[0]
        assert called_user_id == _SAMPLE_USER_ID
        assert called_provider == _SAMPLE_PROVIDER
        assert called_model == _SAMPLE_MODEL

    @pytest.mark.asyncio
    async def test_execute_forwards_request_to_provider(self) -> None:
        service, _, fake_provider = self._build_service()
        request = _make_request(user_prompt="Explain entropy in one sentence.")

        await service.execute(
            user_id=_SAMPLE_USER_ID,
            llm_provider=_SAMPLE_PROVIDER,
            llm_model_name=_SAMPLE_MODEL,
            request=request,
        )

        assert len(fake_provider.execute_calls) == 1
        assert fake_provider.execute_calls[0] is request

    @pytest.mark.asyncio
    async def test_provider_factory_receives_resolved_config(self) -> None:
        config = _make_config(llm_model_name="gpt-4o-mini")
        captured_configs: list[ProviderInitializationConfig] = []
        fake_provider = FakeProvider(_make_result())

        service = GatewayService(
            resolver=FakeResolver(config),  # type: ignore[arg-type]
            provider_factory=lambda cfg: (captured_configs.append(cfg), fake_provider)[
                1
            ],  # type: ignore[return-value]
        )

        await service.execute(
            user_id=_SAMPLE_USER_ID,
            llm_provider=_SAMPLE_PROVIDER,
            llm_model_name="gpt-4o-mini",
            request=_make_request(),
        )

        assert len(captured_configs) == 1
        assert captured_configs[0] is config

    @pytest.mark.asyncio
    async def test_execute_propagates_entitlement_not_found_error(self) -> None:
        exc = EntitlementNotFoundError(
            str(_SAMPLE_USER_ID), _SAMPLE_PROVIDER, _SAMPLE_MODEL
        )
        service = GatewayService(
            resolver=FailingResolver(exc),  # type: ignore[arg-type]
            provider_factory=lambda _: FakeProvider(_make_result()),  # type: ignore[return-value]
        )

        with pytest.raises(EntitlementNotFoundError):
            await service.execute(
                _SAMPLE_USER_ID, _SAMPLE_PROVIDER, _SAMPLE_MODEL, _make_request()
            )

    @pytest.mark.asyncio
    async def test_execute_propagates_model_not_active_error(self) -> None:
        exc = ModelNotActiveError(_SAMPLE_PROVIDER, _SAMPLE_MODEL)
        service = GatewayService(
            resolver=FailingResolver(exc),  # type: ignore[arg-type]
            provider_factory=lambda _: FakeProvider(_make_result()),  # type: ignore[return-value]
        )

        with pytest.raises(ModelNotActiveError):
            await service.execute(
                _SAMPLE_USER_ID, _SAMPLE_PROVIDER, _SAMPLE_MODEL, _make_request()
            )

    @pytest.mark.asyncio
    async def test_execute_propagates_provider_runtime_error(self) -> None:
        config = _make_config()
        service = GatewayService(
            resolver=FakeResolver(config),  # type: ignore[arg-type]
            provider_factory=lambda _: FailingProvider(),  # type: ignore[return-value]
        )

        with pytest.raises(RuntimeError, match="Provider API unavailable"):
            await service.execute(
                _SAMPLE_USER_ID, _SAMPLE_PROVIDER, _SAMPLE_MODEL, _make_request()
            )

    @pytest.mark.asyncio
    async def test_execute_calls_resolver_exactly_once_per_call(self) -> None:
        service, fake_resolver, _ = self._build_service()

        await service.execute(
            _SAMPLE_USER_ID, _SAMPLE_PROVIDER, _SAMPLE_MODEL, _make_request()
        )
        await service.execute(
            _SAMPLE_USER_ID, _SAMPLE_PROVIDER, _SAMPLE_MODEL, _make_request()
        )

        assert len(fake_resolver.calls) == 2

    @pytest.mark.asyncio
    async def test_execute_calls_provider_exactly_once_per_call(self) -> None:
        service, _, fake_provider = self._build_service()

        await service.execute(
            _SAMPLE_USER_ID, _SAMPLE_PROVIDER, _SAMPLE_MODEL, _make_request()
        )

        assert len(fake_provider.execute_calls) == 1

    @pytest.mark.parametrize(
        "provider", ["openai", "anthropic", "gemini", "meta", "mistral"]
    )
    @pytest.mark.asyncio
    async def test_execute_routes_all_five_providers(self, provider: str) -> None:
        config = _make_config(llm_provider=provider)
        service, fake_resolver, _ = self._build_service(config=config)

        await service.execute(_SAMPLE_USER_ID, provider, _SAMPLE_MODEL, _make_request())

        assert fake_resolver.calls[0][1] == provider
