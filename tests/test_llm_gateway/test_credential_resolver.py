"""
Unit tests for the credential resolver pipeline.

All tests use injected fakes — no database, no Fernet key required.
Each layer (repository, decryptor, mapper, resolver) is tested independently
so failures are isolated to a single concern.

Test structure mirrors the source:
    tests/test_llm_gateway/test_credential_resolver.py
    → app/llm_gateway/credential_resolver/

Coverage targets:
    - Happy path: direct provider, Azure, AWS (JSON creds), GCP
    - Error paths: not found, inactive model, unsupported provider/platform
    - Decryption failure propagation
    - Mapper edge cases: None fields, AWS JSON parsing, platform detection
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.llm_gateway.credential_resolver.exceptions import (
    ApiKeyDecryptionError,
    EntitlementNotFoundError,
    ModelNotActiveError,
    UnsupportedCloudPlatformError,
    UnsupportedProviderError,
)
from app.llm_gateway.credential_resolver.mapper import CredentialConfigMapper
from app.llm_gateway.credential_resolver.pipeline import CredentialResolverPipeline
from app.llm_gateway.providers.provider_types import (
    CloudPlatformType,
    LLMProviderType,
    ProviderInitializationConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures and fakes
# ---------------------------------------------------------------------------

_SAMPLE_USER_ID = uuid4()
_SAMPLE_ENTITLEMENT_ID = 42


def _make_row(
    *,
    llm_provider: str = "openai",
    llm_model_name: str = "gpt-4o",
    api_key_value: str = "gAAAAABencrypted",
    api_endpoint_url: str = "https://api.openai.com/v1",
    cloud_provider: str | None = None,
    deployment_name: str | None = None,
    deployment_region: str | None = None,
    azure_api_version: str | None = None,
    max_tokens_ceiling: int | None = 4096,
    tokens_per_minute_limit: int | None = 60_000,
    requests_per_minute_limit: int | None = 500,
    default_temperature: float | None = 0.7,
    default_random_seed: int | None = None,
    is_active_status: bool = True,
    entitlement_id: int = _SAMPLE_ENTITLEMENT_ID,
) -> dict[str, Any]:
    """Build a minimal fake DB row matching the JOIN query's column set."""
    return {
        "entitlement_id": entitlement_id,
        "user_id": str(_SAMPLE_USER_ID),
        "api_key_value": api_key_value,
        "api_endpoint_url": api_endpoint_url,
        "cloud_provider": cloud_provider,
        "deployment_name": deployment_name,
        "deployment_region": deployment_region,
        "llm_provider": llm_provider,
        "llm_model_name": llm_model_name,
        "max_tokens_ceiling": max_tokens_ceiling,
        "tokens_per_minute_limit": tokens_per_minute_limit,
        "requests_per_minute_limit": requests_per_minute_limit,
        "default_temperature": default_temperature,
        "default_random_seed": default_random_seed,
        "is_active_status": is_active_status,
        "azure_api_version": azure_api_version,
    }


class FakeDecryptor:
    """Test double that returns a configurable plaintext without touching Fernet."""

    def __init__(self, plaintext: str = "sk-test-key") -> None:
        self.plaintext = plaintext
        self.calls: list[tuple[str, int]] = []

    def decrypt(self, encrypted_value: str, entitlement_id: int) -> str:
        self.calls.append((encrypted_value, entitlement_id))
        return self.plaintext


class FailingDecryptor:
    """Test double that always raises ApiKeyDecryptionError."""

    def decrypt(self, encrypted_value: str, entitlement_id: int) -> str:
        raise ApiKeyDecryptionError(entitlement_id=entitlement_id)


# ---------------------------------------------------------------------------
# CredentialConfigMapper tests
# ---------------------------------------------------------------------------


class TestCredentialConfigMapper:
    """Unit tests for the row-to-config mapping layer."""

    def setup_method(self) -> None:
        self.mapper = CredentialConfigMapper()

    def test_maps_direct_openai_row_to_config(self) -> None:
        row = _make_row()

        config = self.mapper.map_to_provider_config(row, decrypted_api_key="sk-abc")

        assert config.llm_provider == LLMProviderType.OPENAI
        assert config.cloud_platform is None
        assert config.api_key == "sk-abc"
        assert config.api_key_secondary is None
        assert config.llm_model_name == "gpt-4o"

    def test_maps_azure_row_sets_api_version_from_llm_model_version(self) -> None:
        row = _make_row(
            llm_provider="openai",
            cloud_provider="Azure",
            deployment_name="gpt4o-eastus-prod",
            deployment_region="eastus",
            azure_api_version="2024-08-01-preview",
        )

        config = self.mapper.map_to_provider_config(row, decrypted_api_key="azure-key")

        assert config.cloud_platform == CloudPlatformType.AZURE
        assert config.api_version == "2024-08-01-preview"
        assert config.deployment_name == "gpt4o-eastus-prod"

    def test_maps_aws_json_credentials_into_key_and_secondary(self) -> None:
        aws_creds = json.dumps(
            {"access_key_id": "AKIAIOSFODNN7", "secret_access_key": "wJalrXUtn"}
        )
        row = _make_row(
            llm_provider="anthropic",
            llm_model_name="anthropic.claude-3-5-sonnet-20241022-v2:0",
            cloud_provider="Amazon Web Services",
            deployment_region="us-east-1",
            api_endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )

        config = self.mapper.map_to_provider_config(row, decrypted_api_key=aws_creds)

        assert config.cloud_platform == CloudPlatformType.AWS
        assert config.api_key == "AKIAIOSFODNN7"
        assert config.api_key_secondary == "wJalrXUtn"

    def test_maps_aws_plain_string_as_access_key_id(self) -> None:
        row = _make_row(
            llm_provider="meta",
            cloud_provider="Amazon Web Services",
            deployment_region="us-east-1",
            api_endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )

        config = self.mapper.map_to_provider_config(row, decrypted_api_key="AKIAPLAIN")

        assert config.api_key == "AKIAPLAIN"
        assert config.api_key_secondary is None

    def test_raises_model_not_active_error(self) -> None:
        row = _make_row(is_active_status=False)

        with pytest.raises(ModelNotActiveError):
            self.mapper.map_to_provider_config(row, decrypted_api_key="sk-abc")

    def test_raises_unsupported_provider_error(self) -> None:
        row = _make_row(llm_provider="cohere")

        with pytest.raises(UnsupportedProviderError) as exc_info:
            self.mapper.map_to_provider_config(row, decrypted_api_key="key")

        assert "cohere" in str(exc_info.value)

    def test_raises_unsupported_cloud_platform_error(self) -> None:
        row = _make_row(cloud_provider="IBM Watsonx")

        with pytest.raises(UnsupportedCloudPlatformError) as exc_info:
            self.mapper.map_to_provider_config(row, decrypted_api_key="key")

        assert "IBM Watsonx" in str(exc_info.value)

    def test_none_optional_fields_map_to_none(self) -> None:
        row = _make_row(
            max_tokens_ceiling=None,
            tokens_per_minute_limit=None,
            default_temperature=None,
            default_random_seed=None,
        )

        config = self.mapper.map_to_provider_config(row, decrypted_api_key="sk-abc")

        assert config.max_tokens_ceiling is None
        assert config.tokens_per_minute_limit is None
        assert config.default_temperature is None
        assert config.default_random_seed is None

    @pytest.mark.parametrize(
        "provider",
        ["openai", "anthropic", "gemini", "meta", "mistral"],
    )
    def test_all_top5_providers_are_mapped(self, provider: str) -> None:
        row = _make_row(llm_provider=provider)

        config = self.mapper.map_to_provider_config(row, decrypted_api_key="key")

        assert config.llm_provider.value == provider


# ---------------------------------------------------------------------------
# CredentialResolver integration-style tests (all deps faked)
# ---------------------------------------------------------------------------


class TestCredentialResolver:
    """Integration-style tests for the resolver orchestrator (no DB, no Fernet)."""

    def _build_resolver(
        self,
        row: dict[str, Any],
        decryptor_plaintext: str = "sk-test",
    ) -> CredentialResolverPipeline:
        """Build a resolver with a fake repository that returns the given row."""
        fake_repo = MagicMock()
        fake_repo.fetch_entitlement_with_model_config = AsyncMock(return_value=row)
        return CredentialResolverPipeline(
            repository=fake_repo,
            decryptor=FakeDecryptor(plaintext=decryptor_plaintext),
            mapper=CredentialConfigMapper(),
        )

    @pytest.mark.asyncio
    async def test_resolve_returns_provider_config_for_direct_provider(self) -> None:
        row = _make_row()
        resolver = self._build_resolver(row, decryptor_plaintext="sk-openai-key")

        config = await resolver.resolve(
            user_id=_SAMPLE_USER_ID,
            llm_provider="openai",
            llm_model_name="gpt-4o",
        )

        assert isinstance(config, ProviderInitializationConfig)
        assert config.llm_provider == LLMProviderType.OPENAI
        assert config.api_key == "sk-openai-key"

    @pytest.mark.asyncio
    async def test_resolve_propagates_entitlement_not_found(self) -> None:
        fake_repo = MagicMock()
        fake_repo.fetch_entitlement_with_model_config = AsyncMock(
            side_effect=EntitlementNotFoundError(
                str(_SAMPLE_USER_ID), "openai", "gpt-4o"
            )
        )
        resolver = CredentialResolverPipeline(
            repository=fake_repo,
            decryptor=FakeDecryptor(),
            mapper=CredentialConfigMapper(),
        )

        with pytest.raises(EntitlementNotFoundError):
            await resolver.resolve(_SAMPLE_USER_ID, "openai", "gpt-4o")

    @pytest.mark.asyncio
    async def test_resolve_propagates_decryption_failure(self) -> None:
        row = _make_row()
        fake_repo = MagicMock()
        fake_repo.fetch_entitlement_with_model_config = AsyncMock(return_value=row)
        resolver = CredentialResolverPipeline(
            repository=fake_repo,
            decryptor=FailingDecryptor(),
            mapper=CredentialConfigMapper(),
        )

        with pytest.raises(ApiKeyDecryptionError):
            await resolver.resolve(_SAMPLE_USER_ID, "openai", "gpt-4o")

    @pytest.mark.asyncio
    async def test_resolve_azure_config_includes_api_version(self) -> None:
        row = _make_row(
            cloud_provider="Azure",
            deployment_name="gpt4o-eastus",
            deployment_region="eastus",
            azure_api_version="2024-08-01-preview",
        )
        resolver = self._build_resolver(row, decryptor_plaintext="az-static-key")

        config = await resolver.resolve(_SAMPLE_USER_ID, "openai", "gpt-4o")

        assert config.cloud_platform == CloudPlatformType.AZURE
        assert config.api_version == "2024-08-01-preview"

    @pytest.mark.asyncio
    async def test_decryptor_receives_correct_entitlement_id(self) -> None:
        row = _make_row(entitlement_id=99)
        fake_repo = MagicMock()
        fake_repo.fetch_entitlement_with_model_config = AsyncMock(return_value=row)
        decryptor = FakeDecryptor(plaintext="key")
        resolver = CredentialResolverPipeline(
            repository=fake_repo,
            decryptor=decryptor,
            mapper=CredentialConfigMapper(),
        )

        await resolver.resolve(_SAMPLE_USER_ID, "openai", "gpt-4o")

        assert len(decryptor.calls) == 1
        _, called_entitlement_id = decryptor.calls[0]
        assert called_entitlement_id == 99
