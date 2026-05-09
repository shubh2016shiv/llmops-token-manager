from __future__ import annotations

from unittest.mock import AsyncMock

from pydantic import ValidationError
import pytest

from app.models.request_models import LLMModelCreateRequest
from app.persistence.llm_models import LLMModelPersistence


def test_create_request_requires_max_tokens_for_active_models() -> None:
    with pytest.raises(ValidationError, match="max_tokens is required"):
        LLMModelCreateRequest(
            llm_provider="openai",
            llm_model_name="gpt-4o",
            api_key_variable_name="OPENAI_API_KEY_GPT4O",
            api_endpoint_url="https://api.openai.com/v1",
            is_active_status=True,
        )


def test_create_request_allows_missing_max_tokens_for_inactive_models() -> None:
    request = LLMModelCreateRequest(
        llm_provider="openai",
        llm_model_name="gpt-4o",
        api_key_variable_name="OPENAI_API_KEY_GPT4O",
        api_endpoint_url="https://api.openai.com/v1",
        is_active_status=False,
    )

    assert request.max_tokens is None
    assert request.is_active_status is False


@pytest.mark.asyncio
async def test_persistence_create_rejects_active_models_without_capacity() -> None:
    service = LLMModelPersistence(AsyncMock())

    with pytest.raises(ValueError, match="max_tokens is required"):
        await service.create_llm_model(
            llm_provider="openai",
            llm_model_name="gpt-4o",
            api_key_variable_name="OPENAI_API_KEY_GPT4O",
            api_endpoint_url="https://api.openai.com/v1",
            is_active_status=True,
        )


@pytest.mark.asyncio
async def test_persistence_update_rejects_activation_without_capacity(
    monkeypatch,
) -> None:
    service = LLMModelPersistence(AsyncMock())
    monkeypatch.setattr(
        service,
        "get_llm_model_by_provider_and_model",
        AsyncMock(
            return_value={
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "llm_model_version": None,
                "max_tokens": None,
                "is_active_status": False,
            }
        ),
    )

    with pytest.raises(ValueError, match="max_tokens is required"):
        await service.update_llm_model(
            llm_provider="openai",
            llm_model_name="gpt-4o",
            is_active_status=True,
        )
