"""
Test credential validation logic for LLM models and user entitlements.

This module tests the validation rules for LLM model and user entitlement
request models to ensure proper credential combinations are enforced.
"""

import pytest
from pydantic import ValidationError
from app.models.request_models import (
    LLMModelCreateRequest,
    UserEntitlementCreateRequest,
    LLMProvider,
    CloudProvider,
)


class TestLLMModelCredentialValidation:
    """Test credential validation for LLM model creation requests."""

    def test_direct_llm_with_complete_credentials(self):
        """Test direct LLM configuration with all core fields."""
        # This should succeed
        model = LLMModelCreateRequest(
            llm_provider=LLMProvider.OPENAI,
            llm_model_name="gpt-4o",
            api_endpoint_url="https://api.openai.com/v1",
            api_key_variable_name="OPENAI_API_KEY_GPT4O",
            temperature=0.7,
            top_p=1.0,
        )
        # Validation should pass
        assert model.llm_provider == LLMProvider.OPENAI
        assert model.llm_model_name == "gpt-4o"
        assert model.api_endpoint_url == "https://api.openai.com/v1"
        assert model.api_key_variable_name == "OPENAI_API_KEY_GPT4O"

    def test_direct_llm_with_missing_core_fields(self):
        """Test direct LLM configuration with missing core fields."""
        # Missing api_endpoint_url
        with pytest.raises(ValidationError) as exc_info:
            LLMModelCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_key_variable_name="OPENAI_API_KEY_GPT4O",
            )
        assert "api_endpoint_url" in str(exc_info.value)

        # Missing api_key_variable_name
        with pytest.raises(ValidationError) as exc_info:
            LLMModelCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://api.openai.com/v1",
            )
        assert "api_key_variable_name" in str(exc_info.value)

    def test_cloud_llm_with_complete_credentials(self):
        """Test cloud LLM configuration with all fields."""
        # This should succeed
        model = LLMModelCreateRequest(
            llm_provider=LLMProvider.OPENAI,
            llm_model_name="gpt-4o",
            api_endpoint_url="https://eastus.api.azure.com/openai",
            api_key_variable_name="AZURE_OPENAI_API_KEY",
            cloud_provider=CloudProvider.AZURE,
            deployment_name="gpt-4o-eastus",
            deployment_region="eastus",
        )
        # Validation should pass
        assert model.cloud_provider == CloudProvider.AZURE
        assert model.deployment_name == "gpt-4o-eastus"
        assert model.deployment_region == "eastus"

    def test_cloud_llm_with_partial_cloud_fields(self):
        """Test cloud LLM configuration with partial cloud fields."""
        # Missing deployment_name
        with pytest.raises(ValidationError) as exc_info:
            LLMModelCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://eastus.api.azure.com/openai",
                api_key_variable_name="AZURE_OPENAI_API_KEY",
                cloud_provider=CloudProvider.AZURE,
                deployment_region="eastus",
            )
        assert "deployment_name" in str(exc_info.value)

        # Missing cloud_provider
        with pytest.raises(ValidationError) as exc_info:
            LLMModelCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://eastus.api.azure.com/openai",
                api_key_variable_name="AZURE_OPENAI_API_KEY",
                deployment_name="gpt-4o-eastus",
                deployment_region="eastus",
            )
        assert "cloud_provider" in str(exc_info.value)

        # Missing deployment_region
        with pytest.raises(ValidationError) as exc_info:
            LLMModelCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://eastus.api.azure.com/openai",
                api_key_variable_name="AZURE_OPENAI_API_KEY",
                cloud_provider=CloudProvider.AZURE,
                deployment_name="gpt-4o-eastus",
            )
        assert "deployment_region" in str(exc_info.value)


class TestUserEntitlementCredentialValidation:
    """Test credential validation for user entitlement creation requests."""

    def test_direct_entitlement_with_complete_credentials(self):
        """Test direct LLM entitlement with all core fields."""
        # This should succeed
        entitlement = UserEntitlementCreateRequest(
            llm_provider=LLMProvider.OPENAI,
            llm_model_name="gpt-4o",
            api_endpoint_url="https://api.openai.com/v1",
            api_key_value="sk-1234567890abcdefgh",
        )
        # Validation should pass
        assert entitlement.llm_provider == LLMProvider.OPENAI
        assert entitlement.llm_model_name == "gpt-4o"
        assert entitlement.api_endpoint_url == "https://api.openai.com/v1"
        assert entitlement.api_key_value == "sk-1234567890abcdefgh"

    def test_direct_entitlement_with_missing_core_fields(self):
        """Test direct LLM entitlement with missing core fields."""
        # Missing api_endpoint_url
        with pytest.raises(ValidationError) as exc_info:
            UserEntitlementCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_key_value="sk-1234567890abcdefgh",
            )
        assert "api_endpoint_url" in str(exc_info.value)

        # Missing api_key_value
        with pytest.raises(ValidationError) as exc_info:
            UserEntitlementCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://api.openai.com/v1",
            )
        assert "api_key_value" in str(exc_info.value)

    def test_cloud_entitlement_with_complete_credentials(self):
        """Test cloud LLM entitlement with all fields."""
        # This should succeed
        entitlement = UserEntitlementCreateRequest(
            llm_provider=LLMProvider.OPENAI,
            llm_model_name="gpt-4o",
            api_endpoint_url="https://eastus.api.azure.com/openai",
            api_key_value="sk-1234567890abcdefgh",
            cloud_provider=CloudProvider.AZURE,
            deployment_name="gpt-4o-eastus",
            deployment_region="eastus",
        )
        # Validation should pass
        assert entitlement.cloud_provider == CloudProvider.AZURE
        assert entitlement.deployment_name == "gpt-4o-eastus"
        assert entitlement.deployment_region == "eastus"

    def test_cloud_entitlement_with_partial_cloud_fields(self):
        """Test cloud LLM entitlement with partial cloud fields."""
        # Missing deployment_name
        with pytest.raises(ValidationError) as exc_info:
            UserEntitlementCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://eastus.api.azure.com/openai",
                api_key_value="sk-1234567890abcdefgh",
                cloud_provider=CloudProvider.AZURE,
                deployment_region="eastus",
            )
        assert "deployment_name" in str(exc_info.value)

        # Missing cloud_provider
        with pytest.raises(ValidationError) as exc_info:
            UserEntitlementCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://eastus.api.azure.com/openai",
                api_key_value="sk-1234567890abcdefgh",
                deployment_name="gpt-4o-eastus",
                deployment_region="eastus",
            )
        assert "cloud_provider" in str(exc_info.value)

        # Missing deployment_region
        with pytest.raises(ValidationError) as exc_info:
            UserEntitlementCreateRequest(
                llm_provider=LLMProvider.OPENAI,
                llm_model_name="gpt-4o",
                api_endpoint_url="https://eastus.api.azure.com/openai",
                api_key_value="sk-1234567890abcdefgh",
                cloud_provider=CloudProvider.AZURE,
                deployment_name="gpt-4o-eastus",
            )
        assert "deployment_region" in str(exc_info.value)
