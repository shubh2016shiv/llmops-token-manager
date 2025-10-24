"""
Test credential validation for LLM model update operations.

This module tests the validation rules for LLMModelUpdateRequest to ensure
proper credential combinations are enforced during update operations.
"""

import pytest
from pydantic import ValidationError
from app.models.request_models import (
    LLMModelUpdateRequest,
    CloudProvider,
)


class TestLLMModelUpdateCredentialValidation:
    """Test credential validation for LLM model update operations."""

    def test_update_non_credential_fields(self):
        """Test updating non-credential fields (should succeed)."""
        # This should succeed - no credential fields are being updated
        update_request = LLMModelUpdateRequest(
            max_tokens=16384,
            tokens_per_minute_limit=200000,
            is_active_status=True,
            temperature=0.5,
            top_p=0.9,
        )
        # Validation should pass
        assert update_request.max_tokens == 16384
        assert update_request.tokens_per_minute_limit == 200000
        assert update_request.is_active_status is True
        assert update_request.api_endpoint_url is None
        assert update_request.api_key_variable_name is None

    def test_update_both_credential_fields(self):
        """Test updating both credential fields together (should succeed)."""
        # This should succeed - both credential fields are provided
        update_request = LLMModelUpdateRequest(
            api_endpoint_url="https://api.openai.com/v1",
            api_key_variable_name="OPENAI_API_KEY_GPT4O",
            temperature=0.7,
        )
        # Validation should pass
        assert update_request.api_endpoint_url == "https://api.openai.com/v1"
        assert update_request.api_key_variable_name == "OPENAI_API_KEY_GPT4O"
        assert update_request.temperature == 0.7

    def test_update_only_api_endpoint_url(self):
        """Test updating only api_endpoint_url (should fail)."""
        # This should fail - missing api_key_variable_name
        with pytest.raises(ValidationError) as exc_info:
            LLMModelUpdateRequest(
                api_endpoint_url="https://api.openai.com/v1",
                temperature=0.7,
            )
        # Check error message
        error_msg = str(exc_info.value)
        assert "Incomplete credential update" in error_msg
        assert (
            "api_endpoint_url was provided but api_key_variable_name is missing"
            in error_msg
        )

    def test_update_only_api_key_variable_name(self):
        """Test updating only api_key_variable_name (should fail)."""
        # This should fail - missing api_endpoint_url
        with pytest.raises(ValidationError) as exc_info:
            LLMModelUpdateRequest(
                api_key_variable_name="OPENAI_API_KEY_GPT4O",
                temperature=0.7,
            )
        # Check error message
        error_msg = str(exc_info.value)
        assert "Incomplete credential update" in error_msg
        assert (
            "api_key_variable_name was provided but api_endpoint_url is missing"
            in error_msg
        )

    def test_update_all_cloud_fields(self):
        """Test updating all cloud fields together (should succeed)."""
        # This should succeed - all cloud fields are provided
        update_request = LLMModelUpdateRequest(
            cloud_provider=CloudProvider.AZURE,
            deployment_name="gpt-4o-eastus",
            deployment_region="eastus",
            temperature=0.7,
        )
        # Validation should pass
        assert update_request.cloud_provider == CloudProvider.AZURE
        assert update_request.deployment_name == "gpt-4o-eastus"
        assert update_request.deployment_region == "eastus"
        assert update_request.temperature == 0.7

    def test_update_partial_cloud_fields_missing_provider(self):
        """Test updating partial cloud fields - missing provider (should fail)."""
        # This should fail - missing cloud_provider
        with pytest.raises(ValidationError) as exc_info:
            LLMModelUpdateRequest(
                deployment_name="gpt-4o-eastus",
                deployment_region="eastus",
                temperature=0.7,
            )
        # Check error message
        error_msg = str(exc_info.value)
        assert "Incomplete cloud configuration" in error_msg
        assert "cloud_provider" in error_msg

    def test_update_partial_cloud_fields_missing_name(self):
        """Test updating partial cloud fields - missing deployment name (should fail)."""
        # This should fail - missing deployment_name
        with pytest.raises(ValidationError) as exc_info:
            LLMModelUpdateRequest(
                cloud_provider=CloudProvider.AZURE,
                deployment_region="eastus",
                temperature=0.7,
            )
        # Check error message
        error_msg = str(exc_info.value)
        assert "Incomplete cloud configuration" in error_msg
        assert "deployment_name" in error_msg

    def test_update_partial_cloud_fields_missing_region(self):
        """Test updating partial cloud fields - missing region (should fail)."""
        # This should fail - missing deployment_region
        with pytest.raises(ValidationError) as exc_info:
            LLMModelUpdateRequest(
                cloud_provider=CloudProvider.AZURE,
                deployment_name="gpt-4o-eastus",
                temperature=0.7,
            )
        # Check error message
        error_msg = str(exc_info.value)
        assert "Incomplete cloud configuration" in error_msg
        assert "deployment_region" in error_msg

    def test_update_both_credential_and_cloud_fields(self):
        """Test updating both credential and cloud fields together (should succeed)."""
        # This should succeed - both credential and cloud fields are complete
        update_request = LLMModelUpdateRequest(
            api_endpoint_url="https://eastus.api.azure.com/openai",
            api_key_variable_name="AZURE_OPENAI_API_KEY",
            cloud_provider=CloudProvider.AZURE,
            deployment_name="gpt-4o-eastus",
            deployment_region="eastus",
            temperature=0.7,
        )
        # Validation should pass
        assert update_request.api_endpoint_url == "https://eastus.api.azure.com/openai"
        assert update_request.api_key_variable_name == "AZURE_OPENAI_API_KEY"
        assert update_request.cloud_provider == CloudProvider.AZURE
        assert update_request.deployment_name == "gpt-4o-eastus"
        assert update_request.deployment_region == "eastus"
        assert update_request.temperature == 0.7
