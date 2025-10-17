from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class LLMModel(BaseModel):
    """
    Pydantic model for the 'llm_models' table.

    Catalog of LLM models with configurations and usage metrics.
    Maps to: llm_models_config_schema.sql

    IMPORTANT: Field names with 'model_' prefix have been renamed to 'llm_*' to avoid
    conflicts with Pydantic's protected namespaces. These fields map to the database
    columns with their original names.
    """

    # Renamed from model_id to llm_id to avoid Pydantic namespace conflict
    llm_id: Optional[UUID] = Field(
        default=None,
        description="Unique identifier for the LLM model (auto-generated)",
        alias="model_id",  # Maps to database column 'model_id'
    )
    provider: str = Field(
        default="openai", description="LLM provider: openai, gemini, or anthropic"
    )
    # Renamed from model_name to llm_name
    llm_name: str = Field(
        ...,
        description="Name of the LLM model (e.g., GPT-4)",
        alias="model_name",  # Maps to database column 'model_name'
    )
    deployment_name: Optional[str] = Field(
        default=None, description="Name of the LLM deployment (e.g., gpt-4o)"
    )
    api_key_vault_id: Optional[str] = Field(
        default=None, description="Reference to the API key vault entry"
    )
    api_endpoint: Optional[str] = Field(
        default=None, description="API endpoint for the selected LLM instance"
    )
    # Renamed from model_version to llm_version
    llm_version: Optional[str] = Field(
        default=None,
        description="Specific version of the model",
        alias="model_version",  # Maps to database column 'model_version'
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens the model can process in a single request",
    )
    tokens_per_minute_limit: Optional[int] = Field(
        default=None, description="Token rate limit per minute"
    )
    requests_per_minute_limit: Optional[int] = Field(
        default=None, description="Request rate limit per minute"
    )
    is_active: bool = Field(
        default=True, description="Indicates if the model is available for use"
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature setting for the LLM model"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed value for reproducible LLM outputs"
    )
    region: Optional[str] = Field(
        default=None,
        description="Geographic region of the LLM instance (e.g., eastus2, westus2)",
    )
    total_requests: int = Field(
        default=0, description="Total number of requests processed by this model"
    )
    total_tokens_processed: int = Field(
        default=0, description="Total number of tokens processed by this model"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the model was added"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Timestamp of the last update to model configuration"
    )
    last_used_at: Optional[datetime] = Field(
        default=None, description="Timestamp of the last time this model was used"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider matches database CHECK constraint."""
        allowed_providers = ["openai", "gemini", "anthropic"]
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {', '.join(allowed_providers)}")
        return v

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "llm_name": "gpt-4",  # Updated field name
                "deployment_name": "gpt-4-turbo",
                "max_tokens": 8192,
                "is_active": True,
            }
        }
