from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


########################################################
# Token Estimation Model
########################################################
class InputType(Enum):
    """Types of inputs supported for token estimation."""

    SIMPLE_STRING = "simple_string"
    CHAT_MESSAGES = "chat_messages"
    UNKNOWN = "unknown"


class TokenEstimation(BaseModel):
    """Result of token estimation with validation."""

    input_type: InputType = Field(..., description="Type of input being processed")
    model: str = Field(..., min_length=1, description="Name of the LLM model")
    total_tokens: int = Field(
        ..., ge=0, description="Total number of tokens in the input"
    )
    text_tokens: int = Field(
        ..., ge=0, description="Number of tokens from text content"
    )
    image_tokens: int = Field(
        ..., ge=0, description="Number of tokens from image content"
    )
    tool_tokens: int = Field(..., ge=0, description="Number of tokens from tool calls")
    message_count: int = Field(..., ge=0, description="Number of messages in the input")
    image_count: int = Field(..., ge=0, description="Number of images in the input")
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Processing time in milliseconds"
    )

    @model_validator(mode="after")
    def validate_total_tokens(self) -> "TokenEstimation":
        """Validate that total tokens equals sum of component tokens."""
        text = self.text_tokens
        image = self.image_tokens
        tool = self.tool_tokens
        if self.total_tokens != text + image + tool:
            raise ValueError(
                "Total tokens must equal sum of text, image and tool tokens"
            )
        return self

    def __str__(self) -> str:
        """Pretty print estimation result."""
        lines = [
            "Token Estimation Result:",
            f" Input Type: {self.input_type.value}",
            f" Model: {self.model}",
            f" Total Tokens: {self.total_tokens}",
        ]
        if self.text_tokens > 0:
            lines.append(f" Text Tokens: {self.text_tokens}")
        if self.image_tokens > 0:
            lines.append(
                f" Image Tokens: {self.image_tokens} ({self.image_count} images)"
            )
        if self.tool_tokens > 0:
            lines.append(f" Tool Tokens: {self.tool_tokens}")
        if self.message_count > 1:
            lines.append(f" Messages: {self.message_count}")
        lines.append(f" Processing Time: {self.processing_time_ms:.2f}ms")
        return "\n".join(lines)


########################################################
# Token Allocation Model
########################################################
class TokenAllocation(BaseModel):
    """
    Pydantic model for the 'token_manager' table.

    Central gateway for LLM token allocation requests, ensuring fair usage,
    cost control, and deployment resilience.
    Maps to: token_manager_schema.sql

    Note: This table is named 'token_manager' in the database but represents
    token allocations, so the model is named TokenAllocation for clarity.

    IMPORTANT: Field names with 'model_' prefix have been renamed to 'llm_*' to avoid
    conflicts with Pydantic's protected namespaces. These fields map to the database
    columns with their original names.
    """

    token_request_id: str = Field(
        ..., description="Unique identifier for the token allocation request"
    )
    user_id: UUID = Field(..., description="Reference to the user requesting tokens")
    # Renamed from model_name to llm_name
    llm_model_name: str = Field(
        ...,
        description="Name of the LLM model (e.g., GPT-4)",
        alias="model_name",  # Maps to database column 'model_name'
    )
    # Renamed from model_id to llm_id
    llm_id: Optional[UUID] = Field(
        default=None,
        description="Reference to the specific LLM model in llm_models table",
        alias="model_id",  # Maps to database column 'model_id'
    )
    deployment_name: Optional[str] = Field(
        default=None, description="Specific deployment of the model, if applicable"
    )
    cloud_provider: Optional[str] = Field(
        default=None, description="Cloud provider hosting the LLM (e.g., Azure, AWS)"
    )
    api_endpoint: Optional[str] = Field(
        default=None, description="API endpoint for the selected LLM instance"
    )
    region: Optional[str] = Field(
        default=None,
        description="Geographic region of the LLM instance (e.g., eastus2, westus2)",
    )
    token_count: int = Field(
        ..., description="Number of tokens allocated for this request", gt=0
    )
    allocation_status: str = Field(
        default="ACQUIRED",
        description="Current status: ACQUIRED, RELEASED, EXPIRED, PAUSED, or FAILED",
    )
    allocated_at: datetime = Field(
        ..., description="Timestamp when tokens were allocated"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Optional expiration time for the allocation lock"
    )
    request_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context data in JSON format (e.g., team, application)",
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature setting for this specific request"
    )
    top_p: Optional[float] = Field(
        default=None, description="Top P (nucleus sampling) parameter for this request"
    )
    seed: Optional[float] = Field(
        default=None, description="Seed value for reproducible LLM outputs"
    )

    @field_validator("allocation_status")
    @classmethod
    def validate_allocation_status(cls, v: str) -> str:
        """Validate allocation status matches database CHECK constraint."""
        allowed_statuses = ["ACQUIRED", "RELEASED", "EXPIRED", "PAUSED", "FAILED"]
        if v not in allowed_statuses:
            raise ValueError(
                f"Allocation status must be one of: {', '.join(allowed_statuses)}"
            )
        return v

    @field_validator("token_count")
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        """Validate token count is positive (matches database CHECK constraint)."""
        if v <= 0:
            raise ValueError("Token count must be greater than 0")
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
                "token_request_id": "req_abc123",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "llm_model_name": "gpt-4",  # Updated field name
                "token_count": 1000,
                "allocation_status": "ACQUIRED",
                "allocated_at": "2025-10-13T23:00:00Z",
            }
        }
