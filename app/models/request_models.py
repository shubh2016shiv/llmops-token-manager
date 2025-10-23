"""
LLM Token Management Request Models
===================================

This module defines Pydantic request models for Large Language Model (LLM)
token resource allocation management across multiple cloud providers and model types.

Key Capabilities:
- Request Model for Token capacity allocation and release operations
- Request Model for Multi-provider LLM configuration Registration
- RBAC based request validation for each request type
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field, field_validator, EmailStr
from enum import Enum
from uuid import UUID
from datetime import datetime


class LLMProvider(str, Enum):
    """
    LLM Provider Types - The actual AI model provider/creator.
    Matches database CHECK constraint for llm_provider column.
    """

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    META = "meta"
    HUGGING_FACE = "hugging_face"
    TOGETHER_AI = "together_ai"
    FIREWORKS_AI = "fireworks_ai"
    REPLICATE = "replicate"
    XAI = "xai"
    DEEPINFRA = "deepinfra"
    NOVITA = "novita"
    ON_PREMISE = "on_premise"


class CloudProvider(str, Enum):
    """
    Cloud Provider Types - Infrastructure hosting the LLM deployment.
    Matches database CHECK constraint for cloud_provider column.
    """

    AZURE = "Azure"
    GOOGLE_CLOUD_PLATFORM = "Google Cloud Platform"
    AMAZON_WEB_SERVICES = "Amazon Web Services"
    IBM_WATSONX = "IBM Watsonx"
    ORACLE = "Oracle"
    ON_PREMISE = "On Premise"


class UserRole(str, Enum):
    """User roles for role-based access control"""

    DEVELOPER = "developer"  # End users - can request/release tokens only
    # Requests: TokenAllocationRequest, TokenReleaseRequest

    OPERATOR = "operator"  # Operations team - can manage deployments
    # Requests: PauseDeploymentRequest, ResumeDeploymentRequest

    ADMIN = "admin"  # System administrators - can configure deployments
    # Requests: DeploymentConfigCreate, DeploymentConfigUpdate

    OWNER = "owner"  # Full system access - can perform all requests
    # Requests: All requests

    VIEWER = "viewer"  # Read-only access - can view system status
    USER = "user"  # Basic user access


class UserStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class UserCreateRequest(BaseModel):
    """Request model for creating a new user with additional fields."""

    """API request model - only fields client provides"""

    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=8)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: EmailStr) -> EmailStr:
        return v.lower().strip()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        # Only alphanumeric, underscore, and hyphen
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscore, and hyphen"
            )
        return v.lower().strip()

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "username": "johndoe",
                "email": "john.doe@example.com",
                "password": "SecurePass123",
            }
        }


class UserUpdateRequest(BaseModel):
    """Request model for updating user information."""

    email: Optional[EmailStr] = Field(None, description="New email address")
    first_name: Optional[str] = Field(
        None, description="New first name", min_length=1, max_length=50
    )
    last_name: Optional[str] = Field(
        None, description="New last name", min_length=1, max_length=50
    )
    username: Optional[str] = Field(
        None, description="New username", min_length=1, max_length=50
    )
    password: Optional[str] = Field(None, description="New password", min_length=8)
    role: Optional[UserRole] = Field(
        None, description="New role: owner, admin, developer, or viewer"
    )
    status: Optional[UserStatus] = Field(
        None, description="New status: active, suspended, or inactive"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "email": "john.updated@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "username": "john.doe",
                "role": UserRole.ADMIN,
                "status": "active",
            }
        }


class TokenAllocationClientRequest(BaseModel):
    """
    Minimal client request for token allocation.

    Client provides only essential fields - system derives the rest.
    """

    llm_provider: LLMProvider = Field(
        ..., description="LLM provider type - determines routing"
    )

    llm_model_name: str = Field(
        ...,
        description="The LOGICAL model identifier representing the AI capability requested",
        min_length=1,
        max_length=100,
        alias="model_name",
    )

    input_data: Union[str, List[Dict[str, Any]]] = Field(
        ..., description="The actual input text or structured data for token estimation"
    )

    deployment_name: Optional[str] = Field(
        default=None,
        description="Optional physical deployment instance identifier",
        max_length=100,
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Cloud provider hosting the deployment",
    )

    deployment_region: Optional[str] = Field(
        default=None,
        description="Preferred deployment region for deployment selection",
        max_length=50,
    )

    request_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata for tracking (team, project, batch_id, etc.)",
    )

    class Config:
        protected_namespaces = ()
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4.1",
                "input_data": "Your prompt text here for token estimation",
                "cloud_provider": "Azure",
                "deployment_region": "eastus2",
                "request_context": {"project": "medical-qa", "team": "research"},
            }
        }


class TokenAllocationRequest(BaseModel):
    """
    Reserve token capacity before making LLM calls.

    Flow: Client estimates tokens → calls acquire → receives deployment config → makes LLM call → releases tokens
    """

    user_id: UUID = Field(..., description="Reference to the user requesting tokens")

    user_role: UserRole = Field(
        description="User role - determines access to requests",
        default=UserRole.DEVELOPER,
    )

    llm_provider: LLMProvider = Field(
        ..., description="LLM provider type - determines routing"
    )
    llm_model_name: str = Field(
        ...,
        description=(
            "The LOGICAL model identifier representing the AI capability requested "
            "(e.g., 'gpt-4-turbo-2024-04-09', 'claude-3-opus', 'llama-3-70b'). "
            "This specifies WHAT AI model capability you want to use, independent of "
            "WHERE or HOW it's deployed."
        ),
        min_length=1,
        max_length=100,
        alias="model_name",  # Maps to database column 'model_name'
    )

    token_count: int = Field(
        ...,
        description="Estimated total tokens to reserve based on the user's payload size",
        gt=0,
        le=3000000,
    )

    deployment_name: Optional[str] = Field(
        default=None,
        description=(
            "The PHYSICAL deployment instance identifier (e.g., 'azure-gpt4-eastus-prod-01', "
            "'aws-claude-west-instance-3'). This specifies WHICH specific running instance "
            "or endpoint to use. If omitted, the system automatically selects the least-busy "
            "physical deployment that provides the requested model_name capability. "
            "Think of this as the 'server name' or 'endpoint address' that hosts the model."
        ),
        max_length=100,
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Cloud provider hosting the deployment",
    )

    deployment_region: Optional[str] = Field(
        default=None,
        description="Preferred deployment region - system tries this region first if capacity available",
        max_length=50,
    )

    request_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata for tracking (team, project, batch_id, etc.)",
    )

    @field_validator("token_count")
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Token count must be positive")
        if v > 3000000:
            raise ValueError(f"Token count {v} exceeds maximum limit")
        return v

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4-turbo-2024-04-09-gp",  # Updated field name
                "token_count": 5000,
                "deployment_name": None,
                "cloud_provider": "Azure",
                "deployment_region": "eastus2",
                "request_context": {"team": "research", "project": "medical-qa"},
            }
        }


class TokenReleaseRequest(BaseModel):
    """
    Release allocated tokens back to pool.

    Critical: Must call immediately after LLM call completes to free capacity.
    """

    token_request_id: str = Field(..., description="Token request ID to release")

    user_role: UserRole = Field(
        description="User role - determines access to requests",
        default=UserRole.DEVELOPER,
    )

    class Config:
        json_schema_extra = {"example": {"token_request_id": "abc123def456"}}


class PauseDeploymentRequest(BaseModel):
    """
    Emergency failover - pause a failing deployment.

    Use when: Provider outage, rate limits, high errors, maintenance.
    """

    user_role: UserRole = Field(
        description="User role - determines access to requests",
        default=UserRole.OPERATOR,
    )

    llm_provider: LLMProvider = Field(..., description="LLM provider to pause")

    llm_model_name: str = Field(
        ...,
        description="Model to pause",
        min_length=1,
        max_length=100,
        alias="model_name",  # Maps to database column 'model_name'
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Cloud provider to pause (if cloud-deployed)",
    )

    api_endpoint_url: Optional[str] = Field(
        default=None, description="Endpoint URL to pause", min_length=1, max_length=500
    )

    pause_reason: str = Field(
        ...,
        description="Reason for pause - logged for audit",
        min_length=1,
        max_length=1000,
    )

    pause_duration_minutes: Optional[int] = Field(
        default=None,
        description="How long to pause - if omitted, uses config default",
        gt=0,
        le=1440,
    )

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4-turbo-2024-04-09-gp",  # Updated field name
                "cloud_provider": "Azure",
                "api_endpoint_url": "https://<deployment>-<region>.openai.azure.com/",
                "pause_reason": "Azure region outage - 503 errors",
                "pause_duration_minutes": 60,
            }
        }


class ResumeDeploymentRequest(BaseModel):
    """
    Resume a paused deployment.

    Use when: Issue resolved - manually un-pause before auto-expiry.
    """

    user_id: UUID = Field(..., description="Reference to the user requesting tokens")

    user_role: UserRole = Field(
        description="User role - determines access to requests",
        default=UserRole.OPERATOR,
    )

    llm_provider: LLMProvider = Field(..., description="LLM provider to resume")

    llm_model_name: str = Field(
        ...,
        description="Model to resume",
        min_length=1,
        max_length=100,
        alias="model_name",  # Maps to database column 'model_name'
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Cloud provider to resume (if cloud-deployed)",
    )

    api_base: str = Field(
        ..., description="Endpoint URL to resume", min_length=1, max_length=500
    )

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4-turbo-2024-04-09-gp",  # Updated field name
                "cloud_provider": "Azure",
                "api_base": "https://<deployment>-<region>.openai.azure.com/",
            }
        }


class DeploymentConfigCreate(BaseModel):
    """
    Register new LLM deployment for load balancing.

    Use when: Admin adds new provider endpoint to the pool.
    """

    user_id: UUID = Field(..., description="Reference to the user requesting tokens")

    user_role: UserRole = Field(
        description="User role - determines access to requests",
        default=UserRole.ADMIN,
    )

    llm_provider: LLMProvider = Field(..., description="LLM provider type")

    llm_model_name: str = Field(
        ...,
        description="Model identifier",
        min_length=1,
        max_length=100,
        alias="model_name",  # Maps to database column 'model_name'
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Cloud provider hosting this deployment",
    )

    api_version: str = Field(
        ...,
        description="Provider API version (e.g., '2023-03-15' for Azure OpenAI)",
        min_length=1,
        max_length=50,
    )

    deployment_name: str = Field(
        ...,
        description="Deployment-specific name (Azure deployment name, etc.)",
        min_length=1,
        max_length=100,
    )

    api_endpoint_url: Optional[str] = Field(
        default=None,
        description="Base URL for API calls",
        min_length=1,
        max_length=500,
    )

    api_key_identifier: str = Field(
        ...,
        description="Identifier for the API key like OPENAI_API_KEY_GPT4T or ANTHROPIC_API_KEY_CLAUDE3OPUS",
        min_length=1,
        max_length=200,
    )

    region: str = Field(
        ..., description="Geographic region for routing", min_length=1, max_length=50
    )

    max_tokens: int = Field(
        ...,
        description="Maximum tokens per minute (TPM) quota for this deployment",
        gt=0,
        le=100000000,
    )

    max_token_lock_time_secs: Optional[int] = Field(
        default=70,
        description="Default reservation expiry time in seconds",
        gt=0,
        le=3600,
    )

    temperature: Optional[float] = Field(
        default=0.0,
        description="Default temperature for this deployment",
        ge=0.0,
        le=2.0,
    )

    seed: Optional[int] = Field(
        default=42, description="Default seed for this deployment", ge=0
    )

    is_active: bool = Field(
        default=True, description="Enable/disable traffic to this deployment"
    )

    created_at: Optional[datetime] = Field(
        default=datetime.now(),
        description="Timestamp when the deployment configuration was created",
    )

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4-turbo-2024-04-09-gp",  # Updated field name
                "cloud_provider": "Azure",
                "api_version": "2023-03-15",
                "deployment_name": "gpt-4-turbo-2024-04-09-gp",
                "api_endpoint_url": "https://<deployment>-<region>.openai.azure.com/",
                "api_key_identifier": "AZURE_OPENAI_API_KEY_GPT4T",
                "region": "eastus2",
                "max_tokens": 120000,
                "max_token_lock_time_secs": 70,
                "temperature": 0.0,
                "seed": 42,
                "is_active": True,
                "created_at": "2025-10-17T10:30:00Z",
            }
        }


class DeploymentConfigUpdate(BaseModel):
    """
    Update existing deployment configuration.

    Use when: Modify rate limits, defaults, endpoints, or toggle active status.
    All fields are optional - only provided fields will be updated.
    """

    user_id: UUID = Field(..., description="Reference to the user requesting tokens")

    user_role: UserRole = Field(
        description="User role - determines access to requests",
        default=UserRole.ADMIN,
    )

    llm_model_name: Optional[str] = Field(
        default=None,
        description="Updated model identifier",
        min_length=1,
        max_length=100,
        alias="model_name",  # Maps to database column 'model_name'
    )

    api_version: Optional[str] = Field(
        default=None,
        description="Updated provider API version",
        min_length=1,
        max_length=50,
    )

    deployment_name: Optional[str] = Field(
        default=None,
        description="Updated deployment-specific name",
        min_length=1,
        max_length=100,
    )

    api_base: Optional[str] = Field(
        default=None,
        description="Updated base URL for API calls",
        min_length=1,
        max_length=500,
    )

    api_key_identifier: Optional[str] = Field(
        default=None,
        description="Updated API key identifier",
        min_length=1,
        max_length=200,
    )

    region: Optional[str] = Field(
        default=None,
        description="Updated geographic region for routing",
        min_length=1,
        max_length=50,
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Updated maximum tokens per minute (TPM) quota",
        gt=0,
        le=100000000,
    )

    max_token_lock_time_secs: Optional[int] = Field(
        default=None,
        description="Updated default reservation expiry time in seconds",
        gt=0,
        le=3600,
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Updated default temperature for this deployment",
        ge=0.0,
        le=2.0,
    )

    seed: Optional[int] = Field(
        default=None, description="Updated default seed for this deployment", ge=0
    )

    is_active: Optional[bool] = Field(
        default=None, description="Updated enable/disable traffic to this deployment"
    )

    class Config:
        # Disable protected namespaces to avoid conflicts with model_ prefix fields
        protected_namespaces = ()
        # Allow population by field name or alias
        populate_by_name = True
        json_schema_extra = {
            "examples": [
                # Update rate limits only
                {"max_tokens": 150000, "is_active": True},
                # Update endpoint and credentials
                {
                    "api_base": "https://<deployment>-<region>.openai.azure.com/",
                    "api_key_identifier": "AZURE_OPENAI_API_KEY_GPT4T_V2",
                },
                # Update defaults
                {"temperature": 0.5, "seed": 123, "max_token_lock_time_secs": 120},
            ],
            "example": {"max_tokens": 150000, "is_active": True},
        }


# ============================================================================
# LLM MODEL CONFIGURATION REQUEST MODELS
# ============================================================================


class LLMModelCreateRequest(BaseModel):
    """
    Request model for creating a new LLM model configuration.

    This model defines all required and optional parameters for registering
    a new LLM model in the system for token allocation and rate limiting.
    """

    llm_provider: LLMProvider = Field(
        ...,
        description="LLM provider type (e.g., openai, anthropic, gemini)",
    )

    llm_model_name: str = Field(
        ...,
        description="Name of the LLM model (e.g., 'gpt-4o', 'claude-3.5-sonnet')",
        min_length=1,
        max_length=100,
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Cloud provider hosting the LLM",
    )

    api_key_variable_name: Optional[str] = Field(
        default=None,
        description="Environment variable name for the API key (e.g., 'OPENAI_API_KEY_GPT4O')",
        min_length=1,
        max_length=200,
    )

    llm_model_version: Optional[str] = Field(
        default=None,
        description="Optional model version (e.g., '2024-08', 'v1.0')",
        max_length=50,
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens per request",
        gt=0,
        le=1000000,
    )

    tokens_per_minute_limit: Optional[int] = Field(
        default=None,
        description="Token rate limit per minute",
        gt=0,
        le=10000000,
    )

    requests_per_minute_limit: Optional[int] = Field(
        default=None,
        description="Request rate limit per minute",
        gt=0,
        le=10000,
    )

    deployment_name: Optional[str] = Field(
        default=None,
        description="Optional deployment identifier",
        max_length=100,
    )

    api_endpoint_url: Optional[str] = Field(
        default=None,
        description="Optional API endpoint URL",
        max_length=500,
    )

    is_active_status: bool = Field(
        default=True,
        description="Whether the model is active for token allocation",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Default temperature setting (0.0 to 2.0)",
        ge=0.0,
        le=2.0,
    )

    random_seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for reproducible results",
        ge=0,
    )

    deployment_region: Optional[str] = Field(
        default=None,
        description="Optional geographic deployment region",
        max_length=50,
    )

    @field_validator("llm_model_name")
    @classmethod
    def validate_llm_model_name(cls, v: str) -> str:
        """Validate LLM model name format."""
        if not v or not v.strip():
            raise ValueError("LLM model name cannot be empty")
        # Allow alphanumeric, hyphens, underscores, and dots
        import re

        if not re.match(r"^[a-zA-Z0-9._-]+$", v):
            raise ValueError(
                "LLM model name can only contain letters, numbers, dots, underscores, and hyphens"
            )
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_key_variable_name": "OPENAI_API_KEY_GPT4O",
                "llm_model_version": "2024-08",
                "max_tokens": 8192,
                "tokens_per_minute_limit": 100000,
                "requests_per_minute_limit": 1000,
                "deployment_name": "gpt-4o-eastus",
                "api_endpoint_url": "https://api.openai.com/v1",
                "is_active_status": True,
                "temperature": 0.7,
                "random_seed": 42,
                "deployment_region": "eastus2",
            }
        }


class LLMModelUpdateRequest(BaseModel):
    """
    Request model for updating an existing LLM model configuration.

    All fields are optional - only provided fields will be updated.
    Used for modifying rate limits, endpoints, settings, or activation status.
    """

    llm_provider: Optional[LLMProvider] = Field(
        default=None,
        description="Updated LLM provider type",
    )

    llm_model_name: Optional[str] = Field(
        default=None,
        description="Updated LLM model name",
        min_length=1,
        max_length=100,
    )

    cloud_provider: Optional[CloudProvider] = Field(
        default=None,
        description="Updated cloud provider hosting the LLM",
    )

    api_key_variable_name: Optional[str] = Field(
        default=None,
        description="Updated API key environment variable name",
        min_length=1,
        max_length=200,
    )

    llm_model_version: Optional[str] = Field(
        default=None,
        description="Updated model version",
        max_length=50,
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Updated maximum tokens per request",
        gt=0,
        le=1000000,
    )

    tokens_per_minute_limit: Optional[int] = Field(
        default=None,
        description="Updated token rate limit per minute",
        gt=0,
        le=10000000,
    )

    requests_per_minute_limit: Optional[int] = Field(
        default=None,
        description="Updated request rate limit per minute",
        gt=0,
        le=10000,
    )

    deployment_name: Optional[str] = Field(
        default=None,
        description="Updated deployment identifier",
        max_length=100,
    )

    api_endpoint_url: Optional[str] = Field(
        default=None,
        description="Updated API endpoint URL",
        max_length=500,
    )

    is_active_status: Optional[bool] = Field(
        default=None,
        description="Updated activation status",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Updated temperature setting (0.0 to 2.0)",
        ge=0.0,
        le=2.0,
    )

    random_seed: Optional[int] = Field(
        default=None,
        description="Updated random seed for reproducible results",
        ge=0,
    )

    deployment_region: Optional[str] = Field(
        default=None,
        description="Updated geographic deployment region",
        max_length=50,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_tokens": 16384,
                "tokens_per_minute_limit": 200000,
                "is_active_status": True,
                "temperature": 0.5,
                "deployment_region": "westus2",
            }
        }


# ============================================================================
# TOKEN RETRY REQUEST MODEL
# ============================================================================


class TokenRetryRequest(BaseModel):
    """
    Request model for retrying token allocation.

    Used when a token allocation is in WAITING status and needs to be retried
    to check if capacity is now available.
    """

    token_request_id: str = Field(
        ..., description="Token request ID to retry", min_length=1, max_length=100
    )

    class Config:
        json_schema_extra = {"example": {"token_request_id": "req_abc123xyz"}}


# ============================================================================
# USER LLM ENTITLEMENTS REQUEST MODELS
# ============================================================================


class UserEntitlementCreateRequest(BaseModel):
    """
    Request model for creating a new user LLM entitlement.

    Only users with admin or owner role can create entitlements.
    API keys are encrypted before storage using bcrypt hashing.
    The target user ID is taken from the URL path, not from the request body.
    """

    llm_provider: LLMProvider = Field(..., description="LLM provider type")
    llm_model_name: str = Field(
        ...,
        description="Logical model name (must exist in llm_models table)",
        min_length=1,
        max_length=100,
    )
    api_key_variable_name: Optional[str] = Field(
        None,
        description="Environment variable name for the API key",
        max_length=200,
    )
    api_key_value: str = Field(
        ...,
        description="Plain API key value (will be encrypted before storage)",
        min_length=1,
    )
    api_endpoint_url: Optional[str] = Field(
        None, description="Specific API endpoint URL", max_length=500
    )
    cloud_provider: Optional[CloudProvider] = Field(
        None,
        description="Cloud provider hosting the LLM",
    )
    deployment_name: Optional[str] = Field(
        None,
        description="Physical deployment identifier for cloud providers",
        max_length=100,
    )
    region: Optional[str] = Field(
        None,
        description="Geographic region where model is deployed",
        max_length=50,
    )

    @field_validator("api_key_value")
    @classmethod
    def validate_api_key_value(cls, v: str) -> str:
        """Validate API key format."""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        if len(v) < 10:
            raise ValueError("API key must be at least 10 characters")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "llm_provider": "openai",
                "llm_model_name": "gpt-4o",
                "api_key_variable_name": "OPENAI_API_KEY",
                "api_key_value": "sk-1234567890abcdefgh",
                "api_endpoint_url": "https://api.openai.com/v1",
                "cloud_provider": None,
                "deployment_name": None,
                "deployment_region": "us-east-1",
            }
        }


# class UserEntitlementUpdateRequest(BaseModel):
#     """
#     Request model for updating an existing user LLM entitlement.

#     Only users with admin or owner role can update entitlements.
#     All fields are optional - only provided fields will be updated.
#     """

#     api_key: Optional[str] = Field(
#         None,
#         description="New API key (will be encrypted before storage)",
#         min_length=10,
#     )
#     api_endpoint_url: Optional[str] = Field(
#         None, description="Updated API endpoint URL", max_length=500
#     )
#     cloud_provider: Optional[str] = Field(
#         None, description="Updated cloud provider", max_length=50
#     )
#     deployment_name: Optional[str] = Field(
#         None, description="Updated deployment identifier", max_length=100
#     )
#     region: Optional[str] = Field(
#         None, description="Updated geographic region", max_length=50
#     )

#     @field_validator("api_key")
#     @classmethod
#     def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
#         """Validate API key format if provided."""
#         if v is not None:
#             if not v or not v.strip():
#                 raise ValueError("API key cannot be empty")
#             if len(v) < 10:
#                 raise ValueError("API key must be at least 10 characters")
#             return v.strip()
#         return v

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "api_key": "sk-new1234567890abcdefgh",
#                 "api_endpoint_url": "https://api.openai.com/v1",
#                 "region": "us-west-2",
#             }
#         }
