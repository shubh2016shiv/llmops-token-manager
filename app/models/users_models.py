from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    """
    Pydantic model for the 'users' table.

    Manages user identities and roles for LLM token allocation.
    Maps to: users_schema.sql
    """

    user_id: Optional[UUID] = Field(
        default=None, description="Unique identifier for the user (auto-generated)"
    )
    username: str = Field(..., description="Unique username for login", max_length=50)
    email: str = Field(..., description="Unique email address for user identification")
    first_name: str = Field(..., description="User's first name", max_length=50)
    last_name: str = Field(..., description="User's last name", max_length=50)
    role: str = Field(
        default="developer", description="User role: owner, admin, developer, or viewer"
    )
    status: str = Field(
        default="active", description="User status: active, suspended, or inactive"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Timestamp of the user's creation"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Timestamp of the user's last update"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role matches database CHECK constraint."""
        allowed_roles = ["owner", "admin", "developer", "viewer", "user", "operator"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status matches database CHECK constraint."""
        allowed_statuses = ["active", "suspended", "inactive"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}")
        return v

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",  # Added for completeness
                "username": "johndoe",  # Added
                "email": "user@example.com",
                "first_name": "John",  # Added
                "last_name": "Doe",  # Added
                "role": "developer",
                "status": "active",
                "created_at": "2025-10-13T10:00:00Z",  # Added for completeness
                "updated_at": "2025-10-13T10:00:00Z",  # Added for completeness
            }
        }
