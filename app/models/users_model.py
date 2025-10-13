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
    email: str = Field(..., description="Unique email address for user identification")
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
        allowed_roles = ["owner", "admin", "developer", "viewer"]
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
                "email": "user@example.com",
                "role": "developer",
                "status": "active",
            }
        }
