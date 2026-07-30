"""
Pydantic models for infrastructure connectivity and service status reporting.

These models define the structured result returned by connectivity probes
(e.g. database, Redis) and consumed by console reporting and API health routes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """
    Connection status for a single infrastructure dependency.

    Returned by connectivity probes (e.g. database, Redis) and consumed by
    both console startup reporting and the `/health` API route.
    """

    name: str = Field(..., description="Human-readable dependency name.")
    status: Literal["connected", "failed", "skipped"] = Field(
        ..., description="Result of the connectivity probe."
    )
    error_message: str | None = Field(
        default=None, description="Error detail when status is 'failed'."
    )
    suggestion: str | None = Field(
        default=None, description="Actionable suggestion to resolve a failure."
    )
    connection_details: dict[str, str] | None = Field(
        default=None, description="Host/port/database identifiers for diagnostics."
    )
