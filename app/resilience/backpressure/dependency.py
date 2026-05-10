"""
Backpressure dependency - FastAPI Layer 1 request admission entrypoint.

Architecture:
-------------
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │ token_manager_endpoints  │────▶│ dependency.py            │
    │ Depends(backpressure)    │     │ Layer 1 FastAPI gate     │
    └──────────────────────────┘     └─────────────┬────────────┘
                                                   │
                                                   ▼
                                  ┌─────────────────────────────────┐
                                  │ evaluator.py + decision mapper │
                                  └─────────────────────────────────┘

Dependencies:
    - app/models/resilience_models.py - typed decision contract
    - app/resilience/backpressure/evaluator.py - Layer 1 evaluation logic
    - app/resilience/backpressure/backpressure_http_response.py - 503 conversion

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

from app.resilience.backpressure.backpressure_http_response import (
    raise_for_backpressure_decision,
)
from app.resilience.backpressure.evaluator import evaluate_backpressure


async def backpressure_dependency(_request: Request) -> None:
    """
    Enforce Layer 1 system-health admission control for hot-path endpoints.

    The `Request` parameter is intentionally reserved for future priority routing
    or operator bypass rules without changing the dependency signature.
    """
    decision = await evaluate_backpressure()
    raise_for_backpressure_decision(decision)
