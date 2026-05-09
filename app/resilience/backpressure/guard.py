"""
Backpressure guard - compatibility facade for Layer 1 admission control.

Architecture:
-------------
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │ legacy callers           │────▶│ guard.py                 │
    │ BackPressureGuard.check  │     │ compatibility facade     │
    └──────────────────────────┘     └─────────────┬────────────┘
                                                   │
                                                   ▼
                                  ┌─────────────────────────────────┐
                                  │ evaluator.py + decision mapper │
                                  └─────────────────────────────────┘

Dependencies:
    - app/models/resilience_models.py - BackpressureDecision
    - app/resilience/backpressure/dependency.py - FastAPI dependency alias

Author: Engineering Team
Last Updated: 2026-05-09
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.resilience.backpressure.decision_to_http import (
    raise_for_backpressure_decision,
)
from app.resilience.backpressure.evaluator import evaluate_backpressure

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request

    from app.models.resilience_models import BackpressureDecision


class BackPressureGuard:
    """Compatibility facade for the new Layer 1 backpressure package."""

    @staticmethod
    async def evaluate() -> BackpressureDecision:
        """Return the typed Layer 1 admission decision for the current request."""
        return await evaluate_backpressure()

    @staticmethod
    async def check() -> None:
        """Raise a FastAPI 503 response when Layer 1 rejects the request."""
        decision = await BackPressureGuard.evaluate()
        raise_for_backpressure_decision(decision)

    @staticmethod
    def as_dependency() -> Callable[[Request], Awaitable[None]]:
        """Return the FastAPI dependency callable for compatibility with older docs."""
        from app.resilience.backpressure.dependency import backpressure_dependency

        return backpressure_dependency
