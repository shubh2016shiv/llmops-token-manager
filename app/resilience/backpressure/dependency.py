"""
Backpressure dependency — the FastAPI front door (Layer 1 admission control).

WHAT THIS FILE IS
-----------------
This is THE single public entry point of the backpressure package. FastAPI routes
mount it with `Depends(backpressure_dependency)`; the `/acquire` endpoint does
exactly that. Everything else in this package is an internal collaborator reached
only through here.

It is deliberately tiny — it just wires two collaborators together:

    request ─▶ backpressure_dependency()
                    │  1. ask the evaluator for a verdict (pure data)
                    ▼
              evaluate_backpressure()  ──▶  BackpressureDecision
                    │  2. hand the verdict to the HTTP translator
                    ▼
              raise_for_backpressure_decision()
                    │
                    ├─ verdict says ACCEPT → returns quietly, request proceeds
                    └─ verdict says REJECT → raises HTTPException(503), request stops

Keeping this seam thin is intentional: the "decide" and "translate" steps each
live in their own file and are independently testable; this file only connects them.

Author: Engineering Team
Last Updated: 2026-07-23
"""

from fastapi import Request

# The decision brain: reads the three gauges and returns a typed verdict.
from app.resilience.backpressure.evaluator import evaluate_backpressure

# The HTTP translator: turns a verdict into a 503 (or does nothing on accept).
from app.resilience.backpressure.http_response import raise_for_backpressure_decision


async def backpressure_dependency(_request: Request) -> None:
    """
    Enforce Layer 1 system-health admission control for hot-path endpoints.

    Runs before the endpoint body: if the system is saturated it raises a 503 and
    the endpoint never executes; otherwise it returns and the request proceeds.

    The `Request` parameter is intentionally accepted but unused for now — it
    reserves a stable signature for future priority routing or operator-bypass
    rules (e.g. "let health-check callers through") without breaking call sites.
    """
    # Step 1 — DECIDE: read the gauges and get a verdict (no HTTP involved yet).
    decision = await evaluate_backpressure()
    # Step 2 — TRANSLATE: raise 503 if the verdict rejects; otherwise return.
    raise_for_backpressure_decision(decision)
