"""
Backpressure package — Layer 1 system-health admission control.

WHAT THIS PACKAGE DOES (the one-sentence version)
--------------------------------------------------
Before the `/acquire` endpoint does any expensive token-allocation work, it asks
this package a single yes/no question: "is the system already too saturated to
safely accept more work?" If yes, we reject the request *immediately* with an
HTTP 503 + Retry-After, instead of accepting work we cannot process. That fast,
honest "no" is what "backpressure" means. See README.md in this folder for the
full mental model, and PROPOSED_DESIGN.md for the design decisions behind it.

HOW THE PIECES FIT (the request-time flow — "Flow A")
-----------------------------------------------------
    ┌────────────────────────────┐     ┌────────────────────────────┐
    │ FastAPI `/acquire` route   │────▶│ dependency.py              │
    │ Depends(backpressure_...)  │     │ THE front door (this API)  │
    └────────────────────────────┘     └──────────────┬─────────────┘
                                                       │ asks for a verdict
                                                       ▼
                                    ┌──────────────────────────────────────┐
                                    │ evaluator.py                         │
                                    │ the rule: 3 gauges, first-red-wins   │
                                    └───────┬───────────┬───────────┬──────┘
                                            ▼           ▼           ▼
                                    queue_depth    db_pool      circuit_state
                                    probe          probe        probe
                                            └───────────┴───────────┘
                                                        │ returns a typed decision
                                                        ▼
                                    ┌──────────────────────────────────────┐
                                    │ http_response.py                     │
                                    │ translates the verdict → HTTP 503    │
                                    └──────────────────────────────────────┘

    The three gauges live in the probes/ sub-package (queue_depth, db_pool,
    circuit_state) — each a tiny reader of one signal.

    Separately, a background worker ("Flow B") keeps the queue-depth gauge fresh:
    publisher.py measures RabbitMQ and writes the number into Redis on a timer.
    Flow A only ever *reads* that number. The two flows meet at one Redis key and
    never call each other.

WHAT WE EXPORT (and why so little)
----------------------------------
Only `backpressure_dependency` is part of this package's public surface — it is
the single, real entry point that FastAPI wires via `Depends(...)`. Everything
else (probes, evaluator, translator, publisher) is an internal collaborator that
callers should not reach into directly. Keeping the public surface tiny is what
lets the internals evolve without breaking anyone.

Note: an older `BackPressureGuard` facade used to live here for "legacy callers"
that never actually existed; it was removed because it presented a confusing
second front door. `backpressure_dependency` is the one and only entry point.

Dependencies (the "seams" you would swap to reuse this pattern elsewhere):
    - app/core/config.py               — thresholds and retry tuning (settings.bp_*)
    - app/models/resilience_models.py  — BackpressureDecision contract
    - app/core/redis.py                — queue-depth gauge storage
    - app/core/database.py             — DB pool gauge source
    - app/resilience/circuit_breaker   — DB breaker gauge source

Author: Engineering Team
Last Updated: 2026-07-23
"""

# The ONLY public export. `backpressure_dependency` is the FastAPI dependency the
# `/acquire` route mounts with `Depends(...)`. Import it from the package root so
# call sites read `from app.resilience.backpressure import backpressure_dependency`
# and never need to know which internal module it physically lives in.
from app.resilience.backpressure.dependency import backpressure_dependency

__all__ = [
    "backpressure_dependency",
]
