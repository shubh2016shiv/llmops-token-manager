"""
Backpressure package - Layer 1 system-health admission control.

Architecture:
-------------
    ┌────────────────────────────┐     ┌────────────────────────────┐
    │ FastAPI `/acquire` route   │────▶│ backpressure/dependency.py │
    │ (app/api/)                 │     │ Layer 1 admission gate     │
    └────────────────────────────┘     └──────────────┬─────────────┘
                                                      │
                                                      ▼
                                   ┌──────────────────────────────────────┐
                                   │ evaluator.py + probes               │
                                   │ queue depth / pool / CB state only  │
                                   └──────────────┬───────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────────────┐
                                   │ queue_depth_publisher.py            │
                                   │ worker-side queue telemetry writer  │
                                   └──────────────────────────────────────┘

Dependencies:
    - app/core/config.py - backpressure thresholds and retry settings
    - app/models/resilience_models.py - BackpressureDecision contract
    - app/resilience/circuit_breaker - DB breaker state snapshot
    - app/resilience/token_worker.py - periodic queue-depth publication

Author: Engineering Team
Last Updated: 2026-05-09
"""

from app.resilience.backpressure.dependency import backpressure_dependency
from app.resilience.backpressure.guard import BackPressureGuard

__all__ = [
    "BackPressureGuard",
    "backpressure_dependency",
]
