"""
Backpressure constants - package-local identifiers and reason codes.

Architecture:
-------------
    ┌────────────────────┐     ┌──────────────────────┐
    │ evaluator/probes   │────▶│ constants.py         │
    │ dependency/worker  │     │ shared Layer 1 names │
    └────────────────────┘     └──────────────────────┘

Dependencies:
    - none - this module exists to avoid literal duplication

Author: Engineering Team
Last Updated: 2026-05-09
"""

QUEUE_DEPTH_REDIS_KEY = "token_alloc:queue_depth"

QUEUE_DEPTH_EXCEEDED_REASON = "queue_depth_exceeded"
DB_POOL_SATURATED_REASON = "db_pool_saturated"
DB_CIRCUIT_BREAKER_OPEN_REASON = "db_circuit_breaker_open"

QUEUE_DEPTH_PUBLISH_TASK_NAME = (
    "app.resilience.token_worker.publish_backpressure_queue_depth"
)
QUEUE_DEPTH_PUBLISH_SCHEDULE_NAME = "publish-backpressure-queue-depth"
QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER = 3
