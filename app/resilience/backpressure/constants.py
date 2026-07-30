"""
Backpressure constants — names and reason codes shared across the package.

WHY A CONSTANTS FILE
--------------------
Several files must agree on the exact same strings. If they each hard-coded their
own copy, a typo in one place would silently break the system (e.g. the publisher
writing to one Redis key while the probe reads a different one). Defining each
string ONCE here makes that class of bug impossible.

Nothing here has logic — these are just the shared vocabulary of the package.

Author: Engineering Team
Last Updated: 2026-07-23
"""

# --- The Redis key that links the two flows ---------------------------------
# publisher.py WRITES the queue depth to this key; probes/queue_depth.py READS
# it. They must use the identical string, so it lives here and nowhere else.
QUEUE_DEPTH_REDIS_KEY = "token_alloc:queue_depth"

# --- Machine-readable rejection reason codes --------------------------------
# The evaluator stamps one of these onto a rejection verdict; http_response.py
# maps each to a human message and echoes it in the `X-Backpressure-Reason`
# header. Stable codes let clients branch on the cause without parsing prose.
QUEUE_DEPTH_EXCEEDED_REASON = "queue_depth_exceeded"
DB_POOL_SATURATED_REASON = "db_pool_saturated"
DB_CIRCUIT_BREAKER_OPEN_REASON = "db_circuit_breaker_open"

# --- Background publish-task identifiers ------------------------------------
# The scheduler (Celery beat) registers the queue-depth publish job under these
# names. Kept here so the task, its schedule entry, and the tests all match.
QUEUE_DEPTH_PUBLISH_TASK_NAME = (
    "app.resilience.token_maintenance.publish_backpressure_queue_depth"
)
QUEUE_DEPTH_PUBLISH_SCHEDULE_NAME = "publish-backpressure-queue-depth"

# How long the published queue-depth value lives in Redis, expressed as a
# multiple of the publish interval. TTL = interval × this. If the publisher stops
# running, the key expires after this many cycles and the reader sees "unknown"
# instead of a stale value. 3 gives ~2 missed cycles of grace before expiry.
QUEUE_DEPTH_PUBLISH_TTL_MULTIPLIER = 3
