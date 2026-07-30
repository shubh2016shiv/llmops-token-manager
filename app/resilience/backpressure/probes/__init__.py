"""
Probes — the three health "gauges" the backpressure evaluator reads.

A *probe* is a tiny, single-purpose reader: it measures ONE signal about system
health and returns a clean value, or `None` when it cannot read (the fail-open
rule — see any probe's docstring). Probes never make policy decisions; they only
report. The policy ("is this value too high? reject?") lives in evaluator.py.

    evaluator.py  ──reads──▶  queue_depth   (gauge #1: work waiting in the queue)
                  ──reads──▶  db_pool       (gauge #2: DB connection saturation)
                  ──reads──▶  circuit_state (gauge #3: is the DB breaker open?)

This package groups the three gauges so the folder's structure mirrors the mental
model: read (probes) → decide (evaluator) → translate (http_response).

The public reader functions are re-exported here so callers can write
`from app.resilience.backpressure.probes import read_queue_depth` without needing
to know which file each one lives in.
"""

from app.resilience.backpressure.probes.circuit_state import (
    read_db_circuit_breaker_snapshot,
)
from app.resilience.backpressure.probes.db_pool import read_db_pool_utilization_pct
from app.resilience.backpressure.probes.queue_depth import (
    estimate_queue_retry_after_seconds,
    read_queue_depth,
)

__all__ = [
    "read_queue_depth",
    "estimate_queue_retry_after_seconds",
    "read_db_pool_utilization_pct",
    "read_db_circuit_breaker_snapshot",
]
