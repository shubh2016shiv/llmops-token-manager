"""
Redis counter result enums - public result contracts for token counter flows.

Architecture:
-------------
    ┌──────────────────────────────┐     ┌──────────────────────────────┐
    │ API / worker callers         │────▶│ counter_results.py           │
    │ compare operation outcomes   │     │ public operation enums       │
    └──────────────────────────────┘     └──────────────────────────────┘

Dependencies:
    - Python stdlib enum - enum base types

Author: Engineering Team
Last Updated: 2026-05-09
"""

from enum import IntEnum


class TokenReservationResult(IntEnum):
    """Result values returned by token reservation attempts."""

    ALLOCATED = 1
    EXHAUSTED = 0
    COUNTER_MISS = -1


class CounterReconciliationResult(IntEnum):
    """Result values returned by Redis counter reconciliation."""

    UNCHANGED = 0
    DELTA_APPLIED = 1
    RESEEDED_PARTIAL = 2
    INITIALIZED_MISSING = 3
