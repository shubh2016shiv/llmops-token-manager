"""
Redis counter result enums — the typed contract with the Lua scripts.

Each Lua script in `lua_script_definitions.py` returns a small integer. Rather than
pass raw numbers around, callers get these named enums. The **integer values here
match the Lua return codes exactly** — that 1:1 mapping is the clean seam between
the algorithm (Lua) and the application (Python). If you change a Lua return code,
change it here too.

Author: Engineering Team
"""

from enum import IntEnum


class TokenReservationResult(IntEnum):
    """What a reserve attempt did (values mirror the RESERVE Lua's returns)."""

    ALLOCATED = 1  # room existed; tokens were claimed -> success
    EXHAUSTED = 0  # granting would exceed the limit -> no capacity
    COUNTER_MISS = -1  # counter/limit not in Redis -> caller falls back to the DB


class CounterReconciliationResult(IntEnum):
    """What a reconcile did (values mirror the RECONCILE Lua's returns)."""

    UNCHANGED = 0  # Redis already matched the DB -> nothing to do
    DELTA_APPLIED = 1  # counter and/or limit was corrected toward the DB value
    RESEEDED_PARTIAL = 2  # one key was missing and had to be re-seeded
    INITIALIZED_MISSING = 3  # both keys were missing -> cold-start seed
