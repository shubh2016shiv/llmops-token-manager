"""
Public-API contract test for the backpressure package.

Backpressure exposes exactly ONE public symbol: `backpressure_dependency`, the
FastAPI dependency the `/acquire` route mounts. This test pins that contract so a
future refactor cannot silently drop the export or the parent-package re-export.

(The removed `BackPressureGuard` facade used to be asserted here too; it was
deleted as dead code, so those assertions are intentionally gone.)
"""

from __future__ import annotations

import app.resilience as resilience_package
from app.resilience.backpressure import backpressure_dependency


def test_package_root_exports_backpressure_dependency() -> None:
    # The backpressure package must expose its single public entry point.
    assert backpressure_dependency is not None


def test_resilience_root_reexports_backpressure_dependency() -> None:
    # The resilience package root must re-export the *same* object (identity, not
    # just equality) so `from app.resilience import backpressure_dependency` and
    # `from app.resilience.backpressure import backpressure_dependency` agree.
    assert resilience_package.backpressure_dependency is backpressure_dependency
