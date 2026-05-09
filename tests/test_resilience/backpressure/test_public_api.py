from __future__ import annotations

import app.resilience as resilience_package
from app.resilience.backpressure import BackPressureGuard, backpressure_dependency


def test_package_root_exports_backpressure_symbols() -> None:
    assert BackPressureGuard is not None
    assert backpressure_dependency is not None


def test_resilience_root_reexports_backpressure_symbols() -> None:
    assert resilience_package.BackPressureGuard is BackPressureGuard
    assert resilience_package.backpressure_dependency is backpressure_dependency
