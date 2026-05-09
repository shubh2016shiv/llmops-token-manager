from __future__ import annotations


def test_runtime_modules_resolve_breakers_from_new_package() -> None:
    from app import app as app_module
    from app.api import token_manager_endpoints
    from app.resilience import backpressure, redis_token_counter, token_queue

    assert token_manager_endpoints.get_db_circuit_breaker.__module__ == (
        "app.resilience.circuit_breaker.breaker_registry"
    )
    assert backpressure.get_db_circuit_breaker.__module__ == (
        "app.resilience.circuit_breaker.breaker_registry"
    )
    assert redis_token_counter.get_redis_circuit_breaker.__module__ == (
        "app.resilience.circuit_breaker.breaker_registry"
    )
    assert token_queue.get_rmq_circuit_breaker.__module__ == (
        "app.resilience.circuit_breaker.breaker_registry"
    )
    assert app_module.close_circuit_breaker_redis_client.__module__ == (
        "app.resilience.circuit_breaker"
    )
