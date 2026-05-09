"""
Token queue consumer health check.

This validates that the raw Kombu consumer runtime can import its modules and
reach the configured RabbitMQ broker.
"""

from app.resilience.token_queue.topology import TOKEN_BROKER_CONNECTION


def inspect_token_queue_consumer_readiness(
    max_retries: int = 1,
) -> tuple[bool, str | None]:
    """Validate raw consumer broker connectivity."""
    try:
        with TOKEN_BROKER_CONNECTION.clone() as conn:
            conn.ensure_connection(max_retries=max_retries)
    except Exception as exc:
        return False, f"Token queue consumer connectivity check failed: {exc}"
    return True, None


def main() -> int:
    """Return a shell-compatible exit status for container health checks."""
    is_ready, _ = inspect_token_queue_consumer_readiness(max_retries=1)
    return 0 if is_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
