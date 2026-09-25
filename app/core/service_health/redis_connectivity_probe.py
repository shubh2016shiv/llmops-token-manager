"""Verifies live connectivity to the Redis cache."""

import inspect

from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.config import settings
from app.core.redis import redis_manager
from app.models.service_health_models import ServiceStatus


async def verify_redis_connectivity() -> ServiceStatus:
    """
    Verify Redis connectivity with detailed error reporting.

    Deliberately calls `redis_manager.client.ping()` directly rather than
    `redis_manager.ping()`. The manager's own `ping()` is a general-purpose
    health primitive used elsewhere in the app (e.g. resilience checks) and
    intentionally swallows every exception, returning a bare bool — a good
    contract for a caller that only wants "is it up?". That same swallowing
    would make the except branches below permanently unreachable: every real
    outage would already have been caught and turned into `False` before it
    got here, so this probe would only ever report the generic "did not
    respond to ping" message and never the specific, more actionable ones.
    Going through `.client` directly lets the real redis-py exception surface
    to this probe, whose whole job is to report *why* Redis is unreachable.
    """
    connection_details = {
        "host": settings.redis_host,
        "port": str(settings.redis_port),
        "database": str(settings.redis_db),
    }
    try:
        # redis-py's stubs type Redis.ping() as returning either a bool or an
        # Awaitable[bool] (it shares a base class with the sync client), so a
        # bare `await` doesn't type-check even though the async client always
        # returns an awaitable at runtime. app/core/redis.py's own ping()
        # already works around this the same way — mirrored here rather than
        # reinventing it.
        pong = redis_manager.client.ping()
        if inspect.isawaitable(pong):
            pong = await pong
        if not pong:
            return ServiceStatus(
                name="Redis",
                status="failed",
                error_message="Redis server did not respond to ping",
                suggestion="Check if Redis server is running and accessible",
                connection_details=connection_details,
            )
        return ServiceStatus(
            name="Redis",
            status="connected",
            connection_details=connection_details,
        )
    except RedisConnectionError:
        # redis-py's own exception hierarchy — NOT the builtin
        # ConnectionRefusedError, which redis-py never raises. This also
        # covers AuthenticationError (a ConnectionError subclass), so a bad
        # password reports the same "can't connect" family as a refused TCP
        # connection, which is the right level of detail for a health probe.
        return ServiceStatus(
            name="Redis",
            status="failed",
            error_message=(
                "Connection refused - Redis is not running or not accessible"
            ),
            suggestion=(
                "Start Redis server with: redis-server or check if it's running on "
                f"{settings.redis_host}:{settings.redis_port}"
            ),
            connection_details=connection_details,
        )
    except RuntimeError as e:
        # redis_manager.client raises this specific, already-clear message
        # when RedisManager.initialize() hasn't run yet — a startup-ordering
        # bug, not an infrastructure outage. Surface it verbatim rather than
        # relabeling it as a generic connectivity failure.
        return ServiceStatus(
            name="Redis",
            status="failed",
            error_message=str(e),
            suggestion="Ensure RedisManager.initialize() runs before this probe",
            connection_details=connection_details,
        )
    except Exception as e:
        return ServiceStatus(
            name="Redis",
            status="failed",
            error_message=str(e),
            suggestion="Check Redis configuration in .env file and verify credentials",
            connection_details={
                "host": settings.redis_host,
                "port": str(settings.redis_port),
            },
        )
