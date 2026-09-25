"""
Redis-backed HTTP rate limiting with lifecycle-managed storage.

The limiter uses the TCP peer by default. Forwarding headers are considered
only when both trusted proxy hops and trusted peer networks are configured.
Transient Redis outages fail open; unexpected implementation errors propagate.
"""

from app.core.redis_rate_limiter.endpoint_limiters import (
    auth_token_generate_rate_limiter,
    auth_token_refresh_rate_limiter,
    token_acquire_rate_limiter,
)
from app.core.redis_rate_limiter.moving_window_limiter import (
    RateLimiterManager,
    rate_limiter_manager,
)
from app.core.redis_rate_limiter.rate_limit_enforcement import (
    rate_limit_dependency,
    register_rate_limit_exception_handler,
)
from app.core.redis_rate_limiter.rate_limit_keys import get_client_ip, ip_only_key

__all__ = [
    "RateLimiterManager",
    "auth_token_generate_rate_limiter",
    "auth_token_refresh_rate_limiter",
    "get_client_ip",
    "ip_only_key",
    "rate_limit_dependency",
    "rate_limiter_manager",
    "register_rate_limit_exception_handler",
    "token_acquire_rate_limiter",
]
