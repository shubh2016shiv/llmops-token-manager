# Redis rate limiter

FastAPI dependencies use the `limits` asynchronous moving-window strategy.
`RateLimiterManager` creates storage during application startup and closes
its Redis pool at shutdown. Rule factories in `endpoint_limiters.py` supply
limits and bucket keys; `rate_limit_enforcement.py` returns HTTP 429 with
`Retry-After` when a bucket is exhausted.

The default bucket identity is the TCP peer address. To honor
`X-Forwarded-For`, configure both `rate_limit_trusted_proxy_hops` and
`rate_limit_trusted_proxy_networks`. Only configured proxy peers can supply
forwarded addresses; malformed chains fall back to the peer. The unverified
`X-Service-Id` header never creates a separate acquisition bucket.

Transient Redis connection and timeout errors fail open and are logged.
Unexpected errors propagate, so implementation failures cannot silently
disable rate limiting. If the manager was not initialized, requests fail
loudly instead of bypassing the limiter. For isolated tests, set
`RATE_LIMIT_STORAGE=memory` before initializing the manager.
