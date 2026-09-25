"""
FastAPI Application Entry Point — the composition root.

PRODUCTION PATTERN: this is the ONLY place the FastAPI() instance is built.
Everything that runs the app — the local dev launcher (`index.py`, not
shipped to production) and the container's `uvicorn` CMD (see
`app/Dockerfile`) — both target the same object: `app.app:app`. Centralizing
assembly here means there is exactly one wiring of middleware/routers/
lifespan to reason about; dev and prod are structurally guaranteed to run
the identical app, not two hand-maintained copies that can drift apart.

Main application initialization and configuration.
Registers routers, middleware, and lifecycle handlers.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import api_router
from app.core.config import settings
from app.core.database import db_manager
from app.core.redis import redis_manager
from app.core.redis_rate_limiter import (
    rate_limiter_manager,
    register_rate_limit_exception_handler,
)
from app.core.request_tracing import correlation_id_middleware
from app.core.service_health import (
    display_service_info,
    display_startup_failure,
    verify_database_connectivity,
    verify_redis_connectivity,
)

# Gateway-owned Celery app/health checks disabled: the token manager is a
# ledger — it never fires LLM calls and no longer shares llm_gateway's Celery app.
# from app.llm_client_provisioning.llm_client_request_queue import celery_app
# from app.llm_client_provisioning.service_health import (
#     display_provisioning_service_info,
#     verify_celery_worker_readiness,
#     verify_rabbitmq_connectivity,
# )
from app.persistence.token_maintenance import TokenMaintenancePersistence
from app.resilience.circuit_breaker import close_circuit_breaker_redis_client
from app.resilience.redis_token_counter import (
    close_shared_redis_token_counter_service,
    get_shared_redis_token_counter_service,
)
from app.resilience.token_maintenance.health import (
    verify_token_maintenance_readiness,
)
from app.resilience.token_maintenance.runner import maintenance_runner
from app.resilience.token_queue import declare_token_queues

# -----------------------------------------------------------------------------
# APP BOOTSTRAP EXPLANATION (for future maintainers)
# -----------------------------------------------------------------------------
# "Application bootstrap" is the startup wiring layer where we assemble the app:
# - create the FastAPI instance
# - register middleware
# - register routers/endpoints
# - define lifecycle hooks (startup/shutdown)
#
# In enterprise systems, bootstrap should stay thin and declarative:
# - it should compose modules, not contain business logic
# - it should be easy to scan and reason about quickly
# - it should minimize edit hotspots that cause merge conflicts
#
# This project follows an "aggregated router" pattern:
# - `app.api` exposes a single `api_router` that already includes all endpoint routers
# - bootstrap performs one include: `app.include_router(api_router)`
#
# Why this is considered an enterprise best practice:
# 1) Consistency: route registration happens in one dedicated API package module.
# 2) Maintainability: new endpoint modules usually only touch `app/api/__init__.py`.
# 3) Safety: inclusion order is centralized, helping avoid FastAPI route precedence
#    surprises.
# 4) Readability: bootstrap remains focused on system assembly (middleware + lifecycle).
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with graceful error handling."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    service_statuses = []

    # Check PostgreSQL
    logger.info("Checking PostgreSQL connectivity...")
    await db_manager.initialize()
    postgres_status = await verify_database_connectivity()
    service_statuses.append(postgres_status)

    if postgres_status.status == "connected":
        logger.info("[SUCCESS] PostgreSQL connected and ready")
    else:
        logger.error("[FAILED] PostgreSQL connectivity check")

    # Check Redis
    logger.info("Checking Redis connectivity...")
    redis_manager.initialize()
    redis_status = await verify_redis_connectivity()
    service_statuses.append(redis_status)

    if redis_status.status == "connected":
        logger.info("[SUCCESS] Redis connected and ready")
    else:
        logger.error("[FAILED] Redis connectivity check")

    # Initialize the rate limiter INSIDE the serving event loop.
    # This binds the coredis connection pool to the loop that will handle
    # requests (see RateLimiterManager). A failure here is a permanent
    # misconfiguration (missing/incompatible coredis driver, bad DSN), never a
    # transient outage — so we let it abort startup rather than boot an app
    # whose rate limiter silently fails open on every request.
    logger.info("Initializing rate limiter...")
    try:
        rate_limiter_manager.initialize()
        logger.info("[SUCCESS] Rate limiter initialized")
    except Exception:
        logger.error(
            "[FAILED] Rate limiter initialization failed. This is a deploy-time "
            "driver/config fault, not a transient outage"
        )
        await _close_runtime_resources()
        raise

    logger.info("Checking token maintenance readiness...")
    token_maintenance_status = await verify_token_maintenance_readiness()
    service_statuses.append(token_maintenance_status)

    if token_maintenance_status.status == "connected":
        logger.info("[SUCCESS] Token maintenance runtime ready")
    else:
        logger.error("[FAILED] Token maintenance schedule is not ready")

    startup_blockers = [
        service for service in service_statuses if service.status == "failed"
    ]

    if startup_blockers:
        display_startup_failure(startup_blockers)
        logger.error(
            "Application startup failed: "
            f"{len(startup_blockers)} service(s) unavailable"
        )
        await _close_runtime_resources()
        raise RuntimeError("Required startup dependencies are unavailable")

    # All services connected - display success info
    display_service_info()

    # ----------------------------------------------------------------
    # Resilience layer startup
    # ----------------------------------------------------------------
    # 1. Declare RabbitMQ token allocation queues (idempotent)
    logger.info("Declaring token allocation queues...")
    try:
        declare_token_queues()
        logger.info("[SUCCESS] Token allocation queues declared")
    except Exception:
        logger.warning("[DEGRADED] Token queue declaration failed")

    # 2. Seed Redis token counters from PostgreSQL ground truth
    logger.info("Seeding Redis token counters from PostgreSQL...")
    try:
        await _seed_token_counters()
        logger.info("[SUCCESS] Redis token counters seeded")
    except Exception:
        logger.warning(
            "[DEGRADED] Token counter seeding failed; reconciliation will retry"
        )

    # 3. Run reconciliation, queue-depth publishing, and expiry cleanup in
    # the serving loop. The runner owns cancellation during shutdown.
    try:
        maintenance_runner.start()
    except Exception:
        await _close_runtime_resources()
        raise

    logger.info("[SUCCESS] Application startup complete")

    try:
        yield
    finally:
        logger.info("Shutting down application")
        await _close_runtime_resources()
        logger.info("Application shutdown complete")


async def _close_runtime_resources() -> None:
    """Close independent clients even if one close fails during shutdown."""
    results = await asyncio.gather(
        maintenance_runner.close(),
        rate_limiter_manager.close(),
        db_manager.close(),
        redis_manager.close(),
        close_shared_redis_token_counter_service(),
        return_exceptions=True,
    )
    for resource, result in zip(
        ("maintenance", "rate limiter", "database", "redis", "counter"),
        results,
        strict=True,
    ):
        if isinstance(result, BaseException):
            # Exception CLASS name only, never str(exc) — a close() failure's
            # message can carry connection details (same discipline as
            # token_maintenance/runner.py's job-failure logging). The class
            # name alone is still enough to tell "ConnectionError" apart from
            # "TimeoutError" in the logs without that risk.
            logger.error("{} shutdown failed: {}", resource, type(result).__name__)
    try:
        close_circuit_breaker_redis_client()
    except Exception as exc:
        logger.error("Circuit-breaker client shutdown failed: {}", type(exc).__name__)


async def _seed_token_counters() -> None:
    """
    Seed Redis token counters with current PostgreSQL allocation sums.

    Queries all active (ACQUIRED + PAUSED) allocations grouped by model/endpoint
    and seeds the Redis fast-path counters so the first requests after startup
    use the fast path immediately (no cold-start DB read cascade).

    A failed startup seed is retried by the in-process reconciliation job.
    """
    shared_token_counter_service = get_shared_redis_token_counter_service()
    maintenance_persistence = TokenMaintenancePersistence()
    invalid_active_models = (
        await maintenance_persistence.list_invalid_active_models_without_capacity()
    )
    for invalid_model in invalid_active_models:
        logger.error(
            "Active deployment is missing max_tokens "
            "and is excluded from startup counter seeding",
            llm_provider=invalid_model.llm_provider,
            llm_model_name=invalid_model.llm_model_name,
            api_endpoint_url=invalid_model.api_endpoint_url,
            deployment_name=invalid_model.deployment_name,
            deployment_region=invalid_model.deployment_region,
        )
    seed_records = await maintenance_persistence.list_startup_counter_seed_snapshots()

    seeded = 0
    for seed_record in seed_records:
        await shared_token_counter_service.seed_counter(
            model_name=seed_record.llm_model_name,
            api_endpoint_url=seed_record.api_endpoint_url,
            current_allocated=seed_record.allocated_tokens,
            max_limit=seed_record.max_tokens,
        )
        seeded += 1

    logger.info(f"Seeded {seeded} Redis token counter(s) from PostgreSQL")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=("LLM token management system with multi-provider support"),
    lifespan=lifespan,
    debug=settings.debug,
    # Enable Swagger UI and ReDoc in development, configurable for production
    docs_url="/api/docs",  # Professional API path
    redoc_url="/api/redoc",  # Professional API path
    openapi_url="/api/openapi.json",  # Professional API path
    swagger_ui_parameters={"displayRequestDuration": True},  # Enhanced Swagger UI
)

# Register standardized rate-limit error handler.
register_rate_limit_exception_handler(app)

# CORS middleware.
#
# allow_credentials is deliberately False, not the more common-looking True:
# this API has no cookie/session auth anywhere (every caller authenticates
# via a bearer JWT or an X-Service-Id header, both ordinary headers covered
# by allow_headers) — grepping the codebase turns up zero set_cookie or
# SessionMiddleware usage. allow_origins=["*"] combined with
# allow_credentials=True is a well-known antipattern: browsers require it
# (per the CORS spec, "*" + credentials is actually disallowed), so
# Starlette's CORSMiddleware falls back to echoing the request's own Origin
# header back as an exact match — which, combined with a wildcard allowlist,
# means it accepts a credentialed cross-origin request from literally any
# origin. Since nothing here needs credentialed CORS, disabling it removes
# the antipattern instead of requiring us to guess at a real origin
# allowlist we don't have visibility into.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID middleware. Registered AFTER CORSMiddleware so it becomes
# the OUTERMOST layer: Starlette builds the middleware stack in reverse
# registration order (the middleware added last wraps everything added
# before it — verified against Starlette's actual dispatch order, not
# assumed), so this is what actually makes every response — including a
# CORS preflight short-circuited by CORSMiddleware before reaching any
# route — carry an X-Correlation-Id and get logged with request-tracing
# context. Registering it first (as this file used to) put it INSIDE
# CORSMiddleware instead, the opposite of the "wraps all routes/middleware"
# intent its own docstring states.
app.middleware("http")(correlation_id_middleware)

# Register routers
# Enterprise pattern: register one aggregated API router from `app.api`.
app.include_router(api_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/api/docs",  # Always show docs path
        "redoc": "/api/redoc",
        "openapi": "/api/openapi.json",
    }
