"""
FastAPI Application Entry Point.

Main application initialization and configuration.
Registers routers, middleware, and lifecycle handlers.
"""

# Import everything else
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import api_router
from app.core.config import settings
from app.core.database import db_manager
from app.core.rate_limiter import register_rate_limit_exception_handler
from app.core.redis import redis_manager
from app.core.request_tracing import correlation_id_middleware
from app.core.service_health import (
    display_service_info,
    display_startup_failure,
    verify_database_connectivity,
    verify_redis_connectivity,
)
from app.llm_client_provisioning.llm_client_request_queue import celery_app
from app.llm_client_provisioning.service_health import (
    display_provisioning_service_info,
    verify_celery_worker_readiness,
    verify_rabbitmq_connectivity,
)
from app.persistence.token_maintenance_persistence import TokenMaintenancePersistence
from app.resilience.circuit_breaker import close_circuit_breaker_redis_client
from app.resilience.redis_token_counter import (
    close_shared_redis_token_counter_service,
    get_shared_redis_token_counter_service,
)
from app.resilience.token_maintenance.schedule_registry import register_beat_schedule
from app.resilience.token_maintenance.service_health import (
    verify_token_maintenance_readiness,
)
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
        logger.error(f"[FAILED] PostgreSQL: {postgres_status.error_message}")

    # Check Redis
    logger.info("Checking Redis connectivity...")
    redis_manager.initialize()
    redis_status = await verify_redis_connectivity()
    service_statuses.append(redis_status)

    if redis_status.status == "connected":
        logger.info("[SUCCESS] Redis connected and ready")
    else:
        logger.error(f"[FAILED] Redis: {redis_status.error_message}")

    # Check RabbitMQ
    logger.info("Checking RabbitMQ connectivity...")
    rabbitmq_status = await verify_rabbitmq_connectivity()
    service_statuses.append(rabbitmq_status)

    if rabbitmq_status.status == "connected":
        logger.info("[SUCCESS] RabbitMQ broker connected and ready")
    else:
        logger.error(f"[FAILED] RabbitMQ: {rabbitmq_status.error_message}")

    # Check Celery worker readiness separately from broker connectivity.
    logger.info("Checking Celery worker readiness...")
    celery_worker_status = await verify_celery_worker_readiness()
    service_statuses.append(celery_worker_status)

    if celery_worker_status.status == "connected":
        logger.info("[SUCCESS] Celery worker ready for async execution")
    elif settings.require_celery_worker_on_startup:
        logger.error(f"[FAILED] Celery worker: {celery_worker_status.error_message}")
    else:
        logger.warning(
            "[DEGRADED] Celery worker unavailable: "
            f"{celery_worker_status.error_message}. "
            "FastAPI will continue startup because "
            "REQUIRE_CELERY_WORKER_ON_STARTUP is false."
        )

    logger.info("Checking token maintenance readiness...")
    token_maintenance_status = await verify_token_maintenance_readiness()
    service_statuses.append(token_maintenance_status)

    if token_maintenance_status.status == "connected":
        logger.info("[SUCCESS] Token maintenance runtime ready")
    else:
        logger.error(
            f"[FAILED] Token maintenance: {token_maintenance_status.error_message}"
        )

    startup_blockers = [
        service
        for service in service_statuses
        if service.status == "failed"
        and (
            service.name != "Celery worker" or settings.require_celery_worker_on_startup
        )
    ]

    if startup_blockers:
        display_startup_failure(startup_blockers)
        logger.error(
            "Application startup failed: "
            f"{len(startup_blockers)} service(s) unavailable"
        )
        import os

        os._exit(1)  # Exit immediately without traceback

    # All services connected - display success info
    display_service_info()
    display_provisioning_service_info()

    # ----------------------------------------------------------------
    # Resilience layer startup
    # ----------------------------------------------------------------
    # 1. Declare RabbitMQ token allocation queues (idempotent)
    logger.info("Declaring token allocation queues...")
    try:
        declare_token_queues()
        logger.info("[SUCCESS] Token allocation queues declared")
    except Exception as e:
        logger.warning(f"[DEGRADED] Token queue declaration failed (non-fatal): {e}")

    # 2. Seed Redis token counters from PostgreSQL ground truth
    logger.info("Seeding Redis token counters from PostgreSQL...")
    try:
        await _seed_token_counters()
        logger.info("[SUCCESS] Redis token counters seeded")
    except Exception as e:
        logger.warning(
            f"[DEGRADED] Token counter seeding failed (non-fatal — "
            f"will self-correct on first reconcile run): {e}"
        )

    # 3. Register Celery beat schedule for periodic reconciliation tasks
    register_beat_schedule(celery_app)
    logger.info("[SUCCESS] Celery beat schedule registered")

    logger.info("[SUCCESS] Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")
    try:
        await db_manager.close()
        await redis_manager.close()
        await close_shared_redis_token_counter_service()
        close_circuit_breaker_redis_client()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


async def _seed_token_counters() -> None:
    """
    Seed Redis token counters with current PostgreSQL allocation sums.

    Queries all active (ACQUIRED + PAUSED) allocations grouped by model/endpoint
    and seeds the Redis fast-path counters so the first requests after startup
    use the fast path immediately (no cold-start DB read cascade).

    Non-fatal: if PostgreSQL or Redis are unavailable at this point,
    the counters will be seeded lazily by the first reconcile beat task.
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
    description=(
        "Production-ready LLM token management system with multi-provider support"
    ),
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

# Correlation ID middleware (register early so it wraps all routes/middleware)
app.middleware("http")(correlation_id_middleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
