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
from app.core.config_manager import settings
from app.core.correlation_id import correlation_id_middleware
from app.core.database_connection import db_manager
from app.core.rate_limiter import register_rate_limit_exception_handler
from app.core.redis_connection import redis_manager
from app.core.startup_diagnostics import (
    display_service_info,
    display_startup_failure,
    verify_celery_worker_readiness,
    verify_database_connectivity,
    verify_rabbitmq_connectivity,
    verify_redis_connectivity,
)

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
    logger.info("[SUCCESS] Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")
    try:
        await db_manager.close()
        await redis_manager.close()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


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
