"""
FastAPI Application Entry Point
-------------------------------
Main application initialization and configuration.
Registers routers, middleware, and lifecycle handlers.
"""

# Import everything else
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config_manager import settings
from app.core.database_connection import db_manager
from app.core.redis_connection import redis_manager
from app.core.startup_diagnostics import (
    display_startup_failure,
    display_service_info,
    verify_database_connectivity,
    verify_redis_connectivity,
    verify_rabbitmq_connectivity,
)
from app.api import (
    health_endpoints,
    user_endpoints,
    llm_configuration_endpoints,
    token_manager_endpoints,
    user_entitlement_endpoints,
)
from app.api.auth_endpoints import router as auth_router


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
        logger.info("[SUCCESS] RabbitMQ connected and ready")
    else:
        logger.error(f"[FAILED] RabbitMQ: {rabbitmq_status.error_message}")

    # Check if any service failed
    failed_services = [s for s in service_statuses if s.status == "failed"]

    if failed_services:
        display_startup_failure(failed_services)
        logger.error(
            f"Application startup failed: {len(failed_services)} service(s) unavailable"
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
    description="Production-ready LLM token management system with multi-provider support",
    lifespan=lifespan,
    debug=settings.debug,
    # Enable Swagger UI and ReDoc in development, configurable for production
    docs_url="/api/docs",  # Professional API path
    redoc_url="/api/redoc",  # Professional API path
    openapi_url="/api/openapi.json",  # Professional API path
    swagger_ui_parameters={"displayRequestDuration": True},  # Enhanced Swagger UI
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_endpoints.router)
app.include_router(auth_router)  # JWT authentication endpoints
app.include_router(llm_configuration_endpoints.router)
app.include_router(user_endpoints.router)
app.include_router(user_entitlement_endpoints.router)  # User LLM entitlements
app.include_router(token_manager_endpoints.router)


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
