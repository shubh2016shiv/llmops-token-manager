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
from app.api import health_endpoints, user_endpoints


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.

    Startup:
        - Initialize database connection pool
        - Initialize Redis connection pool
        - Log application start

    Shutdown:
        - Close database connections
        - Close Redis connections
        - Log application shutdown
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    try:
        # Initialize database
        await db_manager.initialize()
        logger.info("Database initialized")

        # Initialize Redis
        redis_manager.initialize()
        logger.info("Redis initialized")

        # Display service URLs in formatted tables
        # Use print for properly aligned tables, with consistent width
        border_line = "═" * 80
        header_line = "─" * 80

        print("\n" + border_line)
        print("SERVICE ENDPOINTS & CONNECTION INFORMATION")
        print(border_line)

        # FastAPI URLs table - properly aligned with fixed width columns
        # Show localhost URLs for local access (works reliably on Windows)
        local_api_base = f"http://localhost:{settings.fastapi_port}"
        print("FASTAPI SERVICE")
        print(header_line)
        print(f"{'Service':<20} | {'URL':<57}")
        print(f"{header_line}")
        print(f"{'Main API':<20} | {local_api_base + '/':<57}")
        print(f"{'API Documentation':<20} | {local_api_base + '/api/docs':<57}")
        print(f"{'ReDoc Interface':<20} | {local_api_base + '/api/redoc':<57}")
        print(f"{'OpenAPI Schema':<20} | {local_api_base + '/api/openapi.json':<57}")
        print(f"{'Health Check':<20} | {local_api_base + '/api/v1/health':<57}")
        print(header_line)

        # Database connection table - properly aligned
        print("\nPOSTGRESQL DATABASE")
        print(header_line)
        print(f"{'Parameter':<20} | {'Value':<57}")
        print(f"{header_line}")
        print(f"{'Host':<20} | {settings.database_host:<57}")
        print(f"{'Port':<20} | {str(settings.database_port):<57}")
        print(f"{'Database':<20} | {settings.database_name:<57}")
        print(
            f"{'Connection Pool':<20} | {f'{settings.database_pool_size} connections (+ {settings.database_max_overflow} overflow)':<57}"
        )
        print(header_line)

        # Redis connection table - properly aligned
        print("\nREDIS CACHE")
        print(header_line)
        print(f"{'Parameter':<20} | {'Value':<57}")
        print(f"{header_line}")
        print(f"{'Host':<20} | {settings.redis_host:<57}")
        print(f"{'Port':<20} | {str(settings.redis_port):<57}")
        print(f"{'Database':<20} | {str(settings.redis_db):<57}")
        print(f"{'Max Connections':<20} | {str(settings.redis_max_connections):<57}")
        print(header_line)
        print(border_line + "\n")

        # Still log a simple message for the log file
        logger.info("Service endpoints and connection information displayed")

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")

    try:
        await db_manager.close()
        logger.info("Database connections closed")

        await redis_manager.close()
        logger.info("Redis connections closed")

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
# app.include_router(llm_endpoints.router)
app.include_router(user_endpoints.router)


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
