"""
Local development launcher.

PRODUCTION PATTERN: entrypoint vs. composition root.
-----------------------------------------------------
This file is a *dev-only launcher*, not the application. The actual FastAPI
app is built in `app/app.py` (the composition root: it creates the FastAPI()
instance and wires lifespan, middleware, and routers in one place). Both this
script and the container's `uvicorn` command below point at the same target,
`app.app:app`, so there is exactly one place the app is assembled — dev and
prod can never wire it differently by accident.

Why this file exists at all: it saves a developer from typing out uvicorn
flags (host/port/reload/log-level) by hand every time, sourcing them from
`app/core/config.py` instead. In production this file is not used and is not
even shipped: `app/Dockerfile`'s CMD calls `uvicorn app.app:app` directly,
and `.dockerignore` excludes `index.py` from the build context. This mirrors
the standard split seen in other frameworks (e.g. Django's `manage.py` vs.
`wsgi.py`/`asgi.py`): a throwaway CLI convenience script for local iteration,
separate from the composition root that production actually runs.
"""


def run_server() -> None:
    """Start the FastAPI app with Uvicorn."""
    import uvicorn

    from app.core.config import settings

    # Bind to 0.0.0.0 for external access, but display localhost URLs for local access
    # This ensures the server accepts connections from all interfaces while showing
    # reliable localhost URLs that work consistently on Windows
    uvicorn.run(
        # Import `app` from `app/app.py` as the ASGI app entry point.
        app="app.app:app",
        host="0.0.0.0",  # Bind to all interfaces for external access
        port=settings.fastapi_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
