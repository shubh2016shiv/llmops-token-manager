"""
Uvicorn startup script.

FastAPI application startup script.
"""

import os
import sys

# Add project root to Python path and import sitecustomize BEFORE anything else
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
