"""
Uvicorn Startup Script
----------------------
FastAPI application startup script.
"""

import sys
import os

# Add project root to Python path and import sitecustomize BEFORE anything else
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import sitecustomize to set Windows event loop policy

import uvicorn
from app.core.config_manager import settings


if __name__ == "__main__":
    # Bind to 0.0.0.0 for external access, but display localhost URLs for local access
    # This ensures the server accepts connections from all interfaces while showing
    # reliable localhost URLs that work consistently on Windows
    uvicorn.run(
        app="app.app:app",  # "app.app:app" means: import the variable `app` from the module `app/app.py` (i.e. `app/app.py` file), so FastAPI/Uvicorn knows where to find the ASGI app instance.
        host="0.0.0.0",  # Bind to all interfaces for external access
        port=settings.fastapi_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
