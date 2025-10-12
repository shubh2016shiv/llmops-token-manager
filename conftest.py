"""
Pytest configuration for LLM Token Manager tests.
Sets up the Python path and common test fixtures.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for all tests
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up test environment variables
os.environ.setdefault("DATABASE_HOST", "localhost")
os.environ.setdefault("DATABASE_PORT", "5432")
os.environ.setdefault("DATABASE_NAME", "mydb")
os.environ.setdefault("DATABASE_USER", "myuser")
os.environ.setdefault("DATABASE_PASSWORD", "mypassword")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
