"""Package-scoped constants for app.core.config."""

import os
from pathlib import Path

CONFIG_YAML_DIR = Path(os.environ.get("CONFIG_YAML_DIR", "app/core/config/yaml"))
