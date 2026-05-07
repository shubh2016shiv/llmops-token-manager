"""
Run pyright only on pre-commit staged Python files.

Why this exists:
- pre-commit passes staged paths to hooks, but `pyright --project ...` can still
  expand analysis scope.
- This wrapper enforces an explicit file list, so type-checking is limited to the
  to-be-committed Python files.
- It also removes any file matched by `.gitignore` using `git check-ignore`.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DEFAULT_PYRIGHT_INCLUDES = ["app"]
DEFAULT_PYRIGHT_EXCLUDES = [
    "**/__pycache__",
    "**/.venv",
    "**/venv",
    "**/migrations",
    "tests",
    "**/*_DELETE.py",
    "**/*_OBSELETE.py",
]


def _load_pyright_config() -> tuple[list[str], list[str]]:
    """Load include/exclude globs from pyproject's pyright section."""
    if tomllib is None or not PYPROJECT_PATH.exists():
        return DEFAULT_PYRIGHT_INCLUDES, DEFAULT_PYRIGHT_EXCLUDES

    with PYPROJECT_PATH.open("rb") as pyproject_file:
        config = tomllib.load(pyproject_file)

    pyright_config = config.get("tool", {}).get("pyright", {})
    includes = pyright_config.get("include") or DEFAULT_PYRIGHT_INCLUDES
    excludes = pyright_config.get("exclude") or DEFAULT_PYRIGHT_EXCLUDES
    return includes, excludes


def _is_git_ignored(path: str) -> bool:
    """Return True if path is ignored by gitignore rules."""
    result = subprocess.run(
        ["git", "check-ignore", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _is_python_file(path: str) -> bool:
    """Return True for existing .py files."""
    p = REPO_ROOT / path
    return p.suffix == ".py" and p.exists()


def _matches_any_glob(path: str, patterns: list[str]) -> bool:
    """Return True if the POSIX-style relative path matches any configured glob."""
    normalized_path = Path(path).as_posix()
    return any(fnmatch(normalized_path, pattern) for pattern in patterns)


def _is_in_pyright_scope(path: str, includes: list[str], excludes: list[str]) -> bool:
    """Respect the repository's pyright include/exclude settings."""
    normalized_path = Path(path).as_posix()

    include_patterns = []
    for include in includes:
        include_path = Path(include).as_posix().rstrip("/")
        include_patterns.extend([include_path, f"{include_path}/**"])

    if include_patterns and not _matches_any_glob(normalized_path, include_patterns):
        return False

    return not _matches_any_glob(normalized_path, excludes)


def main() -> int:
    """
    Run pyright on staged, non-ignored Python files only.

    pre-commit injects candidate filenames as CLI args.
    """
    includes, excludes = _load_pyright_config()
    staged_candidates = sys.argv[1:]
    python_files = [p for p in staged_candidates if _is_python_file(p)]
    target_files = [
        p
        for p in python_files
        if not _is_git_ignored(p) and _is_in_pyright_scope(p, includes, excludes)
    ]

    if not target_files:
        print("pyright (staged): no eligible Python files to check.")
        return 0

    cmd = ["uv", "run", "pyright", *target_files]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
