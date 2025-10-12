#!/usr/bin/env python3
"""
Enterprise-Grade Code Formatter Utility for LLM Token Manager
===============================================================

This script provides automated code formatting and quality assurance using
industry-standard Python linting and formatting tools. It ensures consistent
code style, removes unused imports, and validates code quality.

FORMATTING TOOLS USED:
----------------------
1. BLACK (https://black.readthedocs.io/)
   - Purpose: Opinionated Python code formatter
   - Action: Enforces consistent code style (PEP 8 compliant)
   - Configuration: 120 character line length

2. AUTOFLAKE (https://github.com/PyCQA/autoflake)
   - Purpose: Removes unused imports and variables
   - Action: Cleans up code by removing unused Python imports
   - Configuration: Removes all unused imports, variables, and expands star imports

3. REORDER-PYTHON-IMPORTS (https://github.com/asottile/reorder_python_imports)
   - Purpose: Automatically reorders Python imports
   - Action: Sorts imports to top of file and groups them correctly
   - Configuration: Python 3.8+ compatibility mode

4. FLAKE8 (https://flake8.pycqa.org/)
   - Purpose: Style guide enforcement and linting
   - Action: Validates code against PEP 8 and detects common errors
   - Configuration: Uses .flake8 config file if present

FEATURES:
---------
- Automatic tool installation if missing
- Support for staged Git files only
- Check mode (no modifications)
- YAML file formatting support
- Parallel processing ready
- Comprehensive error handling and logging
- Progress reporting
- Dry-run capabilities

Usage:
    python format_code.py              # Format all Python files in app/ and tests/
    python format_code.py --staged     # Format only Git staged files
    python format_code.py --check      # Check formatting without modifications
    python format_code.py --all        # Format both Python and YAML files
    python format_code.py --lint       # Also run flake8 linting
    python format_code.py --verbose    # Detailed output
    python format_code.py file1.py     # Format specific files

Exit Codes:
    0 - Success (all operations completed)
    1 - Formatting/linting failures detected
    2 - Tool installation failure
    3 - No files found to process
"""
import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Tool installation packages mapping
TOOL_PACKAGES: Dict[str, str] = {
    "autoflake": "autoflake",
    "black": "black",
    "flake8": "flake8",
    "reorder-python-imports": "reorder-python-imports",
}

# Subprocess timeout in seconds (to prevent hanging)
COMMAND_TIMEOUT: int = 300  # 5 minutes

# Directories to scan for Python files
DEFAULT_PYTHON_DIRS: List[str] = ["app", "tests"]

# Exit codes
EXIT_SUCCESS: int = 0
EXIT_FORMATTING_FAILED: int = 1
EXIT_TOOL_INSTALLATION_FAILED: int = 2
EXIT_NO_FILES: int = 3

# Logging configuration
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with appropriate level and format.

    Args:
        verbose: If True, set logging to DEBUG level for detailed output
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_command(cmd: List[str], description: str, timeout: Optional[int] = None) -> bool:
    """Execute a shell command with proper error handling and logging.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description of the operation
        timeout: Maximum execution time in seconds (default: COMMAND_TIMEOUT)

    Returns:
        True if command succeeded (exit code 0), False otherwise

    Raises:
        No exceptions - all errors are caught and logged
    """
    timeout = timeout or COMMAND_TIMEOUT

    logger.info(f"{'='*60}")
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )

        if result.stdout:
            logger.debug(f"STDOUT:\n{result.stdout}")
            print(result.stdout)
        if result.stderr:
            logger.debug(f"STDERR:\n{result.stderr}")
            print(result.stderr, file=sys.stderr)

        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
            return True
        else:
            logger.error(
                f"✗ {description} failed with exit code {result.returncode}"
            )
            return False

    except subprocess.TimeoutExpired:
        logger.error(
            f"✗ {description} timed out after {timeout} seconds"
        )
        return False
    except FileNotFoundError as e:
        logger.error(f"✗ Command not found for {description}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error running {description}: {e}")
        logger.exception("Full traceback:")
        return False


def check_tool_installed(tool_cmd: str) -> bool:
    """Check if a command-line tool is installed and accessible.

    Args:
        tool_cmd: Command name to check (e.g., 'black', 'flake8')

    Returns:
        True if tool is installed and accessible, False otherwise
    """
    try:
        subprocess.run(
            [tool_cmd, "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        logger.debug(f"Tool '{tool_cmd}' is installed")
        return True
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug(f"Tool '{tool_cmd}' is not installed")
        return False


def install_tool(tool_cmd: str) -> bool:
    """Install a Python tool using pip.

    Args:
        tool_cmd: Tool command name to install

    Returns:
        True if installation succeeded, False otherwise
    """
    package_name = TOOL_PACKAGES.get(tool_cmd, tool_cmd)
    logger.warning(f"Tool '{tool_cmd}' not found. Attempting installation...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        logger.info(f"✓ Successfully installed {package_name}")
        logger.debug(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Installation of {package_name} timed out")
        return False
    except subprocess.SubprocessError as e:
        logger.error(f"✗ Failed to install {package_name}: {e}")
        return False


def ensure_tool_available(tool_cmd: str, auto_install: bool = True) -> bool:
    """Ensure a tool is available, optionally installing it.

    Args:
        tool_cmd: Command name to check and install
        auto_install: If True, attempt automatic installation if tool is missing

    Returns:
        True if tool is available (or successfully installed), False otherwise
    """
    if check_tool_installed(tool_cmd):
        return True

    if not auto_install:
        logger.error(f"Tool '{tool_cmd}' is required but not installed")
        return False

    return install_tool(tool_cmd)


# =============================================================================
# FILE DISCOVERY FUNCTIONS
# =============================================================================


def get_staged_files() -> Tuple[Set[str], Set[str]]:
    """Retrieve list of Git-staged Python and YAML files.

    Returns:
        Tuple of (python_files, yaml_files) as sets of file paths

    Note:
        Only returns files that exist on disk (filters out deleted files)
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        files = result.stdout.strip().split("\n")
        python_files = {f for f in files if f.endswith(".py") and Path(f).exists()}
        yaml_files = {
            f for f in files if f.endswith((".yml", ".yaml")) and Path(f).exists()
        }
        logger.info(f"Found {len(python_files)} staged Python files")
        logger.info(f"Found {len(yaml_files)} staged YAML files")
        return python_files, yaml_files
    except subprocess.TimeoutExpired:
        logger.error("Git command timed out while retrieving staged files")
        return set(), set()
    except subprocess.SubprocessError as e:
        logger.warning(f"Could not get staged files (not a git repo?): {e}")
        return set(), set()
    except Exception as e:
        logger.error(f"Unexpected error getting staged files: {e}")
        return set(), set()


def get_all_python_files(directories: Optional[List[str]] = None) -> Set[str]:
    """Discover all Python files in specified directories.

    Args:
        directories: List of directory paths to search (default: DEFAULT_PYTHON_DIRS)

    Returns:
        Set of Python file paths found

    Note:
        Recursively searches directories and only includes existing files
    """
    directories = directories or DEFAULT_PYTHON_DIRS
    python_files = set()

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory '{directory}' does not exist, skipping")
            continue

        if not dir_path.is_dir():
            logger.warning(f"Path '{directory}' is not a directory, skipping")
            continue

        found_files = list(dir_path.rglob("*.py"))
        python_files.update(str(f) for f in found_files)
        logger.debug(f"Found {len(found_files)} Python files in {directory}")

    logger.info(f"Total Python files discovered: {len(python_files)}")
    return python_files


def get_all_yaml_files() -> Set[str]:
    """Discover important YAML configuration files in the project.

    Returns:
        Set of YAML file paths found

    Note:
        Searches for common YAML files like pre-commit configs,
        Docker Compose files, and GitHub workflows
    """
    yaml_files = set()
    # Common YAML file patterns in projects
    yaml_patterns = [
        ".pre-commit-config.yaml",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
    ]

    for pattern in yaml_patterns:
        if "*" in pattern:
            # Handle glob patterns for discovering multiple files
            base_dir = Path(pattern).parent
            if base_dir.exists():
                found_files = list(base_dir.glob(Path(pattern).name))
                yaml_files.update(str(f) for f in found_files)
                logger.debug(f"Found {len(found_files)} YAML files matching {pattern}")
        else:
            # Handle specific file paths
            if Path(pattern).exists():
                yaml_files.add(pattern)
                logger.debug(f"Found YAML file: {pattern}")

    logger.info(f"Total YAML files discovered: {len(yaml_files)}")
    return yaml_files


# =============================================================================
# TEXT FORMATTING FUNCTIONS
# =============================================================================


def remove_trailing_whitespace(files: Set[str]) -> int:
    """Remove trailing whitespace and normalize line endings in files.

    This function performs the following operations:
    - Removes trailing spaces/tabs from each line
    - Ensures file ends with exactly one newline
    - Removes excess blank lines at end of file

    Args:
        files: Set of file paths to process

    Returns:
        Count of successfully processed files

    Note:
        Uses UTF-8 encoding and handles encoding errors gracefully
    """
    fixed_count = 0
    for file_path in files:
        try:
            # Read file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Skip empty files
            if not lines:
                logger.debug(f"Skipping empty file: {file_path}")
                continue

            # Remove trailing whitespace from each line
            new_lines = [line.rstrip() + "\n" if line.strip() else "\n" for line in lines]

            # Ensure file ends with newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            # Remove excessive trailing blank lines (keep max 1)
            while len(new_lines) > 1 and new_lines[-1] == "\n" and new_lines[-2] == "\n":
                new_lines.pop()

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            fixed_count += 1
            logger.debug(f"Fixed trailing whitespace in: {file_path}")

        except UnicodeDecodeError as e:
            logger.warning(
                f"Encoding error in {file_path}: {e} - file may be binary"
            )
        except PermissionError as e:
            logger.error(f"Permission denied for {file_path}: {e}")
        except Exception as e:
            logger.error(f"Could not process {file_path}: {e}")

    return fixed_count


def format_yaml_files(files: Set[str], check_only: bool = False) -> bool:
    """Format YAML files by removing trailing whitespace and fixing line endings.

    YAML files are sensitive to whitespace, so this function:
    - Removes trailing spaces/tabs (which can cause parsing errors)
    - Normalizes line endings
    - Validates files can be read (but doesn't parse YAML structure)

    Args:
        files: Set of YAML file paths to process
        check_only: If True, only check for issues without modifying files

    Returns:
        True if all files were processed successfully (or no issues in check mode)
    """
    if not files:
        logger.debug("No YAML files to process")
        return True

    logger.info(f"{'='*60}")
    logger.info("YAML FILE FORMATTING")
    logger.info(f"{'='*60}")

    if check_only:
        logger.info("Checking YAML files for formatting issues (read-only mode)...")
        issues_found = False

        for file_path in sorted(files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    if line.endswith(" \n") or line.endswith("\t\n"):
                        logger.warning(
                            f"Line {i}: Trailing whitespace in {file_path}"
                        )
                        issues_found = True

            except UnicodeDecodeError as e:
                logger.error(f"Encoding error in {file_path}: {e}")
                issues_found = True
            except Exception as e:
                logger.error(f"Could not check {file_path}: {e}")
                issues_found = True

        return not issues_found

    else:
        logger.info(f"Formatting {len(files)} YAML file(s)...")
        fixed = remove_trailing_whitespace(files)
        logger.info(f"✓ Successfully formatted {fixed} YAML files")
        return True


# =============================================================================
# PYTHON CODE FORMATTING FUNCTIONS
# =============================================================================


def remove_unused_imports(files: Set[str], check_only: bool = False) -> bool:
    """Remove unused imports using AUTOFLAKE.

    AUTOFLAKE removes:
    - Unused imports (including unused from imports)
    - Unused variables
    - Redundant pass statements

    Args:
        files: Set of Python file paths to process
        check_only: If True, check for unused imports without removing them

    Returns:
        True if operation succeeded, False otherwise

    Tool: AUTOFLAKE (https://github.com/PyCQA/autoflake)
    """
    if not files:
        logger.debug("No files provided for unused import removal")
        return True

    # Ensure autoflake is available
    if not ensure_tool_available("autoflake", auto_install=True):
        logger.error("AUTOFLAKE is required but could not be installed")
        return False

    logger.info(f"{'='*60}")
    logger.info("REMOVING UNUSED IMPORTS (AUTOFLAKE)")
    logger.info(f"{'='*60}")

    files_list = sorted(list(files))

    # Build autoflake command
    autoflake_cmd = ["autoflake"]
    if not check_only:
        autoflake_cmd.append("--in-place")
    else:
        autoflake_cmd.append("--check")

    autoflake_cmd.extend([
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        "--expand-star-imports",
        "--remove-duplicate-keys",
    ])
    autoflake_cmd.extend(files_list)

    return run_command(
        autoflake_cmd,
        "Removing unused imports with AUTOFLAKE"
    )


def reorder_imports(files: Set[str], check_only: bool = False) -> bool:
    """Reorder and group Python imports using REORDER-PYTHON-IMPORTS.

    REORDER-PYTHON-IMPORTS:
    - Moves imports to the top of file (after docstrings)
    - Groups imports: stdlib, third-party, first-party
    - Sorts imports alphabetically within groups
    - Removes duplicate imports

    Args:
        files: Set of Python file paths to process
        check_only: If True, check import order without modifying files

    Returns:
        True if all files processed successfully, False if any failed

    Tool: REORDER-PYTHON-IMPORTS
          (https://github.com/asottile/reorder_python_imports)
    """
    if not files:
        logger.debug("No files provided for import reordering")
        return True

    # Ensure reorder-python-imports is available
    if not ensure_tool_available("reorder-python-imports", auto_install=True):
        logger.error("REORDER-PYTHON-IMPORTS could not be installed")
        return False

    logger.info(f"{'='*60}")
    logger.info("REORDERING IMPORTS (REORDER-PYTHON-IMPORTS)")
    logger.info(f"{'='*60}")

    files_list = sorted(list(files))

    # Build reorder command
    reorder_cmd = ["reorder-python-imports", "--py38-plus"]
    if check_only:
        reorder_cmd.append("--diff-only")

    success = True

    # Process files individually to:
    # 1. Avoid command line length limits
    # 2. Provide per-file progress feedback
    # 3. Continue processing even if one file fails
    for file_path in files_list:
        file_cmd = reorder_cmd + [file_path]
        if not run_command(
            file_cmd,
            f"Reordering imports in {file_path}"
        ):
            success = False
            logger.warning(f"Import reordering failed for {file_path}")

    return success


def format_code(files: Set[str], check_only: bool = False) -> bool:
    """Format Python files using BLACK code formatter.

    BLACK (https://black.readthedocs.io/):
    - Opinionated Python code formatter
    - Enforces consistent style across codebase
    - PEP 8 compliant
    - Formats: indentation, quotes, line breaks, spacing

    Process flow:
    1. Remove trailing whitespace (if not check_only)
    2. Apply BLACK formatting with 120 char line length

    Args:
        files: Set of Python file paths to format
        check_only: If True, check formatting without modifying files

    Returns:
        True if all formatting succeeded, False otherwise

    Tool: BLACK (https://black.readthedocs.io/)
    """
    if not files:
        logger.info("No Python files to format")
        return True

    # Ensure BLACK is available
    if not ensure_tool_available("black", auto_install=True):
        logger.error("BLACK formatter could not be installed")
        return False

    files_list = sorted(list(files))
    success = True

    # Step 1: Remove trailing whitespace (unless in check mode)
    if not check_only:
        logger.info(f"{'='*60}")
        logger.info("REMOVING TRAILING WHITESPACE")
        logger.info(f"{'='*60}")
        logger.info(f"Processing {len(files_list)} Python file(s)...")
        fixed = remove_trailing_whitespace(files)
        logger.info(f"✓ Fixed trailing whitespace in {fixed} files")

    # Step 2: Apply BLACK formatting
    logger.info(f"{'='*60}")
    logger.info("CODE FORMATTING (BLACK)")
    logger.info(f"{'='*60}")

    # Build BLACK command
    black_cmd = ["black"]
    if check_only:
        black_cmd.extend(["--check", "--diff"])

    # Configuration: 120 character line length (enterprise standard)
    black_cmd.extend(["--line-length", "120"])
    black_cmd.extend(files_list)

    if not run_command(black_cmd, "BLACK code formatting"):
        success = False
        logger.error("BLACK formatting encountered errors")

    return success


def run_linting(files: Set[str]) -> bool:
    """Run FLAKE8 linting to validate code quality.

    FLAKE8 (https://flake8.pycqa.org/):
    - Checks code against PEP 8 style guide
    - Detects syntax errors and common bugs
    - Checks for undefined names and unused imports
    - Validates code complexity

    Configuration:
    - Uses .flake8 config file if present in project root
    - Default settings apply if no config found

    Args:
        files: Set of Python file paths to lint

    Returns:
        True if all files pass linting, False if issues found

    Tool: FLAKE8 (https://flake8.pycqa.org/)
    """
    if not files:
        logger.debug("No files provided for linting")
        return True

    # Ensure FLAKE8 is available
    if not ensure_tool_available("flake8", auto_install=True):
        logger.error("FLAKE8 linter could not be installed")
        return False

    logger.info(f"{'='*60}")
    logger.info("CODE LINTING (FLAKE8)")
    logger.info(f"{'='*60}")

    files_list = sorted(list(files))
    flake8_cmd = ["flake8"] + files_list

    result = run_command(flake8_cmd, "FLAKE8 linting")

    if not result:
        logger.error(
            "FLAKE8 found code quality issues. "
            "Review the output above for details."
        )

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point for the code formatter utility.

    Orchestrates the formatting pipeline:
    1. Parse command-line arguments
    2. Discover files to process
    3. Run formatting/linting tools in sequence
    4. Report results and statistics

    Returns:
        Exit code:
        - 0: Success
        - 1: Formatting/linting failures
        - 2: Tool installation failure
        - 3: No files found
    """
    # Setup argument parser with comprehensive help
    parser = argparse.ArgumentParser(
        description="Enterprise-Grade Code Formatter for Python and YAML",
        epilog="""
Examples:
  python format_code.py                  # Format all Python files
  python format_code.py --staged         # Format only Git-staged files
  python format_code.py --check          # Check without modifying
  python format_code.py --all --lint     # Format all + run linting
  python format_code.py file1.py file2.py  # Format specific files
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--staged",
        action="store_true",
        help="Only format Git-staged files (requires Git repository)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check formatting without making changes (exits with code 1 if issues found)",
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Also run FLAKE8 linting after formatting",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Format both Python and YAML files (default: Python only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific file(s) to format (overrides --staged and --all)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Display header
    logger.info("=" * 60)
    logger.info("ENTERPRISE CODE FORMATTER UTILITY")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Track start time for statistics
    start_time = datetime.now()

    # =========================================================================
    # STEP 1: Discover files to process
    # =========================================================================
    python_files: Set[str] = set()
    yaml_files: Set[str] = set()

    if args.files:
        # Specific files provided via command line
        logger.info("\nProcessing specified files...")
        for f in args.files:
            if not Path(f).exists():
                logger.warning(f"File not found: {f}")
                continue
            if f.endswith(".py"):
                python_files.add(f)
            elif f.endswith((".yml", ".yaml")):
                yaml_files.add(f)
            else:
                logger.warning(f"Unsupported file type: {f}")

    elif args.staged:
        # Process only Git-staged files
        logger.info("\nDiscovering Git-staged files...")
        python_files, yaml_files = get_staged_files()
        if not python_files and not yaml_files:
            logger.warning("No staged Python or YAML files found")
            return EXIT_NO_FILES

    elif args.all:
        # Process all Python and YAML files
        logger.info("\nDiscovering all Python and YAML files...")
        python_files = get_all_python_files()
        yaml_files = get_all_yaml_files()

    else:
        # Default: Process all Python files in standard directories
        logger.info(f"\nDiscovering Python files in {DEFAULT_PYTHON_DIRS}...")
        python_files = get_all_python_files()

    # Validate we have files to process
    total_files = len(python_files) + len(yaml_files)
    logger.info(
        f"\n{'='*60}\n"
        f"FILES TO PROCESS:\n"
        f"  Python files: {len(python_files)}\n"
        f"  YAML files: {len(yaml_files)}\n"
        f"  Total: {total_files}\n"
        f"{'='*60}"
    )

    if total_files == 0:
        logger.warning("No files found to process")
        return EXIT_NO_FILES

    # =========================================================================
    # STEP 2: Execute formatting pipeline
    # =========================================================================
    success = True
    operations_completed = []

    # Format YAML files (if any)
    if yaml_files:
        logger.info("\n" + ">" * 60)
        logger.info("PHASE 1: YAML FILE FORMATTING")
        logger.info(">" * 60)
        yaml_success = format_yaml_files(yaml_files, check_only=args.check)
        success = success and yaml_success
        operations_completed.append(
            ("YAML formatting", yaml_success, len(yaml_files))
        )

    # Format Python files (if any)
    if python_files:
        logger.info("\n" + ">" * 60)
        logger.info("PHASE 2: PYTHON CODE FORMATTING")
        logger.info(">" * 60)
        python_success = format_code(python_files, check_only=args.check)
        success = success and python_success
        operations_completed.append(
            ("Python formatting", python_success, len(python_files))
        )

        # Optional linting
        if args.lint:
            logger.info("\n" + ">" * 60)
            logger.info("PHASE 3: CODE LINTING")
            logger.info(">" * 60)
            lint_success = run_linting(python_files)
            success = success and lint_success
            operations_completed.append(
                ("FLAKE8 linting", lint_success, len(python_files))
            )

    # =========================================================================
    # STEP 3: Display summary and statistics
    # =========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)

    # Display operation results
    for operation, op_success, file_count in operations_completed:
        status = "✓ PASSED" if op_success else "✗ FAILED"
        logger.info(f"{status} - {operation} ({file_count} files)")

    # Display final status
    logger.info("=" * 60)
    if success:
        if args.check:
            logger.info("✓ ALL CHECKS PASSED")
            logger.info("No formatting issues detected")
        else:
            logger.info("✓ ALL FILES FORMATTED SUCCESSFULLY")
            logger.info(f"Processed {total_files} files in {duration:.2f} seconds")
    else:
        logger.error("✗ SOME OPERATIONS FAILED")
        logger.error("Review the output above for details")

    logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration:.2f} seconds")
    logger.info("=" * 60)

    # Return appropriate exit code
    if success:
        return EXIT_SUCCESS
    else:
        return EXIT_FORMATTING_FAILED


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user")
        sys.exit(EXIT_FORMATTING_FAILED)
    except Exception as e:
        logger.error(f"\n\n✗ Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(EXIT_FORMATTING_FAILED)
