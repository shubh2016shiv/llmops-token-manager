#!/usr/bin/env python3
"""
Code formatter utility for LLM Token Manager
Automatically fixes formatting issues without changing logic

Usage:
    python format_code.py              # Format all Python files
    python format_code.py --staged     # Format only staged files
    python format_code.py --check      # Check without fixing
    python format_code.py --all        # Format Python + YAML files
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Set
from typing import Tuple


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False


def get_staged_files() -> Tuple[Set[str], Set[str]]:
    """Get list of staged Python and YAML files"""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.strip().split("\n")
        python_files = {f for f in files if f.endswith(".py") and Path(f).exists()}
        yaml_files = {f for f in files if f.endswith((".yml", ".yaml")) and Path(f).exists()}
        return python_files, yaml_files
    except Exception as e:
        print(f"Warning: Could not get staged files: {e}")
        return set(), set()


def get_all_python_files() -> Set[str]:
    """Get all Python files in app/ and tests/"""
    python_files = set()
    for directory in ["app", "tests"]:
        if Path(directory).exists():
            for py_file in Path(directory).rglob("*.py"):
                python_files.add(str(py_file))
    return python_files


def get_all_yaml_files() -> Set[str]:
    """Get important YAML files"""
    yaml_files = set()
    # Check for common YAML files
    yaml_patterns = [
        ".pre-commit-config.yaml",
        "docker-compose.yml",
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
    ]

    for pattern in yaml_patterns:
        if "*" in pattern:
            # Handle glob patterns
            base_dir = Path(pattern).parent
            if base_dir.exists():
                for yaml_file in base_dir.glob(Path(pattern).name):
                    yaml_files.add(str(yaml_file))
        else:
            # Handle specific files
            if Path(pattern).exists():
                yaml_files.add(pattern)

    return yaml_files


def remove_trailing_whitespace(files: Set[str]) -> int:
    """Remove trailing whitespace from files"""
    fixed_count = 0
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = [line.rstrip() + "\n" if line.strip() else "\n" for line in lines]

            # Ensure file ends with newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            # Remove trailing blank lines
            while len(new_lines) > 1 and new_lines[-1] == "\n" and new_lines[-2] == "\n":
                new_lines.pop()

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            fixed_count += 1
            print(f"  ✓ Fixed: {file_path}")
        except Exception as e:
            print(f"  ✗ Warning: Could not fix {file_path}: {e}")

    return fixed_count


def format_yaml_files(files: Set[str], check_only: bool = False) -> bool:
    """Format YAML files by removing trailing whitespace and fixing line endings"""
    if not files:
        return True

    print(f"\n{'='*60}")
    print("Processing YAML files")
    print(f"{'='*60}")

    if check_only:
        print("Checking YAML files (read-only)...")
        # Just check for trailing whitespace
        issues_found = False
        for file_path in sorted(files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if line.endswith(" \n") or line.endswith("\t\n"):
                        print(f"  Line {i}: Trailing whitespace in {file_path}")
                        issues_found = True
            except Exception as e:
                print(f"  ✗ Could not check {file_path}: {e}")
        return not issues_found
    else:
        print(f"Fixing {len(files)} YAML file(s)...")
        fixed = remove_trailing_whitespace(files)
        print(f"✓ Fixed {fixed} YAML files")
        return True


def remove_unused_imports(files: Set[str], check_only: bool = False) -> bool:
    """Remove unused imports using autoflake"""
    if not files:
        return True

    # Check if autoflake is installed
    try:
        subprocess.run(["autoflake", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: autoflake not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "autoflake"], check=True)
        except subprocess.SubprocessError:
            print("Failed to install autoflake. Skipping unused import removal.")
            return True

    print(f"\n{'='*60}")
    print("Removing unused imports")
    print(f"{'='*60}")

    files_list = sorted(list(files))

    autoflake_cmd = ["autoflake"]
    if not check_only:
        autoflake_cmd.append("--in-place")
    autoflake_cmd.extend(["--remove-all-unused-imports", "--remove-unused-variables", "--expand-star-imports"])
    autoflake_cmd.extend(files_list)

    return run_command(autoflake_cmd, "Removing unused imports")


def reorder_imports(files: Set[str], check_only: bool = False) -> bool:
    """Reorder imports to top of file using reorder-python-imports"""
    if not files:
        return True

    # Check if reorder-python-imports is installed
    try:
        subprocess.run(["reorder-python-imports", "--help"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: reorder-python-imports not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "reorder-python-imports"], check=True)
        except subprocess.SubprocessError:
            print("Failed to install reorder-python-imports. Skipping import reordering.")
            return True

    print(f"\n{'='*60}")
    print("Reordering imports to top of file")
    print(f"{'='*60}")

    files_list = sorted(list(files))

    reorder_cmd = ["reorder-python-imports", "--py38-plus"]
    if check_only:
        reorder_cmd.append("--diff-only")

    success = True
    # Process files one by one to avoid command line length limits
    for file_path in files_list:
        file_cmd = reorder_cmd + [file_path]
        if not run_command(file_cmd, f"Reordering imports in {file_path}"):
            success = False

    return success


def format_code(files: Set[str], check_only: bool = False) -> bool:
    """Format Python files using black"""
    if not files:
        print("No Python files to format")
        return True

    files_list = sorted(list(files))
    success = True

    # Remove trailing whitespace
    if not check_only:
        print(f"\n{'='*60}")
        print("Removing trailing whitespace from Python files")
        print(f"{'='*60}")
        print(f"Processing {len(files_list)} file(s)...")
        fixed = remove_trailing_whitespace(files)
        print(f"✓ Fixed trailing whitespace in {fixed} files")

    # For simplicity, we'll just run black directly and skip the complex import handling
    # This avoids conflicts with pre-commit hooks

    # Black formatting (must run after import processing)
    black_cmd = ["black"]
    if check_only:
        black_cmd.extend(["--check", "--diff"])
    black_cmd.extend(["--line-length", "120"])  # Increased line length
    black_cmd.extend(files_list)

    if not run_command(black_cmd, "Black code formatting"):
        success = False

    return success


def run_linting(files: Set[str]) -> bool:
    """Run flake8 linting"""
    if not files:
        return True

    files_list = sorted(list(files))

    flake8_cmd = ["flake8"] + files_list

    return run_command(flake8_cmd, "Flake8 linting")


def main() -> int:
    """Main function to process files"""
    parser = argparse.ArgumentParser(description="Format code automatically without changing logic")
    parser.add_argument("--staged", action="store_true", help="Only format staged files")
    parser.add_argument("--check", action="store_true", help="Check formatting without making changes")
    parser.add_argument("--lint", action="store_true", help="Also run linting checks")
    parser.add_argument("--all", action="store_true", help="Format both Python and YAML files")
    parser.add_argument("files", nargs="*", help="Specific files to format")

    args = parser.parse_args()

    print("=" * 60)
    print("Code Formatter Utility")
    print("=" * 60)

    # Determine which files to process
    python_files = set()
    yaml_files = set()

    if args.files:
        for f in args.files:
            if not Path(f).exists():
                continue
            if f.endswith(".py"):
                python_files.add(f)
            elif f.endswith((".yml", ".yaml")):
                yaml_files.add(f)
    elif args.staged:
        print("\nGetting staged files...")
        python_files, yaml_files = get_staged_files()
        if not python_files and not yaml_files:
            print("No staged Python or YAML files found")
            return 0
    elif args.all:
        print("\nGetting all Python and YAML files...")
        python_files = get_all_python_files()
        yaml_files = get_all_yaml_files()
    else:
        print("\nGetting all Python files in app/ and tests/...")
        python_files = get_all_python_files()

    total_files = len(python_files) + len(yaml_files)
    print(f"\nFound {len(python_files)} Python file(s) and {len(yaml_files)} YAML file(s) to process")

    if total_files == 0:
        print("No files to process")
        return 0

    success = True

    # Format YAML files
    if yaml_files:
        yaml_success = format_yaml_files(yaml_files, check_only=args.check)
        success = success and yaml_success

    # Format Python files
    if python_files:
        python_success = format_code(python_files, check_only=args.check)
        success = success and python_success

        # Optional linting
        if args.lint:
            lint_success = run_linting(python_files)
            success = success and lint_success

    # Summary
    print("\n" + "=" * 60)
    if success:
        if args.check:
            print("✓ All checks passed!")
        else:
            print("✓ All files formatted successfully!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some operations failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
