# Code Formatting Guide

**Enterprise-grade code formatting and quality assurance for the LLM Token Manager project.**

This guide explains how to use the `code_formatter.py` utility to maintain consistent code style, enforce quality standards, and ensure your code passes all CI/CD checks.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Formatting Tools](#formatting-tools)
- [Command Reference](#command-reference)
- [Recommended Workflows](#recommended-workflows)
- [What Gets Changed](#what-gets-changed)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [CI/CD Integration](#cicd-integration)

---

## Quick Start

```bash
# Format all Python code before committing (most common)
python code_formatter.py

# Format only Git-staged files (Python + YAML) - fastest for commits
python code_formatter.py --staged

# Format everything (Python + YAML) - comprehensive
python code_formatter.py --all

# Check formatting without changing files - safe preview
python code_formatter.py --check

# Format and run linting - full quality check
python code_formatter.py --lint

# Verbose output for debugging
python code_formatter.py --verbose

# Format specific files
python code_formatter.py app/models/database_models.py tests/test_api.py
```

---

## Formatting Tools

The `code_formatter.py` utility uses **four industry-standard tools** to ensure code quality:

### 1. **BLACK** - Code Formatter
- **Purpose**: Opinionated Python code formatter
- **What it does**: Enforces consistent code style (PEP 8 compliant)
- **Configuration**: 120 character line length (enterprise standard)
- **Website**: https://black.readthedocs.io/
- **Auto-installs**: Yes, if missing

**Changes:**
- Indentation and spacing
- Quote styles (converts to double quotes)
- Line breaks and wrapping
- Bracket and parenthesis placement

### 2. **AUTOFLAKE** - Import Cleaner
- **Purpose**: Removes unused imports and variables
- **What it does**: Cleans up code by removing unused Python imports
- **Configuration**: Removes all unused imports, variables, expands star imports
- **Website**: https://github.com/PyCQA/autoflake
- **Auto-installs**: Yes, if missing

**Changes:**
- Removes unused imports
- Removes unused variables
- Expands `from module import *`
- Removes duplicate dictionary keys

### 3. **REORDER-PYTHON-IMPORTS** - Import Organizer
- **Purpose**: Automatically reorders Python imports
- **What it does**: Sorts imports to top of file and groups them correctly
- **Configuration**: Python 3.8+ compatibility mode
- **Website**: https://github.com/asottile/reorder_python_imports
- **Auto-installs**: Yes, if missing

**Changes:**
- Moves imports to top (after docstrings)
- Groups: stdlib → third-party → first-party
- Sorts alphabetically within groups
- Removes duplicate imports

### 4. **FLAKE8** - Code Linter (Optional)
- **Purpose**: Style guide enforcement and linting
- **What it does**: Validates code against PEP 8 and detects common errors
- **Configuration**: Uses `.flake8` config file if present
- **Website**: https://flake8.pycqa.org/
- **Auto-installs**: Yes, if missing
- **Usage**: Add `--lint` flag to enable

**Checks:**
- PEP 8 style violations
- Syntax errors
- Undefined names
- Code complexity
- Common programming errors

### YAML File Formatting

For YAML files (`.yml`, `.yaml`), the formatter:
- Removes trailing whitespace (critical for YAML parsing)
- Normalizes line endings
- Ensures files end with single newline
- Processes: `.pre-commit-config.yaml`, `docker-compose.yml`, `.github/workflows/*.yml`

---

## Command Reference

| Command | Description | Use Case |
|---------|-------------|----------|
| `python code_formatter.py` | Format all Python files in `app/` and `tests/` | Daily development |
| `python code_formatter.py --staged` | Format only Git-staged files (Python + YAML) | Pre-commit formatting |
| `python code_formatter.py --all` | Format all Python + YAML files | Project-wide cleanup |
| `python code_formatter.py --check` | Check formatting without changes | CI/CD validation |
| `python code_formatter.py --lint` | Format + run FLAKE8 linting | Full quality check |
| `python code_formatter.py --verbose` | Detailed logging output | Debugging issues |
| `python code_formatter.py file.py` | Format specific file(s) | Targeted fixes |
| `python code_formatter.py --staged --check` | Check staged files only | Pre-commit validation |
| `python code_formatter.py --all --lint` | Full format + lint everything | Release preparation |

### Exit Codes

The formatter returns specific exit codes for automation:

- **0**: Success - all operations completed
- **1**: Formatting/linting failures detected
- **2**: Tool installation failure
- **3**: No files found to process

---

## Recommended Workflows

### Workflow 1: Pre-Commit (Fastest) ⚡

**Best for**: Daily commits, quick fixes

```bash
# 1. Stage your changes
git add .

# 2. Format only staged files (Python + YAML)
python code_formatter.py --staged

# 3. Review changes (optional)
git diff

# 4. Add formatted changes
git add .

# 5. Commit
git commit -m "feat: add new feature"
```

**Time**: ~2-5 seconds for typical changes

---

### Workflow 2: Full Quality Check 🔍

**Best for**: Before pull requests, major commits

```bash
# 1. Format all Python files
python code_formatter.py

# 2. Run full linting check
python code_formatter.py --lint

# 3. Review and commit
git add .
git commit -m "style: format code and fix linting issues"
```

**Time**: ~10-30 seconds depending on project size

---

### Workflow 3: YAML Configuration Fixes 📝

**Best for**: Fixing pre-commit config, Docker Compose, GitHub workflows

```bash
# Fix specific YAML file
python code_formatter.py .pre-commit-config.yaml

# Or fix all YAML files
python code_formatter.py --all

# Review and commit
git add .
git commit -m "fix: format YAML configuration files"
```

---

### Workflow 4: Safe Preview (No Changes) 👀

**Best for**: Checking what would change before committing

```bash
# Check all files without modifying
python code_formatter.py --check

# Check only staged files
python code_formatter.py --staged --check

# Verbose output for details
python code_formatter.py --check --verbose
```

**Returns exit code 1 if issues found, 0 if clean**

---

### Workflow 5: One-Line Commit 🚀

**Best for**: Quick commits with auto-formatting

```bash
# Format staged files and commit in one command
python code_formatter.py --staged && git add . && git commit -m "your message"

# With linting
python code_formatter.py --staged --lint && git add . && git commit -m "your message"
```

---

### Workflow 6: Emergency Commit (Skip Hooks) 🚨

**Best for**: Urgent fixes, work-in-progress commits

```bash
# Skip pre-commit hooks (use sparingly!)
git commit --no-verify -m "WIP: urgent fix"

# Format later
python code_formatter.py
git add .
git commit -m "style: format code"
```

**⚠️ Warning**: Only use `--no-verify` for genuine emergencies

---

## What Gets Changed

### ✅ Safe Changes (100% Safe)

The formatter **ONLY** modifies code style and formatting. It **NEVER** changes:

**Will NOT Change:**
- ❌ Business logic or algorithms
- ❌ Function behavior or return values
- ❌ Variable names or identifiers
- ❌ Data structures or types
- ❌ Control flow (if/else/loops)
- ❌ YAML structure or values
- ❌ Comments or docstrings (content)
- ❌ String literals (content)

**Will Change:**
- ✅ Whitespace and indentation
- ✅ Line breaks and wrapping
- ✅ Quote styles (single → double quotes in Python)
- ✅ Import order and grouping
- ✅ Trailing spaces and blank lines
- ✅ Bracket/parenthesis placement
- ✅ Unused imports (removed)

### Example Transformations

**Before:**
```python
from os import path
import sys
from typing import List,Dict
import requests

def my_function(x,y,z):
    result=x+y+z
    return result
```

**After:**
```python
import sys
from os import path
from typing import Dict
from typing import List

import requests


def my_function(x, y, z):
    result = x + y + z
    return result
```

---

## Advanced Usage

### Automatic Tool Installation

The formatter automatically installs missing tools:
- **BLACK** - Code formatter
- **AUTOFLAKE** - Import cleaner
- **REORDER-PYTHON-IMPORTS** - Import organizer
- **FLAKE8** - Linter (if `--lint` used)

**Timeout protection**: 120 seconds per tool installation

### Verbose Logging

Enable detailed logging for debugging:

```bash
python code_formatter.py --verbose
```

**Shows:**
- Tool installation attempts
- File discovery process
- Command execution details
- Full error tracebacks
- Timing statistics

### Processing Statistics

The formatter provides execution summaries:

```
============================================================
EXECUTION SUMMARY
============================================================
✓ PASSED - YAML formatting (3 files)
✓ PASSED - Python formatting (45 files)
✓ PASSED - FLAKE8 linting (45 files)
============================================================
✓ ALL FILES FORMATTED SUCCESSFULLY
Processed 48 files in 12.34 seconds
Completed at: 2025-10-13 04:30:15
Total duration: 12.34 seconds
============================================================
```

### Custom Directory Scanning

By default, scans `app/` and `tests/` directories. To format other directories:

```bash
# Format specific files from any directory
python code_formatter.py path/to/file1.py path/to/file2.py

# Use --all to include YAML files from project root
python code_formatter.py --all
```

---

## Troubleshooting

### Issue 1: "Pre-commit complains about trailing whitespace"

**Solution:**
```bash
# Fix all YAML files
python code_formatter.py --all

# Or fix specific file
python code_formatter.py .pre-commit-config.yaml

# Then stage and commit
git add .
git commit -m "fix: format YAML files"
```

---

### Issue 2: "Tool installation failed"

**Error:** `✗ Installation of black timed out` or `✗ Failed to install black`

**Solutions:**
```bash
# Manual installation
pip install black autoflake reorder-python-imports flake8

# Or use requirements.txt
pip install -r requirements.txt

# Check installation
black --version
flake8 --version
```

---

### Issue 3: "Want to preview changes without modifying files"

**Solution:**
```bash
# Check all files (dry-run)
python code_formatter.py --check

# Check only staged files
python code_formatter.py --staged --check

# Verbose output for details
python code_formatter.py --check --verbose
```

**Exit codes:**
- `0` = No issues found
- `1` = Formatting issues detected

---

### Issue 4: "Formatter is too slow"

**Solutions:**
```bash
# Format only staged files (fastest)
python code_formatter.py --staged

# Format specific files
python code_formatter.py app/models/database_models.py

# Skip linting (faster)
python code_formatter.py  # without --lint flag
```

**Performance tips:**
- Use `--staged` for incremental commits
- Avoid `--all` unless necessary
- Run `--lint` only before pull requests

---

### Issue 5: "Git not found error when using --staged"

**Error:** `Could not get staged files (not a git repo?)`

**Solutions:**
```bash
# Initialize git repository
git init

# Or format all files instead
python code_formatter.py

# Or format specific files
python code_formatter.py app/*.py
```

---

### Issue 6: "Command timed out"

**Error:** `✗ BLACK code formatting timed out after 300 seconds`

**Solutions:**
```bash
# Format smaller batches
python code_formatter.py app/models/*.py
python code_formatter.py app/api/*.py

# Check for infinite loops or very large files
python code_formatter.py --verbose

# Increase timeout (modify COMMAND_TIMEOUT in code_formatter.py)
```

---

## CI/CD Integration

### GitHub Actions

The formatter integrates with CI/CD pipelines:

```yaml
# .github/workflows/ci.yml
- name: Check code formatting
  run: python code_formatter.py --check --lint
```

**Benefits:**
- Validates formatting before merge
- Prevents style inconsistencies
- Catches linting issues early

### Pre-commit Hooks

Integrate with `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: code-formatter
        name: Format Python code
        entry: python code_formatter.py --staged
        language: system
        pass_filenames: false
```

**Runs automatically** on every `git commit`

---

## Best Practices

### ✅ DO:
- Run `--staged` before every commit
- Use `--check` in CI/CD pipelines
- Run `--lint` before pull requests
- Format incrementally (not all at once)
- Review changes with `git diff`

### ❌ DON'T:
- Skip formatting for "quick fixes"
- Commit without formatting
- Use `--no-verify` regularly
- Format files outside `app/` and `tests/` without review
- Ignore linting errors

---

## Quick Reference Card

```bash
# Daily Development
python code_formatter.py --staged        # Before each commit

# Quality Checks
python code_formatter.py --lint          # Before pull request
python code_formatter.py --check         # CI/CD validation

# Troubleshooting
python code_formatter.py --verbose       # Debug issues
python code_formatter.py --all           # Fix everything

# Emergency
git commit --no-verify -m "WIP"          # Skip hooks (rare!)
```

---

## Getting Help

**Formatter help:**
```bash
python code_formatter.py --help
```

**Tool documentation:**
- BLACK: https://black.readthedocs.io/
- AUTOFLAKE: https://github.com/PyCQA/autoflake
- REORDER-PYTHON-IMPORTS: https://github.com/asottile/reorder_python_imports
- FLAKE8: https://flake8.pycqa.org/

**Project-specific issues:**
- Check `.flake8` config file for linting rules
- Review `pyproject.toml` for tool configurations
- See `ARCHITECTURE.md` for code standards

---

## Summary

The `code_formatter.py` utility is your **one-stop solution** for maintaining code quality:

1. **Automatic**: Installs tools, discovers files, applies fixes
2. **Safe**: Only changes formatting, never logic
3. **Fast**: Processes staged files in seconds
4. **Comprehensive**: Handles Python and YAML files
5. **Enterprise-ready**: Robust error handling, logging, and reporting

**Remember**: Consistent code style makes collaboration easier and reduces review friction. Format early, format often! 🚀
