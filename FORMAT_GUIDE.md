# Code Formatting Guide

Quick reference for formatting code in this project.

## Quick Commands

```bash
# Format all Python code before committing
python format_code.py

# Format only staged files (Python + YAML)
python format_code.py --staged

# Format everything (Python + YAML)
python format_code.py --all

# Check formatting without changing files
python format_code.py --check

# Format and lint Python files
python format_code.py --lint

# Format specific files
python format_code.py app/models/database_models.py .pre-commit-config.yaml
```

## What It Does

The `format_code.py` utility automatically fixes:

### Python Files
1. **Trailing whitespace** - Removes spaces at end of lines
2. **File endings** - Ensures files end with single newline
3. **Code formatting** - Uses Black for consistent style
4. **Optional linting** - Runs flake8 with `--lint` flag

> **Note:** For simplicity, we've removed import reordering and unused import detection to avoid conflicts with pre-commit hooks.

### YAML Files
1. **Trailing whitespace** - Removes spaces at end of lines
2. **File endings** - Ensures files end with single newline
3. **Fixes:** `.pre-commit-config.yaml`, `docker-compose.yml`, `.github/workflows/*.yml`

## Recommended Workflow

### Before Committing (Fastest)

```bash
# 1. Stage your changes
git add .

# 2. Format staged files (Python + YAML)
python format_code.py --staged

# 3. Review changes
git diff

# 4. Add formatted changes
git add .

# 5. Commit
git commit -m "your message"
```

### Quick Fix Everything

```bash
# Format all Python files
python format_code.py

# Format all Python + YAML files
python format_code.py --all

# Review and commit
git add .
git commit -m "style: format code"
```

### Fix Pre-commit Config Issues

```bash
# Specifically fix .pre-commit-config.yaml
python format_code.py .pre-commit-config.yaml

# Or fix all YAML files
python format_code.py --all
```

### Skip Pre-commit Hooks (Urgent Commits)

```bash
git commit --no-verify -m "urgent fix"
```

## Command Reference

| Command | Description |
|---------|-------------|
| `python format_code.py` | Format all Python files in app/ and tests/ |
| `python format_code.py --staged` | Format only staged Python and YAML files |
| `python format_code.py --all` | Format all Python + YAML files |
| `python format_code.py --check` | Check formatting without making changes |
| `python format_code.py --lint` | Format and run linting on Python files |
| `python format_code.py file.py` | Format specific file(s) |

## What Won't Change

This utility is 100% safe and will NOT change:
- Business logic
- Function behavior
- Variable names
- Data structures
- Control flow
- Algorithms
- YAML structure or values

It ONLY changes:
- Whitespace
- Line breaks
- Quote styles (single to double in Python)
- Import order (Python)
- Trailing spaces

## Integration with Pre-commit

You can run this before pre-commit hooks to avoid issues:

```bash
# Format staged files and commit in one line
python format_code.py --staged && git add . && git commit -m "message"
```

## CI Integration

The CI workflow will also auto-format code, but running locally is faster and gives you immediate feedback.

## Troubleshooting

### Pre-commit complains about trailing whitespace

```bash
# Fix all YAML files
python format_code.py --all

# Or just fix the specific file
python format_code.py .pre-commit-config.yaml
```

### Want to see what would change without changing it

```bash
python format_code.py --check
python format_code.py --staged --check
```

### Black not found

```bash
# Install formatting tools
pip install black flake8
```
