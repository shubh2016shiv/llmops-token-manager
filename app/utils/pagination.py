"""
Pagination Utilities.

Shared helpers for consistent API pagination behavior across endpoints.

Enterprise pattern:
- Centralize pagination math to avoid drift and subtle off-by-one errors.
- Keep handlers focused on business logic; pagination is a cross-cutting concern.
- Provide a single "source of truth" that endpoints and tests can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaginationParams:
    """Canonical page-based pagination parameters."""

    page: int = 1
    page_size: int = 50


def compute_offset(page: int, page_size: int) -> int:
    """Convert 1-based page params to a 0-based offset."""
    return (page - 1) * page_size


def compute_has_previous(page: int) -> bool:
    """Whether there is a previous page."""
    return page > 1


def compute_has_next_from_offset(total_count: int, offset: int, limit: int) -> bool:
    """
    Whether there is a next page, computed from offset/limit.

    Note: total_count should reflect the total available records, not the page size.
    """
    return (offset + limit) < total_count


def compute_pagination(
    *,
    total_count: int,
    page: int,
    page_size: int,
) -> tuple[int, int, int, int, bool, bool]:
    """
    Compute pagination metadata and parameters.

    Returns:
        (offset, limit, page, page_size, has_next, has_previous)
    """
    offset = compute_offset(page, page_size)
    limit = page_size
    has_previous = compute_has_previous(page)
    has_next = compute_has_next_from_offset(total_count, offset, limit)
    return offset, limit, page, page_size, has_next, has_previous


def compute_pagination_from_limit_offset(
    *,
    total_count: int,
    limit: int,
    offset: int,
) -> tuple[int, int, int, int, bool, bool]:
    """
    Legacy helper for limit/offset-based pagination.

    Returns:
        (offset, limit, page, page_size, has_next, has_previous)
    """
    page = (offset // limit) + 1 if limit > 0 else 1
    page_size = limit
    has_previous = offset > 0
    has_next = compute_has_next_from_offset(total_count, offset, limit)
    return offset, limit, page, page_size, has_next, has_previous
