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
    """
    Convert 1-based page params to a 0-based offset.

    Raises:
        ValueError: If page < 1 or page_size < 1. Without this guard,
            page=0 silently produced a NEGATIVE offset ((0-1)*page_size),
            which PostgreSQL rejects outright ("OFFSET must not be
            negative") — better to fail here, at the one place that does
            the arithmetic, than let every caller rediscover it via a DB
            error.
    """
    if page < 1:
        raise ValueError(f"page must be >= 1, got {page}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
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

    Raises:
        ValueError: If offset < 0 or limit < 1. The previous version let
            limit <= 0 through silently (falling back to page=1 while still
            returning the invalid limit as page_size), and a negative
            offset would reach PostgreSQL as an invalid OFFSET clause.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    page = (offset // limit) + 1
    page_size = limit
    has_previous = offset > 0
    has_next = compute_has_next_from_offset(total_count, offset, limit)
    return offset, limit, page, page_size, has_next, has_previous
