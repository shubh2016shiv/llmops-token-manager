"""
Unit tests for app.utils.pagination.

No tests previously existed for this module. It currently has no live
callers (its only two references were app/api/*_OBSELETE.py — unwired,
retired endpoints — see app/api/__init__.py's commented-out router
registration), but it is a plain, side-effect-free arithmetic module that
should be internally correct regardless of whether it's wired up today, and
its validation gaps were real latent bugs (negative offsets reaching
PostgreSQL) independent of who calls it.
"""

from __future__ import annotations

import pytest

from app.utils.pagination import (
    PaginationParams,
    compute_has_next_from_offset,
    compute_has_previous,
    compute_offset,
    compute_pagination,
    compute_pagination_from_limit_offset,
)


class TestPaginationParams:
    def test_defaults(self):
        params = PaginationParams()

        assert params.page == 1
        assert params.page_size == 50

    def test_is_frozen(self):
        params = PaginationParams()

        with pytest.raises(AttributeError):
            params.page = 2  # type: ignore[misc]


class TestComputeOffset:
    def test_first_page_has_zero_offset(self):
        assert compute_offset(page=1, page_size=50) == 0

    def test_second_page_offset(self):
        assert compute_offset(page=2, page_size=50) == 50

    def test_arbitrary_page(self):
        assert compute_offset(page=5, page_size=20) == 80

    @pytest.mark.parametrize("invalid_page", [0, -1, -100])
    def test_rejects_non_positive_page(self, invalid_page):
        """
        Regression test: page=0 previously produced offset=-50 (a negative
        offset) silently instead of raising — PostgreSQL rejects a negative
        OFFSET outright, so this used to surface as a confusing DB error
        far from the actual mistake.
        """
        with pytest.raises(ValueError, match="page must be >= 1"):
            compute_offset(page=invalid_page, page_size=50)

    @pytest.mark.parametrize("invalid_page_size", [0, -1])
    def test_rejects_non_positive_page_size(self, invalid_page_size):
        with pytest.raises(ValueError, match="page_size must be >= 1"):
            compute_offset(page=1, page_size=invalid_page_size)


class TestComputeHasPrevious:
    def test_first_page_has_no_previous(self):
        assert compute_has_previous(page=1) is False

    def test_later_page_has_previous(self):
        assert compute_has_previous(page=2) is True


class TestComputeHasNextFromOffset:
    def test_more_records_remain(self):
        assert compute_has_next_from_offset(total_count=100, offset=0, limit=50) is True

    def test_exactly_at_the_end(self):
        assert (
            compute_has_next_from_offset(total_count=100, offset=50, limit=50) is False
        )

    def test_past_the_end(self):
        assert (
            compute_has_next_from_offset(total_count=100, offset=90, limit=50) is False
        )


class TestComputePagination:
    def test_first_page_of_many(self):
        offset, limit, page, page_size, has_next, has_previous = compute_pagination(
            total_count=120, page=1, page_size=50
        )

        assert (offset, limit, page, page_size) == (0, 50, 1, 50)
        assert has_next is True
        assert has_previous is False

    def test_middle_page(self):
        offset, limit, page, page_size, has_next, has_previous = compute_pagination(
            total_count=120, page=2, page_size=50
        )

        assert offset == 50
        assert has_next is True
        assert has_previous is True

    def test_last_page(self):
        offset, limit, page, page_size, has_next, has_previous = compute_pagination(
            total_count=120, page=3, page_size=50
        )

        assert offset == 100
        assert has_next is False
        assert has_previous is True

    def test_invalid_page_propagates_from_compute_offset(self):
        with pytest.raises(ValueError, match="page must be >= 1"):
            compute_pagination(total_count=100, page=0, page_size=50)


class TestComputePaginationFromLimitOffset:
    def test_first_page(self):
        result = compute_pagination_from_limit_offset(
            total_count=120, limit=50, offset=0
        )

        offset, limit, page, page_size, has_next, has_previous = result
        assert (offset, limit, page, page_size) == (0, 50, 1, 50)
        assert has_next is True
        assert has_previous is False

    def test_non_aligned_offset_still_computes_a_page_number(self):
        # offset=75 with limit=50 isn't a "clean" page boundary — this
        # legacy helper exists for limit/offset callers that don't think in
        # pages at all, so the page number is only an approximation.
        result = compute_pagination_from_limit_offset(
            total_count=200, limit=50, offset=75
        )

        offset, limit, page, page_size, has_next, has_previous = result
        assert page == 2  # (75 // 50) + 1
        assert has_previous is True

    def test_rejects_negative_offset(self):
        """
        Regression test: a negative offset previously passed straight
        through to has_next_from_offset without validation, and would have
        reached PostgreSQL as an invalid OFFSET clause.
        """
        with pytest.raises(ValueError, match="offset must be >= 0"):
            compute_pagination_from_limit_offset(total_count=100, limit=50, offset=-1)

    @pytest.mark.parametrize("invalid_limit", [0, -1])
    def test_rejects_non_positive_limit(self, invalid_limit):
        """
        Regression test: limit=0 previously fell back to page=1 while still
        returning limit=0 as page_size — an internally inconsistent result
        (page 1 of size 0) instead of a clear failure.
        """
        with pytest.raises(ValueError, match="limit must be >= 1"):
            compute_pagination_from_limit_offset(
                total_count=100, limit=invalid_limit, offset=0
            )
