"""Unit tests for suba.sparse_rendering."""

import numpy as np
import pytest

from suba.sparse_rendering import (
    find_overlapping_gene_indices,
    hilbert_d_to_xy,
    render_binned,
    render_signal,
)


# ---------------------------------------------------------------------------
# find_overlapping_gene_indices
# ---------------------------------------------------------------------------

class TestFindOverlapping:
    def setup_method(self):
        # Three non-overlapping genes at [10,20), [30,40), [50,60)
        self.starts = np.array([10, 30, 50], dtype=np.int64)
        self.ends   = np.array([20, 40, 60], dtype=np.int64)
        self.sort_idx = np.argsort(self.starts, kind="stable")

    def test_no_overlap_before(self):
        idx = find_overlapping_gene_indices(0, 10, self.starts, self.ends, self.sort_idx)
        assert len(idx) == 0

    def test_no_overlap_after(self):
        idx = find_overlapping_gene_indices(60, 100, self.starts, self.ends, self.sort_idx)
        assert len(idx) == 0

    def test_single_overlap(self):
        idx = find_overlapping_gene_indices(15, 25, self.starts, self.ends, self.sort_idx)
        assert set(idx.tolist()) == {0}

    def test_all_overlap(self):
        idx = find_overlapping_gene_indices(0, 100, self.starts, self.ends, self.sort_idx)
        assert set(idx.tolist()) == {0, 1, 2}

    def test_boundary_exclusive(self):
        # Query [20, 30) touches neither gene 0 (ends at 20) nor gene 1 (starts at 30)
        idx = find_overlapping_gene_indices(20, 30, self.starts, self.ends, self.sort_idx)
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# render_signal
# ---------------------------------------------------------------------------

class TestRenderSignal:
    def test_no_genes_returns_zeros(self):
        sig = render_signal(0, 10, np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        assert sig.shape == (10,)
        assert (sig == 0).all()

    def test_single_gene_full_overlap(self):
        starts = np.array([0], dtype=np.int64)
        ends   = np.array([10], dtype=np.int64)
        sig = render_signal(0, 10, starts, ends)
        assert sig.shape == (10,)
        assert (sig == 1).all()

    def test_single_gene_partial_overlap(self):
        # Gene [3, 7), query [0, 10)
        starts = np.array([3], dtype=np.int64)
        ends   = np.array([7], dtype=np.int64)
        sig = render_signal(0, 10, starts, ends)
        np.testing.assert_array_equal(sig, [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

    def test_two_overlapping_genes(self):
        # Gene A [0,6), Gene B [3,10) — overlap at [3,6)
        starts = np.array([0, 3], dtype=np.int64)
        ends   = np.array([6, 10], dtype=np.int64)
        sig = render_signal(0, 10, starts, ends)
        expected = [1, 1, 1, 2, 2, 2, 1, 1, 1, 1]
        np.testing.assert_array_equal(sig, expected)

    def test_repeated_start_positions(self):
        # Two genes both starting at position 2 — np.add.at must accumulate
        starts = np.array([2, 2], dtype=np.int64)
        ends   = np.array([5, 8], dtype=np.int64)
        sig = render_signal(0, 10, starts, ends)
        # positions [2,5): both genes → 2; [5,8): one gene → 1
        assert sig[0] == 0
        assert sig[1] == 0
        assert sig[2] == 2
        assert sig[4] == 2
        assert sig[5] == 1
        assert sig[8] == 0

    def test_gene_ends_at_query_boundary(self):
        # Gene [0, 10) with query [0, 10) — end marker falls at index 10 (absorbed)
        starts = np.array([0], dtype=np.int64)
        ends   = np.array([10], dtype=np.int64)
        sig = render_signal(0, 10, starts, ends)
        assert len(sig) == 10
        assert (sig == 1).all()

    def test_dtype_is_int8(self):
        sig = render_signal(0, 5, np.array([0], np.int64), np.array([5], np.int64))
        assert sig.dtype == np.int8


# ---------------------------------------------------------------------------
# render_binned
# ---------------------------------------------------------------------------

class TestRenderBinned:
    def test_even_bins(self):
        # Gene [0,10), query [0,10), step=2 → 5 bins each == 1.0
        starts = np.array([0], dtype=np.int64)
        ends   = np.array([10], dtype=np.int64)
        out = render_binned(0, 10, 2, starts, ends)
        assert len(out) == 5
        np.testing.assert_allclose(out, 1.0)

    def test_partial_last_bin(self):
        # Gene [0,10), query [0,10), step=3 → bins of size 3,3,3,1
        starts = np.array([0], dtype=np.int64)
        ends   = np.array([10], dtype=np.int64)
        out = render_binned(0, 10, 3, starts, ends)
        assert len(out) == 4          # ceil(10/3) = 4
        np.testing.assert_allclose(out, 1.0)

    def test_half_gene(self):
        # Gene [0,5), query [0,10), step=5 → [1.0, 0.0]
        starts = np.array([0], dtype=np.int64)
        ends   = np.array([5], dtype=np.int64)
        out = render_binned(0, 10, 5, starts, ends)
        assert len(out) == 2
        np.testing.assert_allclose(out[0], 1.0)
        np.testing.assert_allclose(out[1], 0.0)

    def test_step_1_same_as_render_signal(self):
        starts = np.array([2, 5], dtype=np.int64)
        ends   = np.array([7, 9], dtype=np.int64)
        ref = render_signal(0, 10, starts, ends).astype(np.float64)
        out = render_binned(0, 10, 1, starts, ends)
        np.testing.assert_allclose(out, ref)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            render_binned(0, 10, 0, np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# hilbert_d_to_xy
# ---------------------------------------------------------------------------

class TestHilbertD2XY:
    def test_order_2_all_points(self):
        # 2x2 grid, 4 points.
        # Any valid Hilbert traversal must: start at (0,0), cover all 4 unique cells,
        # and stay within [0,2). The specific orientation is not prescribed.
        x, y = hilbert_d_to_xy(2, np.arange(4))
        coords = list(zip(x.tolist(), y.tolist()))
        assert coords[0] == (0, 0), "Hilbert curve must start at origin"
        assert len(set(coords)) == 4, "All 4 cells must be visited exactly once"
        assert all(0 <= cx < 2 and 0 <= cy < 2 for cx, cy in coords)

    def test_order_4_unique_coords(self):
        n = 4
        d = np.arange(n * n)
        x, y = hilbert_d_to_xy(n, d)
        assert x.shape == (n * n,)
        assert y.shape == (n * n,)
        # All coordinates within [0, n)
        assert (x >= 0).all() and (x < n).all()
        assert (y >= 0).all() and (y < n).all()
        # Each (x,y) pair is unique
        pairs = set(zip(x.tolist(), y.tolist()))
        assert len(pairs) == n * n

    def test_not_power_of_two_raises(self):
        with pytest.raises(ValueError):
            hilbert_d_to_xy(3, np.array([0]))

    def test_empty_input(self):
        x, y = hilbert_d_to_xy(4, np.array([], dtype=np.int64))
        assert len(x) == 0 and len(y) == 0
