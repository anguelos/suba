"""
suba.sparse_rendering
~~~~~~~~~~~~~~~~~~~~~
Pure stateless functions for sparse genomic signal rendering.

All functions are side-effect-free and operate only on NumPy arrays.
They are used by :class:`suba.Genome` and by every subclass so that the
core algorithms remain independently testable.

Coordinate convention
---------------------
All coordinates are **0-based, half-open** ``[start, end)``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Interval overlap query
# ---------------------------------------------------------------------------

def find_overlapping_gene_indices(
    q_start: int,
    q_end: int,
    gene_starts: np.ndarray,
    gene_ends: np.ndarray,
    start_sort_idx: np.ndarray,
) -> np.ndarray:
    """Return indices of genes whose span overlaps the query range ``[q_start, q_end)``.

    A gene at ``[g_start, g_end)`` overlaps ``[q_start, q_end)`` iff::

        g_start < q_end  AND  g_end > q_start

    The search is O(log n + k) where k is the number of candidates with
    start < q_end (not all of which necessarily overlap).

    Parameters
    ----------
    q_start, q_end:
        Query range, 0-based half-open universal coordinates.
    gene_starts:
        Gene start positions (universal coordinates, any order).
    gene_ends:
        Gene end positions (universal coordinates, exclusive, any order).
    start_sort_idx:
        ``np.argsort(gene_starts, kind="stable")`` — indices that sort
        *gene_starts* in ascending order.

    Returns
    -------
    np.ndarray of int
        Indices into *gene_starts* / *gene_ends* of all overlapping genes.
    """
    sorted_starts = gene_starts[start_sort_idx]
    # All genes with start < q_end (searchsorted left = first index >= q_end)
    n = int(np.searchsorted(sorted_starts, q_end, side="left"))
    candidates = start_sort_idx[:n]
    # Keep only those whose end is strictly after q_start
    return candidates[gene_ends[candidates] > q_start]


# ---------------------------------------------------------------------------
# Parallel prefix sum (cumsum)
# ---------------------------------------------------------------------------

def parallel_cumsum(arr: np.ndarray, n_threads: int = 0) -> np.ndarray:
    """Parallel prefix sum using the 3-phase work-efficient algorithm.

    For arrays smaller than ``_PARALLEL_CUMSUM_THRESHOLD`` the standard
    ``np.cumsum`` is used (less overhead).  Above that threshold the array
    is split into *n_threads* chunks and processed as follows:

    1. **Phase 1 (parallel):** ``np.cumsum`` each chunk independently.
    2. **Phase 2 (sequential):** accumulate per-chunk carry values
       — only *n_threads* additions, negligible cost.
    3. **Phase 3 (parallel):** add the carry offset to each chunk.

    numpy releases the GIL during array operations so Python threads give
    true CPU parallelism here.

    Parameters
    ----------
    arr:
        1D integer or float array.
    n_threads:
        Number of worker threads.  ``0`` (default) uses
        ``os.cpu_count()``.

    Returns
    -------
    np.ndarray
        Same dtype and shape as *arr*.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    n = len(arr)
    if n_threads <= 0:
        n_threads = os.cpu_count() or 4

    if n < _PARALLEL_CUMSUM_THRESHOLD or n_threads == 1:
        return np.cumsum(arr)

    chunk_size = (n + n_threads - 1) // n_threads
    chunk_starts = list(range(0, n, chunk_size))
    chunks_in = [arr[s : s + chunk_size] for s in chunk_starts]

    # Phase 1: local cumsum — parallel
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        local_cumsums = list(pool.map(np.cumsum, chunks_in))

    # Phase 2: carry offsets — sequential, O(n_threads)
    carries = [arr.dtype.type(0)]
    for lc in local_cumsums[:-1]:
        carries.append(carries[-1] + lc[-1])

    # Phase 3: apply carry to each chunk — parallel
    result = np.empty(n, dtype=arr.dtype)

    def _apply(args):
        lc, carry, out = args
        np.add(lc, carry, out=out)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(
            _apply,
            [
                (lc, carry, result[s : s + chunk_size])
                for lc, carry, s in zip(local_cumsums, carries, chunk_starts)
            ],
        ))

    return result


# Minimum array length before switching to parallel cumsum.
# Below this threshold np.cumsum has lower overhead.
_PARALLEL_CUMSUM_THRESHOLD: int = 5_000_000


# ---------------------------------------------------------------------------
# Signal rendering
# ---------------------------------------------------------------------------

def render_signal(
    q_start: int,
    q_end: int,
    gene_starts: np.ndarray,
    gene_ends: np.ndarray,
) -> np.ndarray:
    """Render a piecewise-constant gene-overlap signal for ``[q_start, q_end)``.

    Uses the start/end increment + cumulative-sum algorithm:

    * ``+1`` at each gene's (clipped) start position
    * ``-1`` at each gene's (clipped) end position
    * ``np.add.at`` is used so that repeated start/end positions are
      accumulated correctly (avoids the silent ``+=`` overwrite bug).
    * ``cumsum`` integrates the increments into a coverage count.

    Parameters
    ----------
    q_start, q_end:
        Query range, 0-based half-open universal coordinates.
    gene_starts, gene_ends:
        Universal coordinates of the genes to render (already filtered to
        overlap the query range, though extra genes are harmless).

    Returns
    -------
    np.ndarray of int8, shape ``(q_end - q_start,)``
        Value at index ``i`` equals the number of genes that overlap position
        ``q_start + i``.  0 = intergenic, 1 = within one gene, 2 = two
        overlapping genes, etc.
    """
    length = q_end - q_start
    # int8 uses 4x less memory than int32, giving ~4x better throughput on the
    # cumsum (memory-bandwidth-bound).  Gene overlap counts exceeding 127 are
    # not representable; in practice that never occurs for gene-body annotations.
    # Allocate one extra element so end markers equal to `length` are in-bounds.
    arr = np.zeros(length + 1, dtype=np.int8)

    eff_starts = np.clip(gene_starts, q_start, q_end) - q_start
    eff_ends   = np.clip(gene_ends,   q_start, q_end) - q_start

    # Discard genes whose clipped span has zero length (fully outside range).
    mask = eff_ends > eff_starts
    eff_starts = eff_starts[mask]
    eff_ends   = eff_ends[mask]

    np.add.at(arr, eff_starts, 1)
    np.add.at(arr, eff_ends,  -1)

    # dtype=np.int8 keeps the output compact and fast (numpy would otherwise
    # upcast to int64 by default, wasting bandwidth).
    return np.cumsum(arr, dtype=np.int8)[:length]


def render_binned(
    q_start: int,
    q_end: int,
    step: int,
    gene_starts: np.ndarray,
    gene_ends: np.ndarray,
) -> np.ndarray:
    """Render a binned-average gene-overlap signal for ``[q_start, q_end)``.

    Each output element is the mean of *step* consecutive base-resolution
    values from :func:`render_signal`.  The last bin, if smaller than *step*,
    is averaged over its actual number of elements (not zero-padded).

    Parameters
    ----------
    q_start, q_end:
        Query range, 0-based half-open universal coordinates.
    step:
        Bin size in bases.  Must be >= 1.
    gene_starts, gene_ends:
        Universal coordinates of genes to render.

    Returns
    -------
    np.ndarray of float64, shape ``(ceil((q_end - q_start) / step),)``
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    signal = render_signal(q_start, q_end, gene_starts, gene_ends).astype(np.float64)
    length = len(signal)
    n_full = length // step
    remainder = length % step

    result_size = n_full + (1 if remainder else 0)
    result = np.empty(result_size, dtype=np.float64)

    if n_full > 0:
        result[:n_full] = signal[: n_full * step].reshape(n_full, step).mean(axis=1)
    if remainder:
        result[n_full] = signal[n_full * step :].mean()

    return result



def render_binned_direct(
    q_start: int,
    q_end: int,
    step: int,
    gene_starts: np.ndarray,
    gene_ends: np.ndarray,
) -> np.ndarray:
    """Render a binned-average gene-overlap signal without a base-resolution intermediate.

    Produces **identical output** to :func:`render_binned` but runs in
    O(n_genes) time and O(n_bins) memory instead of O(n_bases) for both.

    The idea: for each gene ``[gs, ge)`` the contribution to bin ``b``
    (covering ``[q_start + b*step, q_start + (b+1)*step)``) is exactly::

        min(ge, b_end) - max(gs, b_start)      (clamped to [0, step])

    All arithmetic stays in base coordinates (64-bit integers), so there
    are no rounding errors vs. the cumsum path.  The final ``/ step``
    (or ``/ remainder``) is the only floating-point operation, matching
    what :func:`render_binned` does.

    Multi-bin genes (spanning three or more bins) are handled with the
    same start/end increment + cumsum trick used in :func:`render_signal`,
    applied at bin resolution rather than base resolution.

    Parameters
    ----------
    q_start, q_end:
        Query range, 0-based half-open universal coordinates.
    step:
        Bin size in bases.  Must be >= 1.
    gene_starts, gene_ends:
        Universal coordinates of genes to render.

    Returns
    -------
    np.ndarray of float64, shape ``(ceil((q_end - q_start) / step),)``
    """
    import math as _math

    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    length = q_end - q_start
    n_full = length // step
    remainder = length % step
    n_bins = n_full + (1 if remainder else 0)

    # Clip gene coordinates to the query window and shift to 0-based local coords
    eff_starts = np.clip(gene_starts, q_start, q_end) - q_start
    eff_ends   = np.clip(gene_ends,   q_start, q_end) - q_start
    mask = eff_ends > eff_starts
    eff_starts = eff_starts[mask]
    eff_ends   = eff_ends[mask]

    # Bin index of the first and last base of each clipped gene
    first_bins = eff_starts // step
    last_bins  = (eff_ends - 1) // step  # inclusive last bin

    # Accumulate genic-base counts per bin (integer, exact)
    counts = np.zeros(n_bins, dtype=np.int64)

    single = first_bins == last_bins
    multi  = ~single

    # ── Single-bin genes ──────────────────────────────────────────────
    np.add.at(counts, first_bins[single], eff_ends[single] - eff_starts[single])

    # ── Multi-bin genes ───────────────────────────────────────────────
    if multi.any():
        ms = eff_starts[multi]
        me = eff_ends[multi]
        fb = first_bins[multi]
        lb = last_bins[multi]

        # Partial first bin: [ms .. (fb+1)*step)
        np.add.at(counts, fb, (fb + 1) * step - ms)
        # Partial last bin:  [lb*step .. me)
        np.add.at(counts, lb, me - lb * step)

        # Fully-covered middle bins: use start/end increment + cumsum
        # so that a gene spanning k bins costs O(1) not O(k).
        diff = np.zeros(n_bins + 1, dtype=np.int64)
        np.add.at(diff, fb + 1,  step)   # each middle range starts after fb
        np.add.at(diff, lb,     -step)   # each middle range ends before lb
        counts += np.cumsum(diff)[:n_bins]

    # ── Convert counts to mean overlap ────────────────────────────────
    result = np.empty(n_bins, dtype=np.float64)
    if n_full > 0:
        result[:n_full] = counts[:n_full] / step
    if remainder:
        result[n_full] = counts[n_full] / remainder

    return result

# ---------------------------------------------------------------------------
# Hilbert curve — pure NumPy, no external dependency
# ---------------------------------------------------------------------------

def hilbert_d_to_xy(n: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Hilbert curve 1D indices to 2D ``(x, y)`` coordinates.

    Implements the standard bit-manipulation algorithm in a fully vectorised
    NumPy form — no Python loops over individual indices.

    Parameters
    ----------
    n:
        Grid side length; **must be a power of 2**.  The Hilbert curve fills
        an ``n × n`` grid, so valid indices are in ``[0, n*n)``.
    d:
        1D Hilbert curve indices, shape ``(m,)``.

    Returns
    -------
    x, y : np.ndarray of int64, each shape ``(m,)``
        2D coordinates corresponding to each index in *d*.

    Examples
    --------
    >>> x, y = hilbert_d_to_xy(2, np.arange(4))
    >>> list(zip(x.tolist(), y.tolist()))
    [(0, 0), (1, 0), (1, 1), (0, 1)]
    """
    if n & (n - 1):
        raise ValueError(f"n must be a power of 2, got {n}")

    d = np.asarray(d, dtype=np.int64)
    x = np.zeros_like(d)
    y = np.zeros_like(d)
    q = d.copy()

    s = 1
    while s < n:
        rx = (q >> 1) & 1
        ry = (q ^ rx) & 1

        # Where ry == 0: rotate the quadrant
        rot = ry == 0
        flip = rot & (rx == 1)

        # Flip around the anti-diagonal when rx==1 and ry==0
        x[flip] = s - 1 - x[flip]
        y[flip] = s - 1 - y[flip]

        # Transpose (swap x and y) where ry == 0
        tmp = x[rot].copy()
        x[rot] = y[rot]
        y[rot] = tmp

        x += s * rx
        y += s * ry

        q >>= 2
        s <<= 1

    return x, y
