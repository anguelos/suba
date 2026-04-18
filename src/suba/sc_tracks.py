"""
suba.sc_tracks
~~~~~~~~~~~~~~
Abstract base class :class:`SCTracks` and concrete :class:`CountMatrixTracks`
for projecting single-cell count matrices onto the universal genome coordinate
space.

Memory guard
------------
The module-level variable :attr:`max_dense_object_size` (default 8 GB) limits
how large a dense result array :meth:`SCTracks.__getitem__` is allowed to
allocate.  Raise it before calling if you have the RAM::

    import suba.sc_tracks as st
    st.max_dense_object_size = 16 * 1024**3

Indexing convention
-------------------
``tracks[cell_key, coord_key]``

*cell_key* may be:

* ``int`` — integer index into :attr:`barcodes`
* ``str`` — barcode string
* ``list`` / ``np.ndarray`` of ints or barcode strings
* ``slice`` — integer slice over cell axis

*coord_key* may be:

* ``int`` — single universal coordinate → scalar per selected cell
* ``slice(a, b)`` — base-resolution signal, shape ``(..., b-a)``
* ``slice(a, b, s)`` — binned average, shape ``(..., ceil((b-a)/s))``
* ``slice(None, None, s)`` — whole-genome binned
* ``str`` — chromosome or gene name (resolved to its coordinate range)

A scalar *cell_key* (int or single string) drops the cell dimension from the
output; a non-scalar key keeps it.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from suba.genome import Genome, _chr_sort_key
from suba.sparse_rendering import find_overlapping_gene_indices


# ---------------------------------------------------------------------------
# Module-level memory guard
# ---------------------------------------------------------------------------

#: Maximum number of bytes a dense result array from :meth:`SCTracks.__getitem__`
#: may occupy.  Raise this limit if you have sufficient RAM.
max_dense_object_size: int = 8 * 1024 ** 3  # 8 GB


def _check_size(shape: tuple, dtype) -> None:
    """Raise MemoryError if the described array would exceed *max_dense_object_size*."""
    import suba.sc_tracks as _st
    n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if n_bytes > _st.max_dense_object_size:
        gb = n_bytes / 1024 ** 3
        limit_gb = _st.max_dense_object_size / 1024 ** 3
        raise MemoryError(
            f"Result would require {gb:.2f} GB "
            f"(limit={limit_gb:.2f} GB). "
            "Adjust suba.sc_tracks.max_dense_object_size to override."
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SCTracks(Genome, ABC):
    """Abstract 2D genomic track: cells × universal coordinates.

    Inherits the universal address space from :class:`~suba.genome.Genome`.
    Subclasses must implement :attr:`barcodes` and :meth:`__getitem__`.

    The ``__getitem__`` interface accepts a 2-tuple
    ``(cell_key, coord_key)`` — see module docstring for details.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def barcodes(self) -> np.ndarray:
        """1D object array of cell barcode strings, length :attr:`n_cells`."""

    @abstractmethod
    def __getitem__(self, key) -> np.ndarray:
        """Return signal values for the given ``(cell_key, coord_key)``."""

    @abstractmethod
    def create_count_matrix(self) -> tuple[np.ndarray, np.ndarray, Any]:
        """Return ``(barcodes, gene_names, matrix)`` from stored data.

        *matrix* has shape ``(n_cells, n_genes)``.  The values match
        whatever was passed to the constructor (raw counts or normalised).
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of cells (rows)."""
        return len(self.barcodes)

    def _resolve_cell_key(self, cell_key) -> tuple[np.ndarray, bool]:
        """Resolve *cell_key* to integer indices and a scalar flag.

        Returns ``(indices, scalar)`` where *scalar* is ``True`` when the
        key was a single int or barcode string (causing the cell dimension
        to be squeezed in the output).
        """
        barcodes = self.barcodes
        if isinstance(cell_key, (int, np.integer)):
            idx = int(cell_key) % self.n_cells
            return np.array([idx], dtype=np.intp), True
        if isinstance(cell_key, str):
            idx = self._barcode_to_idx[cell_key]
            return np.array([idx], dtype=np.intp), True
        if isinstance(cell_key, (list, np.ndarray)):
            arr = np.asarray(cell_key)
            if arr.dtype.kind in ("U", "S", "O"):
                indices = np.array(
                    [self._barcode_to_idx[str(b)] for b in arr], dtype=np.intp
                )
            else:
                indices = arr.astype(np.intp)
            return indices, False
        if isinstance(cell_key, slice):
            return np.arange(self.n_cells, dtype=np.intp)[cell_key], False
        raise TypeError(
            f"Unsupported cell key type: {type(cell_key).__name__}. "
            "Expected int, str, list, or slice."
        )


# ---------------------------------------------------------------------------
# CountMatrixTracks
# ---------------------------------------------------------------------------

class CountMatrixTracks(SCTracks):
    """Project a single-cell count matrix onto the universal genome address space.

    Each gene in the count matrix is mapped to its genomic span (from the
    :class:`~suba.genome.Genome` annotation).  The signal at a position is:

    * **Raw counts** (``length_normalize=True``, the default for integer
      matrices): ``count / gene_length_bp`` per base, so integrating
      over the gene body recovers the original count.
    * **Pre-normalised** (``length_normalize=False``, the default for float
      matrices): the stored value is spread flat across the gene body.
      Querying the mean over the gene body recovers the original value.

    For positions covered by multiple genes:

    * Raw / length-normalised mode: values are **summed** (total density).
    * Pre-normalised mode: values are **averaged** (weighted by coverage
      fraction in each bin).

    Parameters
    ----------
    matrix:
        Count matrix, shape ``(n_cells, n_genes)``.  Converted to
        ``scipy.sparse.csr_matrix`` internally.
    barcodes:
        Cell barcode strings, length ``n_cells``.
    gene_names:
        Gene name strings, length ``n_genes``.  Should match the HGNC symbols
        used in the genome annotation (without strand suffix).
    genome:
        Pre-built :class:`~suba.genome.Genome`.  If ``None``, one is
        constructed using *genome_kwargs*.
    length_normalize:
        ``True`` → values are raw counts, divide by gene length for signal.
        ``False`` → values are pre-normalised, use flat.
        ``None`` (default) → auto-detect from *matrix* dtype (integer →
        ``True``, float → ``False``).
    **genome_kwargs:
        Forwarded to :class:`~suba.genome.Genome` when *genome* is ``None``.
    """

    def __init__(
        self,
        matrix,
        barcodes,
        gene_names,
        genome: Optional[Genome] = None,
        length_normalize: Optional[bool] = None,
        **genome_kwargs,
    ) -> None:
        from scipy.sparse import issparse, csr_matrix

        # ── Build or reuse Genome ─────────────────────────────────────
        if genome is not None:
            # Copy genome internals rather than calling __init__ (avoids IO)
            self.__dict__.update(
                {k: v for k, v in genome.__dict__.items()}
            )
        else:
            Genome.__init__(self, **genome_kwargs)

        # ── Store count matrix ────────────────────────────────────────
        m = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
        self._matrix = m.astype(np.float32)
        self._barcodes: np.ndarray = np.asarray(barcodes, dtype=object)
        self._gene_names_matrix: np.ndarray = np.asarray(gene_names, dtype=object)

        if length_normalize is None:
            length_normalize = np.issubdtype(matrix.dtype, np.integer)
        self._length_normalize: bool = bool(length_normalize)

        # ── Barcode lookup ────────────────────────────────────────────
        self._barcode_to_idx: dict[str, int] = {
            str(b): i for i, b in enumerate(self._barcodes)
        }

        # ── Map count-matrix columns → genome gene indices ────────────
        # genome gene names have a strand suffix (BRCA1+, BRCA1-)
        # count matrix gene names do not (BRCA1)
        # Per spec: spread value across all strand variants as if one gene
        n_genome_genes = len(self._gene_names)
        # genome_gene_to_col[g] = matrix column index, or -1 if not in matrix
        genome_gene_to_col = np.full(n_genome_genes, -1, dtype=np.int32)

        matrix_gene_to_col: dict[str, int] = {
            str(gn): i for i, gn in enumerate(self._gene_names_matrix)
        }

        for g_idx, gname in enumerate(self._gene_names):
            # Strip strand suffix
            base = str(gname).rstrip("+-")
            col = matrix_gene_to_col.get(base, -1)
            genome_gene_to_col[g_idx] = col

        self._genome_gene_to_col: np.ndarray = genome_gene_to_col

    # ------------------------------------------------------------------
    # SCTracks abstract interface
    # ------------------------------------------------------------------

    @property
    def barcodes(self) -> np.ndarray:
        return self._barcodes

    def create_count_matrix(self) -> tuple[np.ndarray, np.ndarray, Any]:
        """Return ``(barcodes, gene_names, matrix)`` as stored.

        The returned *matrix* is the original ``(n_cells, n_genes)``
        sparse matrix before any genomic projection.  Use this for
        reconstruction / round-trip tests.
        """
        return self._barcodes, self._gene_names_matrix, self._matrix

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """``tracks[cell_key, coord_key]`` — see module docstring."""
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                "SCTracks requires a 2-tuple key: tracks[cell_key, coord_key]. "
                f"Got: {type(key).__name__}"
            )
        cell_key, coord_key = key
        cell_indices, scalar_cell = self._resolve_cell_key(cell_key)
        start, stop, step = self._resolve_coord_key(coord_key)

        if step is None or step == 1:
            result = self._render_base(cell_indices, start, stop)
        else:
            result = self._render_binned(cell_indices, start, stop, step)

        if scalar_cell:
            return result[0]  # squeeze cell axis
        return result

    def _resolve_coord_key(self, coord_key) -> tuple[int, int, Optional[int]]:
        """Return ``(start, stop, step)`` from a coord key.

        *step* is ``None`` for base-resolution queries.
        """
        if isinstance(coord_key, (int, np.integer)):
            pos = int(coord_key)
            if pos < 0:
                pos += self._total_length
            return pos, pos + 1, None

        if isinstance(coord_key, str):
            s, e = self._resolve_name(coord_key)
            return s, e, None

        if isinstance(coord_key, slice):
            raw_start = coord_key.start
            raw_stop  = coord_key.stop
            raw_step  = coord_key.step

            start = int(raw_start) if raw_start is not None else 0
            stop  = int(raw_stop)  if raw_stop  is not None else self._total_length
            step  = int(raw_step)  if raw_step  is not None and raw_step != 1 else None
            start = max(0, start)
            stop  = min(self._total_length, stop)
            return start, stop, step

        raise TypeError(
            f"Unsupported coord key type: {type(coord_key).__name__}. "
            "Expected int, slice, or str."
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_base(
        self, cell_indices: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        """Base-resolution signal, shape ``(n_sel, stop-start)``."""
        length = stop - start
        n_sel = len(cell_indices)
        _check_size((n_sel, length), np.float32)

        result = np.zeros((n_sel, length), dtype=np.float32)

        gene_idx = find_overlapping_gene_indices(
            start, stop,
            self._gene_start_addr, self._gene_end_addr,
            self._gene_start_sort_idx,
        )
        if len(gene_idx) == 0:
            return result

        # For average mode: accumulate coverage count per base position
        if not self._length_normalize:
            coverage = np.zeros(length, dtype=np.int32)

        for g in gene_idx:
            col = int(self._genome_gene_to_col[g])
            if col < 0:
                continue
            gene_len = int(self._gene_end_addr[g]) - int(self._gene_start_addr[g])
            eff_s = int(max(self._gene_start_addr[g], start)) - start
            eff_e = int(min(self._gene_end_addr[g],  stop))  - start

            # Extract values for selected cells — shape (n_sel,)
            vals = np.asarray(
                self._matrix[cell_indices, col].todense()
            ).ravel()

            if self._length_normalize:
                per_base = vals / gene_len  # shape (n_sel,)
                result[:, eff_s:eff_e] += per_base[:, np.newaxis]
            else:
                result[:, eff_s:eff_e] += vals[:, np.newaxis]
                coverage[eff_s:eff_e] += 1

        if not self._length_normalize:
            # Average where multiple genes overlap
            multi = coverage > 1
            if multi.any():
                result[:, multi] /= coverage[multi]

        return result

    def _render_binned(
        self,
        cell_indices: np.ndarray,
        start: int,
        stop: int,
        step: int,
    ) -> np.ndarray:
        """Binned signal via sparse weight matrix multiply.

        Builds a sparse weight matrix ``W`` of shape
        ``(n_matrix_genes, n_bins)`` where ``W[col, b]`` is the
        contribution weight of gene *col* to bin *b*, then computes
        ``matrix_selected @ W``.

        For raw counts (length_normalize):  ``W[col,b] = overlap_bases / gene_len``
        For pre-normalised:                 ``W[col,b] = overlap_bases``
        (divided by total coverage per bin at the end for averaging).
        """
        length = stop - start
        n_bins = math.ceil(length / step)
        n_sel  = len(cell_indices)
        _check_size((n_sel, n_bins), np.float32)

        from scipy.sparse import coo_matrix

        gene_idx = find_overlapping_gene_indices(
            start, stop,
            self._gene_start_addr, self._gene_end_addr,
            self._gene_start_sort_idx,
        )
        if len(gene_idx) == 0:
            return np.zeros((n_sel, n_bins), dtype=np.float32)

        n_cols = self._matrix.shape[1]

        # Build weight matrix W (COO → CSR)
        rows, cols_w, data_w = [], [], []
        total_coverage = np.zeros(n_bins, dtype=np.float64)

        for g in gene_idx:
            col = int(self._genome_gene_to_col[g])
            if col < 0:
                continue
            gene_len = int(self._gene_end_addr[g]) - int(self._gene_start_addr[g])
            eff_s = int(max(self._gene_start_addr[g], start)) - start
            eff_e = int(min(self._gene_end_addr[g],  stop))  - start

            fb = eff_s // step
            lb = (eff_e - 1) // step

            # Compute overlap_bases per touched bin
            if fb == lb:
                bin_range = [fb]
                overlaps  = [eff_e - eff_s]
            else:
                bin_range = list(range(fb, lb + 1))
                overlaps  = [0] * len(bin_range)
                overlaps[0]  = (fb + 1) * step - eff_s
                overlaps[-1] = eff_e - lb * step
                for j in range(1, len(bin_range) - 1):
                    overlaps[j] = step

            for b, ov in zip(bin_range, overlaps):
                w = ov / gene_len if self._length_normalize else float(ov)
                rows.append(col)
                cols_w.append(b)
                data_w.append(w)
                if not self._length_normalize:
                    total_coverage[b] += ov

        if not rows:
            return np.zeros((n_sel, n_bins), dtype=np.float32)

        W = coo_matrix(
            (data_w, (rows, cols_w)), shape=(n_cols, n_bins), dtype=np.float32
        ).tocsr()

        mat_sel = self._matrix[cell_indices, :]  # (n_sel, n_cols) sparse
        result = (mat_sel @ W).toarray().astype(np.float32)  # (n_sel, n_bins)

        if not self._length_normalize:
            # Weighted average: divide by total coverage per bin
            denom = total_coverage.astype(np.float32)
            mask = denom > 0
            result[:, mask] /= denom[mask]

        # Correct last bin if partial (mean over actual elements, not step)
        remainder = length % step
        if remainder and self._length_normalize:
            result[:, -1] *= step / remainder

        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CountMatrixTracks("
            f"n_cells={self.n_cells}, "
            f"n_matrix_genes={len(self._gene_names_matrix)}, "
            f"length_normalize={self._length_normalize}, "
            f"genome={Genome.__repr__(self)})"
        )
