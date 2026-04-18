"""
suba.genome
~~~~~~~~~~~
:class:`Genome` — base class representing a genome as a universal 1D address
space.

Every chromosome is laid end-to-end with fixed inter-chromosomal padding so
that all genomic signals can share a single integer coordinate.  No dense
signal array is allocated; gene annotations are stored sparsely and rendered
on demand via :mod:`suba.sparse_rendering`.

Coordinate convention
---------------------
**0-based, half-open** ``[start, end)`` everywhere.  GTF input (1-based
inclusive) is converted on parse::

    uc_start = chr_offset + (gtf_start - 1)
    uc_end   = chr_offset +  gtf_end

Padding value
-------------
Positions inside inter-chromosomal padding return ``Genome.PADDING_VALUE``
(``-1``) for the base class.  Subclasses may override this.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from suba.sparse_rendering import (
    find_overlapping_gene_indices,
    render_binned,
    render_binned_direct,
    render_signal,
)
from suba.io.genome import (
    infer_chrom_sizes_url as _infer_chrom_sizes_url,
    ensure_cached as _ensure_cached_io,
    parse_gtf_genes as _parse_gtf_genes,
    parse_chrom_sizes as _parse_chrom_sizes,
    npz_cache_path as _npz_cache_path,
    save_gene_cache as _save_gene_cache,
    load_gene_cache as _load_gene_cache,
)

_ALT_PATTERN = re.compile(r"_(random|alt|fix|hap\d+|Un)", re.IGNORECASE)
_CHRUN_PATTERN = re.compile(r"^chrUn_", re.IGNORECASE)

_DEFAULT_GTF_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/"
    "hg38.ncbiRefSeq.gtf.gz"
)


def _chr_sort_key(name: str) -> tuple:
    """Natural-sort key: numeric chromosomes first, then alphabetical."""
    stripped = name.lower()
    for prefix in ("chromosome", "chrom", "chr"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped.isdigit():
        return (0, int(stripped), "")
    return (1, 0, stripped)


def _is_alt_chromosome(name: str) -> bool:
    """Return True if *name* looks like an alt/patch/random contig."""
    return bool(_ALT_PATTERN.search(name)) or bool(_CHRUN_PATTERN.match(name))


class Genome:
    """Universal 1D genome address space built from a GTF annotation.

    All chromosomes (excluding alt/patch contigs by default) are concatenated
    in natural sort order with a fixed inter-chromosomal padding gap.  Gene
    annotations are stored as sparse arrays; signals are rendered on demand
    via :mod:`suba.sparse_rendering`.

    Parameters
    ----------
    gtf_url:
        URL or local file path to a GTF (or ``.gtf.gz``) annotation.
        Defaults to UCSC hg38 knownGene.
    chrom_sizes_url:
        URL or local path to a UCSC chrom.sizes file.  If ``None`` and the
        GTF URL looks like a UCSC URL, the matching chrom.sizes is inferred
        automatically.
    padding:
        Number of bases inserted between consecutive chromosomes (default
        1 Mbp).  No padding after the last chromosome.
    cache_dir:
        Directory for downloaded files.  Created if absent.
    include_alt:
        If ``False`` (default), exclude alt/patch/random contigs.

    Notes
    -----
    **Coordinate convention**: 0-based half-open ``[start, end)`` throughout.

    **Padding**: inter-chromosomal positions return
    :attr:`PADDING_VALUE` (``-1``) when queried.

    **Gene names**: duplicate gene names on the same strand are merged
    (union of spans).  Strand is appended as a suffix: ``BRCA1+``,
    ``BRCA1-``.  Querying ``genome['BRCA1']`` (no suffix) raises
    ``KeyError`` with a helpful message.
    """

    PADDING_VALUE: int = -1

    DEFAULT_GTF_URL: str = _DEFAULT_GTF_URL

    def __init__(
        self,
        gtf_url: str = _DEFAULT_GTF_URL,
        chrom_sizes_url: Optional[str] = None,
        padding: int = 1_000_000,
        cache_dir: str = "./tmp/cache",
        include_alt: bool = False,
        transcript_id_prefixes: Optional[list] = ("NM_",),
    ) -> None:
        self._padding = padding
        self._include_alt = include_alt
        self._transcript_id_prefixes = transcript_id_prefixes
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve chrom sizes URL
        if chrom_sizes_url is None:
            chrom_sizes_url = _infer_chrom_sizes_url(gtf_url)

        # Ensure files are cached locally
        gtf_path = self._ensure_cached(gtf_url)
        chrom_sizes: dict[str, int] = {}
        if chrom_sizes_url:
            sizes_path = self._ensure_cached(chrom_sizes_url)
            chrom_sizes = _parse_chrom_sizes(sizes_path)

        # Parse gene records from GTF
        npz_path = _npz_cache_path(gtf_path, transcript_id_prefixes)
        if npz_path.exists() and npz_path.stat().st_mtime >= gtf_path.stat().st_mtime:
            raw_genes = _load_gene_cache(npz_path)
        else:
            raw_genes = _parse_gtf_genes(gtf_path, transcript_id_prefixes=transcript_id_prefixes)
            _save_gene_cache(npz_path, raw_genes)

        # Determine chromosome list
        if chrom_sizes:
            all_chroms = list(chrom_sizes.keys())
        else:
            all_chroms = sorted(
                {g[0] for g in raw_genes}, key=_chr_sort_key
            )

        # Filter alt chromosomes
        if not include_alt:
            all_chroms = [c for c in all_chroms if not _is_alt_chromosome(c)]

        # Natural sort
        chromosomes = sorted(all_chroms, key=_chr_sort_key)

        # Build chromosome address arrays
        n_chrs = len(chromosomes)
        chr_starts = np.zeros(n_chrs, dtype=np.int64)
        chr_ends = np.zeros(n_chrs, dtype=np.int64)

        offset = 0
        for i, chrom in enumerate(chromosomes):
            if chrom in chrom_sizes:
                size = chrom_sizes[chrom]
            else:
                # Infer from GTF max end coordinate
                ends = [g[2] for g in raw_genes if g[0] == chrom]
                size = max(ends) if ends else 0
                if size == 0:
                    warnings.warn(
                        f"Chromosome {chrom!r} has no gene records; size set to 0.",
                        stacklevel=2,
                    )
            chr_starts[i] = offset
            chr_ends[i] = offset + size
            offset += size
            if i < n_chrs - 1:
                offset += padding

        self._total_length: int = offset
        self._chromosome_start_addr: np.ndarray = chr_starts
        self._chromosome_end_addr: np.ndarray = chr_ends
        self._chromosome_name: np.ndarray = np.array(chromosomes, dtype=object)
        self._chr_name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(chromosomes)
        }

        # Build gene arrays
        chr_offset_map = {chrom: chr_starts[i] for i, chrom in enumerate(chromosomes)}

        # gene_key -> [uc_start, uc_end, chr_idx, strand_val]
        gene_dict: dict[str, list] = {}
        for chrom, start, end, strand, gene_name in raw_genes:
            if chrom not in chr_offset_map:
                continue
            chr_offset = chr_offset_map[chrom]
            chr_idx = self._chr_name_to_idx[chrom]
            strand_val = 1 if strand == "+" else (-1 if strand == "-" else 0)

            if strand == "+":
                gene_key = f"{gene_name}+"
            elif strand == "-":
                gene_key = f"{gene_name}-"
            else:
                gene_key = gene_name

            uc_start = chr_offset + start
            uc_end = chr_offset + end

            if gene_key in gene_dict:
                existing = gene_dict[gene_key]
                existing[0] = min(existing[0], uc_start)
                existing[1] = max(existing[1], uc_end)
            else:
                gene_dict[gene_key] = [uc_start, uc_end, chr_idx, strand_val]

        if gene_dict:
            keys = list(gene_dict.keys())
            vals = list(gene_dict.values())
            self._gene_names: np.ndarray = np.array(keys, dtype=object)
            self._gene_start_addr: np.ndarray = np.array(
                [v[0] for v in vals], dtype=np.int64
            )
            self._gene_end_addr: np.ndarray = np.array(
                [v[1] for v in vals], dtype=np.int64
            )
            self._gene_chr_idx: np.ndarray = np.array(
                [v[2] for v in vals], dtype=np.int32
            )
            self._gene_strand: np.ndarray = np.array(
                [v[3] for v in vals], dtype=np.int8
            )
        else:
            self._gene_names = np.array([], dtype=object)
            self._gene_start_addr = np.array([], dtype=np.int64)
            self._gene_end_addr = np.array([], dtype=np.int64)
            self._gene_chr_idx = np.array([], dtype=np.int32)
            self._gene_strand = np.array([], dtype=np.int8)

        self._gene_start_sort_idx: np.ndarray = np.argsort(
            self._gene_start_addr, kind="stable"
        )
        self._gene_end_sort_idx: np.ndarray = np.argsort(
            self._gene_end_addr, kind="stable"
        )
        self._gene_name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._gene_names)
        }

    # ------------------------------------------------------------------
    # Class method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        chromosome_names: list[str],
        chromosome_sizes: list[int],
        gene_names: list[str],
        gene_chroms: list[str],
        gene_local_starts: list[int],
        gene_local_ends: list[int],
        gene_strands: list[str],
        padding: int = 1_000_000,
    ) -> "Genome":
        """Construct a :class:`Genome` directly from pre-parsed arrays.

        Bypasses all file I/O — useful for unit tests and synthetic genomes.

        Parameters
        ----------
        chromosome_names:
            Ordered list of chromosome names.
        chromosome_sizes:
            Size in bases for each chromosome (same order).
        gene_names:
            Base gene names (strand suffix will be appended automatically).
        gene_chroms:
            Chromosome name for each gene.
        gene_local_starts:
            Gene start in 0-based chromosomal (local) coordinates.
        gene_local_ends:
            Gene end in 0-based half-open chromosomal coordinates.
        gene_strands:
            Strand per gene: ``"+"`` , ``"-"``, or ``"."``.
        padding:
            Inter-chromosomal padding in bases.
        """
        obj = cls.__new__(cls)
        obj._padding = padding
        obj._include_alt = False
        obj._transcript_id_prefixes = None
        obj._cache_dir = Path(".")

        # Sort chromosomes using the same natural-sort key as __init__
        paired = sorted(zip(chromosome_names, chromosome_sizes), key=lambda t: _chr_sort_key(t[0]))
        chromosome_names = [p[0] for p in paired]
        chromosome_sizes = [p[1] for p in paired]

        n_chrs = len(chromosome_names)
        chr_starts = np.zeros(n_chrs, dtype=np.int64)
        chr_ends = np.zeros(n_chrs, dtype=np.int64)

        offset = 0
        for i, (name, size) in enumerate(zip(chromosome_names, chromosome_sizes)):
            chr_starts[i] = offset
            chr_ends[i] = offset + size
            offset += size
            if i < n_chrs - 1:
                offset += padding

        obj._total_length = offset
        obj._chromosome_start_addr = chr_starts
        obj._chromosome_end_addr = chr_ends
        obj._chromosome_name = np.array(chromosome_names, dtype=object)
        obj._chr_name_to_idx = {n: i for i, n in enumerate(chromosome_names)}

        chr_offset_map = {n: chr_starts[i] for i, n in enumerate(chromosome_names)}

        gene_dict: dict[str, list] = {}
        for gname, chrom, ls, le, strand in zip(
            gene_names, gene_chroms, gene_local_starts, gene_local_ends, gene_strands
        ):
            if chrom not in chr_offset_map:
                continue
            off = chr_offset_map[chrom]
            chr_idx = obj._chr_name_to_idx[chrom]
            strand_val = 1 if strand == "+" else (-1 if strand == "-" else 0)
            gene_key = (
                f"{gname}+" if strand == "+"
                else (f"{gname}-" if strand == "-" else gname)
            )
            uc_start, uc_end = off + ls, off + le
            if gene_key in gene_dict:
                gene_dict[gene_key][0] = min(gene_dict[gene_key][0], uc_start)
                gene_dict[gene_key][1] = max(gene_dict[gene_key][1], uc_end)
            else:
                gene_dict[gene_key] = [uc_start, uc_end, chr_idx, strand_val]

        if gene_dict:
            keys = list(gene_dict.keys())
            vals = list(gene_dict.values())
            obj._gene_names = np.array(keys, dtype=object)
            obj._gene_start_addr = np.array([v[0] for v in vals], dtype=np.int64)
            obj._gene_end_addr = np.array([v[1] for v in vals], dtype=np.int64)
            obj._gene_chr_idx = np.array([v[2] for v in vals], dtype=np.int32)
            obj._gene_strand = np.array([v[3] for v in vals], dtype=np.int8)
        else:
            obj._gene_names = np.array([], dtype=object)
            obj._gene_start_addr = np.array([], dtype=np.int64)
            obj._gene_end_addr = np.array([], dtype=np.int64)
            obj._gene_chr_idx = np.array([], dtype=np.int32)
            obj._gene_strand = np.array([], dtype=np.int8)

        obj._gene_start_sort_idx = np.argsort(obj._gene_start_addr, kind="stable")
        obj._gene_end_sort_idx = np.argsort(obj._gene_end_addr, kind="stable")
        obj._gene_name_to_idx = {n: i for i, n in enumerate(obj._gene_names)}
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> int:
        """Total length of the universal address space in bases."""
        return self._total_length

    @property
    def n_chromosomes(self) -> int:
        """Number of chromosomes in this genome."""
        return len(self._chromosome_name)

    @property
    def n_genes(self) -> int:
        """Number of gene entries (strand-suffixed)."""
        return len(self._gene_names)

    @property
    def padding(self) -> int:
        """Inter-chromosomal padding size in bases."""
        return self._padding

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def genomic_to_universal(self, chrom: str, pos: int) -> int:
        """Convert a chromosomal position to a universal coordinate.

        Parameters
        ----------
        chrom:
            Chromosome name (e.g. ``"chr1"``).
        pos:
            0-based position within the chromosome.

        Returns
        -------
        int
            Universal coordinate.
        """
        if chrom not in self._chr_name_to_idx:
            raise KeyError(f"Unknown chromosome: {chrom!r}")
        idx = self._chr_name_to_idx[chrom]
        uc = int(self._chromosome_start_addr[idx]) + pos
        if uc >= self._chromosome_end_addr[idx]:
            raise ValueError(
                f"Position {pos} is beyond the end of {chrom!r} "
                f"(size {int(self._chromosome_end_addr[idx] - self._chromosome_start_addr[idx])})"
            )
        return uc

    def universal_to_genomic(self, uc: int) -> tuple[str, int]:
        """Convert a universal coordinate to ``(chromosome, local_pos)``.

        Parameters
        ----------
        uc:
            Universal coordinate.

        Returns
        -------
        tuple[str, int]
            ``(chromosome_name, 0-based_local_position)``

        Raises
        ------
        ValueError
            If *uc* is in a padding region or out of range.
        """
        if uc < 0 or uc >= self._total_length:
            raise ValueError(f"Universal coordinate {uc} is out of range.")
        idx = int(np.searchsorted(self._chromosome_start_addr, uc, side="right")) - 1
        if idx < 0 or uc >= self._chromosome_end_addr[idx]:
            raise ValueError(
                f"Universal coordinate {uc} is in an inter-chromosomal padding region."
            )
        chrom = str(self._chromosome_name[idx])
        local = uc - int(self._chromosome_start_addr[idx])
        return chrom, local

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_cached(self, url_or_path: str) -> Path:
        """Return a local Path to the file, downloading if necessary."""
        return _ensure_cached_io(url_or_path, self._cache_dir)

    def _is_padding(self, uc: int) -> bool:
        """Return True if *uc* falls in an inter-chromosomal padding region."""
        if uc < 0 or uc >= self._total_length:
            return False  # out of range, handled separately
        idx = int(np.searchsorted(self._chromosome_start_addr, uc, side="right")) - 1
        if idx < 0:
            return True
        return uc >= self._chromosome_end_addr[idx]

    def _get_padding_mask(self, start: int, stop: int) -> np.ndarray:
        """Boolean array, True where position is in inter-chromosomal padding."""
        mask = np.ones(stop - start, dtype=bool)
        for cs, ce in zip(self._chromosome_start_addr, self._chromosome_end_addr):
            lo = max(0, int(cs) - start)
            hi = min(stop - start, int(ce) - start)
            if lo < hi:
                mask[lo:hi] = False
        return mask

    def _render(self, start: int, stop: int, step: Optional[int]) -> np.ndarray:
        """Core rendering: find overlapping genes and produce signal array.

        For unbinned queries (step is None / 1) the full ``[start, stop)``
        signal is materialised as an ``int32`` array — callers should avoid
        whole-genome unbinned access.

        For binned queries (step > 1) the range is processed in chunks of
        ``_RENDER_CHUNK_BASES`` bases to cap peak memory at approximately
        ``_RENDER_CHUNK_BASES * 4`` bytes regardless of query length.
        """
        if step is None or step == 1:
            gene_idx = find_overlapping_gene_indices(
                start, stop,
                self._gene_start_addr, self._gene_end_addr,
                self._gene_start_sort_idx,
            )
            g_starts = self._gene_start_addr[gene_idx]
            g_ends = self._gene_end_addr[gene_idx]
            sig = render_signal(start, stop, g_starts, g_ends)
            pad_mask = self._get_padding_mask(start, stop)
            sig[pad_mask] = self.PADDING_VALUE
            return sig
        else:
            return self._render_binned_chunked(start, stop, step)

    # Maximum bases rendered into memory at once during binned queries.
    # At int32 this caps the peak signal buffer at ~400 MB.
    def _render_binned_chunked(
        self, start: int, stop: int, step: int
    ) -> np.ndarray:
        """Binned rendering via direct coordinate arithmetic (no base-resolution array).

        Delegates to :func:`render_binned_direct` which computes each bin's
        mean overlap by clipping gene coordinates to bin boundaries — O(n_genes)
        time and O(n_bins) memory regardless of the query span.
        """
        gene_idx = find_overlapping_gene_indices(
            start, stop,
            self._gene_start_addr, self._gene_end_addr,
            self._gene_start_sort_idx,
        )
        return render_binned_direct(
            start, stop, step,
            self._gene_start_addr[gene_idx],
            self._gene_end_addr[gene_idx],
        )

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """Retrieve signal values by universal coordinate, name, or slice.

        Supported key forms
        -------------------
        ``genome[i]``
            Scalar integer — returns a single int value.
            Padding positions return :attr:`PADDING_VALUE`.
        ``genome[a:b]``
            Slice — returns ``int32`` array of shape ``(b-a,)``.
        ``genome[a:b:s]``
            Slice with step — returns ``float64`` binned-average array.
        ``genome[::s]``
            Whole-genome binned array (used by CLI for Hilbert pixels).
        ``genome['chr1']``
            Full chromosome signal.
        ``genome['BRCA1+']``
            Signal over the gene body.
        ``genome['chr1', 1000:2000]``
            Signal for a local range within chr1 (0-based local coords).
        ``genome['BRCA1+', 100:500]``
            Signal for a local range within the gene (0-based local coords).
        """
        if isinstance(key, (int, np.integer)):
            return self._getitem_scalar(int(key))
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if isinstance(key, str):
            return self._getitem_name(key)
        if isinstance(key, tuple) and len(key) == 2:
            name, slc = key
            if not isinstance(name, str):
                raise TypeError(
                    f"First element of tuple key must be str, got {type(name).__name__}"
                )
            if not isinstance(slc, slice):
                raise TypeError(
                    f"Second element of tuple key must be slice, got {type(slc).__name__}"
                )
            return self._getitem_name_slice(name, slc)
        raise TypeError(f"Unsupported key type: {type(key).__name__}")

    def _getitem_scalar(self, pos: int) -> int:
        if pos < 0:
            pos = self._total_length + pos
        if pos < 0 or pos >= self._total_length:
            raise IndexError(
                f"Position {pos} is out of range [0, {self._total_length})."
            )
        if self._is_padding(pos):
            return self.PADDING_VALUE
        sig = self._render(pos, pos + 1, None)
        return int(sig[0])

    def _getitem_slice(self, key: slice) -> np.ndarray:
        start, stop, step = key.indices(self._total_length)
        if start >= stop:
            return np.array([], dtype=np.int32)
        return self._render(start, stop, step if step != 1 else None)

    def _getitem_name(self, name: str) -> np.ndarray:
        base_start, base_end = self._resolve_name(name)
        return self._render(base_start, base_end, None)

    def _getitem_name_slice(self, name: str, slc: slice) -> np.ndarray:
        base_start, base_end = self._resolve_name(name)
        base_len = base_end - base_start
        local_start, local_stop, step = slc.indices(base_len)
        uc_start = base_start + local_start
        uc_stop = base_start + local_stop
        if uc_start >= uc_stop:
            return np.array([], dtype=np.int32)
        return self._render(uc_start, uc_stop, step if step != 1 else None)

    def _resolve_name(self, name: str) -> tuple[int, int]:
        """Return (uc_start, uc_end) for a chromosome or gene name."""
        if name in self._chr_name_to_idx:
            idx = self._chr_name_to_idx[name]
            return int(self._chromosome_start_addr[idx]), int(self._chromosome_end_addr[idx])
        if name in self._gene_name_to_idx:
            idx = self._gene_name_to_idx[name]
            return int(self._gene_start_addr[idx]), int(self._gene_end_addr[idx])
        # Helpful error for missing strand suffix
        candidates = [k for k in (f"{name}+", f"{name}-") if k in self._gene_name_to_idx]
        if candidates:
            raise KeyError(
                f"{name!r} is ambiguous. Use a strand-suffixed key: "
                + " or ".join(repr(c) for c in candidates)
            )
        raise KeyError(f"{name!r} is not a known chromosome or gene.")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------


    def chromosome_label_signal(self, step: int = 1) -> np.ndarray:
        """Return a chromosome-index array for Hilbert-curve chromosome plots.

        Each element gives the **1-based** index of the chromosome that
        contains the start of that bin, or ``0`` for inter-chromosomal
        padding.

        Parameters
        ----------
        step:
            Bin size in bases (same as the slice step in ``genome[::step]``).

        Returns
        -------
        np.ndarray of int32, shape ``(ceil(total_length / step),)``
            Values in ``[0, n_chromosomes]``.  0 = padding.
        """
        import math as _math
        n_bins = _math.ceil(self._total_length / step)
        bin_starts = np.arange(n_bins, dtype=np.int64) * step

        # Binary-search to find which chromosome each bin start falls in
        chr_idx = (
            np.searchsorted(self._chromosome_start_addr, bin_starts, side="right") - 1
        )
        chr_idx = np.clip(chr_idx, 0, self.n_chromosomes - 1)

        # Mask bins that fall in padding (past a chromosome's end)
        in_chrom = bin_starts < self._chromosome_end_addr[chr_idx]

        return np.where(in_chrom, chr_idx + 1, 0).astype(np.int32)
    def gene_label_signal(self, step: int = 1) -> np.ndarray:
        """Return a gene-index array for Hilbert-curve gene-identity plots.

        Each element gives the **1-based** index (in genomic start order) of a
        gene whose body covers the start of that bin, or ``0`` for intergenic /
        padding positions.  When multiple genes overlap the same bin the one
        with the latest start position is chosen (it is the most specific).

        The index space uses the gene's rank in ``_gene_start_sort_idx``
        (ascending start order), so that consecutive label integers correspond
        to genomically adjacent genes — ideal for the golden-ratio colormap
        which ensures consecutive labels receive perceptually distant colours.

        Parameters
        ----------
        step:
            Bin size in bases (same as the slice step in ``genome[::step]``).

        Returns
        -------
        np.ndarray of int32, shape ``(ceil(total_length / step),)``
            Values in ``[0, n_genes]``.  0 = intergenic or padding.
        """
        import math as _math
        n_bins = _math.ceil(self._total_length / step)
        bin_starts = np.arange(n_bins, dtype=np.int64) * step

        sorted_starts = self._gene_start_addr[self._gene_start_sort_idx]

        # For each bin, find the rank of the last gene that started at or
        # before bin_start.  searchsorted(..., side='right') gives the number
        # of starts that are <= bin_start, so rank = that - 1.
        rank = np.searchsorted(sorted_starts, bin_starts, side="right") - 1

        # rank == -1 means no gene has started yet — treat as intergenic
        valid = rank >= 0
        safe_rank = np.where(valid, rank, 0)

        gene_orig_idx = self._gene_start_sort_idx[safe_rank]

        # The gene overlaps the bin iff its end is strictly after bin_start
        overlaps = valid & (self._gene_end_addr[gene_orig_idx] > bin_starts)

        # Label = 1-based rank in start-sort order (so adjacent labels ->
        # adjacent genes -> golden-ratio hue gives maximum contrast)
        return np.where(overlaps, safe_rank + 1, 0).astype(np.int32)

    def __repr__(self) -> str:
        prefix_info = (
            f", transcript_filter={self._transcript_id_prefixes!r}"
            if self._transcript_id_prefixes is not None
            else ""
        )
        return (
            f"Genome(n_chromosomes={self.n_chromosomes}, "
            f"n_genes={self.n_genes}, "
            f"total_length={self.total_length:,}, "
            f"padding={self.padding:,}"
            f"{prefix_info})"
        )

    def __len__(self) -> int:
        return self._total_length
