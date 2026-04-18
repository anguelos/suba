"""
tests.unit.test_sc_tracks
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for SCTracks / CountMatrixTracks using a small synthetic genome
(no network access required).
"""
import math
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import suba.sc_tracks as st_mod
from suba.sc_tracks import CountMatrixTracks
from suba.genome import Genome


# ---------------------------------------------------------------------------
# Shared synthetic genome + count matrix
# ---------------------------------------------------------------------------

def _make_genome():
    """Two chromosomes, three genes."""
    return Genome.from_arrays(
        chromosome_names=["chr1", "chr2"],
        chromosome_sizes=[10_000, 8_000],
        gene_names=  ["ALPHA", "BETA", "GAMMA"],
        gene_chroms= ["chr1",  "chr1", "chr2"],
        gene_local_starts=[100,  400,  200],
        gene_local_ends=  [300,  600,  500],
        gene_strands=     ["+",  "-",  "+"],
        padding=1_000,
    )


def _make_tracks(genome=None, raw=True):
    """
    3 cells × 3 genes count matrix.
    ALPHA  BETA  GAMMA
      2     0     4
      0     6     0
      3     3     1
    """
    mat = csr_matrix(np.array([
        [2, 0, 4],
        [0, 6, 0],
        [3, 3, 1],
    ], dtype=np.int32 if raw else np.float32))
    barcodes   = np.array(["A", "B", "C"], dtype=object)
    gene_names = np.array(["ALPHA", "BETA", "GAMMA"], dtype=object)
    if genome is None:
        genome = _make_genome()
    return CountMatrixTracks(
        mat, barcodes, gene_names,
        genome=genome,
        length_normalize=raw,
    )


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

def test_n_cells():
    assert _make_tracks().n_cells == 3


def test_barcodes():
    t = _make_tracks()
    assert list(t.barcodes) == ["A", "B", "C"]


def test_create_count_matrix_roundtrip():
    t = _make_tracks()
    barcodes, gene_names, matrix = t.create_count_matrix()
    assert list(barcodes) == ["A", "B", "C"]
    assert list(gene_names) == ["ALPHA", "BETA", "GAMMA"]
    arr = matrix.toarray()
    np.testing.assert_array_equal(arr, [[2,0,4],[0,6,0],[3,3,1]])


# ---------------------------------------------------------------------------
# Cell index resolution
# ---------------------------------------------------------------------------

def test_cell_index_int():
    t = _make_tracks()
    indices, scalar = t._resolve_cell_key(0)
    assert scalar is True
    assert list(indices) == [0]


def test_cell_index_barcode_str():
    t = _make_tracks()
    indices, scalar = t._resolve_cell_key("B")
    assert scalar is True
    assert list(indices) == [1]


def test_cell_index_list_of_barcodes():
    t = _make_tracks()
    indices, scalar = t._resolve_cell_key(["A", "C"])
    assert scalar is False
    assert list(indices) == [0, 2]


def test_cell_index_slice():
    t = _make_tracks()
    indices, scalar = t._resolve_cell_key(slice(0, 2))
    assert scalar is False
    assert list(indices) == [0, 1]


def test_cell_index_all_slice():
    t = _make_tracks()
    indices, scalar = t._resolve_cell_key(slice(None))
    assert scalar is False
    assert len(indices) == 3


# ---------------------------------------------------------------------------
# Signal values (base resolution)
# ---------------------------------------------------------------------------

def _gene_len(genome, key):
    idx = genome._gene_name_to_idx[key]
    return int(genome._gene_end_addr[idx] - genome._gene_start_addr[idx])


def test_signal_inside_alpha_gene():
    """Cell 0 has count=2 for ALPHA (length 200).  Per base = 2/200 = 0.01."""
    g = _make_genome()
    t = _make_tracks(g)
    # ALPHA+ on chr1, local [100, 300), universal [100, 300)
    sig = t[0, 150:160]          # 10 bases inside ALPHA+
    assert sig.shape == (10,)
    np.testing.assert_allclose(sig, 2 / 200, rtol=1e-5)


def test_signal_barcode_access():
    t = _make_tracks()
    sig_int = t[1, 150:160]
    sig_str = t["B", 150:160]
    np.testing.assert_array_equal(sig_int, sig_str)


def test_signal_intergenic_is_zero():
    t = _make_tracks()
    sig = t[0, 350:400]   # between ALPHA end (300) and BETA start (400) on chr1
    assert np.all(sig == 0.0)


def test_signal_multi_cell():
    """tracks[[A,C], range] returns 2D array."""
    t = _make_tracks()
    sig = t[["A", "C"], 150:160]
    assert sig.shape == (2, 10)
    # Cell A: 2/200 per base in ALPHA
    np.testing.assert_allclose(sig[0], 2 / 200, rtol=1e-5)
    # Cell C: 3/200 per base in ALPHA
    np.testing.assert_allclose(sig[1], 3 / 200, rtol=1e-5)


def test_signal_all_cells():
    t = _make_tracks()
    sig = t[:, 150:160]
    assert sig.shape == (3, 10)


# ---------------------------------------------------------------------------
# Binned signal
# ---------------------------------------------------------------------------

def test_binned_step_covers_gene():
    """Bin exactly covers ALPHA gene body → bin value = count / gene_len * gene_len / step.

    With step >= gene_len the bin value = count * overlap / (gene_len * step/step)
    = count * gene_len_in_bin / gene_len.  Here gene fully in bin → value = count/step.
    Actually: result = matrix_sel @ W, W[col,b] = overlap/gene_len.
    overlap = min(gene_end, bin_end) - max(gene_start, bin_start).
    For a 1000-base bin [0,1000) covering ALPHA [100,300): overlap=200, gene_len=200
    => W[ALPHA_col, 0] = 200/200 = 1.0.
    Cell 0 count for ALPHA = 2, so result[0,0] = 2.0.
    """
    t = _make_tracks()
    # chr1 starts at universal 0.  Step 1000 -> bin 0 covers [0,1000).
    sig = t[0, 0:1000:1000]
    assert sig.shape == (1,)
    # ALPHA fully in bin: weight=1.0, count=2 -> 2.0
    # BETA fully in bin: weight=1.0, count=0 -> 0
    np.testing.assert_allclose(sig[0], 2.0, rtol=1e-4)


def test_binned_whole_genome_shape():
    g = _make_genome()
    t = _make_tracks(g)
    step = 500
    n_bins = math.ceil(g.total_length / step)
    sig = t[:, ::step]
    assert sig.shape == (3, n_bins)


# ---------------------------------------------------------------------------
# max_dense_object_size guard
# ---------------------------------------------------------------------------

def test_max_dense_size_raises():
    t = _make_tracks()
    old = st_mod.max_dense_object_size
    try:
        st_mod.max_dense_object_size = 1  # 1 byte — always trips
        with pytest.raises(MemoryError, match="max_dense_object_size"):
            t[:, 0:100]
    finally:
        st_mod.max_dense_object_size = old


# ---------------------------------------------------------------------------
# IO round-trip
# ---------------------------------------------------------------------------

def test_npz_roundtrip(tmp_path):
    from suba.io import save_count_matrix_npz, load_count_matrix_npz
    t = _make_tracks()
    barcodes, gene_names, matrix = t.create_count_matrix()
    path = tmp_path / "test.npz"
    save_count_matrix_npz(path, barcodes, gene_names, matrix, length_normalize=True)
    b2, g2, m2, ln = load_count_matrix_npz(path)
    assert list(b2) == list(barcodes)
    assert list(g2) == list(gene_names)
    np.testing.assert_array_equal(m2.toarray(), matrix.toarray())
    assert ln is True


def test_save_load_dispatcher(tmp_path):
    from suba.io import save_count_matrix, load_count_matrix
    t = _make_tracks()
    barcodes, gene_names, matrix = t.create_count_matrix()
    path = tmp_path / "test.npz"
    save_count_matrix(path, barcodes, gene_names, matrix)
    b2, g2, m2, ln = load_count_matrix(path)
    np.testing.assert_array_equal(m2.toarray(), matrix.toarray())
