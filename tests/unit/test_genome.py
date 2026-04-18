"""Unit tests for suba.Genome (using from_arrays — no network required)."""

import numpy as np
import pytest

from suba.genome import Genome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def g():
    """Tiny two-chromosome genome with three genes."""
    return Genome.from_arrays(
        chromosome_names=["chr1", "chr2"],
        chromosome_sizes=[1000, 500],
        gene_names=["GENE_A", "GENE_B", "GENE_C"],
        gene_chroms=["chr1", "chr1", "chr2"],
        gene_local_starts=[100, 250, 0],
        gene_local_ends=[300, 400, 200],
        gene_strands=["+", "-", "+"],
        padding=100,
    )


# ---------------------------------------------------------------------------
# Address space layout
# ---------------------------------------------------------------------------

class TestLayout:
    def test_total_length(self, g):
        # chr1=1000, padding=100, chr2=500  → 1600
        assert g.total_length == 1600

    def test_n_chromosomes(self, g):
        assert g.n_chromosomes == 2

    def test_n_genes(self, g):
        # GENE_A+, GENE_B-, GENE_C+
        assert g.n_genes == 3

    def test_chr1_start(self, g):
        assert g._chromosome_start_addr[0] == 0

    def test_chr1_end(self, g):
        assert g._chromosome_end_addr[0] == 1000

    def test_chr2_start(self, g):
        assert g._chromosome_start_addr[1] == 1100

    def test_chr2_end(self, g):
        assert g._chromosome_end_addr[1] == 1600


# ---------------------------------------------------------------------------
# Gene universal addresses
# ---------------------------------------------------------------------------

class TestGeneAddresses:
    def test_gene_a_start(self, g):
        idx = g._gene_name_to_idx["GENE_A+"]
        assert g._gene_start_addr[idx] == 100   # chr1 offset 0 + local 100

    def test_gene_a_end(self, g):
        idx = g._gene_name_to_idx["GENE_A+"]
        assert g._gene_end_addr[idx] == 300

    def test_gene_c_start(self, g):
        idx = g._gene_name_to_idx["GENE_C+"]
        assert g._gene_start_addr[idx] == 1100  # chr2 offset 1100 + local 0

    def test_gene_strand(self, g):
        assert g._gene_strand[g._gene_name_to_idx["GENE_A+"]] == 1
        assert g._gene_strand[g._gene_name_to_idx["GENE_B-"]] == -1


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

class TestCoordConversion:
    def test_genomic_to_universal(self, g):
        assert g.genomic_to_universal("chr1", 0) == 0
        assert g.genomic_to_universal("chr1", 500) == 500
        assert g.genomic_to_universal("chr2", 0) == 1100

    def test_universal_to_genomic(self, g):
        assert g.universal_to_genomic(0) == ("chr1", 0)
        assert g.universal_to_genomic(999) == ("chr1", 999)
        assert g.universal_to_genomic(1100) == ("chr2", 0)

    def test_padding_raises(self, g):
        with pytest.raises(ValueError):
            g.universal_to_genomic(1050)  # in padding [1000,1100)

    def test_out_of_range_raises(self, g):
        with pytest.raises(ValueError):
            g.universal_to_genomic(1600)


# ---------------------------------------------------------------------------
# __getitem__ — scalar
# ---------------------------------------------------------------------------

class TestScalarAccess:
    def test_intergenic_returns_zero(self, g):
        # chr1 position 50 (before GENE_A which starts at 100)
        assert g[50] == 0

    def test_genic_returns_one(self, g):
        assert g[150] == 1   # inside GENE_A+ [100,300)

    def test_overlap_returns_two(self, g):
        # GENE_A+ [100,300) and GENE_B- [250,400) overlap at [250,300)
        assert g[275] == 2

    def test_padding_returns_minus_one(self, g):
        assert g[1050] == Genome.PADDING_VALUE

    def test_out_of_range_raises(self, g):
        with pytest.raises(IndexError):
            _ = g[1600]


# ---------------------------------------------------------------------------
# __getitem__ — slice
# ---------------------------------------------------------------------------

class TestSliceAccess:
    def test_intergenic_slice_zeros(self, g):
        sig = g[0:100]
        assert sig.shape == (100,)
        assert (sig == 0).all()

    def test_genic_slice(self, g):
        # GENE_A+ spans [100,300); GENE_B- starts at 250, so [250,300) = 2.
        # Query [100,250) is purely within GENE_A+.
        sig = g[100:250]
        assert sig.shape == (150,)
        assert (sig == 1).all()
        # The overlap zone [250,300) correctly returns 2.
        sig_overlap = g[250:300]
        assert (sig_overlap == 2).all()

    def test_overlap_slice(self, g):
        sig = g[250:300]
        assert sig.shape == (50,)
        assert (sig == 2).all()

    def test_padding_in_slice(self, g):
        sig = g[1000:1100]
        assert (sig == Genome.PADDING_VALUE).all()

    def test_whole_genome_slice(self, g):
        sig = g[0:g.total_length]
        assert len(sig) == g.total_length

    def test_empty_slice(self, g):
        sig = g[100:100]
        assert len(sig) == 0


# ---------------------------------------------------------------------------
# __getitem__ — binned (step)
# ---------------------------------------------------------------------------

class TestBinnedAccess:
    def test_step_whole_genome(self, g):
        out = g[::100]
        # 1600 / 100 = 16 bins exactly
        assert len(out) == 16

    def test_binned_genic_region(self, g):
        # GENE_A+ [100,300), GENE_B- [250,400) → overlap at [250,300)
        # Bin 1: [100,200) → all GENE_A+ → mean 1.0
        # Bin 2: [200,300) → [200,250)=1, [250,300)=2 → mean 1.5
        out = g[100:300:100]
        assert len(out) == 2
        np.testing.assert_allclose(out[0], 1.0)
        np.testing.assert_allclose(out[1], 1.5)


# ---------------------------------------------------------------------------
# __getitem__ — name-based
# ---------------------------------------------------------------------------

class TestNameAccess:
    def test_chromosome_access(self, g):
        sig = g["chr1"]
        assert sig.shape == (1000,)

    def test_gene_access(self, g):
        sig = g["GENE_A+"]
        assert sig.shape == (200,)  # [100,300)
        # [100,250) → only GENE_A+ → 1; [250,300) → GENE_A+ and GENE_B- → 2
        assert (sig[:150] == 1).all()
        assert (sig[150:] == 2).all()

    def test_gene_no_strand_raises(self, g):
        with pytest.raises(KeyError, match="GENE_A"):
            _ = g["GENE_A"]

    def test_unknown_name_raises(self, g):
        with pytest.raises(KeyError):
            _ = g["NOTEXIST"]


# ---------------------------------------------------------------------------
# __getitem__ — name + local slice
# ---------------------------------------------------------------------------

class TestNameSliceAccess:
    def test_chr_local_slice(self, g):
        # chr1 local [100,300) → universal [100,300)
        # Overlap at [250,300): both GENE_A+ and GENE_B-
        sig = g["chr1", 100:300]
        assert sig.shape == (200,)
        assert (sig[:150] == 1).all()
        assert (sig[150:] == 2).all()

    def test_gene_local_slice(self, g):
        # GENE_A+ is 200 bp; local [50:100) → universal [150,200)
        sig = g["GENE_A+", 50:100]
        assert sig.shape == (50,)
        assert (sig == 1).all()

    def test_name_slice_with_step(self, g):
        out = g["GENE_A+", ::10]
        assert len(out) == 20   # 200 bp / 10 = 20 bins


# ---------------------------------------------------------------------------
# Chromosome ordering
# ---------------------------------------------------------------------------

class TestChromosomeOrdering:
    def test_numeric_before_alpha(self):
        g2 = Genome.from_arrays(
            chromosome_names=["chrX", "chr2", "chr10", "chr1"],
            chromosome_sizes=[100, 200, 300, 400],
            gene_names=[],
            gene_chroms=[],
            gene_local_starts=[],
            gene_local_ends=[],
            gene_strands=[],
            padding=10,
        )
        names = g2._chromosome_name.tolist()
        assert names == ["chr1", "chr2", "chr10", "chrX"]
