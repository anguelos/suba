"""
Integration tests — require network access and download hg38 files.

Skipped automatically when the SUBA_INTEGRATION environment variable is not
set to "1" or when the cached files are not present.

Run with::

    SUBA_INTEGRATION=1 pytest tests/integration/
"""

import os
import pytest

INTEGRATION = os.environ.get("SUBA_INTEGRATION", "0") == "1"


@pytest.mark.skipif(not INTEGRATION, reason="SUBA_INTEGRATION not set")
class TestHg38Download:
    def test_genome_loads(self, tmp_path):
        from suba.genome import Genome
        g = Genome(cache_dir=str(tmp_path))
        assert g.n_chromosomes >= 24   # at least chr1-22, chrX, chrY
        assert g.total_length > 3_000_000_000

    def test_known_gene_present(self, tmp_path):
        from suba.genome import Genome
        g = Genome(cache_dir=str(tmp_path))
        # TP53 should exist on the + or - strand
        found = any(k.startswith("TP53") for k in g._gene_name_to_idx)
        assert found, "TP53 not found in gene index"

    def test_slice_across_padding(self, tmp_path):
        from suba.genome import Genome
        g = Genome(cache_dir=str(tmp_path))
        # Slice that includes the end of chr1 and the start of padding
        chr1_end = int(g._chromosome_end_addr[g._chr_name_to_idx["chr1"]])
        sig = g[chr1_end - 10 : chr1_end + 10]
        assert len(sig) == 20
        # First 10 bases are chromosome, last 10 are padding (-1)
        assert all(v >= 0 for v in sig[:10])
        assert all(v == -1 for v in sig[10:])
