"""
Shared pytest fixtures.

The synthetic genome used throughout unit tests has two chromosomes:
  chr1  length 1000
  chr2  length  500
padding = 100

Universal address space:
  [0,    1000)  chr1
  [1000, 1100)  padding
  [1100, 1600)  chr2
  total_length = 1600

Genes (0-based local coords, strand-suffixed):
  GENE_A+   chr1  [100, 300)    (length 200)
  GENE_B-   chr1  [250, 400)    (length 150)  overlaps GENE_A+ by 50 bases
  GENE_C+   chr2  [0,   200)    (length 200)
"""

import pytest
from suba.genome import Genome


@pytest.fixture(scope="module")
def tiny_genome() -> Genome:
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
