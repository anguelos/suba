"""
suba.io
~~~~~~~
File-format I/O for genomes and single-cell count matrices.
"""
from suba.io.resumable_download import download_resumable
from suba.io.genome import (
    infer_chrom_sizes_url,
    ensure_cached,
    parse_gtf_genes,
    parse_chrom_sizes,
    npz_cache_path,
    save_gene_cache,
    load_gene_cache,
)
from suba.io.count_matrix import (
    load_count_matrix_10x,
    load_count_matrix_npz,
    load_count_matrix_h5,
    load_count_matrix,
    save_count_matrix_npz,
    save_count_matrix_h5,
    save_count_matrix,
)

__all__ = [
    "download_resumable",
    "infer_chrom_sizes_url",
    "ensure_cached",
    "parse_gtf_genes",
    "parse_chrom_sizes",
    "npz_cache_path",
    "save_gene_cache",
    "load_gene_cache",
    "load_count_matrix_10x",
    "load_count_matrix_npz",
    "load_count_matrix_h5",
    "load_count_matrix",
    "save_count_matrix_npz",
    "save_count_matrix_h5",
    "save_count_matrix",
]
