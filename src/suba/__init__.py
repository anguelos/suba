"""
suba — Single cell Unidimensional Base Annotation.

Presents genomic data modalities as 1D numpy-like objects where every position
maps to a single base in a universal coordinate space formed by concatenating
all chromosomes with fixed inter-chromosomal padding.
"""

from suba.genome import Genome
from suba.colors import label_colormap
from suba.sc_tracks import SCTracks, CountMatrixTracks, max_dense_object_size
from suba.io import (
    load_count_matrix,
    load_count_matrix_10x,
    load_count_matrix_npz,
    load_count_matrix_h5,
    save_count_matrix,
    save_count_matrix_npz,
    save_count_matrix_h5,
)
from suba.sparse_rendering import (
    find_overlapping_gene_indices,
    render_signal,
    render_binned,
    hilbert_d_to_xy,
    parallel_cumsum,
    render_binned_direct,
)

__version__ = "0.1.0"
__all__ = [
    "Genome",
    "find_overlapping_gene_indices",
    "render_signal",
    "render_binned",
    "hilbert_d_to_xy",
    "parallel_cumsum",
    "render_binned_direct",
    "label_colormap",
    "SCTracks",
    "CountMatrixTracks",
    "max_dense_object_size",
    "load_count_matrix",
    "load_count_matrix_10x",
    "load_count_matrix_npz",
    "load_count_matrix_h5",
    "save_count_matrix",
    "save_count_matrix_npz",
    "save_count_matrix_h5",
]
