"""
suba.io.count_matrix
~~~~~~~~~~~~~~~~~~~~
Load and save single-cell count matrices in multiple formats.

Supported formats
-----------------
* **10x MEX** (directory containing ``matrix.mtx.gz``, ``barcodes.tsv.gz``,
  ``features.tsv.gz`` or ``genes.tsv.gz``) — always treated as raw integer
  counts (``length_normalize=True``).
* **suba .npz** — stores CSR sparse matrix components + metadata in a single
  compressed NumPy archive.
* **HDF5 .h5** — stores the same data in an HDF5 file (requires ``h5py``).

Tuple convention
----------------
All load functions return ``(barcodes, gene_names, matrix, length_normalize)``
where *matrix* is a ``scipy.sparse.csr_matrix`` of shape
``(n_cells, n_genes)``.  *length_normalize* is ``True`` when the values are
raw integer counts (divide by gene length to get per-base signal) and
``False`` when the values are already normalised (use flat per-gene value).

All save functions accept the same leading four arguments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_length_normalize(matrix) -> bool:
    """Infer normalisation from dtype: integer -> raw counts, float -> normalised."""
    import numpy as np
    dtype = getattr(matrix, "dtype", np.dtype("float32"))
    return np.issubdtype(dtype, np.integer)


def _to_csr(matrix):
    """Ensure *matrix* is a scipy CSR sparse matrix."""
    from scipy.sparse import issparse, csr_matrix
    if issparse(matrix):
        return matrix.tocsr()
    return csr_matrix(matrix)


# ---------------------------------------------------------------------------
# 10x MEX format
# ---------------------------------------------------------------------------

def load_count_matrix_10x(
    dir_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, object, bool]:
    """Load a 10x Genomics MEX directory.

    Expects ``matrix.mtx.gz``, ``barcodes.tsv.gz``, and either
    ``features.tsv.gz`` (Cell Ranger ≥ 3) or ``genes.tsv.gz`` (Cell Ranger 2).

    The matrix in the ``.mtx`` file is stored genes × cells; this function
    transposes it to the standard cells × genes orientation.

    Gene names are taken from column 2 of the features file (HGNC symbol).

    Returns
    -------
    (barcodes, gene_names, matrix, length_normalize)
        *length_normalize* is always ``True`` for MEX files (raw integer counts).
    """
    import gzip
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    dir_path = Path(dir_path)

    # Locate the three required files
    matrix_file = dir_path / "matrix.mtx.gz"
    if not matrix_file.exists():
        matrix_file = dir_path / "matrix.mtx"
    if not matrix_file.exists():
        raise FileNotFoundError(f"No matrix.mtx(.gz) found in {dir_path}")

    barcodes_file = dir_path / "barcodes.tsv.gz"
    if not barcodes_file.exists():
        barcodes_file = dir_path / "barcodes.tsv"

    features_file = dir_path / "features.tsv.gz"
    if not features_file.exists():
        features_file = dir_path / "features.tsv"
    if not features_file.exists():
        features_file = dir_path / "genes.tsv.gz"
    if not features_file.exists():
        features_file = dir_path / "genes.tsv"
    if not features_file.exists():
        raise FileNotFoundError(f"No features.tsv(.gz) or genes.tsv(.gz) in {dir_path}")

    # Read barcodes
    opener = gzip.open if str(barcodes_file).endswith(".gz") else open
    with opener(barcodes_file, "rt") as fh:
        barcodes = np.array([line.strip() for line in fh if line.strip()], dtype=object)

    # Read gene names (column 2 = HGNC symbol; column 1 = Ensembl ID)
    opener = gzip.open if str(features_file).endswith(".gz") else open
    gene_names = []
    with opener(features_file, "rt") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            # Use symbol (col 2) if present, else col 1
            gene_names.append(parts[1] if len(parts) >= 2 else parts[0])
    gene_names = np.array(gene_names, dtype=object)

    # Read sparse matrix (genes x cells in MEX), transpose to cells x genes
    opener = gzip.open if str(matrix_file).endswith(".gz") else open
    with opener(matrix_file, "rb") as fh:
        mat = mmread(fh).T  # cells x genes
    matrix = csr_matrix(mat)

    return barcodes, gene_names, matrix, True


# ---------------------------------------------------------------------------
# suba .npz format
# ---------------------------------------------------------------------------

def save_count_matrix_npz(
    path: str | Path,
    barcodes: np.ndarray,
    gene_names: np.ndarray,
    matrix,
    length_normalize: Optional[bool] = None,
) -> None:
    """Save a count matrix to a suba-native compressed .npz file.

    All data (sparse matrix components, barcodes, gene names, and the
    ``length_normalize`` flag) are stored in a single ``.npz`` archive.

    Parameters
    ----------
    path:
        Output file path (should end in ``.npz``).
    barcodes, gene_names:
        1D object arrays of strings.
    matrix:
        Count matrix, cells × genes.  Converted to CSR if necessary.
    length_normalize:
        Whether the values are raw counts (``True``) or pre-normalised
        (``False``).  ``None`` auto-detects from *matrix* dtype.
    """
    path = Path(path)
    m = _to_csr(matrix)
    if length_normalize is None:
        length_normalize = _detect_length_normalize(m)

    np.savez_compressed(
        path,
        # CSR components
        data=m.data,
        indices=m.indices,
        indptr=m.indptr,
        shape=np.array(m.shape, dtype=np.int64),
        # Metadata
        barcodes=np.asarray(barcodes, dtype=object),
        gene_names=np.asarray(gene_names, dtype=object),
        length_normalize=np.array([length_normalize]),
        format=np.array(["csr"]),
    )


def load_count_matrix_npz(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, object, bool]:
    """Load a count matrix from a suba-native .npz file.

    Returns
    -------
    (barcodes, gene_names, matrix, length_normalize)
    """
    from scipy.sparse import csr_matrix

    path = Path(path)
    d = np.load(path, allow_pickle=True)
    matrix = csr_matrix(
        (d["data"], d["indices"], d["indptr"]),
        shape=tuple(d["shape"].tolist()),
    )
    length_normalize = bool(d["length_normalize"][0])
    return d["barcodes"], d["gene_names"], matrix, length_normalize


# ---------------------------------------------------------------------------
# HDF5 .h5 format
# ---------------------------------------------------------------------------

def save_count_matrix_h5(
    path: str | Path,
    barcodes: np.ndarray,
    gene_names: np.ndarray,
    matrix,
    length_normalize: Optional[bool] = None,
) -> None:
    """Save a count matrix to an HDF5 file.

    Requires ``h5py``.  The file stores CSR components, barcodes, gene names,
    and the ``length_normalize`` flag as attributes.

    Parameters
    ----------
    path:
        Output file path (should end in ``.h5``).
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 support: pip install h5py") from exc

    path = Path(path)
    m = _to_csr(matrix)
    if length_normalize is None:
        length_normalize = _detect_length_normalize(m)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "csr"
        f.attrs["source"] = "suba"
        f.attrs["length_normalize"] = bool(length_normalize)
        f.attrs["shape_0"] = m.shape[0]
        f.attrs["shape_1"] = m.shape[1]
        f.create_dataset("data", data=m.data, compression="gzip")
        f.create_dataset("indices", data=m.indices, compression="gzip")
        f.create_dataset("indptr", data=m.indptr, compression="gzip")
        # Store strings as bytes for portability
        f.create_dataset(
            "barcodes",
            data=np.asarray(barcodes, dtype=object).astype("S"),
            compression="gzip",
        )
        f.create_dataset(
            "gene_names",
            data=np.asarray(gene_names, dtype=object).astype("S"),
            compression="gzip",
        )


def load_count_matrix_h5(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, object, bool]:
    """Load a count matrix from a suba HDF5 file.

    Returns
    -------
    (barcodes, gene_names, matrix, length_normalize)
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 support: pip install h5py") from exc
    from scipy.sparse import csr_matrix

    path = Path(path)
    with h5py.File(path, "r") as f:
        shape = (int(f.attrs["shape_0"]), int(f.attrs["shape_1"]))
        length_normalize = bool(f.attrs["length_normalize"])
        matrix = csr_matrix(
            (f["data"][:], f["indices"][:], f["indptr"][:]),
            shape=shape,
        )
        barcodes = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["barcodes"][:]],
            dtype=object,
        )
        gene_names = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["gene_names"][:]],
            dtype=object,
        )
    return barcodes, gene_names, matrix, length_normalize


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def save_count_matrix(
    path: str | Path,
    barcodes: np.ndarray,
    gene_names: np.ndarray,
    matrix,
    length_normalize: Optional[bool] = None,
) -> None:
    """Save a count matrix, choosing format from the file extension.

    ``.npz`` → :func:`save_count_matrix_npz` (default when no extension).
    ``.h5`` or ``.hdf5`` → :func:`save_count_matrix_h5`.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".h5", ".hdf5"):
        save_count_matrix_h5(path, barcodes, gene_names, matrix, length_normalize)
    else:
        if ext not in (".npz",):
            path = path.with_suffix(".npz")
        save_count_matrix_npz(path, barcodes, gene_names, matrix, length_normalize)


def load_count_matrix(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, object, bool]:
    """Load a count matrix, dispatching on path type and file extension.

    * A **directory** is treated as a 10x MEX directory.
    * ``.npz`` → :func:`load_count_matrix_npz`.
    * ``.h5`` / ``.hdf5`` → :func:`load_count_matrix_h5`.

    Returns
    -------
    (barcodes, gene_names, matrix, length_normalize)
    """
    path = Path(path)
    if path.is_dir():
        return load_count_matrix_10x(path)
    ext = path.suffix.lower()
    if ext in (".h5", ".hdf5"):
        return load_count_matrix_h5(path)
    return load_count_matrix_npz(path)
