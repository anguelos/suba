# suba — Single-cell Universal Base Addressing

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Repo size](https://img.shields.io/github/repo-size/anguelos/suba)](https://github.com/anguelos/suba)
[![Tests](https://img.shields.io/badge/tests-75%20passing-brightgreen.svg)](tests/)
[![Coverage](tmp/coverage.svg)](htmlcov/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**suba** presents genomic data as 1D numpy-like objects where every integer
position refers to a single base in a *universal coordinate space* — all
chromosomes concatenated end-to-end with fixed inter-chromosomal padding.
Single-cell count matrices can be projected onto this space, turning each
cell into a genomic signal track.

---

## Key concepts

| Term | Meaning |
|---|---|
| **Universal coordinate** | A 0-based integer offset into the concatenated genome. Every base in every chromosome has exactly one universal coordinate. |
| **Inter-chromosomal padding** | Fixed gap (default 1 Mbp) between consecutive chromosomes. Padding positions return `−1`. |
| **Sparse rendering** | No dense array is ever allocated. Gene annotations are stored sparsely; signals are rendered on demand via a start/end increment + cumsum algorithm. |
| **SCTracks** | Abstract 2D track class (cells × coordinates) built on top of `Genome`. |

---

## Installation

```bash
pip install suba                  # core (numpy, requests, tqdm)
pip install "suba[cli]"           # + CLI plotting (fargv, matplotlib, seaborn)
pip install "suba[dev]"           # + tests and docs
```

---

## Quick start

### Genome annotation

```python
from suba import Genome

# Load hg38 — downloads and caches GTF + chrom.sizes on first run (~60 ms on subsequent runs)
genome = Genome()
print(genome)
# Genome(n_chromosomes=25, n_genes=19336, total_length=3,209,286,105, padding=1,000,000)

# Base-resolution signal over a range (0-based half-open)
signal = genome[1_000_000:2_000_000]   # int8 array, length 1,000,000

# Binned average — 4,096 bp per bin
binned = genome[::4096]                # float64 array, ~780 k elements

# Access by name
chr1  = genome["chr1"]                 # full chromosome
brca1 = genome["BRCA1+"]              # forward-strand gene body
local = genome["chr1", 1_000_000:2_000_000:100]  # local range, 100 bp bins
```

### Single-cell count matrix

```python
from suba import CountMatrixTracks
from suba.io import load_count_matrix

barcodes, gene_names, matrix, length_normalize = load_count_matrix(
    "filtered_feature_bc_matrix/"   # 10x MEX directory
)

tracks = CountMatrixTracks(matrix, barcodes, gene_names)

# Signal for one cell over a genomic range
sig = tracks["AAACCTGA-1", 1_000_000:2_000_000]   # 1D array

# Signal for all cells, whole genome, 100 kbp bins
binned = tracks[:, ::100_000]   # 2D array (n_cells × n_bins)
```

### Hilbert-curve visualisation

```bash
# Gene-density map
suba --output genome.png

# Each chromosome a different colour
suba --color_by chromosome --output chromosomes.png

# Each gene a different colour
suba --color_by gene --output genes.png
```

---

## Project layout

```
suba/
├── src/suba/
│   ├── genome.py           # Genome base class
│   ├── sparse_rendering.py # Stateless rendering algorithms
│   ├── sc_tracks.py        # SCTracks ABC + CountMatrixTracks
│   ├── colors.py           # Golden-ratio label colormap
│   ├── cli.py              # Hilbert-curve CLI
│   └── io/
│       ├── genome.py       # GTF parsing + .npz cache
│       ├── count_matrix.py # 10x MEX / .npz / .h5 I/O
│       └── resumable_download.py
├── tests/
│   ├── unit/               # Fast synthetic-genome tests (no network)
│   └── integration/        # hg38 tests (requires SUBA_INTEGRATION=1)
└── docs/                   # Sphinx + MyST source
```

---

## Development

```bash
make test      # run all tests
make testcov   # unit tests + coverage report
make docs      # build HTML docs → docs/_build/html/
make clean     # remove build artefacts
```

---

## Coordinate conventions

All coordinates are **0-based, half-open** `[start, end)` throughout —
internally and in the public API.

```
chr1:1000-2000  →  length 1000  →  genome["chr1", 1000:2000]
```

Padding positions between chromosomes return `Genome.PADDING_VALUE` (`−1`).

---

## License

MIT © Anguelos Nicolaou
