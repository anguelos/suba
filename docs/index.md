# suba

**Single cell Unidimensional Base Annotation**

`suba` presents genomic data as 1D numpy-like objects where every integer
position refers to a single base in a *universal coordinate space* — all
chromosomes concatenated end-to-end with fixed inter-chromosomal padding.

## Contents

```{toctree}
:maxdepth: 2

autoapi/suba/index
```

## Quick start

```python
from suba import Genome

# Load hg38 (downloads and caches GTF + chrom.sizes on first run)
genome = Genome()

# Slice by universal coordinates (0-based half-open)
signal = genome[1_000_000:2_000_000]   # int32 array, length 1_000_000

# Binned average (4096 bp per bin)
binned = genome[::4096]

# Access by name (chromosome or gene)
chr1   = genome["chr1"]
brca1  = genome["BRCA1+"]

# Local slice within a chromosome or gene
region = genome["chr1", 1_000_000:2_000_000]
```

## Coordinate conventions

All coordinates are **0-based, half-open** `[start, end)` — internally and
in the public API.  Genomic coordinate `chr1:1000-2000` has length 1000.
Inter-chromosomal padding positions return `Genome.PADDING_VALUE` (`-1`).

## CLI

```
suba --help
suba --resolution=1024 --colormap=plasma --output=genome.png
```
