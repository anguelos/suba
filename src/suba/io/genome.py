"""
suba.io.genome
~~~~~~~~~~~~~~
GTF parsing, chrom.sizes loading, and .npz gene-cache helpers.

All public functions are pure (no class state).  Used by
:class:`suba.genome.Genome` and callable directly for lower-level access.
"""
from __future__ import annotations

import gzip
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from suba.io.resumable_download import download_resumable


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def infer_chrom_sizes_url(gtf_url: str) -> Optional[str]:
    """Derive the UCSC chrom.sizes URL from a UCSC GTF URL, or return None."""
    m = re.match(
        r"(https?://hgdownload\.soe\.ucsc\.edu/goldenPath/(\w+)/)", gtf_url
    )
    if m:
        base, asm = m.group(1), m.group(2)
        return f"{base}bigZips/{asm}.chrom.sizes"
    return None


def ensure_cached(url_or_path: str, cache_dir: Path) -> Path:
    """Return a local Path, downloading into *cache_dir* if needed."""
    p = Path(url_or_path)
    if p.exists():
        return p
    filename = url_or_path.rstrip("/").split("/")[-1]
    dest = cache_dir / filename
    if not dest.exists():
        download_resumable(url_or_path, dest)
    return dest


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def _parse_gtf_attr(attrs: str, key: str) -> Optional[str]:
    m = re.search(rf'{key} "([^"]+)"', attrs)
    return m.group(1) if m else None


def parse_gtf_genes(
    path: Path,
    transcript_id_prefixes: Optional[tuple] = None,
) -> list[tuple]:
    """Parse gene records from a GTF file (gzipped or plain).

    Returns a list of ``(chrom, start, end, strand, gene_name)`` tuples
    where coordinates are **0-based half-open**.

    Falls back to ``transcript`` records and infers gene spans if no explicit
    ``gene`` feature rows are found.  *transcript_id_prefixes* filters which
    transcripts are used during the fallback (e.g. ``("NM_",)``).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    gene_records: list[tuple] = []
    transcript_records: list[tuple] = []

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, feature = fields[0], fields[2]
            start = int(fields[3]) - 1
            end   = int(fields[4])
            strand, attrs = fields[6], fields[8]

            gene_name = (
                _parse_gtf_attr(attrs, "gene_name")
                or _parse_gtf_attr(attrs, "gene_id")
                or ""
            )

            if feature == "gene":
                gene_records.append((chrom, start, end, strand, gene_name))
            elif feature == "transcript":
                if transcript_id_prefixes is not None:
                    tid = _parse_gtf_attr(attrs, "transcript_id") or ""
                    if not any(tid.startswith(p) for p in transcript_id_prefixes):
                        continue
                transcript_records.append((chrom, start, end, strand, gene_name))

    if gene_records:
        return gene_records

    gene_dict: dict[tuple, list] = {}
    for chrom, start, end, strand, name in transcript_records:
        key = (name, strand, chrom)
        if key not in gene_dict:
            gene_dict[key] = [start, end]
        else:
            gene_dict[key][0] = min(gene_dict[key][0], start)
            gene_dict[key][1] = max(gene_dict[key][1], end)

    return [
        (chrom, span[0], span[1], strand, name)
        for (name, strand, chrom), span in gene_dict.items()
    ]


def parse_chrom_sizes(path: Path) -> dict[str, int]:
    """Parse a UCSC chrom.sizes file into ``{name: size}``."""
    sizes: dict[str, int] = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    return sizes


# ---------------------------------------------------------------------------
# .npz gene cache
# ---------------------------------------------------------------------------

def npz_cache_path(gtf_path: Path, transcript_id_prefixes) -> Path:
    """Return the .npz sidecar path for a GTF + prefix-filter combination."""
    if transcript_id_prefixes is None:
        suffix = "all"
    else:
        suffix = "_".join(sorted(transcript_id_prefixes)).replace("_", "")
    return gtf_path.parent / (gtf_path.name.split(".gtf")[0] + f"_genes_{suffix}.npz")


def save_gene_cache(npz_path: Path, raw_genes: list) -> None:
    """Serialise a raw_genes list to a compressed .npz sidecar."""
    if not raw_genes:
        np.savez_compressed(
            npz_path,
            chroms=np.array([], dtype=object),
            starts=np.array([], dtype=np.int64),
            ends=np.array([], dtype=np.int64),
            strands=np.array([], dtype=object),
            names=np.array([], dtype=object),
        )
        return
    np.savez_compressed(
        npz_path,
        chroms=np.array([g[0] for g in raw_genes], dtype=object),
        starts=np.array([g[1] for g in raw_genes], dtype=np.int64),
        ends=np.array([g[2] for g in raw_genes], dtype=np.int64),
        strands=np.array([g[3] for g in raw_genes], dtype=object),
        names=np.array([g[4] for g in raw_genes], dtype=object),
    )


def load_gene_cache(npz_path: Path) -> list:
    """Load a raw_genes list from a .npz sidecar."""
    data = np.load(npz_path, allow_pickle=True)
    return list(zip(
        data["chroms"].tolist(),
        data["starts"].tolist(),
        data["ends"].tolist(),
        data["strands"].tolist(),
        data["names"].tolist(),
    ))
