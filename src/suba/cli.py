"""
suba.cli
~~~~~~~~
Command-line interface for suba.

Renders the genome as a Hilbert-curve heatmap distinguishing genic regions
(gene bodies) from intergenic regions.

Usage
-----
::

    suba [options]
    suba --help

Argument parsing uses :mod:`fargv` with the dataclass style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fargv

from suba.genome import Genome, _DEFAULT_GTF_URL
from suba.sparse_rendering import hilbert_d_to_xy
from suba.colors import label_colormap


# ---------------------------------------------------------------------------
# CLI parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    gtf_url: str = _DEFAULT_GTF_URL
    "URL or local path to a GTF(.gz) genome annotation file."

    cache_dir: str = "./tmp/cache"
    "Directory for caching downloaded files."

    padding: int = 1_000_000
    "Inter-chromosomal padding in bases."

    resolution: int = 1024
    "Hilbert curve grid side length (must be a power of 2). Image is resolution x resolution pixels."

    colormap: str = "viridis"
    "Matplotlib colormap name for the Hilbert curve plot."

    output: str = "/tmp/hilbert_genome.png"
    "Output image file path. Use \'show\' to display interactively."

    dpi: int = 150
    "Output image DPI."

    color_by: str = "density"
    "What to color: 'density' (gene overlap count) or 'chromosome' (each chromosome a distinct colour)."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two_side(n_values: int) -> int:
    """Return smallest power-of-2 p such that p*p >= n_values."""
    p = 1
    while p * p < n_values:
        p <<= 1
    return p


def _build_hilbert_image(
    signal: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Map a 1D signal onto a resolution x resolution Hilbert curve image.

    The signal is padded with zeros to fill ``resolution ** 2`` pixels.
    If the signal is longer, it is truncated.

    Parameters
    ----------
    signal:
        1D array of values (float or int).
    resolution:
        Side length of the output grid; must be a power of 2.

    Returns
    -------
    np.ndarray of float64, shape (resolution, resolution)
    """
    n_pixels = resolution * resolution

    # Pad or truncate to exactly n_pixels
    if len(signal) < n_pixels:
        signal = np.pad(signal.astype(np.float64), (0, n_pixels - len(signal)))
    else:
        signal = signal[:n_pixels].astype(np.float64)

    indices = np.arange(n_pixels, dtype=np.int64)
    x, y = hilbert_d_to_xy(resolution, indices)

    image = np.zeros((resolution, resolution), dtype=np.float64)
    image[y, x] = signal
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``suba`` CLI command."""
    cfg, _ = fargv.parse(Config)

    if not _is_power_of_two(cfg.resolution):
        raise ValueError(
            f"--resolution must be a power of 2, got {cfg.resolution}."
        )

    print(f"Loading genome from {cfg.gtf_url!r} ...")
    genome = Genome(
        gtf_url=cfg.gtf_url,
        cache_dir=cfg.cache_dir,
        padding=cfg.padding,
    )
    print(repr(genome))

    # Compute step size so that genome[::step] gives ~ resolution^2 bins
    n_pixels = cfg.resolution * cfg.resolution
    total = genome.total_length
    step = max(1, total // n_pixels)
    print(
        f"Rendering genome[::{step}] "
        f"({total:,} bases / {step} bp per pixel) ..."
    )
    
    # ── Render signal ─────────────────────────────────────────────────────
    if cfg.color_by == "chromosome":
        raw = genome.chromosome_label_signal(step=step).astype(np.float64)
        n_chrom = genome.n_chromosomes
        cmap = label_colormap(n_chrom)
        vmin, vmax = 0, n_chrom
        cbar_ticks = list(range(1, n_chrom + 1))
        cbar_labels = genome._chromosome_name.tolist()
        title_suffix = "chromosome identity"
    elif cfg.color_by == "gene":
        raw = genome.gene_label_signal(step=step).astype(np.float64)
        n_genes = genome.n_genes
        cmap = label_colormap(n_genes)
        vmin, vmax = 0, n_genes
        cbar_ticks = None
        cbar_labels = None
        title_suffix = "gene identity"
    else:
        raw = genome[::step]
        raw = np.nan_to_num(raw, nan=0.0)
        cmap = cfg.colormap
        vmin, vmax = None, None
        cbar_ticks = None
        cbar_labels = None
        title_suffix = "gene overlap count"

    image = _build_hilbert_image(raw, cfg.resolution)

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "CLI plotting requires the [cli] extra: "
            "pip install suba[cli]"
        ) from exc

    fig, ax = plt.subplots(
        figsize=(cfg.resolution / cfg.dpi, cfg.resolution / cfg.dpi),
        dpi=cfg.dpi,
    )
    im = ax.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labels, fontsize=5)
    ax.set_title(
        f"{title_suffix.capitalize()} — Hilbert curve\n"
        f"{cfg.resolution}x{cfg.resolution} @ {step:,} bp/pixel"
    )

    if cfg.output == "show":
        plt.show()
    else:
        out_path = Path(cfg.output)
        fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
