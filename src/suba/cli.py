"""
suba.cli
~~~~~~~~
Command-line interface for suba — Hilbert-curve genome visualiser.

Subcommands
-----------

``genes``
    Colour each pixel by the rank of the gene whose body it falls in
    (golden-angle label colourmap; intergenic = background colour).

``chromosomes``
    Colour each pixel by its chromosome (golden-angle label colourmap).

``density``
    Colour each pixel by the number of gene bodies that overlap it
    (continuous colourmap).

``coding``
    Binary genic / intergenic mask: one colour for positions inside any
    gene body, another for intergenic regions.

Usage
-----
::

    suba [global options] <subcommand> [subcommand options]

    suba genes --output_file=hilbert.png
    suba chromosomes --output_file="" --color_bar
    suba density --colormap=plasma --output_file=density.png
    suba coding --output_file=""

    # shared flags (before the subcommand)
    suba --resolution=2048 --dpi=200 genes --output_file=hi_res.png
"""
from __future__ import annotations

import fargv

from suba.genome import Genome, _DEFAULT_GTF_URL
from suba.util.colors import label_colormap
from suba.util.hilbert import signal_to_hilbert

import numpy as np


# ---------------------------------------------------------------------------
# Subcommand definitions
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    "genes": {
        # no subcommand-specific parameters
    },
    "chromosomes": {
        # no subcommand-specific parameters
    },
    "density": {
        "colormap": ("viridis", "Matplotlib colormap name for the continuous signal."),
    },
    "coding": {
        "genic_color":      ("#4e9af1", "Colour for positions inside a gene body (any matplotlib colour spec)."),
        "intergenic_color": ("#111111", "Colour for intergenic positions."),
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``suba`` CLI command."""
    p, _ = fargv.parse(
        {
            "gtf_url":    (_DEFAULT_GTF_URL, "URL or local path to a GTF(.gz) annotation file."),
            "cache_dir":  ("./tmp/cache",    "Directory for caching downloaded files."),
            "padding":    (1_000_000,        "Inter-chromosomal padding in bases."),
            "resolution": (1024,             "Hilbert grid side length (power of 2). Output is resolution×resolution pixels."),
            "dpi":        (150,              "Output image DPI."),
            "output_file":("",              'Output path (e.g. hilbert.png). Empty string "" shows interactively.'),
            "color_bar":  (True,             "Attach a colour-bar legend to the figure."),
            "cmd": _SUBCOMMANDS,
        },
        subcommand_return_type="flat",
    )

    if (p.resolution & (p.resolution - 1)) != 0:
        raise SystemExit(
            f"--resolution must be a power of 2, got {p.resolution}."
        )

    print(f"Loading genome from {p.gtf_url!r} ...")
    genome = Genome(
        gtf_url=p.gtf_url,
        cache_dir=p.cache_dir,
        padding=p.padding,
    )
    print(repr(genome))

    n_pixels = p.resolution * p.resolution
    total    = genome.total_length
    step     = max(1, total // n_pixels)
    print(f"Rendering [::{step}] ({total:,} bases, {step:,} bp/pixel) ...")

    # ── Build signal, colourmap, and metadata per subcommand ────────────
    if p.cmd == "genes":
        raw         = genome.gene_label_signal(step=step)
        cmap        = label_colormap(genome.n_genes)
        discrete    = True
        tick_labels = None          # too many genes to label individually
        title       = f"Gene identity — {p.resolution}×{p.resolution} @ {step:,} bp/pixel"

    elif p.cmd == "chromosomes":
        raw         = genome.chromosome_label_signal(step=step)
        cmap        = label_colormap(genome.n_chromosomes)
        discrete    = True
        tick_labels = genome._chromosome_name.tolist()
        title       = f"Chromosome identity — {p.resolution}×{p.resolution} @ {step:,} bp/pixel"

    elif p.cmd == "density":
        raw         = genome[::step].astype(np.float64)
        np.nan_to_num(raw, nan=0.0, copy=False)
        cmap        = p.colormap
        discrete    = False
        tick_labels = None
        title       = f"Gene overlap density — {p.resolution}×{p.resolution} @ {step:,} bp/pixel"

    elif p.cmd == "coding":
        import matplotlib.colors as mcolors
        raw         = (genome[::step] > 0).astype(np.float64)
        cmap        = mcolors.ListedColormap([p.intergenic_color, p.genic_color])
        discrete    = False
        tick_labels = None
        title       = f"Genic / intergenic — {p.resolution}×{p.resolution} @ {step:,} bp/pixel"

    else:
        raise SystemExit(f"Unknown subcommand: {p.cmd!r}")

    # ── Render ──────────────────────────────────────────────────────────
    image, fig = signal_to_hilbert(
        raw,
        colormap    = cmap,
        discrete    = discrete,
        legend      = p.color_bar,
        resolution  = p.resolution,
        dpi         = p.dpi,
        title       = title,
        tick_labels = tick_labels,
    )

    # ── Output ───────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    if p.output_file == "":
        plt.show()
    else:
        from pathlib import Path
        out = Path(p.output_file)
        fig.savefig(out, dpi=p.dpi, bbox_inches="tight")
        print(f"Saved to {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
