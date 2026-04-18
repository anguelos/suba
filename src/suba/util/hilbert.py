"""
suba.util.hilbert
~~~~~~~~~~~~~~~~~
Hilbert-curve rendering utilities.

Provides :func:`signal_to_hilbert` which maps any 1D signal onto a 2D
Hilbert-curve image and returns both the raw image array and a ready-to-use
:class:`matplotlib.figure.Figure`.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from suba.sparse_rendering import hilbert_d_to_xy
from suba.util.colors import label_colormap


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def signal_to_hilbert(
    signal: np.ndarray,
    colormap: Union[str, "matplotlib.colors.Colormap"] = "viridis",
    discrete: bool = False,
    legend: bool = True,
    resolution: Optional[int] = None,
    dpi: int = 150,
    title: Optional[str] = None,
    tick_labels: Optional[list] = None,
) -> tuple[np.ndarray, "matplotlib.figure.Figure"]:
    """Render a 1D signal as a Hilbert-curve 2D image.

    Parameters
    ----------
    signal:
        1D (or squeeze-able to 1D) array of values.  Integer arrays are
        treated as discrete labels; float arrays as continuous unless
        *discrete* is set explicitly.
    colormap:
        Matplotlib colormap name **or** a
        :class:`~matplotlib.colors.Colormap` instance.  When *discrete* is
        ``True`` and *colormap* is the string ``"auto"``, a golden-angle
        :func:`~suba.util.colors.label_colormap` is generated automatically
        with one colour per unique label value.
    discrete:
        ``True`` → treat the signal as integer labels (no interpolation,
        colour-bar shows integer ticks).  Inferred from dtype when ``False``
        is not explicitly passed — integer arrays default to ``True``, float
        arrays to ``False``.
    legend:
        Whether to attach a colour-bar to the figure.
    resolution:
        Side length of the square output grid (must be a power of 2).
        Defaults to the smallest power of 2 whose square fits the signal
        (i.e. ``ceil(sqrt(len(signal)))`` rounded up to the next power of 2).
    dpi:
        Figure DPI.
    title:
        Optional figure title.
    tick_labels:
        For discrete colourmaps, an optional list of string labels
        corresponding to integer values ``1, 2, …, n_labels``.  If ``None``
        the integer values themselves are used.

    Returns
    -------
    image : np.ndarray, shape ``(resolution, resolution)``
        2D Hilbert-curve image array (``float64``).
    fig : matplotlib.figure.Figure
        Ready-to-save / ready-to-show figure.  Call ``fig.savefig(path)`` or
        ``plt.show()`` after this function returns.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.random.rand(4096)
    >>> image, fig = signal_to_hilbert(signal, colormap="plasma", legend=True)
    >>> image.shape
    (64, 64)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError as exc:
        raise ImportError(
            "signal_to_hilbert requires matplotlib.  "
            "Install it with: pip install matplotlib"
        ) from exc

    # ── Normalise signal to 1D ────────────────────────────────────────────
    signal = np.squeeze(np.asarray(signal))
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1D after squeezing; got shape {signal.shape}"
        )

    # ── Infer discrete mode from dtype ───────────────────────────────────
    if not isinstance(discrete, bool):
        discrete = bool(discrete)
    if np.issubdtype(signal.dtype, np.integer) and discrete is False:
        # respect explicit False — caller knows best
        pass
    if np.issubdtype(signal.dtype, np.integer):
        discrete = True  # default for integer dtype

    # ── Choose resolution ─────────────────────────────────────────────────
    if resolution is None:
        side = _next_power_of_two(math.ceil(math.sqrt(len(signal))))
        resolution = max(side, 2)
    elif (resolution & (resolution - 1)) != 0 or resolution < 2:
        raise ValueError(
            f"resolution must be a power of 2 >= 2, got {resolution}"
        )

    n_pixels = resolution * resolution

    # ── Pad / truncate ────────────────────────────────────────────────────
    sig_f = signal.astype(np.float64)
    if len(sig_f) < n_pixels:
        sig_f = np.pad(sig_f, (0, n_pixels - len(sig_f)))
    else:
        sig_f = sig_f[:n_pixels]

    # ── Build 2D image via Hilbert mapping ────────────────────────────────
    indices = np.arange(n_pixels, dtype=np.int64)
    x, y = hilbert_d_to_xy(resolution, indices)
    image = np.zeros((resolution, resolution), dtype=np.float64)
    image[y, x] = sig_f

    # ── Resolve colormap ──────────────────────────────────────────────────
    if discrete:
        n_labels = int(sig_f.max()) if sig_f.size > 0 else 0
        if colormap == "auto" or colormap == "viridis":
            cmap = label_colormap(n_labels)
        elif isinstance(colormap, str):
            cmap = plt.get_cmap(colormap, n_labels + 1)
        else:
            cmap = colormap
        vmin, vmax = 0, n_labels
    else:
        cmap = colormap if not isinstance(colormap, str) else plt.get_cmap(colormap)
        vmin, vmax = None, None

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(resolution / dpi, resolution / dpi),
        dpi=dpi,
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

    if legend:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if discrete and n_labels <= 50:
            ticks = list(range(1, n_labels + 1))
            cbar.set_ticks(ticks)
            if tick_labels is not None:
                cbar.set_ticklabels(tick_labels, fontsize=5)
            else:
                cbar.set_ticklabels([str(t) for t in ticks], fontsize=5)

    if title is not None:
        ax.set_title(title)

    return image, fig
