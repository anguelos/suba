"""
suba.colors
~~~~~~~~~~~
Colormap utilities for genomic visualisation.
"""
from __future__ import annotations

import colorsys
import matplotlib.colors as mcolors
import numpy as np


def label_colormap(
    n_labels: int,
    background_color: tuple = (0.08, 0.08, 0.08),
    saturation_range: tuple = (0.55, 0.90),
    value_range: tuple = (0.75, 0.95),
) -> mcolors.ListedColormap:
    """Build a :class:`~matplotlib.colors.ListedColormap` for integer labels.

    Consecutive label integers are assigned hues separated by the golden
    angle (≈137.5°), which guarantees that no two nearby integers share a
    similar hue regardless of how many labels are used.  Saturation and
    brightness are modulated by independent sub-golden sequences to provide
    extra visual separation when hues are close.

    Label ``0`` is reserved for background (default: near-black).

    Parameters
    ----------
    n_labels:
        Number of distinct labels **excluding** background.  The returned
        colormap has ``n_labels + 1`` entries (index 0 = background).
    background_color:
        RGB tuple for label 0 (background / padding).
    saturation_range:
        ``(min, max)`` saturation for label colours.
    value_range:
        ``(min, max)`` HSV value (brightness) for label colours.

    Returns
    -------
    matplotlib.colors.ListedColormap

    Examples
    --------
    >>> cmap = label_colormap(25)
    >>> # use with imshow / heatmap for up to 25 distinct labels
    """
    golden = (1.0 + 5.0 ** 0.5) / 2.0   # φ ≈ 1.618
    hue_step = 1.0 / golden               # ≈ 0.618 — golden angle in [0,1)
    # Secondary offsets for saturation/value — use independent irrational steps
    # so saturation and value cycles are incommensurate with the hue cycle.
    sat_step = 1.0 / (golden ** 2)        # ≈ 0.382
    val_step = 1.0 / (golden ** 3)        # ≈ 0.236

    s_lo, s_hi = saturation_range
    v_lo, v_hi = value_range

    colors = [background_color]
    for i in range(1, n_labels + 1):
        h = (i * hue_step) % 1.0
        s = s_lo + (s_hi - s_lo) * ((i * sat_step) % 1.0)
        v = v_lo + (v_hi - v_lo) * ((i * val_step) % 1.0)
        colors.append(colorsys.hsv_to_rgb(h, s, v))

    return mcolors.ListedColormap(colors, name="suba_labels")
