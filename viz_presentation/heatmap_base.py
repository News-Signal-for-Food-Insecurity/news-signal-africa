"""
heatmap_base.py — shared drawing primitives for the two news-coverage heatmaps.

Both the temporal (themes x period) and spatial (themes x country) heatmaps use
binned blue cells with a labelled categorical legend, exactly like the talk. The
bin edges are derived from the data quantiles by the caller; nothing here encodes
a result.
"""
from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from viz_theme import BLUE_SEQ, INK, INK_SOFT


def bin_index(value: float, bins: list[float]) -> int:
    """Index of the bin (0..len-2) that ``value`` falls into; clamped."""
    if not np.isfinite(value):
        return 0
    idx = int(np.searchsorted(bins, value, side="right") - 1)
    return max(0, min(idx, len(bins) - 2))


def bin_labels(bins: list[float], *, pct: bool = True) -> list[str]:
    """Human-readable labels for each bin, e.g. '7.2 - 15.6 %'."""
    def f(v):
        return f"{v:.1f}" if pct else f"{v:.2f}"
    n = len(bins) - 1
    out = []
    for i in range(n):
        if i == 0:
            out.append(f"< {f(bins[1])}{' %' if pct else ''}")
        elif i == n - 1:
            out.append(f"≥ {f(bins[i])}{' %' if pct else ''}")
        else:
            out.append(f"{f(bins[i])} – {f(bins[i+1])}{' %' if pct else ''}")
    return out


def draw_cells(ax, matrix: np.ndarray, bins: list[float],
               colours: list[str] | None = None, *, edge="white", lw=1.4) -> None:
    """Draw a grid of binned rectangles. matrix is (n_rows, n_cols) of values."""
    colours = colours or BLUE_SEQ
    n_rows, n_cols = matrix.shape
    for r in range(n_rows):
        for c in range(n_cols):
            ci = bin_index(matrix[r, c], bins)
            ax.add_patch(mpatches.Rectangle(
                (c, n_rows - 1 - r), 1, 1, facecolor=colours[ci],
                edgecolor=edge, linewidth=lw, zorder=2))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def legend_patches(bins: list[float], colours: list[str] | None = None, *,
                   pct: bool = True, title: str | None = None,
                   quantile: bool = False) -> list:
    """Build labelled legend handles for the binned scale."""
    colours = colours or BLUE_SEQ
    labels = bin_labels(bins, pct=pct)
    if quantile:
        labels = [f"Q{i+1}  ({lab})" for i, lab in enumerate(labels)]
    patches = [mpatches.Patch(facecolor=colours[i], edgecolor="#FFFFFF",
                              linewidth=0.6, label=labels[i])
               for i in range(len(labels))]
    return patches
