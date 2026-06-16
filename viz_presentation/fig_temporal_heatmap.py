"""
fig_temporal_heatmap.py — "Temporal Heterogeneity in News Coverage" (talk slide 18).

Rows are news themes, columns are the 11 IPC periods; each cell is the mean
relative coverage of that theme across all districts in that period, binned into a
5-step blue scale. Bin edges come from the data quantiles. From the modelling
dataset only.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import data_access as da
from heatmap_base import draw_cells, legend_patches
from viz_theme import (BLUE_SEQ, INK, INK_SOFT, THEME_LABELS, apply_rcparams,
                       is_bare, save_fig)

# The talk presents the eight substantive themes (the residual "other" is omitted).
DISPLAY_THEMES = ["conflict", "displacement", "economic", "food_security",
                  "governance", "health", "humanitarian", "weather"]


def render() -> None:
    print("[fig] Temporal heterogeneity heatmap")
    apply_rcparams()

    mat_df = da.temporal_theme_matrix().reindex(DISPLAY_THEMES) * 100.0  # -> %
    values = mat_df.to_numpy()
    periods = list(mat_df.columns)

    bins = da.quantile_bins(values.ravel(), [0, 0.45, 0.75, 0.92, 0.985, 1.0])

    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    draw_cells(ax, values, bins, BLUE_SEQ)

    n_rows, n_cols = values.shape
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(periods, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([THEME_LABELS[t] for t in DISPLAY_THEMES[::-1]],
                       fontsize=11, fontweight="bold")
    ax.tick_params(length=0)

    if not is_bare():
        fig.suptitle("Temporal Heterogeneity in News Coverage Across Africa",
                     x=0.5, y=1.02, fontsize=18, fontweight="bold", color=INK)
        ax.set_title("Mean relative coverage of news themes by IPC period",
                     fontsize=11.5, color=INK_SOFT, pad=24, loc="center")

    handles = legend_patches(bins, BLUE_SEQ, pct=True)
    ax.legend(handles=handles, title="Relative\ncoverage",
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9.5, title_fontsize=10, frameon=False,
              handlelength=1.3, handleheight=1.3, labelspacing=0.7)

    if not is_bare():
        fig.text(0.5, -0.01, "Source: GDELT news themes aggregated to IPC periods "
                 "(district mean). Bin edges from data quantiles.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "temporal_heterogeneity_heatmap")


if __name__ == "__main__":
    render()
