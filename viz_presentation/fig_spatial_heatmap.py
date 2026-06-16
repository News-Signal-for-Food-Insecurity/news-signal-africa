"""
fig_spatial_heatmap.py — "Spatial Heterogeneity in News Coverage" (talk slide 19).

Rows are news themes, columns are countries grouped by region (with region headers
and separators). Each cell is the mean relative coverage of a theme in a country,
binned into quintiles of the data. From the modelling dataset only.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import data_access as da
from heatmap_base import bin_index, legend_patches
from viz_theme import (BLUE_SEQ, INK, INK_SOFT, THEME_LABELS, apply_rcparams,
                       is_bare, save_fig)
import matplotlib.patches as mpatches

DISPLAY_THEMES = ["conflict", "displacement", "economic", "food_security",
                  "governance", "health", "humanitarian", "weather"]


def render() -> None:
    print("[fig] Spatial heterogeneity heatmap")
    apply_rcparams()

    mat = da.spatial_theme_matrix().reindex(DISPLAY_THEMES) * 100.0  # -> %
    regions = da.countries_by_region()
    ordered_countries = [c for _, members in regions for c in members]
    mat = mat[ordered_countries]
    values = mat.to_numpy()

    bins = da.quantile_bins(values.ravel(), [0, 0.2, 0.4, 0.6, 0.8, 1.0])  # quintiles

    n_rows = len(DISPLAY_THEMES)
    fig, ax = plt.subplots(figsize=(13.0, 6.6))

    # Draw cells with a small horizontal gap between region blocks.
    GAP = 0.45
    col_x, x = {}, 0.0
    region_spans = []
    for region, members in regions:
        start = x
        for country in members:
            col_x[country] = x
            x += 1.0
        region_spans.append((region, start, x))
        x += GAP
    total_w = x - GAP

    for r, theme in enumerate(DISPLAY_THEMES):
        yrow = n_rows - 1 - r
        for c_idx, country in enumerate(ordered_countries):
            ci = bin_index(values[r, c_idx], bins)
            ax.add_patch(mpatches.Rectangle(
                (col_x[country], yrow), 1, 1, facecolor=BLUE_SEQ[ci],
                edgecolor="white", linewidth=1.3, zorder=2))

    # Region separators + headers
    for region, start, end in region_spans:
        ax.text((start + end) / 2.0, n_rows + 0.35, region, ha="center",
                va="bottom", fontsize=11, fontweight="bold", color=INK)
        if end < total_w:
            ax.axvline(end + GAP / 2.0, color="#BBBBBB", lw=1.0, zorder=1)

    ax.set_xlim(-0.1, total_w + 0.1)
    ax.set_ylim(0, n_rows + 0.9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([THEME_LABELS[t] for t in DISPLAY_THEMES[::-1]],
                       fontsize=11, fontweight="bold")
    ax.set_xticks([col_x[c] + 0.5 for c in ordered_countries])
    ax.set_xticklabels(ordered_countries, rotation=55, ha="right", fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not is_bare():
        fig.suptitle("Spatial Heterogeneity in News Coverage Across Africa",
                     x=0.5, y=1.04, fontsize=18, fontweight="bold", color=INK)
        ax.set_title("Mean relative coverage of news themes by country (grouped by region)",
                     fontsize=11.5, color=INK_SOFT, pad=30)

    handles = legend_patches(bins, BLUE_SEQ, pct=True, quantile=True)
    ax.legend(handles=handles, title="Relative coverage\n(quintile)",
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              title_fontsize=10, frameon=False, handlelength=1.3,
              handleheight=1.3, labelspacing=0.7)

    if not is_bare():
        fig.text(0.5, -0.02, "Source: GDELT news themes by country (district mean). "
                 "Bins are quintiles of the data.", ha="center", fontsize=8,
                 color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "spatial_heterogeneity_heatmap")


if __name__ == "__main__":
    render()
