"""
map_base.py — shared choropleth cartography for the Africa maps.

A single ``draw_choropleth`` used by the news-volume and IPC-prevalence maps:
pale continent backdrop, study ADM2 polygons filled by a binned scale, country
borders + labels, north arrow and a labelled legend. Mirrors the working Fig 1
cartography in 06_paper_figures.py.
"""
from __future__ import annotations

import warnings

import matplotlib.patches as mpatches
import numpy as np

from viz_theme import (BORDER_GREY, INK, LAND_GREY, add_north_arrow, binned_cmap)

# Study-area window over Africa.
XLIM = (-20, 52)
YLIM = (-35, 23)

LABEL_SHORT = {
    "Democratic Republic of the Congo": "DR Congo",
    "Central African Republic": "C.A.R.",
}
LABEL_NUDGE = {
    "Burundi": (1.5, 0.0), "Rwanda": (1.5, 0.5), "Malawi": (1.2, 0.0),
    "South Sudan": (0.0, 0.5), "Democratic Republic of the Congo": (-1.5, -1.5),
}


def draw_choropleth(ax, study_gdf, africa_gdf, value_col, bins, colours,
                    bin_labels, legend_title, *, label_countries=True):
    """Render one choropleth panel on ``ax``."""
    cmap, norm = binned_cmap(colours, bins)

    # Layer 1 — full continent backdrop (pale grey).
    africa_gdf.plot(ax=ax, color=LAND_GREY, linewidth=0.3,
                    edgecolor="#AAAAAA", zorder=1)
    # Layer 2 — study ADM2 polygons.
    has = study_gdf[study_gdf[value_col].notna()].copy()
    nos = study_gdf[study_gdf[value_col].isna()]
    if len(nos):
        nos.plot(ax=ax, color="#CCCCCC", linewidth=0.1, edgecolor="#BBBBBB",
                 zorder=2)
    has.plot(ax=ax, column=value_col, cmap=cmap, norm=norm, linewidth=0.1,
             edgecolor="white", zorder=3)
    # Layer 3 — country borders.
    africa_gdf.plot(ax=ax, color="none", linewidth=0.7, edgecolor=BORDER_GREY,
                    zorder=4)

    ax.set_axis_off()
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    name_col = "COUNTRY" if "COUNTRY" in africa_gdf.columns else "country_name"
    if label_countries:
        for _, row in africa_gdf.iterrows():
            name = row[name_col]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cen = row.geometry.centroid
            dx, dy = LABEL_NUDGE.get(name, (0.0, 0.0))
            ax.text(cen.x + dx, cen.y + dy, LABEL_SHORT.get(name, name),
                    fontsize=5.2, ha="center", va="center", color="#222222",
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              alpha=0.6, edgecolor="none"),
                    zorder=7, clip_on=True)

    add_north_arrow(ax)

    patches = [mpatches.Patch(facecolor=colours[i], edgecolor="#999999",
                              linewidth=0.4, label=bin_labels[i])
               for i in range(len(bin_labels))]
    patches.insert(0, mpatches.Patch(visible=False, label=legend_title))
    patches.append(mpatches.Patch(facecolor="#D8D8D8", edgecolor="#CCCCCC",
                                  label="Not in study"))
    ax.legend(handles=patches, loc="lower left", fontsize=7.8, framealpha=0.96,
              edgecolor="#CCCCCC", handlelength=1.2, handleheight=1.0,
              borderpad=0.7, labelspacing=0.4)
