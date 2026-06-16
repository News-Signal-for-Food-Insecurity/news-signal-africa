"""
fig_datasources_map.py — study-scope / data-sources basemap (talk slide 25).

A greyscale map of the study area: ADM2 district boundaries over ADM0 country
borders, with illustrative flow arrows toward the major crisis corridors (Sahel,
Lake Chad, Horn of Africa, South-East). North arrow, scale bar and a data-source
attribution box. The geometry is real (GADM); the arrows are fixed cartographic
annotations marking where the analysis concentrates, not data values.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import data_access as da
from viz_theme import (CRIMSON, INK, INK_SOFT, add_north_arrow, apply_rcparams,
                       is_bare, save_fig)

# Illustrative flow arrows: (start_lon, start_lat) -> (end_lon, end_lat).
# Endpoints sit over crisis corridors the study covers.
FLOW_ARROWS = [
    ((-2, 26), (2, 15)),     # into the western Sahel (Mali / Niger)
    ((24, 28), (30, 14)),    # into Sudan / South Sudan
    ((52, 14), (45, 6)),     # into the Horn (Somalia)
    ((40, 32), (38, 12)),    # into Ethiopia highlands
    ((36, -28), (34, -18)),  # into south-east (Mozambique / Zimbabwe)
]


def render() -> None:
    print("[fig] Data-sources study-scope map")
    apply_rcparams()

    import geopandas as gpd
    adm2 = gpd.read_file(da.SHAPEFILE_ADM2)
    africa = gpd.read_file(da.SHAPEFILE_ADM0)
    study = adm2[adm2["country_name"].isin(da.study_countries())].copy()

    fig, ax = plt.subplots(figsize=(8.4, 9.0))
    africa.plot(ax=ax, color="#FBFBFB", edgecolor="#C8C8C8", linewidth=0.4,
                zorder=1)
    study.plot(ax=ax, color="#ECECEC", edgecolor="#D2D2D2", linewidth=0.12,
               zorder=2)
    africa.plot(ax=ax, color="none", edgecolor="#666666", linewidth=0.7,
                zorder=3)

    for (x0, y0), (x1, y1) in FLOW_ARROWS:
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle="arc3,rad=0.25",
            arrowstyle="-|>", mutation_scale=18, lw=2.0, color=CRIMSON,
            zorder=6, alpha=0.9))

    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 30)
    ax.set_axis_off()
    add_north_arrow(ax, x=0.93, y0=0.90, y1=0.97)

    # Scale bar (~1000 km per ~9 degrees of longitude), placed in the empty
    # ocean at the bottom-right so it never collides with the legend.
    deg_per_1000km = 1000.0 / 111.0
    x_sb, y_sb = 33.0, -34.0
    for i in range(2):
        ax.add_patch(mpatches.Rectangle(
            (x_sb + i * deg_per_1000km, y_sb), deg_per_1000km, 0.8,
            facecolor="#333333" if i % 2 == 0 else "white",
            edgecolor="#333333", linewidth=0.8, zorder=7))
    for i, lab in enumerate(["0", "1000", "2000 km"]):
        ax.text(x_sb + i * deg_per_1000km, y_sb - 1.0, lab, ha="center",
                va="top", fontsize=7.5, color=INK)

    from matplotlib.lines import Line2D
    legend = [
        mpatches.Patch(facecolor="#ECECEC", edgecolor="#D2D2D2",
                       label="Administrative localities (ADM2)"),
        Line2D([0], [0], color="#666666", lw=1.0, label="Country boundaries"),
        Line2D([0], [0], color=CRIMSON, lw=2.0, label="Analysis focus corridors"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8.5, framealpha=0.96,
              edgecolor="#CCCCCC", borderpad=0.8, handlelength=1.6,
              labelspacing=0.6)

    if not is_bare():
        ax.set_title("Study Scope: News Signals Across African Districts",
                     fontsize=15.5, fontweight="bold", color=INK, pad=10)
        fig.text(0.5, 0.02, "Boundaries: GADM v4.1 (ADM2 districts, ADM0 countries). "
                 "News: GDELT Project (2020–2024). Arrows mark crisis corridors.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "study_scope_datasources_map")


if __name__ == "__main__":
    render()
