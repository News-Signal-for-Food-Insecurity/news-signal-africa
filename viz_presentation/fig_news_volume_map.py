"""
fig_news_volume_map.py — "Volume of news articles 2020-2024" choropleth (slide 16L).

Each study ADM2 district is shaded by its cumulative GDELT article count (blue
quantile bins). Article totals from monthly_gdelt_features; geometry from GADM.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import data_access as da
from map_base import draw_choropleth
from viz_theme import BLUE_SEQ, INK, INK_SOFT, apply_rcparams, is_bare, save_fig


def _fmt_k(v: float) -> str:
    v = int(v)
    return f"{v // 1000}k" if v >= 1000 else str(v)


def render() -> None:
    print("[fig] News-volume choropleth")
    apply_rcparams()

    totals = da.district_article_totals()
    plot, africa = da.choropleth_frame(totals, "articles")
    vals = plot["articles"].dropna().to_numpy()
    bins = da.quantile_bins(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    labels = [f"{_fmt_k(bins[i])} – {_fmt_k(bins[i+1])}" for i in range(len(bins) - 1)]
    # Clip into the top bin so BoundaryNorm covers every value.
    plot["articles"] = plot["articles"].clip(upper=bins[-1] - 1e-6)

    fig, ax = plt.subplots(figsize=(8.0, 9.2))
    draw_choropleth(ax, plot, africa, "articles", bins, BLUE_SEQ, labels,
                    "Cumulative articles\n(2021–2024)")
    if not is_bare():
        ax.set_title("Volume of News Articles, 2020–2024", fontsize=17,
                     fontweight="bold", color=INK, pad=10)
        fig.text(0.5, 0.02, "Source: GDELT Project. District boundaries: GADM v4.1 "
                 "(ADM2). Bins are quintiles of cumulative article volume.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "news_volume_choropleth")


if __name__ == "__main__":
    render()
