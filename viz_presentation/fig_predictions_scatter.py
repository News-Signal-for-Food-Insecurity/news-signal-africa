"""
fig_predictions_scatter.py — "Predictions with vs. without News" (talk slide 29).

Each point is one out-of-sample prediction: P(AR) on x, P(AR+News) on y, coloured
by crisis regime. The diagonal marks "news changes nothing". Legend counts come
straight from the regime value_counts. From fold_predictions.csv only.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import data_access as da
from viz_theme import (INK, INK_SOFT, REGIME_COLOURS, REGIME_ORDER,
                       apply_rcparams, is_bare, save_fig, style_axes)

# Draw crisis (red) regimes on top of non-crisis (green) ones for legibility.
_Z = {"stable": 2, "recovery": 3, "chronic": 4, "onset": 5}


def render() -> None:
    print("[fig] Predictions scatter by regime")
    apply_rcparams()
    fp = da.load_predictions()
    counts = da.regime_counts()

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.plot([0, 1], [0, 1], color="#444444", lw=1.2, linestyle=(0, (4, 3)),
            zorder=6)

    for regime in REGIME_ORDER:
        sub = fp[fp["regime"] == regime]
        ax.scatter(sub["prob_ar"], sub["prob_combined"], s=12,
                   color=REGIME_COLOURS[regime], alpha=0.55, linewidths=0,
                   zorder=_Z[regime])

    handles = [mlines.Line2D([], [], marker="o", linestyle="none",
               markersize=8, markerfacecolor=REGIME_COLOURS[r],
               markeredgecolor="none",
               label=f"{r.capitalize()}  (n = {counts.get(r, 0):,})")
               for r in REGIME_ORDER]
    leg = ax.legend(handles=handles, title="Crisis regime", loc="upper left",
                    fontsize=10, title_fontsize=10.5, framealpha=0.95)
    leg.get_frame().set_linewidth(0.8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("P(AR)  —  probability of crisis without news")
    ax.set_ylabel("P(AR + News)  —  probability with news")
    if not is_bare():
        ax.set_title("Predictions with vs. without News", pad=12)
    style_axes(ax, grid_axis="both")

    if not is_bare():
        fig.text(0.5, 0.005, "Points on the diagonal are unchanged by news; "
                 "dispersion off the diagonal concentrates around transitions.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "predictions_scatter_by_regime")


if __name__ == "__main__":
    render()
