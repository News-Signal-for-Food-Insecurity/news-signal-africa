"""
fig_null_histogram.py — "Distribution of PR-AUC Scores Across N Null Models" (slide 26).

Histogram of the null-permutation PR-AUC scores (sage bars) with three dashed
markers: AR-only, Median-NULL and AR+News. N, the scores, the markers and the
p-value are all read from results/shuffle_test_v3 and results/window_2yr — the
title shows the real N (1,000), not the slide's rounded "100".
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import data_access as da
from viz_theme import (INK, INK_SOFT, NULL_BAR_COLOUR, NULL_BAR_EDGE,
                       apply_rcparams, is_bare, save_fig, style_axes)


def render(metric: str = "pr_auc") -> None:
    print("[fig] Null-distribution histogram")
    apply_rcparams()
    s = da.null_summary(metric)
    scores = s["scores"]

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.hist(scores, bins=34, color=NULL_BAR_COLOUR, edgecolor=NULL_BAR_EDGE,
            linewidth=0.6, zorder=2)
    style_axes(ax, grid_axis="y")

    label = "PR-AUC" if metric == "pr_auc" else "ROC-AUC"
    markers = [("AR only", s["real_ar"], "right", -0.0015),
               ("Median NULL", s["median_null"], "center", 0.0),
               ("AR + News", s["real_full"], "left", 0.0015)]
    ax.set_ylim(top=ax.get_ylim()[1] * 1.16)   # headroom for labels
    ymax = ax.get_ylim()[1]
    for name, xval, ha, dx in markers:
        ax.axvline(xval, color=INK, linestyle=(0, (4, 3)), lw=1.6, zorder=4)
        ax.text(xval + dx, ymax * 0.995, f"{name}\n{xval:.4f}", ha=ha,
                va="top", fontsize=9.5, color=INK, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#DDDDDD", alpha=1.0), zorder=6)

    ax.set_xlabel(f"{label}  (mean across folds)")
    ax.set_ylabel("Frequency")
    if not is_bare():
        ax.set_title(f"Distribution of {label} Scores Across "
                     f"{s['n_models']:,} Null Models", pad=12)

    # p-value annotation, computed from the null test.
    ax.text(0.015, 0.97, f"AR + News vs. null:  p = {s['p_value']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            color=INK_SOFT)

    if not is_bare():
        fig.text(0.5, 0.005, "Each null model retrains on news features whose "
                 "semantic alignment has been destroyed by within-column shuffling.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, f"null_distribution_{metric}")


if __name__ == "__main__":
    render("pr_auc")
    render("roc_auc")
