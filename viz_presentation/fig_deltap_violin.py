"""
fig_deltap_violin.py — ΔP(crisis) by crisis regime (talk slide 30).

For each crisis regime, the distribution of ΔP(crisis) = P(AR+News) − P(AR-only)
as a violin with an embedded boxplot. Crisis regimes (onset, chronic) in red,
non-crisis (recovery, stable) in green. Counts above each violin, zero reference
line. From fold_predictions.csv only.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import data_access as da
from viz_theme import (INK, INK_SOFT, REGIME_COLOURS, REGIME_ORDER,
                       REGIME_SUBTITLES, apply_rcparams, is_bare, save_fig,
                       style_axes)


def render() -> None:
    print("[fig] Delta-P violins by regime")
    apply_rcparams()
    dp = da.delta_prob_by_regime()

    data = [dp.loc[dp["regime"] == r, "delta"].to_numpy() for r in REGIME_ORDER]
    positions = np.arange(1, len(REGIME_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.axhline(0, color="#777777", lw=1.0, linestyle=(0, (4, 3)), zorder=1)

    parts = ax.violinplot(data, positions=positions, widths=0.82,
                          showmeans=False, showextrema=False, showmedians=False)
    for body, regime in zip(parts["bodies"], REGIME_ORDER):
        body.set_facecolor(REGIME_COLOURS[regime])
        body.set_alpha(0.45)
        body.set_edgecolor(REGIME_COLOURS[regime])
        body.set_linewidth(1.0)

    # Embedded boxplots in each regime's colour.
    bp = ax.boxplot(data, positions=positions, widths=0.16, patch_artist=True,
                    showfliers=False, zorder=5,
                    medianprops=dict(color="white", linewidth=2.0),
                    whiskerprops=dict(color="#333333", linewidth=1.2),
                    capprops=dict(color="#333333", linewidth=1.2))
    for patch, regime in zip(bp["boxes"], REGIME_ORDER):
        patch.set_facecolor(REGIME_COLOURS[regime])
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.95)

    ymax = max(np.nanmax(d) for d in data if len(d))
    for pos, regime, d in zip(positions, REGIME_ORDER, data):
        ax.text(pos, ymax * 1.06, f"n = {len(d):,}", ha="center", va="bottom",
                fontsize=10, color=INK)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{r.capitalize()}\n{REGIME_SUBTITLES[r]}" for r in REGIME_ORDER],
        fontsize=10.5)
    ax.set_ylabel("ΔP(crisis) = P(AR + News) − P(AR-only)")
    if not is_bare():
        ax.set_title("Distribution of ΔP(crisis) across crisis regimes", pad=14)
    style_axes(ax, grid_axis="y")
    ax.margins(x=0.04)
    fig.subplots_adjust(bottom=0.18)

    if not is_bare():
        fig.text(0.5, 0.04, "Positive ΔP means news raised the predicted crisis "
                 "probability. Mass concentrates around transition regimes (onset).",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "deltap_violin_by_regime")


if __name__ == "__main__":
    render()
