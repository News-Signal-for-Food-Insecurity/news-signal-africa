"""
fig_pr_comparison.py — the progressive Precision-Recall build (talk slides 15/22/23).

Three states, emitted as three files so the build can be shown step by step:
  pr_1_news      News-only curve + prevalence baseline
  pr_2_news_ar   News-only + AR
  pr_3_all       News-only + AR + AR+News

Scientific basis (consistent with the study, single source of truth):
  * The PR-AUC printed for each model is the study's reported metric — the mean
    (± SD) of the per-fold average precision — read directly from the canonical
    result files (fold_results.csv / news_only_fold_results.csv). It is NOT
    recomputed here.
  * The plotted curve is the MEAN of the per-fold PR curves (with a ±1 SD band),
    so the curve and the label are the same fold-wise aggregation.
  * The baseline is the crisis prevalence on the test set.
Nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import data_access as da
from viz_theme import (BASELINE_COLOUR, INK_SOFT, MODEL_COLOURS,
                       apply_rcparams, is_bare, save_fig, style_axes)

# Draw spec per model: (model_key, display label, colour, linestyle, linewidth)
_SPEC = {
    "News":      ("news", "News", MODEL_COLOURS["News"], "-", 2.4),
    "AR":        ("ar", "AR", MODEL_COLOURS["AR"], (0, (5, 2)), 2.4),
    "AR + News": ("combined", "AR + News", MODEL_COLOURS["AR + News"], "-", 2.6),
}


def _draw(models: list[str], filename: str, *, show_baseline: bool) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(7.4, 6.6))   # plain white panel — no decorative tint

    handles, labels = [], []
    for name in models:
        key, label, colour, ls, lw = _SPEC[name]
        recall, mean_p, sd_p = da.mean_fold_pr_curve(key)   # geometry
        mean_ap, sd_ap = da.reported_pr_auc(key)            # reported metric (from file)
        ax.fill_between(recall, np.clip(mean_p - sd_p, 0, 1),
                        np.clip(mean_p + sd_p, 0, 1), color=colour, alpha=0.12,
                        linewidth=0, zorder=2)
        line, = ax.plot(recall, mean_p, color=colour, linestyle=ls, lw=lw,
                        solid_capstyle="round", zorder=4)
        handles.append(line)
        labels.append(f"{label} (PR-AUC = {mean_ap:.2f} ± {sd_ap:.2f})")

    if show_baseline:
        base = da.crisis_prevalence()
        bl = ax.axhline(base, color=BASELINE_COLOUR, linestyle=(0, (6, 3)),
                        lw=1.8, zorder=3)
        handles.append(bl)
        labels.append("Baseline (prevalence)")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    if not is_bare():
        ax.set_title("Precision–Recall Comparison", pad=12)
    style_axes(ax, grid_axis="both")
    leg = ax.legend(handles, labels, loc="lower left", fontsize=10.5,
                    borderpad=0.7, handlelength=2.2)
    leg.get_frame().set_linewidth(0.8)

    if not is_bare():
        fig.text(0.5, 0.005,
                 "Mean of the per-fold PR curves (shaded ±1 SD across folds). "
                 "PR-AUC = mean ± SD across the 7 rolling-CV folds (from results files).",
                 ha="center", va="bottom", fontsize=8, color=INK_SOFT,
                 fontstyle="italic")
    save_fig(fig, filename)


def render() -> None:
    print("[fig] Precision-Recall build (3 states)")
    _draw(["News"], "pr_1_news", show_baseline=True)
    _draw(["News", "AR"], "pr_2_news_ar", show_baseline=False)
    _draw(["News", "AR", "AR + News"], "pr_3_all", show_baseline=False)


if __name__ == "__main__":
    render()
