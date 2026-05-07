"""
generate_fig7_new.py
====================
Two figures with identical 4-panel layout — one across time, one across space.

Panel order (both figures):
  1. Number of News         — total article count (stacked bar by regime colour)
  2. Observations by Regime — stacked bar: onset / chronic / recovery / stable
  3. Volatility Score       — fraction of fold-pairs where regime switches (0–1)
  4. PR-AUC                 — AR-Only dot vs AR+News square, arrow showing direction

Colour scheme:
  stable   — deep green    #1A7A1A
  recovery — light green   #6CC56C
  chronic  — deep red      #C00000
  onset    — light red     #E87070

Output:
  figures/fig7a_time.pdf  — 7 rows, one per test fold
  figures/fig7b_space.pdf — 18 rows, one per country
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import average_precision_score

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "window_2yr"
FIGURES_DIR = BASE_DIR / "figures"

# ── Colour config ──────────────────────────────────────────────────────────────
REGIME_COLOURS = {
    "stable":   "#1A7A1A",   # deep green
    "recovery": "#6CC56C",   # light green
    "chronic":  "#C00000",   # deep red
    "onset":    "#E87070",   # light red
}
REGIME_ORDER = ["onset", "chronic", "recovery", "stable"]   # stacking order

MODEL_COLOURS = {"AR-Only": "#1f77b4", "AR+News": "#9467bd"}
DELTA_POS     = "#27AE60"
DELTA_NEG     = "#E74C3C"
NEWS_COL      = "#4E79A7"    # neutral blue for article count bars

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":    9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── Country / region mapping ───────────────────────────────────────────────────
COUNTRY_REGION = {
    "Burkina Faso":                      "West Africa",
    "Burundi":                           "East Africa",
    "Cameroon":                          "Central Africa",
    "Chad":                              "Central Africa",
    "Democratic Republic of the Congo":  "Central Africa",
    "Ethiopia":                          "East Africa",
    "Kenya":                             "East Africa",
    "Madagascar":                        "Southern Africa",
    "Malawi":                            "Southern Africa",
    "Mali":                              "West Africa",
    "Mozambique":                        "Southern Africa",
    "Niger":                             "West Africa",
    "Nigeria":                           "West Africa",
    "Somalia":                           "East Africa",
    "South Sudan":                       "East Africa",
    "Sudan":                             "North Africa",
    "Uganda":                            "East Africa",
    "Zimbabwe":                          "Southern Africa",
}
REGION_ORDER = ["East Africa", "West Africa", "Central Africa", "North Africa", "Southern Africa"]
region_rank  = {r: i for i, r in enumerate(REGION_ORDER)}

SHORT = {
    "Democratic Republic of the Congo": "DRC",
    "South Sudan":  "S. Sudan",
    "Burkina Faso": "Burkina F.",
}


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── Load predictions & fold results ───────────────────────────────────────────
preds   = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")
ml      = pd.read_parquet(BASE_DIR / "DATA" / "modelling" / "monthly_gdelt_features.parquet")
ml["month"] = pd.to_datetime(ml["month"])

preds["country"] = preds["district_id"].apply(
    lambda d: [p.strip() for p in str(d).split(",")][-1]
)

fids       = sorted(preds["fold_id"].unique())
n_folds    = len(fids)
fold_dates = dict(zip(fold_df["fold_id"],
                      pd.to_datetime(fold_df["test_start"]).dt.strftime("%b %Y")))
fold_starts = dict(zip(fold_df["fold_id"],
                       pd.to_datetime(fold_df["test_start"])))


def _articles_for_fold(fid):
    """Total article count for districts in this fold, across the 4-month IPC period."""
    sf    = preds[preds.fold_id == fid]
    start = fold_starts[fid]
    end   = start + pd.DateOffset(months=4)
    sub   = ml[
        ml["district_id"].isin(sf["district_id"]) &
        (ml["month"] >= start) & (ml["month"] < end)
    ]
    return int(sub["article_count"].sum())


def _articles_for_country(country):
    """Total article count across all months for districts belonging to this country."""
    dists = preds[preds.country == country]["district_id"].unique()
    return int(ml[ml["district_id"].isin(dists)]["article_count"].sum())


# ── Build per-fold summary rows ────────────────────────────────────────────────
fold_rows = []
for fid in fids:
    sf    = preds[preds.fold_id == fid]
    y     = sf["target_crisis_binary"].values.astype(int)
    n_tot = len(sf)
    n_pos = int(y.sum())
    n_neg = n_tot - n_pos

    pr_ar = pr_full = float("nan")
    if n_pos >= 2 and n_neg >= 2:
        try:
            pr_ar   = average_precision_score(y, sf["prob_ar"].values)
            pr_full = average_precision_score(y, sf["prob_combined"].values)
        except Exception:
            pass

    rc = sf["regime"].value_counts().to_dict()
    fold_rows.append({
        "fold_id":  fid,
        "fold_lbl": fold_dates[fid],
        "n_tot":    n_tot,
        "articles": _articles_for_fold(fid),
        "pr_ar":    pr_ar,
        "pr_full":  pr_full,
        **{r: rc.get(r, 0) for r in REGIME_ORDER},
    })

fdf = pd.DataFrame(fold_rows)

# Volatility per fold: fraction of district-regime pairs that changed across folds
# For fig7a (time view) we compute cross-district mean volatility for each fold pair.
# Simpler and more meaningful: for each fold, report fraction of districts that
# switched regime vs the previous fold (only meaningful from fold 2 onward).
# For fold 1 there is no prior, so we show the inter-fold switch rate vs fold 2.
district_regime = preds.pivot_table(index="district_id", columns="fold_id",
                                    values="regime", aggfunc="first")

fold_volatility = {}
prev_col = None
for fid in fids:
    if prev_col is None:
        fold_volatility[fid] = float("nan")   # no prior fold
    else:
        both = district_regime[[prev_col, fid]].dropna()
        switched = (both[prev_col] != both[fid]).mean()
        fold_volatility[fid] = float(switched)
    prev_col = fid

fdf["volatility"] = fdf["fold_id"].map(fold_volatility)

# ── Build per-country summary rows ────────────────────────────────────────────
country_rows = []
for c in sorted(preds["country"].unique()):
    sub   = preds[preds.country == c]
    y     = sub["target_crisis_binary"].values.astype(int)
    n_pos = int(y.sum()); n_neg = int((1 - y).sum()); n_tot = len(sub)
    can_score = (n_pos >= 2 and n_neg >= 2)
    pr_ar = pr_full = float("nan")
    if can_score:
        try:
            pr_ar   = average_precision_score(y, sub["prob_ar"].values)
            pr_full = average_precision_score(y, sub["prob_combined"].values)
        except Exception:
            can_score = False

    # Volatility: fraction of consecutive fold-pairs where district switches regime
    fold_crisis = sub.groupby("fold_id")["target_crisis_binary"].mean()
    regime_seq  = (fold_crisis >= 0.5).astype(int)
    vol = float((regime_seq.diff().abs().dropna() > 0).mean()) if len(regime_seq) > 1 else 0.0

    rc = sub["regime"].value_counts().to_dict()
    country_rows.append({
        "country":   c,
        "short":     SHORT.get(c, c),
        "region":    COUNTRY_REGION.get(c, "Other"),
        "n_tot":     n_tot,
        "articles":  _articles_for_country(c),
        "can_score": can_score,
        "pr_ar":     pr_ar,
        "pr_full":   pr_full,
        "volatility": vol,
        **{r: rc.get(r, 0) for r in REGIME_ORDER},
    })

cdf = pd.DataFrame(country_rows)
cdf["region_rank"] = cdf["region"].map(region_rank).fillna(99)
cdf = cdf.sort_values(["region_rank", "pr_ar"], ascending=[True, False]).reset_index(drop=True)
n_countries = len(cdf)


# ── Shared panel drawing helpers ───────────────────────────────────────────────
def _fmt_k(n):
    """Format large integers as e.g. 1.2M or 345K."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(int(n))


def _panel_news(ax, df, y_col, art_col, n_rows, ylabel_col=None, labels=None,
                region_breaks=None):
    """Panel 1: horizontal bar of total article count."""
    max_art = max(int(df[art_col].max()), 1)
    for i, row in df.iterrows():
        if region_breaks and i in region_breaks:
            ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
        val = int(row[art_col])
        ax.barh(i, val, height=0.68, color=NEWS_COL, alpha=0.85, zorder=3)
        ax.text(val + max_art * 0.02, i, _fmt_k(val),
                va="center", ha="left", fontsize=6.8, color="#444444")
    ax.set_xlim(0, max_art * 1.22)
    ax.set_xlabel("Total articles", fontsize=8.5, labelpad=5)
    ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
    _despine(ax)
    if labels is not None:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(n_rows - 0.5, -0.5)


def _panel_regime(ax, df, n_rows, region_breaks=None):
    """Panel 2: stacked bar of observations by regime."""
    max_n = max(int(df[[r for r in REGIME_ORDER]].sum(axis=1).max()), 1)
    for i, row in df.iterrows():
        if region_breaks and i in region_breaks:
            ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
        left = 0
        for reg in REGIME_ORDER:
            val = int(row[reg])
            if val > 0:
                ax.barh(i, val, height=0.68, color=REGIME_COLOURS[reg],
                        left=left, zorder=3)
                if val >= 12:
                    ax.text(left + val / 2, i, str(val), ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
                left += val
    ax.set_xlim(0, max_n * 1.05)
    ax.set_xlabel("Observations", fontsize=8.5, labelpad=5)
    ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
    _despine(ax)
    ax.set_ylim(n_rows - 0.5, -0.5)
    # legend
    leg = [mpatches.Patch(color=REGIME_COLOURS[r], label=r.capitalize())
           for r in REGIME_ORDER]
    ax.legend(handles=leg, fontsize=7.5, loc="upper right",
              framealpha=0.95, ncol=2, handlelength=1.0, handletextpad=0.4)


def _panel_volatility(ax, df, n_rows, region_breaks=None):
    """Panel 3: horizontal bar of volatility score (0–1)."""
    for i, row in df.iterrows():
        if region_breaks and i in region_breaks:
            ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
        vol = row["volatility"]
        if not np.isfinite(vol):
            ax.text(0.5, i, "n/a", ha="center", va="center",
                    fontsize=7, color="#AAAAAA", style="italic")
            continue
        ax.barh(i, vol, height=0.68, color="#5B9BD5", alpha=0.85, zorder=3)
        if vol > 0.04:
            ax.text(vol + 0.025, i, f"{vol:.2f}",
                    va="center", ha="left", fontsize=6.8, color="#444444")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Volatility score (0–1)", fontsize=8.5, labelpad=5)
    ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
    _despine(ax)
    ax.set_ylim(n_rows - 0.5, -0.5)


def _panel_prauc(ax, df, n_rows, region_breaks=None, delta_fontsize=7.5):
    """Panel 4: PR-AUC dot+arrow AR-Only -> AR+News."""
    for i, row in df.iterrows():
        if region_breaks and i in region_breaks:
            ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
        if not (np.isfinite(row["pr_ar"]) and np.isfinite(row["pr_full"])):
            ax.text(0.75, i, "insufficient data", ha="center", va="center",
                    fontsize=6.5, color="#AAAAAA", style="italic")
            continue
        ax.annotate("", xy=(row["pr_full"], i), xytext=(row["pr_ar"], i),
                    arrowprops=dict(arrowstyle="->", color="#333333",
                                   lw=1.5, mutation_scale=14))
        ax.scatter(row["pr_ar"],   i, color=MODEL_COLOURS["AR-Only"],
                   s=72, zorder=5, edgecolors="white", lw=0.5)
        ax.scatter(row["pr_full"], i, marker="s", color=MODEL_COLOURS["AR+News"],
                   s=72, zorder=5, edgecolors="white", lw=0.5)
        delta = row["pr_full"] - row["pr_ar"]
        ax.text(1.03, i, f"{delta:+.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=delta_fontsize,
                color=DELTA_POS if delta >= 0 else DELTA_NEG)

    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.75, color="#AAAAAA", lw=0.8, ls=":")
    ax.set_xlabel("PR-AUC", fontsize=8.5, labelpad=5)
    ax.text(1.03, -0.75, "delta", transform=ax.get_yaxis_transform(),
            va="top", ha="left", fontsize=delta_fontsize - 0.5,
            color="#333333", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
    _despine(ax)
    ax.set_ylim(n_rows - 0.5, -0.5)
    leg = [Line2D([0],[0], marker="o", color="w",
                  markerfacecolor=MODEL_COLOURS["AR-Only"], ms=8, label="AR-Only"),
           Line2D([0],[0], marker="s", color="w",
                  markerfacecolor=MODEL_COLOURS["AR+News"], ms=8, label="AR+News")]
    ax.legend(handles=leg, fontsize=7.5, loc="upper left", frameon=True,
              handlelength=1.0, handletextpad=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7a — TIME  (7 rows, one per test fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7a_time ...")

fig, axes = plt.subplots(
    1, 4,
    figsize=(20, 5),
    gridspec_kw={"wspace": 0.08, "width_ratios": [1.2, 1.3, 0.85, 1.2]},
    sharey=True,
)

labels_a = fdf["fold_lbl"].tolist()

_panel_news(axes[0], fdf, "fold_id", "articles", n_folds, labels=labels_a)
_panel_regime(axes[1], fdf, n_folds)
_panel_volatility(axes[2], fdf, n_folds)
_panel_prauc(axes[3], fdf, n_folds)

fig.suptitle(
    "Figure 7a — News volume, regime composition, volatility and PR-AUC across 7 test periods",
    fontsize=10.5, fontweight="bold", y=1.03,
)
fig.savefig(FIGURES_DIR / "fig7a_time.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7a_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7b — SPACE  (18 rows, one per country)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7b_space ...")

# Identify region break rows (first row of each new region after the first)
region_breaks_b = set()
prev_r = None
for i, row in cdf.iterrows():
    r = row["region"]
    if r != prev_r and i > 0:
        region_breaks_b.add(i)
    prev_r = r

fig_h_sp = max(8, n_countries * 0.50)
fig, axes = plt.subplots(
    1, 4,
    figsize=(22, fig_h_sp),
    gridspec_kw={"wspace": 0.06, "width_ratios": [1.2, 1.3, 0.85, 1.2]},
    sharey=True,
)

labels_b = cdf["short"].tolist()

_panel_news(axes[0], cdf, "country", "articles", n_countries,
            labels=labels_b, region_breaks=region_breaks_b)
_panel_regime(axes[1], cdf, n_countries, region_breaks=region_breaks_b)
_panel_volatility(axes[2], cdf, n_countries, region_breaks=region_breaks_b)
_panel_prauc(axes[3], cdf, n_countries, region_breaks=region_breaks_b, delta_fontsize=7)

fig.suptitle(
    "Figure 7b — News volume, regime composition, volatility and PR-AUC by country",
    fontsize=10.5, fontweight="bold", y=1.003,
)
fig.savefig(FIGURES_DIR / "fig7b_space.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7b_space.pdf")
print("Done.")
