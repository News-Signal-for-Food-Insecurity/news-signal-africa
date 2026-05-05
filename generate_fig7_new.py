"""
generate_fig7_new.py
====================
Two figures extending fig6c to full country-level analysis.
Both share the same 4-panel layout and visual language (horizontal bars /
dot-arrows, countries on y-axis, despined axes with x-axis grid).

  fig7a_time.pdf  — per-fold breakdown (same 4 panels as fig7b, across time)
    Panel T1: Crisis prevalence per fold — mini stacked bars per country per fold
    Panel T2: AR-Only vs AR+News PR-AUC per fold — dot-pairs per fold
    Panel T3: Regime change indicator per fold — coloured dot (change / stable)
    Panel T4: Onset+chronic count per fold — mini bars per country per fold

  fig7b_space.pdf — country-level aggregates (same as fig6c + extras)
    Panel S1: Prevalence stacked bar  (mirrors fig6c left panel)
    Panel S2: PR-AUC dot+arrow AR-Only -> AR+News  (mirrors fig6c right)
    Panel S3: Volatility horizontal bars
    Panel S4: Onset+chronic count horizontal bars

All values from fold_predictions.csv and fold_results.csv. AR+News = prob_combined.
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

MODEL_COLOURS = {"AR-Only": "#1f77b4", "AR+News": "#9467bd"}
CRISIS_COL    = "#C0392B"
NONCRISIS_COL = "#2980B9"
NODATA_COL    = "#CCCCCC"
DELTA_POS     = "#27AE60"
DELTA_NEG     = "#C0392B"

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":    9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

COUNTRY_REGION = {
    "Burkina Faso":   "West Africa",    "Burundi":    "East Africa",
    "Cameroon":       "Central Africa", "Chad":       "Central Africa",
    "Ethiopia":       "East Africa",    "Kenya":      "East Africa",
    "Madagascar":     "East Africa",    "Malawi":     "East Africa",
    "Mali":           "West Africa",    "Mozambique": "East Africa",
    "Niger":          "West Africa",    "Nigeria":    "West Africa",
    "Somalia":        "East Africa",    "South Sudan":"East Africa",
    "Sudan":          "North Africa",
    "The Democratic Republic of the": "Central Africa",
    "Uganda":         "East Africa",    "Zimbabwe":   "East Africa",
}
REGION_ORDER = ["East Africa", "West Africa", "Central Africa", "North Africa", "Southern Africa"]
region_rank  = {r: i for i, r in enumerate(REGION_ORDER)}

SHORT = {
    "The Democratic Republic of the": "DRC",
    "South Sudan":  "S. Sudan",
    "Burkina Faso": "Burkina F.",
}


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _add_region_separators(ax, df):
    prev_region = None
    for i, row in df.iterrows():
        if row["region"] != prev_region and i > 0:
            ax.axhline(i - 0.5, color="#333333", lw=1.2, ls="--", alpha=0.55, zorder=4)
        prev_region = row["region"]


def _common_y(ax, df, show_labels=True):
    n = len(df)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["short"].tolist() if show_labels else [""] * n, fontsize=9)
    ax.set_ylim(n - 0.5, -0.5)


# ── Load & compute ─────────────────────────────────────────────────────────────
preds   = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")

preds["country"] = preds["district_id"].apply(
    lambda d: [p.strip() for p in str(d).split(",")][-1]
)

fids      = sorted(preds["fold_id"].unique())
n_folds   = len(fids)
fold_dates = dict(zip(fold_df["fold_id"],
                      pd.to_datetime(fold_df["test_start"]).dt.strftime("%b %Y")))
fold_lbls = [fold_dates[f] for f in fids]

rows = []
for c in sorted(preds["country"].unique()):
    sub   = preds[preds.country == c]
    y     = sub["target_crisis_binary"].values.astype(int)
    n_pos = int(y.sum()); n_neg = int((1-y).sum()); n_tot = len(sub)
    prev  = n_pos / n_tot if n_tot else 0.0
    can_score = (n_pos >= 2 and n_neg >= 2)
    pr_ar = pr_full = float("nan")
    if can_score:
        try:
            pr_ar   = average_precision_score(y, sub["prob_ar"].values)
            pr_full = average_precision_score(y, sub["prob_combined"].values)
        except Exception:
            can_score = False

    # per-fold stats
    fold_pr_ar = {}; fold_pr_full = {}; fold_prev = {}
    fold_oc    = {}  # onset+chronic count per fold
    fold_regime_majority = {}  # majority regime label per fold (for change detection)

    for fid in fids:
        sf = sub[sub.fold_id == fid]
        yf = sf["target_crisis_binary"].values.astype(int)
        fold_prev[fid] = float(yf.mean()) if len(yf) else float("nan")
        fold_oc[fid]   = int(sf[sf.regime.isin(["onset", "chronic"])].shape[0])
        fold_regime_majority[fid] = int(yf.mean() >= 0.5) if len(yf) else -1
        if yf.sum() >= 1 and (1-yf).sum() >= 1:
            try:
                fold_pr_ar[fid]   = average_precision_score(yf, sf["prob_ar"].values)
                fold_pr_full[fid] = average_precision_score(yf, sf["prob_combined"].values)
            except Exception:
                fold_pr_ar[fid] = fold_pr_full[fid] = float("nan")
        else:
            fold_pr_ar[fid] = fold_pr_full[fid] = float("nan")

    # per-fold regime change flag: did majority flip from previous fold?
    fold_regime_change = {}
    prev_maj = None
    for fid in fids:
        maj = fold_regime_majority[fid]
        if prev_maj is None or maj == -1 or prev_maj == -1:
            fold_regime_change[fid] = False
        else:
            fold_regime_change[fid] = (maj != prev_maj)
        prev_maj = maj if maj != -1 else prev_maj

    oc_total = int(sub[sub.regime.isin(["onset", "chronic"])].shape[0])
    fold_crisis = sub.groupby("fold_id")["target_crisis_binary"].mean()
    regime_seq  = (fold_crisis >= 0.5).astype(int)
    vol = float((regime_seq.diff().abs().dropna() > 0).mean()) if len(regime_seq) > 1 else 0.0

    rows.append({
        "country": c, "short": SHORT.get(c, c),
        "region":  COUNTRY_REGION.get(c, "Other"),
        "n_tot": n_tot, "n_pos": n_pos, "n_neg": n_neg,
        "prev": prev, "can_score": can_score,
        "pr_ar": pr_ar, "pr_full": pr_full,
        "volatility": vol, "onset_chronic": oc_total,
        "fold_pr_ar":          fold_pr_ar,
        "fold_pr_full":        fold_pr_full,
        "fold_prev":           fold_prev,
        "fold_oc":             fold_oc,
        "fold_regime_change":  fold_regime_change,
    })

df = pd.DataFrame(rows)
df["region_rank"] = df["region"].map(region_rank).fillna(99)
df = df.sort_values(["region_rank", "prev"], ascending=[True, False]).reset_index(drop=True)
n_countries = len(df)
y_pos       = np.arange(n_countries)

# Sub-row height for mini-bars within each country row (7 folds)
# Each country occupies 1 unit of y-space; mini-bars share that space
BAR_H    = 0.80 / n_folds   # height of each fold's mini-bar within the country row
FOLD_OFF = np.linspace(-0.38, 0.38, n_folds)  # vertical offsets within country row


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7a — TIME  (same 4-panel layout as fig7b, but per fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7a_time ...")

fig_h = max(8, n_countries * 0.55)
fig, axes = plt.subplots(
    1, 4,
    figsize=(26, fig_h),
    gridspec_kw={"wspace": 0.04, "width_ratios": [1.6, 1.4, 0.9, 0.9]},
    sharey=True,
)

# ── T1: Crisis prevalence per fold — mini stacked bars ───────────────────────
ax = axes[0]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")

for i, row in df.iterrows():
    for j, fid in enumerate(fids):
        fp = row["fold_prev"].get(fid, float("nan"))
        if not np.isfinite(fp):
            continue
        fn  = 1.0 - fp
        yc  = i + FOLD_OFF[j]
        # crisis portion
        ax.barh(yc, fp, height=BAR_H, color=CRISIS_COL,    zorder=3, left=0)
        # non-crisis portion
        ax.barh(yc, fn, height=BAR_H, color=NONCRISIS_COL, zorder=3, left=fp)
        # fold label on the left edge inside the bar
        ax.text(-0.01, yc, fold_lbls[j], ha="right", va="center",
                fontsize=5.5, color="#555555")

_common_y(ax, df, show_labels=True)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of test-set observations", fontsize=8.5, labelpad=5)
ax.set_title("T1 — Crisis prevalence per fold", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df)
prev_legend = [mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC>=3)"),
               mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis")]
ax.legend(handles=prev_legend, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0, -0.08), framealpha=0.95, ncol=2)
_despine(ax)

# ── T2: AR-Only vs AR+News PR-AUC per fold — dot-pairs ───────────────────────
ax = axes[1]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")

for i, row in df.iterrows():
    for j, fid in enumerate(fids):
        par  = row["fold_pr_ar"].get(fid, float("nan"))
        pfull= row["fold_pr_full"].get(fid, float("nan"))
        yc   = i + FOLD_OFF[j]
        if not (np.isfinite(par) and np.isfinite(pfull)):
            ax.text(0.5, yc, "—", ha="center", va="center",
                    fontsize=6, color="#CCCCCC")
            continue
        # arrow AR-Only -> AR+News
        ax.annotate("", xy=(pfull, yc), xytext=(par, yc),
                    arrowprops=dict(arrowstyle="->", color="#888888",
                                   lw=0.9, mutation_scale=8))
        ax.scatter(par,   yc, color=MODEL_COLOURS["AR-Only"],
                   s=28, zorder=5, edgecolors="white", lw=0.3)
        ax.scatter(pfull, yc, color=MODEL_COLOURS["AR+News"],
                   marker="s", s=28, zorder=5, edgecolors="white", lw=0.3)
        # fold label
        ax.text(-0.01, yc, fold_lbls[j], ha="right", va="center",
                fontsize=5.5, color="#555555")

_common_y(ax, df, show_labels=False)
ax.set_xlim(0, 1)
ax.set_xlabel("PR-AUC", fontsize=8.5, labelpad=5)
ax.set_title("T2 — AR-Only vs AR+News PR-AUC per fold", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df)
leg_elems = [Line2D([0],[0], marker="o", color="w",
                    markerfacecolor=MODEL_COLOURS["AR-Only"], ms=7, label="AR-Only"),
             Line2D([0],[0], marker="s", color="w",
                    markerfacecolor=MODEL_COLOURS["AR+News"], ms=7, label="AR+News")]
ax.legend(handles=leg_elems, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0, -0.08), frameon=True, ncol=2)
_despine(ax)

# ── T3: Regime change per fold — coloured dot ────────────────────────────────
CHANGE_COL = "#E67E22"   # orange = regime changed this fold
STABLE_COL = "#7F7F7F"   # grey   = regime stable

ax = axes[2]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")

for i, row in df.iterrows():
    for j, fid in enumerate(fids):
        yc      = i + FOLD_OFF[j]
        changed = row["fold_regime_change"].get(fid, False)
        maj     = row["fold_prev"].get(fid, float("nan"))
        if not np.isfinite(maj):
            continue
        color = CHANGE_COL if changed else STABLE_COL
        ax.scatter(0.5, yc, color=color, s=38, zorder=4,
                   edgecolors="white", lw=0.3)
        ax.text(-0.01, yc, fold_lbls[j], ha="right", va="center",
                fontsize=5.5, color="#555555")

_common_y(ax, df, show_labels=False)
ax.set_xlim(0, 1)
ax.set_xticks([0.5])
ax.set_xticklabels([""], fontsize=7)
ax.set_xlabel("Regime change\nper fold", fontsize=8.5, labelpad=5)
ax.set_title("T3 — Regime change", fontsize=9, fontweight="bold", pad=6)
_add_region_separators(ax, df)
chg_legend = [mpatches.Patch(color=CHANGE_COL, label="Regime changed"),
              mpatches.Patch(color=STABLE_COL, label="Stable")]
ax.legend(handles=chg_legend, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0, -0.08), framealpha=0.95, ncol=1)
_despine(ax)

# ── T4: Onset+chronic count per fold — mini bars ─────────────────────────────
ax = axes[3]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")

max_oc_fold = max(
    max((row["fold_oc"].get(fid, 0) for fid in fids), default=0)
    for _, row in df.iterrows()
)
max_oc_fold = max(max_oc_fold, 1)

for i, row in df.iterrows():
    for j, fid in enumerate(fids):
        oc = row["fold_oc"].get(fid, 0)
        yc = i + FOLD_OFF[j]
        ax.barh(yc, oc, height=BAR_H, color="#E67E22", alpha=0.82, zorder=3)
        ax.text(-0.05, yc, fold_lbls[j], ha="right", va="center",
                fontsize=5.5, color="#555555")

_common_y(ax, df, show_labels=False)
ax.set_xlim(0, max_oc_fold * 1.18)
ax.set_xlabel("Onset + chronic\nobservations per fold", fontsize=8.5, labelpad=5)
ax.set_title("T4 — Crisis frequency per fold", fontsize=9, fontweight="bold", pad=6)
_add_region_separators(ax, df)
_despine(ax)

fig.subplots_adjust(bottom=0.12, left=0.07)
fig.suptitle(
    "Figure 7a — Country-level model performance across 7 folds (time)",
    fontsize=11, fontweight="bold", y=1.003,
)
fig.savefig(FIGURES_DIR / "fig7a_time.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7a_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7b — SPACE  (country-level aggregates, 4 panels)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7b_space ...")

fig_h_sp = max(7, n_countries * 0.44)
fig, axes = plt.subplots(
    1, 4,
    figsize=(22, fig_h_sp),
    gridspec_kw={"wspace": 0.04, "width_ratios": [1.5, 1.3, 0.85, 0.85]},
    sharey=True,
)

# ── S1: Prevalence stacked bar ────────────────────────────────────────────────
ax = axes[0]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
for i, row in df.iterrows():
    tot = row["n_tot"]
    if tot == 0:
        ax.barh(i, 1.0, color=NODATA_COL, height=0.72)
        continue
    fp = row["n_pos"] / tot
    fn = row["n_neg"] / tot
    ax.barh(i, fp, color=CRISIS_COL,    height=0.72, zorder=3)
    ax.barh(i, fn, left=fp, color=NONCRISIS_COL, height=0.72, zorder=3)
    pct = f"{fp*100:.0f}%"
    if fp >= 0.12:
        ax.text(fp / 2, i, pct, ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")
    else:
        ax.text(fp + 0.02, i, pct, ha="left", va="center", fontsize=7.5,
                color=CRISIS_COL, fontweight="bold")
    ax.text(1.03, i, f"n={tot}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=6.5, color="#555555")

_common_y(ax, df, show_labels=True)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of test-set observations", fontsize=8.5, labelpad=5)
ax.set_title("S1 — Crisis Prevalence", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df)
prev_legend = [mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC>=3)"),
               mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis")]
ax.legend(handles=prev_legend, fontsize=7.5, loc="upper right", framealpha=0.95)
_despine(ax)

# ── S2: PR-AUC dot+arrow AR-Only -> AR+News ───────────────────────────────────
ax = axes[1]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
for i, row in df.iterrows():
    if not row["can_score"]:
        ax.text(0.5, i, "insufficient class mix",
                ha="center", va="center", fontsize=7,
                color="#AAAAAA", style="italic")
        continue
    ax.annotate("", xy=(row["pr_full"], i), xytext=(row["pr_ar"], i),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, mutation_scale=14))
    ax.scatter(row["pr_ar"],   i, color=MODEL_COLOURS["AR-Only"],
               s=80, zorder=5, edgecolors="white", lw=0.5)
    ax.scatter(row["pr_full"], i, marker="s", color=MODEL_COLOURS["AR+News"],
               s=80, zorder=5, edgecolors="white", lw=0.5)
    delta = row["pr_full"] - row["pr_ar"]
    ax.text(1.03, i, f"{delta:+.2f}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=7,
            color="#27AE60" if delta >= 0 else "#C0392B")

ax.set_xlim(0, 1)
ax.set_xlabel("PR-AUC", fontsize=8.5, labelpad=5)
ax.set_title("S2 — AR-Only vs AR+News PR-AUC", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df)
leg_elems = [Line2D([0],[0], marker="o", color="w",
                    markerfacecolor=MODEL_COLOURS["AR-Only"], ms=9, label="AR-Only"),
             Line2D([0],[0], marker="s", color="w",
                    markerfacecolor=MODEL_COLOURS["AR+News"], ms=9, label="AR+News")]
ax.legend(handles=leg_elems, fontsize=7.5, loc="upper right", frameon=True)
ax.text(1.03, -0.8, "delta", transform=ax.get_yaxis_transform(),
        va="top", ha="left", fontsize=7, color="#333333", fontweight="bold")
_despine(ax)

# ── S3: Volatility bar ────────────────────────────────────────────────────────
ax = axes[2]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
for i, row in df.iterrows():
    ax.barh(i, row["volatility"], height=0.72, color="#8E44AD", alpha=0.78, zorder=3)
    if row["volatility"] > 0.05:
        ax.text(row["volatility"] + 0.02, i, f"{row['volatility']:.2f}",
                va="center", ha="left", fontsize=6.5, color="#555555")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Fraction of folds\nwith regime change", fontsize=8.5, labelpad=5)
ax.set_title("S3 — Volatility", fontsize=9, fontweight="bold", pad=6)
_add_region_separators(ax, df)
_despine(ax)

# ── S4: Onset + Chronic count bar ─────────────────────────────────────────────
ax = axes[3]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
max_oc = max(int(df["onset_chronic"].max()), 1)
for i, row in df.iterrows():
    ax.barh(i, row["onset_chronic"], height=0.72, color="#E67E22", alpha=0.82, zorder=3)
    if row["onset_chronic"] > 0:
        ax.text(row["onset_chronic"] + max_oc * 0.02, i, str(int(row["onset_chronic"])),
                va="center", ha="left", fontsize=6.5, color="#555555")
ax.set_xlim(0, max_oc * 1.18)
ax.set_xlabel("Onset + chronic\nobservations (all folds)", fontsize=8.5, labelpad=5)
ax.set_title("S4 — Crisis Frequency", fontsize=9, fontweight="bold", pad=6)
_add_region_separators(ax, df)
_despine(ax)

fig.suptitle(
    "Figure 7b — Country-level characteristics and model performance (space)",
    fontsize=11, fontweight="bold", y=1.003,
)
fig.savefig(FIGURES_DIR / "fig7b_space.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7b_space.pdf")
print("Done.")
