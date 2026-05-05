"""
generate_fig7_new.py
====================
Two figures extending fig6c to full country-level analysis.
Both share the same 4-panel visual language as fig6c / fig7b.

  fig7a_time.pdf  — 7 rows, one per test fold (aggregated across all countries)
    Same 4-panel layout as fig7b, but the unit is the test period, not country.
    Panel T1: Crisis prevalence stacked bar per fold
    Panel T2: AR-Only vs AR+News PR-AUC dot+arrow per fold
    Panel T3: Regime breakdown bar per fold (onset / chronic / recovery / stable)
    Panel T4: Onset+chronic count bar per fold

  fig7b_space.pdf — 18 rows, one per country (aggregated across all folds)
    Panel S1: Prevalence stacked bar
    Panel S2: PR-AUC dot+arrow AR-Only -> AR+News
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
DELTA_NEG     = "#E74C3C"

# Matches REGIME_COLOURS in 06_paper_figures.py exactly
REGIME_COLOURS = {
    "onset":    "#D62728",
    "chronic":  "#FF7F0E",
    "recovery": "#2CA02C",
    "stable":   "#7F7F7F",
}

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


# ── Load data ──────────────────────────────────────────────────────────────────
preds   = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")

preds["country"] = preds["district_id"].apply(
    lambda d: [p.strip() for p in str(d).split(",")][-1]
)

fids       = sorted(preds["fold_id"].unique())
n_folds    = len(fids)
fold_dates = dict(zip(fold_df["fold_id"],
                      pd.to_datetime(fold_df["test_start"]).dt.strftime("%b %Y")))

# ── Build per-fold summary rows (for fig7a) ────────────────────────────────────
fold_rows = []
for fid in fids:
    sf    = preds[preds.fold_id == fid]
    y     = sf["target_crisis_binary"].values.astype(int)
    n_tot = len(sf)
    n_pos = int(y.sum())
    n_neg = n_tot - n_pos
    prev  = n_pos / n_tot if n_tot else 0.0

    pr_ar = pr_full = float("nan")
    if n_pos >= 2 and n_neg >= 2:
        try:
            pr_ar   = average_precision_score(y, sf["prob_ar"].values)
            pr_full = average_precision_score(y, sf["prob_combined"].values)
        except Exception:
            pass

    regime_counts = sf["regime"].value_counts().to_dict()

    fold_rows.append({
        "fold_id":   fid,
        "fold_lbl":  fold_dates[fid],
        "n_tot":     n_tot,
        "n_pos":     n_pos,
        "n_neg":     n_neg,
        "prev":      prev,
        "pr_ar":     pr_ar,
        "pr_full":   pr_full,
        "onset":     regime_counts.get("onset",    0),
        "chronic":   regime_counts.get("chronic",  0),
        "recovery":  regime_counts.get("recovery", 0),
        "stable":    regime_counts.get("stable",   0),
    })

fdf = pd.DataFrame(fold_rows)   # 7 rows

# ── Build per-country summary rows (for fig7b) ────────────────────────────────
country_rows = []
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

    fold_crisis = sub.groupby("fold_id")["target_crisis_binary"].mean()
    regime_seq  = (fold_crisis >= 0.5).astype(int)
    vol = float((regime_seq.diff().abs().dropna() > 0).mean()) if len(regime_seq) > 1 else 0.0
    oc  = int(sub[sub.regime.isin(["onset", "chronic"])].shape[0])

    country_rows.append({
        "country": c, "short": SHORT.get(c, c),
        "region":  COUNTRY_REGION.get(c, "Other"),
        "n_tot": n_tot, "n_pos": n_pos, "n_neg": n_neg,
        "prev": prev, "can_score": can_score,
        "pr_ar": pr_ar, "pr_full": pr_full,
        "volatility": vol, "onset_chronic": oc,
    })

cdf = pd.DataFrame(country_rows)
cdf["region_rank"] = cdf["region"].map(region_rank).fillna(99)
cdf = cdf.sort_values(["region_rank", "prev"], ascending=[True, False]).reset_index(drop=True)
n_countries = len(cdf)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7a — TIME  (7 rows, one per test fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7a_time ...")

fig, axes = plt.subplots(
    1, 4,
    figsize=(20, 5),
    gridspec_kw={"wspace": 0.06, "width_ratios": [1.4, 1.3, 1.2, 0.9]},
    sharey=True,
)

y_pos = np.arange(n_folds)

def _fmt(ax, title, xlabel):
    ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5, labelpad=5)
    ax.set_ylim(n_folds - 0.5, -0.5)
    _despine(ax)

# ── T1: Crisis prevalence stacked bar ────────────────────────────────────────
ax = axes[0]
for i, row in fdf.iterrows():
    fp = row["n_pos"] / row["n_tot"]
    fn = row["n_neg"] / row["n_tot"]
    ax.barh(i, fp, height=0.65, color=CRISIS_COL,    zorder=3)
    ax.barh(i, fn, height=0.65, color=NONCRISIS_COL, zorder=3, left=fp)
    pct = f"{fp*100:.0f}%"
    if fp >= 0.12:
        ax.text(fp / 2, i, pct, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    else:
        ax.text(fp + 0.02, i, pct, ha="left", va="center",
                fontsize=8, color=CRISIS_COL, fontweight="bold")
    ax.text(1.03, i, f"n={row['n_tot']}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=7.5, color="#555555")

ax.set_yticks(y_pos)
ax.set_yticklabels(fdf["fold_lbl"].tolist(), fontsize=9)
ax.set_xlim(0, 1)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_fmt(ax, "T1 — Crisis prevalence per test period", "Fraction of observations")
prev_legend = [mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC>=3)"),
               mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis")]
ax.legend(handles=prev_legend, fontsize=8, loc="upper right", framealpha=0.95)

# ── T2: AR-Only vs AR+News PR-AUC dot+arrow ──────────────────────────────────
ax = axes[1]
for i, row in fdf.iterrows():
    if not (np.isfinite(row["pr_ar"]) and np.isfinite(row["pr_full"])):
        continue
    ax.annotate("", xy=(row["pr_full"], i), xytext=(row["pr_ar"], i),
                arrowprops=dict(arrowstyle="->", color="#333333",
                                lw=1.5, mutation_scale=14))
    ax.scatter(row["pr_ar"],   i, color=MODEL_COLOURS["AR-Only"],
               s=80, zorder=5, edgecolors="white", lw=0.5)
    ax.scatter(row["pr_full"], i, marker="s", color=MODEL_COLOURS["AR+News"],
               s=80, zorder=5, edgecolors="white", lw=0.5)
    delta = row["pr_full"] - row["pr_ar"]
    ax.text(1.03, i, f"{delta:+.3f}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=7.5,
            color=DELTA_POS if delta >= 0 else DELTA_NEG)

ax.set_xlim(0.5, 1.0)
ax.axvline(0.75, color="#AAAAAA", lw=0.8, ls=":")
ax.text(1.03, -0.7, "delta", transform=ax.get_yaxis_transform(),
        va="top", ha="left", fontsize=7.5, color="#333333", fontweight="bold")
_fmt(ax, "T2 — AR-Only vs AR+News PR-AUC per test period", "PR-AUC")
leg_elems = [Line2D([0],[0], marker="o", color="w",
                    markerfacecolor=MODEL_COLOURS["AR-Only"], ms=9, label="AR-Only"),
             Line2D([0],[0], marker="s", color="w",
                    markerfacecolor=MODEL_COLOURS["AR+News"], ms=9, label="AR+News")]
ax.legend(handles=leg_elems, fontsize=8, loc="upper left", frameon=True)

# ── T3: Regime breakdown stacked bar ─────────────────────────────────────────
ax = axes[2]
for i, row in fdf.iterrows():
    left = 0
    for reg in ["onset", "chronic", "recovery", "stable"]:
        val = row[reg]
        if val > 0:
            ax.barh(i, val, height=0.65, color=REGIME_COLOURS[reg],
                    left=left, zorder=3)
            if val >= 15:
                ax.text(left + val / 2, i, str(val), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
            left += val

ax.set_xlim(0, fdf[["onset","chronic","recovery","stable"]].sum(axis=1).max() * 1.05)
_fmt(ax, "T3 — Regime breakdown per test period", "Number of observations")
reg_legend = [mpatches.Patch(color=REGIME_COLOURS[r], label=r.capitalize())
              for r in ["onset", "chronic", "recovery", "stable"]]
ax.legend(handles=reg_legend, fontsize=8, loc="upper right", framealpha=0.95, ncol=2)

# ── T4: Onset+chronic count bar ───────────────────────────────────────────────
ax = axes[3]
max_oc = max(int((fdf["onset"] + fdf["chronic"]).max()), 1)
for i, row in fdf.iterrows():
    oc = row["onset"] + row["chronic"]
    ax.barh(i, oc, height=0.65, color=REGIME_COLOURS["chronic"], alpha=0.85, zorder=3)
    ax.text(oc + max_oc * 0.02, i, str(int(oc)),
            va="center", ha="left", fontsize=8, color="#555555")

ax.set_xlim(0, max_oc * 1.25)
_fmt(ax, "T4 — Onset+chronic\nper test period", "Observations")

fig.suptitle(
    "Figure 7a — Model performance and crisis characteristics across 7 test periods",
    fontsize=11, fontweight="bold", y=1.03,
)
fig.savefig(FIGURES_DIR / "fig7a_time.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7a_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7b — SPACE  (18 rows, one per country)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7b_space ...")

fig_h_sp = max(7, n_countries * 0.44)
fig, axes = plt.subplots(
    1, 4,
    figsize=(22, fig_h_sp),
    gridspec_kw={"wspace": 0.04, "width_ratios": [1.5, 1.3, 0.85, 0.85]},
    sharey=True,
)

y_pos = np.arange(n_countries)

# ── S1: Prevalence stacked bar ────────────────────────────────────────────────
ax = axes[0]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
prev_region = None
for i, row in cdf.iterrows():
    if row["region"] != prev_region and i > 0:
        ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
    prev_region = row["region"]
    tot = row["n_tot"]
    fp = row["n_pos"] / tot; fn = row["n_neg"] / tot
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

ax.set_yticks(y_pos)
ax.set_yticklabels(cdf["short"].tolist(), fontsize=9)
ax.set_ylim(n_countries - 0.5, -0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of test-set observations", fontsize=8.5, labelpad=5)
ax.set_title("S1 — Crisis Prevalence", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
prev_legend = [mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC>=3)"),
               mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis")]
ax.legend(handles=prev_legend, fontsize=7.5, loc="upper right", framealpha=0.95)
_despine(ax)

# ── S2: PR-AUC dot+arrow ──────────────────────────────────────────────────────
ax = axes[1]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
prev_region = None
for i, row in cdf.iterrows():
    if row["region"] != prev_region and i > 0:
        ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
    prev_region = row["region"]
    if not row["can_score"]:
        ax.text(0.5, i, "insufficient class mix", ha="center", va="center",
                fontsize=7, color="#AAAAAA", style="italic")
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
            color=DELTA_POS if delta >= 0 else DELTA_NEG)

ax.set_xlim(0, 1)
ax.set_xlabel("PR-AUC", fontsize=8.5, labelpad=5)
ax.set_title("S2 — AR-Only vs AR+News PR-AUC", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
ax.text(1.03, -0.8, "delta", transform=ax.get_yaxis_transform(),
        va="top", ha="left", fontsize=7, color="#333333", fontweight="bold")
leg_elems = [Line2D([0],[0], marker="o", color="w",
                    markerfacecolor=MODEL_COLOURS["AR-Only"], ms=9, label="AR-Only"),
             Line2D([0],[0], marker="s", color="w",
                    markerfacecolor=MODEL_COLOURS["AR+News"], ms=9, label="AR+News")]
ax.legend(handles=leg_elems, fontsize=7.5, loc="upper right", frameon=True)
_despine(ax)

# ── S3: Volatility bar ────────────────────────────────────────────────────────
ax = axes[2]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
prev_region = None
for i, row in cdf.iterrows():
    if row["region"] != prev_region and i > 0:
        ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
    prev_region = row["region"]
    ax.barh(i, row["volatility"], height=0.72, color="#5B9BD5", alpha=0.85, zorder=3)
    if row["volatility"] > 0.05:
        ax.text(row["volatility"] + 0.02, i, f"{row['volatility']:.2f}",
                va="center", ha="left", fontsize=6.5, color="#555555")

ax.set_xlim(0, 1.05)
ax.set_xlabel("Fraction of folds\nwith regime change", fontsize=8.5, labelpad=5)
ax.set_title("S3 — Volatility", fontsize=9, fontweight="bold", pad=6)
_despine(ax)

# ── S4: Onset+chronic count bar ───────────────────────────────────────────────
ax = axes[3]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
max_oc = max(int(cdf["onset_chronic"].max()), 1)
prev_region = None
for i, row in cdf.iterrows():
    if row["region"] != prev_region and i > 0:
        ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
    prev_region = row["region"]
    ax.barh(i, row["onset_chronic"], height=0.72, color=REGIME_COLOURS["chronic"], alpha=0.85, zorder=3)
    if row["onset_chronic"] > 0:
        ax.text(row["onset_chronic"] + max_oc * 0.02, i, str(int(row["onset_chronic"])),
                va="center", ha="left", fontsize=6.5, color="#555555")

ax.set_xlim(0, max_oc * 1.18)
ax.set_xlabel("Onset + chronic\nobservations (all folds)", fontsize=8.5, labelpad=5)
ax.set_title("S4 — Crisis Frequency", fontsize=9, fontweight="bold", pad=6)
_despine(ax)

fig.suptitle(
    "Figure 7b — Country-level characteristics and model performance (space)",
    fontsize=11, fontweight="bold", y=1.003,
)
fig.savefig(FIGURES_DIR / "fig7b_space.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7b_space.pdf")
print("Done.")
