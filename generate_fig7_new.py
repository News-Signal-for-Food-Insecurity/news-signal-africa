"""
generate_fig7_new.py
====================
Two figures extending fig6c to full country-level analysis.
Both share the same visual language as fig6c / fig7b (horizontal bars,
despined axes, x-axis grid, countries on y-axis).

  fig7a_time.pdf  — one row per country-fold (18 countries x 7 folds = 126 rows)
    Same 4-panel layout as fig7b, but each row is one fold for one country.
    Country name shown once (centred over its 7 fold rows); fold date as y-tick.
    Panel T1: Crisis prevalence stacked bar per country-fold
    Panel T2: AR-Only vs AR+News PR-AUC dot+arrow per country-fold
    Panel T3: Regime change indicator per country-fold
    Panel T4: Onset+chronic count bar per country-fold

  fig7b_space.pdf — one row per country (18 rows), country-level aggregates
    Panel S1: Prevalence stacked bar  (mirrors fig6c left)
    Panel S2: PR-AUC dot+arrow AR-Only -> AR+News  (mirrors fig6c right)
    Panel S3: Volatility horizontal bars
    Panel S4: Onset+chronic count horizontal bars

All values from fold_predictions.csv. AR+News = prob_combined.
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
CHANGE_COL    = "#E67E22"
STABLE_COL    = "#AAAAAA"

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


# ── Load & compute ─────────────────────────────────────────────────────────────
preds   = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")

preds["country"] = preds["district_id"].apply(
    lambda d: [p.strip() for p in str(d).split(",")][-1]
)

fids       = sorted(preds["fold_id"].unique())
n_folds    = len(fids)
fold_dates = dict(zip(fold_df["fold_id"],
                      pd.to_datetime(fold_df["test_start"]).dt.strftime("%b %Y")))
fold_lbls  = [fold_dates[f] for f in fids]

# ── Build per-country summary (for fig7b) ─────────────────────────────────────
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

# ── Build per-country-fold rows (for fig7a) ───────────────────────────────────
# Order: same country order as cdf, folds within each country chronologically
time_rows = []
for _, crow in cdf.iterrows():
    c   = crow["country"]
    sub = preds[preds.country == c]

    prev_maj = None
    for j, fid in enumerate(fids):
        sf = sub[sub.fold_id == fid]
        yf = sf["target_crisis_binary"].values.astype(int)
        n_tot_f = len(sf)
        prev_f  = float(yf.mean()) if n_tot_f > 0 else float("nan")
        n_pos_f = int(yf.sum())
        n_neg_f = int((1 - yf).sum())
        oc_f    = int(sf[sf.regime.isin(["onset", "chronic"])].shape[0])

        pr_ar_f = pr_full_f = float("nan")
        can_f   = (n_pos_f >= 1 and n_neg_f >= 1)
        if can_f:
            try:
                pr_ar_f   = average_precision_score(yf, sf["prob_ar"].values)
                pr_full_f = average_precision_score(yf, sf["prob_combined"].values)
            except Exception:
                can_f = False

        maj = int(prev_f >= 0.5) if np.isfinite(prev_f) else None
        changed = (prev_maj is not None and maj is not None and maj != prev_maj)
        prev_maj = maj if maj is not None else prev_maj

        time_rows.append({
            "country":    c,
            "short":      crow["short"],
            "region":     crow["region"],
            "fold_id":    fid,
            "fold_lbl":   fold_dates[fid],
            "fold_idx":   j,
            "n_tot":      n_tot_f,
            "n_pos":      n_pos_f,
            "n_neg":      n_neg_f,
            "prev":       prev_f,
            "can_score":  can_f,
            "pr_ar":      pr_ar_f,
            "pr_full":    pr_full_f,
            "onset_chronic": oc_f,
            "regime_changed": changed,
        })

tdf = pd.DataFrame(time_rows).reset_index(drop=True)
n_rows_t = len(tdf)   # 126


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7a — TIME  (126 rows: country × fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7a_time ...")

ROW_H  = 0.28   # inches per row
fig_h  = max(10, n_rows_t * ROW_H + 2.5)
fig_w  = 22

fig, axes = plt.subplots(
    1, 4,
    figsize=(fig_w, fig_h),
    gridspec_kw={"wspace": 0.04, "width_ratios": [1.6, 1.4, 0.7, 0.8]},
    sharey=True,
)

y_all = np.arange(n_rows_t)

# ── Y-axis: fold date labels; country name centred over its 7-row block ───────
ax0 = axes[0]
ax0.set_yticks(y_all)
ax0.set_yticklabels(tdf["fold_lbl"].tolist(), fontsize=7)
ax0.set_ylim(n_rows_t - 0.5, -0.5)

# Country label centred over each block of 7 rows, placed to the left
country_starts = {}
for i, row in tdf.iterrows():
    c = row["country"]
    if c not in country_starts:
        country_starts[c] = []
    country_starts[c].append(i)

for c, idxs in country_starts.items():
    mid = (idxs[0] + idxs[-1]) / 2.0
    ax0.text(-0.18, mid, SHORT.get(c, c),
             transform=ax0.get_yaxis_transform(),
             ha="right", va="center", fontsize=8, fontweight="bold", color="#222222")

# Separator lines between countries
for c, idxs in country_starts.items():
    if idxs[0] > 0:
        for ax in axes:
            ax.axhline(idxs[0] - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45, zorder=4)

# Alternate background shading per country block for readability
for ci, (c, idxs) in enumerate(country_starts.items()):
    if ci % 2 == 1:
        for ax in axes:
            ax.axhspan(idxs[0] - 0.5, idxs[-1] + 0.5,
                       color="#F5F5F5", zorder=0, lw=0)


def _fmt_axes(ax, title, xlabel):
    ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--", zorder=1)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5, labelpad=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── T1: Crisis prevalence stacked bar ────────────────────────────────────────
ax = axes[0]
for i, row in tdf.iterrows():
    if not np.isfinite(row["prev"]) or row["n_tot"] == 0:
        ax.barh(i, 1.0, height=0.72, color=NODATA_COL, zorder=2)
        continue
    fp = row["n_pos"] / row["n_tot"]
    fn = row["n_neg"] / row["n_tot"]
    ax.barh(i, fp, height=0.72, color=CRISIS_COL,    zorder=2)
    ax.barh(i, fn, height=0.72, color=NONCRISIS_COL, zorder=2, left=fp)
    if fp >= 0.12:
        ax.text(fp / 2, i, f"{fp*100:.0f}%", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")
    elif fp > 0:
        ax.text(fp + 0.02, i, f"{fp*100:.0f}%", ha="left", va="center",
                fontsize=6, color=CRISIS_COL)

ax.set_xlim(0, 1)
ax.axvline(0.5, color="#BBBBBB", lw=0.7, ls=":")
_fmt_axes(ax, "T1 — Crisis prevalence per fold", "Fraction of observations")
prev_legend = [mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC>=3)"),
               mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis")]
ax.legend(handles=prev_legend, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0, -0.04), framealpha=0.95, ncol=2)

# ── T2: AR-Only vs AR+News PR-AUC dot+arrow ──────────────────────────────────
ax = axes[1]
for i, row in tdf.iterrows():
    if not row["can_score"] or not (np.isfinite(row["pr_ar"]) and np.isfinite(row["pr_full"])):
        ax.text(0.5, i, "—", ha="center", va="center", fontsize=7, color="#CCCCCC")
        continue
    ax.annotate("", xy=(row["pr_full"], i), xytext=(row["pr_ar"], i),
                arrowprops=dict(arrowstyle="->", color="#888888",
                                lw=1.0, mutation_scale=10))
    ax.scatter(row["pr_ar"],   i, color=MODEL_COLOURS["AR-Only"],
               s=35, zorder=5, edgecolors="white", lw=0.3)
    ax.scatter(row["pr_full"], i, color=MODEL_COLOURS["AR+News"],
               marker="s", s=35, zorder=5, edgecolors="white", lw=0.3)
    delta = row["pr_full"] - row["pr_ar"]
    ax.text(1.03, i, f"{delta:+.2f}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=6,
            color="#27AE60" if delta >= 0 else "#C0392B")

ax.set_xlim(0, 1)
ax.axvline(0.5, color="#BBBBBB", lw=0.7, ls=":")
ax.text(1.03, -1, "delta", transform=ax.get_yaxis_transform(),
        va="center", ha="left", fontsize=7, color="#333333", fontweight="bold")
_fmt_axes(ax, "T2 — AR-Only vs AR+News PR-AUC per fold", "PR-AUC")
leg_elems = [Line2D([0],[0], marker="o", color="w",
                    markerfacecolor=MODEL_COLOURS["AR-Only"], ms=7, label="AR-Only"),
             Line2D([0],[0], marker="s", color="w",
                    markerfacecolor=MODEL_COLOURS["AR+News"], ms=7, label="AR+News")]
ax.legend(handles=leg_elems, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0, -0.04), frameon=True, ncol=2)

# ── T3: Regime change indicator ───────────────────────────────────────────────
ax = axes[2]
for i, row in tdf.iterrows():
    color = CHANGE_COL if row["regime_changed"] else STABLE_COL
    ax.scatter(0.5, i, color=color, s=55, zorder=3, edgecolors="white", lw=0.3)

ax.set_xlim(0, 1)
ax.set_xticks([])
_fmt_axes(ax, "T3 — Regime\nchange", "")
chg_legend = [mpatches.Patch(color=CHANGE_COL, label="Changed"),
              mpatches.Patch(color=STABLE_COL, label="Stable")]
ax.legend(handles=chg_legend, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0, -0.04), framealpha=0.95, ncol=1)

# ── T4: Onset+chronic count bar ───────────────────────────────────────────────
ax = axes[3]
max_oc = max(int(tdf["onset_chronic"].max()), 1)
for i, row in tdf.iterrows():
    ax.barh(i, row["onset_chronic"], height=0.72,
            color="#E67E22", alpha=0.82, zorder=2)
    if row["onset_chronic"] > 0:
        ax.text(row["onset_chronic"] + max_oc * 0.02, i,
                str(int(row["onset_chronic"])),
                va="center", ha="left", fontsize=6, color="#555555")

ax.set_xlim(0, max_oc * 1.2)
_fmt_axes(ax, "T4 — Onset+chronic\nper fold", "Observations")

fig.suptitle(
    "Figure 7a — Country-level model performance per fold (time)",
    fontsize=11, fontweight="bold", y=1.003,
)
fig.subplots_adjust(left=0.12, bottom=0.06)
fig.savefig(FIGURES_DIR / "fig7a_time.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7a_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7b — SPACE  (18 rows, country-level aggregates)
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
for i, row in cdf.iterrows():
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

ax.set_yticks(y_pos)
ax.set_yticklabels(cdf["short"].tolist(), fontsize=9)
ax.set_ylim(n_countries - 0.5, -0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of test-set observations", fontsize=8.5, labelpad=5)
ax.set_title("S1 — Crisis Prevalence", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")

prev_region = None
for i, row in cdf.iterrows():
    if row["region"] != prev_region and i > 0:
        ax.axhline(i - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.45)
    prev_region = row["region"]

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
            color="#27AE60" if delta >= 0 else "#C0392B")

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
    ax.barh(i, row["volatility"], height=0.72, color="#8E44AD", alpha=0.78, zorder=3)
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
    ax.barh(i, row["onset_chronic"], height=0.72, color="#E67E22", alpha=0.82, zorder=3)
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
