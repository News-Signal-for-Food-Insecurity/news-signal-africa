"""
generate_fig7_new.py
====================
Produces two figures extending fig6c to full country-level analysis:

  fig7_time.pdf  — 3-panel TIME figure (country x fold heatmaps)
    Panel T1: Crisis prevalence per country per fold
    Panel T2: AR+News PR-AUC per country per fold
    Panel T3: Delta PR-AUC (AR+News - AR-Only) per country per fold

  fig7_space.pdf — 4-panel SPACE figure (country-level aggregates)
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
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import average_precision_score

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "window_2yr"
FIGURES_DIR = BASE_DIR / "figures"

MODEL_COLOURS = {"AR-Only": "#1f77b4", "AR+News": "#9467bd"}
CRISIS_COL    = "#C0392B"
NONCRISIS_COL = "#2980B9"
NODATA_COL    = "#CCCCCC"

COUNTRY_REGION = {
    "Burkina Faso":  "West Africa", "Burundi":    "East Africa",
    "Cameroon":      "Central Africa", "Chad":    "Central Africa",
    "Ethiopia":      "East Africa",  "Kenya":     "East Africa",
    "Madagascar":    "East Africa",  "Malawi":    "East Africa",
    "Mali":          "West Africa",  "Mozambique":"East Africa",
    "Niger":         "West Africa",  "Nigeria":   "West Africa",
    "Somalia":       "East Africa",  "South Sudan":"East Africa",
    "Sudan":         "North Africa", "The Democratic Republic of the": "Central Africa",
    "Uganda":        "East Africa",  "Zimbabwe":  "East Africa",
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


def _add_region_separators(ax, df, orientation="h"):
    prev_region = None
    for i, row in df.iterrows():
        if row["region"] != prev_region and i > 0:
            if orientation == "h":
                ax.axhline(i - 0.5, color="#333333", lw=1.2, ls="--", alpha=0.55, zorder=4)
            else:
                ax.axvline(i - 0.5, color="#333333", lw=1.2, ls="--", alpha=0.55, zorder=4)
        prev_region = row["region"]


# ── Load & compute ─────────────────────────────────────────────────────────────
preds   = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")

preds["country"] = preds["district_id"].apply(
    lambda d: [p.strip() for p in str(d).split(",")][-1]
)
preds["delta_prob"] = preds["prob_combined"] - preds["prob_ar"]

fids       = sorted(preds["fold_id"].unique())
fold_dates = dict(zip(fold_df["fold_id"],
                      pd.to_datetime(fold_df["test_start"]).dt.strftime("%b %Y")))

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

    fold_crisis = sub.groupby("fold_id")["target_crisis_binary"].mean()
    regime_seq  = (fold_crisis >= 0.5).astype(int)
    vol = float((regime_seq.diff().abs().dropna() > 0).mean()) if len(regime_seq) > 1 else 0.0
    oc  = int(sub[sub.regime.isin(["onset", "chronic"])].shape[0])

    fold_pr_ar = {}; fold_pr_full = {}; fold_prev = {}
    for fid in fids:
        sf = sub[sub.fold_id == fid]
        yf = sf["target_crisis_binary"].values.astype(int)
        fold_prev[fid] = float(yf.mean()) if len(yf) else float("nan")
        if yf.sum() >= 1 and (1-yf).sum() >= 1:
            try:
                fold_pr_ar[fid]   = average_precision_score(yf, sf["prob_ar"].values)
                fold_pr_full[fid] = average_precision_score(yf, sf["prob_combined"].values)
            except Exception:
                fold_pr_ar[fid] = fold_pr_full[fid] = float("nan")
        else:
            fold_pr_ar[fid] = fold_pr_full[fid] = float("nan")

    rows.append({
        "country": c, "short": SHORT.get(c, c),
        "region":  COUNTRY_REGION.get(c, "Other"),
        "n_tot": n_tot, "n_pos": n_pos, "n_neg": n_neg,
        "prev": prev, "can_score": can_score,
        "pr_ar": pr_ar, "pr_full": pr_full,
        "volatility": vol, "onset_chronic": oc,
        "fold_pr_ar":   fold_pr_ar,
        "fold_pr_full": fold_pr_full,
        "fold_prev":    fold_prev,
    })

df = pd.DataFrame(rows)
df["region_rank"] = df["region"].map(region_rank).fillna(99)
df = df.sort_values(["region_rank", "prev"], ascending=[True, False]).reset_index(drop=True)
n_countries    = len(df)
country_labels = df["short"].tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 TIME  —  3 heatmap panels stacked vertically
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7_time ...")

n_folds   = len(fids)
fold_lbls = [fold_dates[f] for f in fids]

# Build matrices: rows = countries (top-to-bottom), cols = folds
mat_prev    = np.full((n_countries, n_folds), np.nan)
mat_pr_full = np.full((n_countries, n_folds), np.nan)
mat_delta   = np.full((n_countries, n_folds), np.nan)

for i, row in df.iterrows():
    for j, fid in enumerate(fids):
        mat_prev[i, j]    = row["fold_prev"].get(fid, float("nan"))
        mat_pr_full[i, j] = row["fold_pr_full"].get(fid, float("nan"))
        far = row["fold_pr_ar"].get(fid, float("nan"))
        ffu = row["fold_pr_full"].get(fid, float("nan"))
        mat_delta[i, j]   = ffu - far if (np.isfinite(far) and np.isfinite(ffu)) else float("nan")

cmap_prev  = plt.cm.YlOrRd
cmap_prauc = plt.cm.Blues
cmap_delta = LinearSegmentedColormap.from_list(
    "rg", ["#C0392B", "#FFFFFF", "#27AE60"], N=256)

ROW_H   = 0.42          # inches per country row
PAD_TOP = 1.6           # inches for suptitle + x-labels above top panel
PAD_BOT = 0.5           # inches below bottom panel
N_PANELS = 3
PANEL_H  = n_countries * ROW_H
CB_FRAC  = 0.03         # colorbar fraction of panel width
CB_PAD   = 0.008

fig_h = PAD_TOP + N_PANELS * PANEL_H + (N_PANELS - 1) * 0.7 + PAD_BOT
fig_w = 11

fig, axes = plt.subplots(
    3, 1,
    figsize=(fig_w, fig_h),
    gridspec_kw={"hspace": 0.65, "height_ratios": [1, 1, 1]},
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _heatmap_panel(ax, mat, cmap, vmin, vmax, title, cb_label, nodata_col="#E8E8E8"):
    masked = np.ma.masked_invalid(mat)
    cmap_copy = cmap
    if hasattr(cmap, "copy"):
        cmap_copy = cmap.copy()
    cmap_copy.set_bad(color=nodata_col)
    im = ax.imshow(masked, aspect="auto", cmap=cmap_copy, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    # x-axis: fold dates
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(fold_lbls, fontsize=8, rotation=35, ha="right")
    # y-axis: country labels
    ax.set_yticks(range(n_countries))
    ax.set_yticklabels(country_labels, fontsize=8.5)
    ax.set_ylim(n_countries - 0.5, -0.5)   # top-to-bottom, no drift
    # thin white row separators
    for i in range(n_countries - 1):
        ax.axhline(i + 0.5, color="white", lw=0.5, alpha=0.7, zorder=3)
    # region separators (dashed dark)
    _add_region_separators(ax, df, "h")
    ax.spines[:].set_visible(False)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    cb = fig.colorbar(im, ax=ax, fraction=CB_FRAC, pad=CB_PAD, shrink=0.90)
    cb.set_label(cb_label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    return im


_heatmap_panel(axes[0], mat_prev, cmap_prev, 0, 1,
               "T1 — Crisis prevalence per country per fold",
               "Prevalence")

_heatmap_panel(axes[1], mat_pr_full, cmap_prauc, 0, 1,
               "T2 — AR+News PR-AUC per country per fold  (grey = not computable)",
               "PR-AUC")

abs_max = float(np.nanmax(np.abs(mat_delta[np.isfinite(mat_delta)]))) if np.isfinite(mat_delta).any() else 0.3
_heatmap_panel(axes[2], mat_delta, cmap_delta, -abs_max, abs_max,
               "T3 — Delta PR-AUC (AR+News minus AR-Only)  (green = news helps, red = hurts)",
               "Delta PR-AUC")

fig.suptitle(
    "Figure 7 (Time) — Country-level model performance across 7 folds",
    fontsize=11, fontweight="bold", y=1.003
)
fig.savefig(FIGURES_DIR / "fig7_time.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 SPACE  —  4 panels side by side, shared y-axis
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7_space ...")

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

ax.set_yticks(y_pos)
ax.set_yticklabels(country_labels, fontsize=9)
ax.set_ylim(n_countries - 0.5, -0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of test-set observations", fontsize=8.5, labelpad=5)
ax.set_title("S1 — Crisis Prevalence", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df, "h")
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
_add_region_separators(ax, df, "h")
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
_add_region_separators(ax, df, "h")
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
_add_region_separators(ax, df, "h")
_despine(ax)

fig.suptitle(
    "Figure 7 (Space) — Country-level characteristics and model performance",
    fontsize=11, fontweight="bold", y=1.003
)
fig.savefig(FIGURES_DIR / "fig7_space.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7_space.pdf")
print("Done.")
