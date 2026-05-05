"""
generate_fig7_new.py
====================
Produces two figures extending fig6c to full country-level analysis:

  fig7_time.pdf  — 4-panel TIME figure (country x fold)
    Panel T1: Crisis prevalence per country per fold (heatmap)
    Panel T2: AR+News PR-AUC per country per fold (heatmap)
    Panel T3: Delta PR-AUC (AR+News - AR-Only) per country per fold (heatmap)
    Panel T4: Fold-level PR-AUC trajectory lines per country

  fig7_space.pdf — 4-panel SPACE figure (country-level aggregates)
    Panel S1: Prevalence stacked bar (same as fig6c left panel)
    Panel S2: PR-AUC dot+arrow AR-Only -> AR+News (same as fig6c right panel)
    Panel S3: Volatility bar per country
    Panel S4: Onset+chronic count bar per country

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
REGION_ORDER = ["East Africa","West Africa","Central Africa","North Africa","Southern Africa"]
region_rank  = {r: i for i, r in enumerate(REGION_ORDER)}

SHORT = {"The Democratic Republic of the": "DRC", "South Sudan": "S. Sudan",
         "Burkina Faso": "Burkina F."}


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


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
    oc  = int(sub[sub.regime.isin(["onset","chronic"])].shape[0])

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
df = df.sort_values(["region_rank","prev"], ascending=[True,False]).reset_index(drop=True)
n_countries = len(df)
country_labels = df["short"].tolist()


# ── Region separators helper ───────────────────────────────────────────────────
def _add_region_separators(ax, df, orientation="h"):
    prev_region = None
    for i, row in df.iterrows():
        if row["region"] != prev_region and i > 0:
            if orientation == "h":
                ax.axhline(i - 0.5, color="#333333", lw=1.0, ls="--", alpha=0.5)
            else:
                ax.axvline(i - 0.5, color="#333333", lw=1.0, ls="--", alpha=0.5)
        prev_region = row["region"]


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — TIME  (4 panels vertically stacked)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7_time ...")

n_folds    = len(fids)
fold_lbls  = [fold_dates[f] for f in fids]

# Build matrices: rows=countries, cols=folds
mat_prev     = np.full((n_countries, n_folds), np.nan)
mat_pr_full  = np.full((n_countries, n_folds), np.nan)
mat_delta    = np.full((n_countries, n_folds), np.nan)

for i, row in df.iterrows():
    for j, fid in enumerate(fids):
        mat_prev[i, j]    = row["fold_prev"].get(fid, float("nan"))
        mat_pr_full[i, j] = row["fold_pr_full"].get(fid, float("nan"))
        far = row["fold_pr_ar"].get(fid, float("nan"))
        ffu = row["fold_pr_full"].get(fid, float("nan"))
        mat_delta[i, j]   = ffu - far if (np.isfinite(far) and np.isfinite(ffu)) else float("nan")

# Colormaps
cmap_prev  = plt.cm.YlOrRd
cmap_prauc = plt.cm.Blues
# Diverging for delta: red=negative, white=zero, green=positive
cmap_delta = LinearSegmentedColormap.from_list(
    "rg", ["#C0392B","#FFFFFF","#27AE60"], N=256)

fig, axes = plt.subplots(4, 1, figsize=(12, n_countries * 0.38 + 3),
                         gridspec_kw={"hspace": 0.55})

# ── T1: Prevalence heatmap ────────────────────────────────────────────────────
ax = axes[0]
im1 = ax.imshow(mat_prev, aspect="auto", cmap=cmap_prev, vmin=0, vmax=1,
                interpolation="nearest")
ax.set_xticks(range(n_folds)); ax.set_xticklabels(fold_lbls, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(n_countries)); ax.set_yticklabels(country_labels, fontsize=8)
ax.set_title("T1 — Crisis Prevalence per Country per Fold", fontsize=9, fontweight="bold", pad=4)
for i in range(n_countries - 1):
    ax.axhline(i + 0.5, color="white", lw=0.4, alpha=0.6)
_add_region_separators(ax, df, "h")
cb1 = fig.colorbar(im1, ax=ax, pad=0.01, shrink=0.85)
cb1.set_label("Prevalence", fontsize=7); cb1.ax.tick_params(labelsize=7)
ax.spines[:].set_visible(False)

# ── T2: AR+News PR-AUC heatmap ───────────────────────────────────────────────
ax = axes[1]
masked2 = np.ma.masked_invalid(mat_pr_full)
im2 = ax.imshow(masked2, aspect="auto", cmap=cmap_prauc, vmin=0, vmax=1,
                interpolation="nearest")
ax.set_xticks(range(n_folds)); ax.set_xticklabels(fold_lbls, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(n_countries)); ax.set_yticklabels(country_labels, fontsize=8)
ax.set_title("T2 — AR+News PR-AUC per Country per Fold  (grey = not computable)", fontsize=9, fontweight="bold", pad=4)
for i in range(n_countries - 1):
    ax.axhline(i + 0.5, color="white", lw=0.4, alpha=0.6)
_add_region_separators(ax, df, "h")
cb2 = fig.colorbar(im2, ax=ax, pad=0.01, shrink=0.85)
cb2.set_label("PR-AUC", fontsize=7); cb2.ax.tick_params(labelsize=7)
ax.spines[:].set_visible(False)

# ── T3: Delta PR-AUC heatmap ─────────────────────────────────────────────────
ax = axes[2]
masked3 = np.ma.masked_invalid(mat_delta)
abs_max = np.nanmax(np.abs(mat_delta[np.isfinite(mat_delta)])) if np.isfinite(mat_delta).any() else 0.3
im3 = ax.imshow(masked3, aspect="auto", cmap=cmap_delta,
                vmin=-abs_max, vmax=abs_max, interpolation="nearest")
ax.set_xticks(range(n_folds)); ax.set_xticklabels(fold_lbls, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(n_countries)); ax.set_yticklabels(country_labels, fontsize=8)
ax.set_title("T3 — Delta PR-AUC (AR+News minus AR-Only) per Country per Fold  (green = news helps)", fontsize=9, fontweight="bold", pad=4)
for i in range(n_countries - 1):
    ax.axhline(i + 0.5, color="white", lw=0.4, alpha=0.6)
_add_region_separators(ax, df, "h")
cb3 = fig.colorbar(im3, ax=ax, pad=0.01, shrink=0.85)
cb3.set_label("Delta PR-AUC", fontsize=7); cb3.ax.tick_params(labelsize=7)
ax.spines[:].set_visible(False)

# ── T4: Fold trajectory lines ─────────────────────────────────────────────────
ax = axes[3]
ax.grid(True, axis="y", alpha=0.15, lw=0.5)
x_pos = np.arange(n_folds)
cmap_ctry = plt.cm.tab20
for i, row in df.iterrows():
    vals_ar   = [row["fold_pr_ar"].get(f, float("nan"))   for f in fids]
    vals_full = [row["fold_pr_full"].get(f, float("nan")) for f in fids]
    color = cmap_ctry(i / n_countries)
    # plot AR+News as solid, AR-Only as faint dashed
    finite = [j for j,v in enumerate(vals_full) if np.isfinite(v)]
    if len(finite) >= 2:
        xf = [x_pos[j] for j in finite]
        yf = [vals_full[j] for j in finite]
        ya = [vals_ar[j]   for j in finite]
        ax.plot(xf, yf, "-o", color=color, lw=1.4, ms=4, alpha=0.85,
                label=row["short"] if len(finite) >= 2 else None)
        ax.plot(xf, ya, "--", color=color, lw=0.7, ms=0, alpha=0.40)

ax.set_xticks(x_pos); ax.set_xticklabels(fold_lbls, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("PR-AUC", fontsize=8); ax.set_ylim(0, 1.05)
ax.set_title("T4 — PR-AUC Trajectory per Country  (solid = AR+News, dashed = AR-Only)", fontsize=9, fontweight="bold", pad=4)
ax.legend(fontsize=6.5, ncol=3, loc="lower left", framealpha=0.85,
          bbox_to_anchor=(0, -0.38), borderaxespad=0)
_despine(ax)

fig.suptitle("Figure 7 (Time) — Country-Level Model Performance Across 7 Folds",
             fontsize=11, fontweight="bold", y=1.002)
fig.savefig(FIGURES_DIR / "fig7_time.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — SPACE  (4 panels side by side, horizontal bars, country on y-axis)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building fig7_space ...")

fig, axes = plt.subplots(1, 4, figsize=(20, max(7, n_countries * 0.42)),
                         gridspec_kw={"wspace": 0.08,
                                      "width_ratios": [1.4, 1.2, 0.9, 0.9]})

y_pos = np.arange(n_countries)

# ── S1: Prevalence stacked bar (fig6c left panel) ─────────────────────────────
ax = axes[0]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
for i, row in df.iterrows():
    tot = row["n_tot"]
    if tot == 0:
        ax.barh(i, 1.0, color=NODATA_COL, height=0.65)
        continue
    fp = row["n_pos"] / tot; fn = row["n_neg"] / tot
    ax.barh(i, fp, color=CRISIS_COL,    height=0.65, zorder=3)
    ax.barh(i, fn, left=fp, color=NONCRISIS_COL, height=0.65, zorder=3)
    pct = f"{fp*100:.0f}%"
    if fp >= 0.12:
        ax.text(fp/2, i, pct, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")
    else:
        ax.text(fp + 0.02, i, pct, ha="left", va="center", fontsize=7,
                color=CRISIS_COL, fontweight="bold")
    ax.text(1.03, i, f"n={tot}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=6.5, color="#555555")

ax.set_yticks(y_pos); ax.set_yticklabels(country_labels, fontsize=8.5)
ax.set_xlim(0, 1); ax.set_xlabel("Fraction of test-set observations", fontsize=8, labelpad=4)
ax.set_title("S1 — Crisis Prevalence", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df, "h")
prev_legend = [mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC>=3)"),
               mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis")]
ax.legend(handles=prev_legend, fontsize=7, loc="upper right", framealpha=0.95)
_despine(ax)

# ── S2: PR-AUC dot+arrow AR-Only -> AR+News ───────────────────────────────────
ax = axes[1]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
ax.set_yticks(y_pos); ax.set_yticklabels([""] * n_countries)
for i, row in df.iterrows():
    if not row["can_score"]:
        ax.text(0.5, i, "—  insufficient class mix",
                ha="center", va="center", fontsize=7.5,
                color="#999999", style="italic")
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

ax.set_xlim(0, 1); ax.set_xlabel("PR-AUC", fontsize=8, labelpad=4)
ax.set_title("S2 — AR-Only vs AR+News PR-AUC", fontsize=9, fontweight="bold", pad=6)
ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")
_add_region_separators(ax, df, "h")
leg_elems = [Line2D([0],[0], marker="o", color="w",
                    markerfacecolor=MODEL_COLOURS["AR-Only"], ms=9, label="AR Only"),
             Line2D([0],[0], marker="s", color="w",
                    markerfacecolor=MODEL_COLOURS["AR+News"], ms=9, label="AR + News")]
ax.legend(handles=leg_elems, fontsize=7, loc="upper right", frameon=True)
ax.text(1.03, n_countries - 0.3, "delta", transform=ax.get_yaxis_transform(),
        va="bottom", ha="left", fontsize=7, color="#333333", fontweight="bold")
_despine(ax)

# ── S3: Volatility bar ────────────────────────────────────────────────────────
ax = axes[2]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
ax.set_yticks(y_pos); ax.set_yticklabels([""] * n_countries)
for i, row in df.iterrows():
    ax.barh(i, row["volatility"], height=0.65, color="#8E44AD", alpha=0.75, zorder=3)
    if row["volatility"] > 0.05:
        ax.text(row["volatility"] + 0.01, i, f"{row['volatility']:.2f}",
                va="center", ha="left", fontsize=6.5, color="#555555")
ax.set_xlim(0, 1); ax.set_xlabel("Volatility\n(fraction of folds with regime change)", fontsize=8, labelpad=4)
ax.set_title("S3 — Volatility", fontsize=9, fontweight="bold", pad=6)
_add_region_separators(ax, df, "h")
_despine(ax)

# ── S4: Onset + Chronic count bar ─────────────────────────────────────────────
ax = axes[3]
ax.grid(True, axis="x", alpha=0.18, lw=0.5, ls="--")
ax.set_yticks(y_pos); ax.set_yticklabels([""] * n_countries)
max_oc = df["onset_chronic"].max()
for i, row in df.iterrows():
    ax.barh(i, row["onset_chronic"], height=0.65, color="#E67E22", alpha=0.80, zorder=3)
    if row["onset_chronic"] > 0:
        ax.text(row["onset_chronic"] + max_oc*0.01, i, str(row["onset_chronic"]),
                va="center", ha="left", fontsize=6.5, color="#555555")
ax.set_xlabel("Onset + Chronic\nobservations (all folds)", fontsize=8, labelpad=4)
ax.set_title("S4 — Crisis Frequency", fontsize=9, fontweight="bold", pad=6)
_add_region_separators(ax, df, "h")
_despine(ax)

fig.suptitle("Figure 7 (Space) — Country-Level Characteristics and Model Performance",
             fontsize=11, fontweight="bold", y=1.002)
fig.savefig(FIGURES_DIR / "fig7_space.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved fig7_space.pdf")
print("Done.")
