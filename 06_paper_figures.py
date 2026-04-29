"""
06_paper_figures.py
===================
Generates all paper figures as publication-ready PDFs.  No captions — captions
are written in LaTeX.  All values, labels and colour breaks are derived from
the actual results data; nothing is hardcoded beyond colour conventions and the
fixed IPC phase scale (1–5).

Figures produced
----------------
  fig1a  Choropleth: IPC phase at last assessment (ADM2)
  fig1b  Choropleth: cumulative GDELT article count (ADM2)
  fig2a  Heatmap: topic relative-coverage × time (all districts)
  fig2b  Heatmap: overall news coverage × time × country (grouped by region)
  fig3a  Null PR-AUC distribution (temporal shuffle test, n=100 permutations)
  fig3b  Null ROC-AUC distribution  (omitted cleanly if no ROC null data)
  fig4a  Regime scatter: Prob(AR+News) vs Prob(AR-Only), coloured by regime
  fig4b  Regime violin + boxplot: ΔProb by crisis regime
  fig5a  Temporal line: fold-level PR-AUC over test dates, both models
  fig5b  Temporal line: fold-level ROC-AUC over test dates, both models
  fig6a  Dot-arrow: country-level PR-AUC (AR-Only → AR+News), grouped by region
  fig6b  Strip + box: region-level PR-AUC distribution, both models
  fig7a  District scatter: log₁₀(mean articles/month) vs ΔPR-AUC
  fig7b  District scatter: volatility vs AR-Only PR-AUC
  fig7c  District scatter: onset+chronic count vs AR-Only PR-AUC

Run
---
  python 06_paper_figures.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths  (all relative to this script's directory)
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "DATA"
RESULTS_DIR = BASE_DIR / "results" / "window_2yr"    # primary (2-year) model only
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SHAPEFILE   = DATA_DIR / "shapefiles" / "gadm" / "africa_adm2_combined.gpkg"
MONTHLY_PATH = DATA_DIR / "modelling" / "monthly_gdelt_features.parquet"

# ---------------------------------------------------------------------------
# Colour conventions  (never changed)
# ---------------------------------------------------------------------------
REGIME_COLOURS = {
    "onset":    "#D62728",   # red
    "chronic":  "#FF7F0E",   # orange
    "recovery": "#2CA02C",   # green
    "stable":   "#1F77B4",   # blue
}
MODEL_COLOURS = {
    "AR-Only": "#1B4F72",    # dark navy
    "AR+News": "#C0392B",    # deep red
}
REGION_COLOURS = {
    "East Africa":     "#8E44AD",
    "West Africa":     "#E67E22",
    "Central Africa":  "#27AE60",
    "North Africa":    "#2980B9",
    "Southern Africa": "#C0392B",
}
REGION_ORDER = ["East Africa", "West Africa", "Central Africa",
                "North Africa", "Southern Africa"]

# Country → region mapping (all 18 countries actually in the dataset)
COUNTRY_REGION = {
    "Burkina Faso":                       "West Africa",
    "Burundi":                            "East Africa",
    "Cameroon":                           "Central Africa",
    "Chad":                               "Central Africa",
    "Democratic Republic of the Congo":   "Central Africa",
    "Ethiopia":                           "East Africa",
    "Kenya":                              "East Africa",
    "Madagascar":                         "Southern Africa",
    "Malawi":                             "Southern Africa",
    "Mali":                               "West Africa",
    "Mozambique":                         "Southern Africa",
    "Niger":                              "West Africa",
    "Nigeria":                            "West Africa",
    "Somalia":                            "East Africa",
    "South Sudan":                        "East Africa",
    "Sudan":                              "North Africa",
    "Uganda":                             "East Africa",
    "Zimbabwe":                           "Southern Africa",
}

# Theme list derived from dataset at runtime (see _load_themes)
# Fallback if needed:
_THEMES_FALLBACK = [
    "conflict", "displacement", "economic", "food_security",
    "governance", "health", "humanitarian", "weather", "other",
]

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#CCCCCC",
    "axes.linewidth":       0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.25,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "lines.linewidth":      1.8,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "pdf.fonttype":         42,   # embeds fonts as Type-42 (TrueType) in PDF
    "ps.fonttype":          42,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_pdf(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_themes(ds: pd.DataFrame) -> list[str]:
    rel_cols = [c for c in ds.columns if c.endswith("_relative_coverage")]
    themes   = [c.replace("_relative_coverage", "") for c in rel_cols]
    return themes if themes else _THEMES_FALLBACK


def _country_from_district_id(district_id: str) -> str:
    """Extract country from the last comma-separated token of district_id."""
    parts = [p.strip() for p in district_id.split(",")]
    return parts[-1] if parts else "Unknown"


def _sturges_bins(n: int) -> int:
    return max(5, int(np.ceil(1 + np.log2(max(n, 1)))))


def _annotation_box(ax, text, x=0.97, y=0.05, ha="right", va="bottom", fs=9):
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#AAAAAA", alpha=0.9))


def _regression_annotation(ax, x_arr, y_arr, fs=9):
    """Draw OLS line and annotate r, p, n."""
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    n = mask.sum()
    if n < 5:
        return
    slope, intercept, r, p, _ = scipy_stats.linregress(x_arr[mask], y_arr[mask])
    xfit = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 200)
    ax.plot(xfit, slope * xfit + intercept, color="#333333", lw=1.4, ls="--", zorder=5)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    _annotation_box(ax, f"r = {r:.2f},  {p_str},  n = {n}", fs=fs)


# ---------------------------------------------------------------------------
# FIGURE 1 — Choropleth maps (ADM2)
# ---------------------------------------------------------------------------

def figure_1() -> None:
    print("\n[Fig 1] Choropleth maps (ADM2)...")
    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not installed — skipping Fig 1.")
        return
    if not SHAPEFILE.exists():
        print(f"  Shapefile not found at {SHAPEFILE} — skipping Fig 1.")
        return

    gdf = gpd.read_file(SHAPEFILE)

    # Identify geometry join key
    join_key = "GID_2" if "GID_2" in gdf.columns else "NAME_2"

    ds = pd.read_parquet(DATA_DIR / "dataset.parquet")
    ds["ipc_period_start"] = pd.to_datetime(ds["ipc_period_start"])

    # Last observed IPC state per district (ipc_lag_1 = binary crisis indicator)
    last = (ds.sort_values("ipc_period_start")
              .groupby("district_id")[["ipc_lag_1", "spatial_lag"]]
              .last()
              .reset_index())
    # spatial_lag is IDW-weighted neighbour IPC phase — clip to 1–4 for ordinal display
    last["ipc_phase"] = last["spatial_lag"].clip(1, 4).round().astype(int)

    # Cumulative article count per district from monthly features file
    if MONTHLY_PATH.exists():
        mg = pd.read_parquet(MONTHLY_PATH)
        art_agg = (mg.groupby("district_id")["article_count"]
                     .sum()
                     .reset_index()
                     .rename(columns={"article_count": "total_articles"}))
    else:
        art_agg = pd.DataFrame(columns=["district_id", "total_articles"])

    # --- helper: colour districts on a GeoDataFrame ---
    def _plot_choropleth(gdf_base, df_val, val_col, cmap_listed, bins,
                         legend_labels, title_str, fig_name):
        # Try to merge on NAME_2 or GID_2 using the last comma token of district_id
        df_val = df_val.copy()
        df_val["_name_key"] = df_val["district_id"].apply(
            lambda x: x.split(",")[-2].strip() if len(x.split(",")) >= 2 else x.strip()
        )
        gdf_m = gdf_base.copy()
        # Try NAME_2 substring match (best-effort)
        gdf_m[val_col] = np.nan
        name_col = "NAME_2" if "NAME_2" in gdf_m.columns else join_key
        val_dict = df_val.set_index("district_id")[val_col].to_dict()
        # Secondary lookup by partial name
        name_dict = df_val.set_index("_name_key")[val_col].to_dict()

        def _lookup(row):
            for key_col in (name_col,):
                v = val_dict.get(row.get(key_col, ""), np.nan)
                if not np.isnan(v):
                    return v
            n2 = row.get("NAME_2", "")
            return name_dict.get(n2, np.nan)

        gdf_m[val_col] = gdf_m.apply(lambda r: _lookup(r), axis=1)

        norm = BoundaryNorm(bins, cmap_listed.N)
        fig, ax = plt.subplots(figsize=(8, 9))
        # Background (no data)
        gdf_m[gdf_m[val_col].isna()].plot(
            ax=ax, color="#E8E8E8", linewidth=0.25, edgecolor="white")
        # Data districts
        gdf_has = gdf_m[~gdf_m[val_col].isna()]
        if len(gdf_has):
            gdf_has.plot(ax=ax, column=val_col, cmap=cmap_listed, norm=norm,
                         linewidth=0.25, edgecolor="white")
        ax.set_axis_off()
        ax.set_xlim(-20, 52); ax.set_ylim(-35, 22)

        # North arrow (cartographic style)
        ax.annotate("", xy=(0.06, 0.17), xytext=(0.06, 0.11),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=2, mutation_scale=14))
        ax.text(0.06, 0.09, "N", transform=ax.transAxes, ha="center",
                va="top", fontsize=11, fontweight="bold")

        # Manual legend
        patches = [mpatches.Patch(color=cmap_listed.colors[i], label=legend_labels[i])
                   for i in range(len(legend_labels))]
        patches.append(mpatches.Patch(color="#E8E8E8", label="No data"))
        ax.legend(handles=patches, loc="lower left", fontsize=8,
                  title=title_str, title_fontsize=8.5,
                  framealpha=0.95, edgecolor="#CCCCCC")
        fig.tight_layout(pad=0.3)
        save_pdf(fig, fig_name)

    # Fig 1a — IPC phase (proxy via spatial_lag)
    ipc_cmap = ListedColormap(["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"])
    ipc_bins  = [0.5, 1.5, 2.5, 3.5, 4.5]
    ipc_labels = ["Phase 1 (Minimal)", "Phase 2 (Stressed)",
                  "Phase 3 (Crisis)", "Phase 4+ (Emergency)"]
    _plot_choropleth(gdf, last, "ipc_phase", ipc_cmap, ipc_bins,
                     ipc_labels, "IPC Phase", "fig1a_ipc_choropleth")

    # Fig 1b — cumulative news articles
    if len(art_agg):
        news_vals  = art_agg["total_articles"].dropna()
        qs_news    = np.nanpercentile(news_vals, [0, 20, 40, 60, 80, 100])
        news_bins  = sorted(set(qs_news.tolist()))
        n_news_bins = len(news_bins) - 1
        news_cmap   = ListedColormap(
            ["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B"][:n_news_bins])
        news_labels = [f"{int(qs_news[i]/1e3)}k - {int(qs_news[i+1]/1e3)}k"
                       for i in range(n_news_bins)]
        art_agg["total_articles_binned"] = pd.cut(
            art_agg["total_articles"], bins=news_bins,
            labels=range(n_news_bins), include_lowest=True
        ).astype(float)
        _plot_choropleth(gdf, art_agg.rename(
            columns={"total_articles_binned": "total_articles"}),
            "total_articles", news_cmap, list(range(n_news_bins + 1)),
            news_labels, "Articles (cumulative)", "fig1b_news_choropleth")
    else:
        print("  Monthly features not found — skipping fig1b.")


# ---------------------------------------------------------------------------
# FIGURE 2 — Heatmaps
# ---------------------------------------------------------------------------

def figure_2() -> None:
    print("\n[Fig 2] Heatmaps...")
    ds = pd.read_parquet(DATA_DIR / "dataset.parquet")
    ds["ipc_period_start"] = pd.to_datetime(ds["ipc_period_start"])
    ds["period_label"] = ds["ipc_period_start"].dt.strftime("%Y-%m")
    ds["region"]       = ds["ipc_country"].map(COUNTRY_REGION).fillna("Other")

    themes      = _load_themes(ds)
    rel_cols    = [f"{t}_relative_coverage" for t in themes]
    theme_labels = [t.replace("_", " ").title() for t in themes]

    cmap5 = ListedColormap(["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B"])
    cmap5.set_bad(color="#D0D0D0")

    def _quintile_bin(arr2d):
        vals = arr2d[np.isfinite(arr2d) & (arr2d > 0)]
        qs   = np.nanpercentile(vals, [20, 40, 60, 80])
        return np.digitize(arr2d, qs, right=True)  # 0..4

    # ── Fig 2a — topic × time ─────────────────────────────────────────────
    pivot_a = (ds.groupby("period_label")[rel_cols].mean()
                 .sort_index())
    pivot_a.columns = theme_labels
    binned_a = _quintile_bin(pivot_a.values)

    # Use quarterly x-tick labels to avoid label collisions
    periods   = list(pivot_a.index)
    n_periods = len(periods)
    # Show every 3rd label (quarterly cadence for 4-month periods)
    tick_step = 3
    tick_pos  = list(range(0, n_periods, tick_step))
    tick_lbl  = [periods[i] for i in tick_pos]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.grid(False)
    im = ax.imshow(binned_a.T, aspect="auto", cmap=cmap5, vmin=0, vmax=4,
                   interpolation="nearest")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(theme_labels)))
    ax.set_yticklabels(theme_labels, fontsize=9)
    ax.set_xlabel("Assessment period", labelpad=6)
    ax.set_ylabel("News theme", labelpad=6)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], pad=0.015, shrink=0.85)
    cb.ax.set_yticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"],
                           fontsize=8)
    cb.set_label("Relative coverage\n(global quintile)", fontsize=9)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig2a_topic_heatmap")

    # ── Fig 2b — country × time (grouped by region) ───────────────────────
    ds2 = ds.copy()
    country_order = []
    for r in REGION_ORDER:
        cs = sorted(c for c in ds2["ipc_country"].unique()
                    if COUNTRY_REGION.get(c) == r)
        country_order.extend(cs)
    other = [c for c in ds2["ipc_country"].unique() if c not in country_order]
    country_order.extend(sorted(other))

    pivot_b = (ds2.groupby(["ipc_country", "period_label"])[rel_cols]
                  .mean().mean(axis=1)
                  .unstack("period_label")
                  .sort_index(axis=1))
    pivot_b = pivot_b.reindex([c for c in country_order if c in pivot_b.index])

    raw_b = pivot_b.values.astype(float)
    binned_b = np.where(np.isfinite(raw_b), _quintile_bin(raw_b), np.nan)
    masked_b = np.ma.masked_invalid(binned_b)

    n_countries = len(pivot_b)
    region_sizes = []
    for r in REGION_ORDER:
        n = sum(1 for c in pivot_b.index if COUNTRY_REGION.get(c) == r)
        if n:
            region_sizes.append((r, n))

    fig, ax = plt.subplots(figsize=(13, max(6, n_countries * 0.4)))
    ax.grid(False)
    im = ax.imshow(masked_b, aspect="auto", cmap=cmap5, vmin=0, vmax=4,
                   interpolation="nearest")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_countries))
    ax.set_yticklabels(pivot_b.index.tolist(), fontsize=8)
    ax.set_xlabel("Assessment period", labelpad=6)
    ax.set_ylabel("Country", labelpad=6)
    ax.spines[:].set_visible(False)

    # Region separator lines and right-side labels
    cumulative = 0
    ax2 = ax.twinx()
    midpoints, rlabels = [], []
    for (rname, rsize) in region_sizes:
        boundary = cumulative + rsize
        if boundary < n_countries:
            ax.axhline(boundary - 0.5, color="black", lw=1.2, ls="-")
        midpoints.append(cumulative + rsize / 2 - 0.5)
        rlabels.append(rname)
        cumulative = boundary

    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(midpoints)
    ax2.set_yticklabels(rlabels, fontsize=8.5, color="#333333", fontweight="bold")
    ax2.tick_params(right=False, labelright=True)
    ax2.spines[:].set_visible(False)

    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], pad=0.08, shrink=0.6)
    cb.ax.set_yticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=8)
    cb.set_label("Coverage quintile", fontsize=9)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig2b_country_heatmap")


# ---------------------------------------------------------------------------
# FIGURE 3 — Null distribution histograms
# ---------------------------------------------------------------------------

def figure_3() -> None:
    print("\n[Fig 3] Null distribution histograms...")
    null_path = BASE_DIR / "results" / "shuffle_test" / "null_distribution.csv"
    cfg_path  = BASE_DIR / "results" / "shuffle_test" / "config.json"
    if not null_path.exists():
        print("  shuffle_test/null_distribution.csv not found — skipping Fig 3.")
        return

    null_df      = pd.read_csv(null_path)
    fold_results = pd.read_csv(RESULTS_DIR / "fold_results.csv")
    n_perms      = len(null_df)

    # Observed means from the primary (7-fold) model
    real_ar   = fold_results["ar_pr_auc"].mean()
    real_full = fold_results["full_pr_auc"].mean()

    # Null distribution: mean PR-AUC of combined model with shuffled news
    null_pr = null_df["mean_pr_auc"].values
    null_mean = null_pr.mean()
    null_std  = null_pr.std()

    # Effect size (z-score)
    z_ar   = (real_ar   - null_mean) / null_std if null_std > 0 else np.nan
    z_full = (real_full - null_mean) / null_std if null_std > 0 else np.nan

    p_ar   = float((null_pr >= real_ar).mean())
    p_full = float((null_pr >= real_full).mean())
    p_str_full = "p < 0.01" if p_full < 0.01 else f"p = {p_full:.3f}"
    p_str_ar   = "p < 0.01" if p_ar   < 0.01 else f"p = {p_ar:.3f}"

    n_bins = _sturges_bins(n_perms)
    x_lo = min(null_pr.min(), real_ar, real_full) - 0.02
    x_hi = max(null_pr.max(), real_ar, real_full) + 0.02
    bins  = np.linspace(x_lo, x_hi, n_bins + 1)

    # ── Fig 3a — PR-AUC ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_pr, bins=bins, density=True, alpha=0.60,
            color="#AAAACC", edgecolor="white", lw=0.5,
            label=f"Null (n={n_perms} permutations)")
    ax.axvline(real_ar,   color=MODEL_COLOURS["AR-Only"], lw=2.2, ls="--",
               label=f"AR-Only  {real_ar:.3f}  ({p_str_ar})")
    ax.axvline(real_full, color=MODEL_COLOURS["AR+News"],  lw=2.2, ls="-",
               label=f"AR+News  {real_full:.3f}  ({p_str_full})")

    # Null mean ± 1 SD band
    ax.axvspan(null_mean - null_std, null_mean + null_std,
               alpha=0.12, color="#666688", label="Null ±1 SD")
    ax.axvline(null_mean, color="#666688", lw=1.0, ls=":")

    # Effect size annotation
    eff_txt = (f"z(AR-Only) = {z_ar:.1f}\n"
               f"z(AR+News) = {z_full:.1f}\n"
               f"Null: {null_mean:.3f} ± {null_std:.3f}")
    _annotation_box(ax, eff_txt, x=0.02, y=0.97, ha="left", va="top", fs=8)

    ax.set_xlabel("Mean PR-AUC (across folds)", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
    ax.legend(fontsize=8.5, loc="upper right")
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig3a_null_prauc")

    # ── Fig 3b — ROC-AUC (only if real null data exists) ─────────────────
    roc_cols = [c for c in null_df.columns if "roc" in c.lower()]
    if not roc_cols:
        print("  No ROC-AUC null distribution available — omitting fig3b.")
        return

    null_roc = null_df[roc_cols[0]].values
    real_ar_roc   = fold_results["ar_roc_auc"].mean()
    real_full_roc = fold_results["full_roc_auc"].mean()
    p_roc = float((null_roc >= real_full_roc).mean())
    p_roc_str = "p < 0.01" if p_roc < 0.01 else f"p = {p_roc:.3f}"

    bins_r = np.linspace(
        min(null_roc.min(), real_ar_roc, real_full_roc) - 0.01,
        max(null_roc.max(), real_ar_roc, real_full_roc) + 0.01,
        _sturges_bins(n_perms) + 1
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_roc, bins=bins_r, density=True, alpha=0.60,
            color="#AAAACC", edgecolor="white", lw=0.5,
            label=f"Null (n={n_perms} permutations)")
    ax.axvline(real_ar_roc,   color=MODEL_COLOURS["AR-Only"], lw=2.2, ls="--",
               label=f"AR-Only  {real_ar_roc:.3f}")
    ax.axvline(real_full_roc, color=MODEL_COLOURS["AR+News"],  lw=2.2, ls="-",
               label=f"AR+News  {real_full_roc:.3f}  ({p_roc_str})")
    ax.set_xlabel("Mean ROC-AUC (across folds)", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
    ax.legend(fontsize=8.5, loc="upper left")
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig3b_null_rocauc")


# ---------------------------------------------------------------------------
# FIGURE 4 — Crisis regime analysis
# ---------------------------------------------------------------------------

def figure_4() -> None:
    print("\n[Fig 4] Crisis regime analysis...")
    preds = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")

    # Validate regimes present
    regime_order = ["onset", "chronic", "recovery", "stable"]
    preds = preds[preds["regime"].isin(regime_order)].copy()
    preds["delta_prob"] = preds["prob_combined"] - preds["prob_ar"]

    regime_counts = preds["regime"].value_counts()

    # ── Fig 4a — probability scatter ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.5)
    for regime in regime_order:
        grp = preds[preds["regime"] == regime]
        if grp.empty:
            continue
        ax.scatter(grp["prob_ar"], grp["prob_combined"],
                   color=REGIME_COLOURS[regime], alpha=0.35, s=14,
                   edgecolors="none",
                   label=f"{regime.capitalize()} (n={len(grp):,})")
    ax.plot([0, 1], [0, 1], color="#333333", lw=1.4, ls="--", alpha=0.6,
            zorder=10, label="y = x")
    ax.set_xlabel("P(crisis) — AR-Only", labelpad=6)
    ax.set_ylabel("P(crisis) — AR+News", labelpad=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, title="Crisis regime", title_fontsize=8.5,
              markerscale=1.8, loc="upper left")
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig4a_regime_scatter")

    # ── Fig 4b — violin + boxplot: ΔProb by regime ────────────────────────
    data_by_regime = []
    labels_regime  = []
    for r in regime_order:
        arr = preds[preds["regime"] == r]["delta_prob"].dropna().values
        if len(arr) > 1:
            data_by_regime.append(arr)
            labels_regime.append(r)

    n_reg  = len(data_by_regime)
    x_pos  = np.arange(n_reg)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.5)
    parts = ax.violinplot(data_by_regime, positions=x_pos,
                          showmedians=False, showextrema=False, widths=0.75)
    for pc, r in zip(parts["bodies"], labels_regime):
        pc.set_facecolor(REGIME_COLOURS[r])
        pc.set_alpha(0.55)
        pc.set_edgecolor("none")

    bp = ax.boxplot(data_by_regime, positions=x_pos, widths=0.18,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color="#333333", lw=1.2),
                    capprops=dict(color="#333333", lw=1.2),
                    flierprops=dict(marker="o", markersize=2.5,
                                    alpha=0.3, markeredgecolor="none"))
    for patch, r in zip(bp["boxes"], labels_regime):
        patch.set_facecolor(REGIME_COLOURS[r])
        patch.set_alpha(0.90)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)

    ax.axhline(0, color="#333333", lw=1.0, ls="--", alpha=0.7)

    # Sample size annotations above each violin
    for i, (arr, r) in enumerate(zip(data_by_regime, labels_regime)):
        ymax = np.percentile(arr, 95)
        ax.text(i, ymax + 0.01, f"n={len(arr):,}", ha="center",
                va="bottom", fontsize=7.5, color="#444444")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.capitalize() for r in labels_regime], fontsize=10)
    ax.set_xlabel("Crisis regime", labelpad=6)
    ax.set_ylabel("ΔP(crisis)  =  P(AR+News) − P(AR-Only)", labelpad=6)
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig4b_regime_violin")


# ---------------------------------------------------------------------------
# FIGURE 5 — Temporal line plots
# ---------------------------------------------------------------------------

def figure_5() -> None:
    print("\n[Fig 5] Temporal line plots...")
    fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")
    fold_df["test_start"] = pd.to_datetime(fold_df["test_start"])
    x = fold_df["test_start"]

    for metric, col_ar, col_full, fname, ylabel in [
        ("PR-AUC",  "ar_pr_auc",  "full_pr_auc",  "fig5a_temporal_prauc",  "PR-AUC"),
        ("ROC-AUC", "ar_roc_auc", "full_roc_auc", "fig5b_temporal_rocauc", "ROC-AUC"),
    ]:
        y_ar   = fold_df[col_ar].values
        y_full = fold_df[col_full].values

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.5)

        ax.plot(x, y_ar,   "o-", color=MODEL_COLOURS["AR-Only"],
                label="AR-Only", lw=2.0, ms=7, zorder=4)
        ax.plot(x, y_full, "s-", color=MODEL_COLOURS["AR+News"],
                label="AR+News",  lw=2.0, ms=7, zorder=4)

        # Shaded area between models (highlight where news helps)
        ax.fill_between(x, y_ar, y_full,
                        where=(y_full >= y_ar),
                        alpha=0.12, color=MODEL_COLOURS["AR+News"],
                        interpolate=True, label="Combined > AR")
        ax.fill_between(x, y_ar, y_full,
                        where=(y_full < y_ar),
                        alpha=0.12, color="#888888",
                        interpolate=True, label="AR > Combined")

        # n_test labels above the higher of the two lines
        n_vals = fold_df["n_test"].values
        for xi, ya, yf, ni in zip(x, y_ar, y_full, n_vals):
            y_top = max(ya, yf)
            ax.annotate(f"n={ni}", (xi, y_top),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=7, ha="center", color="#555555")

        ax.set_xlabel("Test period (start)", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_ylim(max(0, min(y_ar.min(), y_full.min()) - 0.08), 1.0)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8.5, loc="lower left", ncol=2)
        _despine(ax)
        fig.tight_layout(pad=0.5)
        save_pdf(fig, fname)


# ---------------------------------------------------------------------------
# FIGURE 6 — Country / region performance
# ---------------------------------------------------------------------------

def figure_6() -> None:
    print("\n[Fig 6] Country/region performance...")
    preds = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")

    # Reliable country extraction: last comma-separated token of district_id
    preds["ipc_country"] = preds["district_id"].apply(_country_from_district_id)

    from sklearn.metrics import average_precision_score
    rows = []
    for country, grp in preds.groupby("ipc_country"):
        y = grp["target_crisis_binary"].values.astype(int)
        n_pos = int(y.sum()); n_neg = int((1 - y).sum())
        if n_pos < 2 or n_neg < 2:
            continue
        try:
            pr_ar   = average_precision_score(y, grp["prob_ar"].values)
            pr_full = average_precision_score(y, grp["prob_combined"].values)
        except Exception:
            continue
        rows.append({
            "country":    country,
            "region":     COUNTRY_REGION.get(country, "Other"),
            "prauc_ar":   pr_ar,
            "prauc_full": pr_full,
            "n_obs":      len(grp),
            "n_pos":      n_pos,
        })
    cdf = pd.DataFrame(rows)

    if cdf.empty:
        print("  No qualifying countries — skipping Fig 6.")
        return

    region_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    cdf["region_rank"] = cdf["region"].map(region_rank).fillna(99)
    cdf = cdf.sort_values(["region_rank", "prauc_ar"]).reset_index(drop=True)

    print(f"  Countries in fig6: {len(cdf)}  —  {sorted(cdf['country'].tolist())}")

    # ── Fig 6a — dot + arrow (AR-Only → AR+News) per country ─────────────
    n_ctry = len(cdf)
    fig, ax = plt.subplots(figsize=(7.5, max(5.5, n_ctry * 0.42)))
    ax.grid(True, axis="x", alpha=0.20, linestyle="--", linewidth=0.5)

    for i, row in enumerate(cdf.itertuples()):
        col = REGION_COLOURS.get(row.region, "#888888")
        # Arrow: AR-Only → AR+News
        ax.annotate("",
                    xy     =(row.prauc_full, i),
                    xytext =(row.prauc_ar,   i),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.8, mutation_scale=10))
        ax.plot(row.prauc_ar,   i, "o", color=MODEL_COLOURS["AR-Only"],
                ms=8, zorder=5, markeredgewidth=0.5, markeredgecolor="white")
        ax.plot(row.prauc_full, i, "s", color=MODEL_COLOURS["AR+News"],
                ms=8, zorder=5, markeredgewidth=0.5, markeredgecolor="white")
        # n annotation on right margin
        ax.text(1.02, i, f"n={row.n_obs}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7, color="#666666")

    ax.set_yticks(range(n_ctry))
    ax.set_yticklabels(cdf["country"].tolist(), fontsize=8.5)
    ax.set_xlabel("PR-AUC", labelpad=6)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")

    # Region separator lines
    prev_region = None
    for i, row in enumerate(cdf.itertuples()):
        if row.region != prev_region and i > 0:
            ax.axhline(i - 0.5, color="#555555", lw=0.8, ls="--", alpha=0.5)
        prev_region = row.region

    # Legend: models + region colours
    legend_elems = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=MODEL_COLOURS["AR-Only"], ms=8, label="AR-Only"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor=MODEL_COLOURS["AR+News"],  ms=8, label="AR+News"),
    ]
    for r in REGION_ORDER:
        if r in cdf["region"].values:
            legend_elems.append(
                mpatches.Patch(color=REGION_COLOURS[r], label=r, alpha=0.85))
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right",
              title="Model / Region", title_fontsize=8)
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig6a_country_prauc")

    # ── Fig 6b — strip + box: PR-AUC by region ───────────────────────────
    regions_present = [r for r in REGION_ORDER if r in cdf["region"].values]
    n_reg = len(regions_present)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.grid(True, axis="y", alpha=0.20, linestyle="--", linewidth=0.5)

    gap    = 2.2
    bw     = 0.6
    offset = 0.35

    for ri, region in enumerate(regions_present):
        sub = cdf[cdf["region"] == region]
        xc  = ri * gap
        col = REGION_COLOURS[region]

        for model_key, col_name, xoff, mshape in [
            ("AR-Only", "prauc_ar",   -offset, "o"),
            ("AR+News",  "prauc_full", +offset, "s"),
        ]:
            vals = sub[col_name].values
            xj   = xc + xoff + np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
            mc   = MODEL_COLOURS[model_key]
            ax.scatter(xj, vals, color=mc, alpha=0.8, s=50, zorder=4,
                       marker=mshape, edgecolors="white", linewidths=0.5)
            if len(vals) >= 3:
                bp = ax.boxplot(vals, positions=[xc + xoff], widths=bw * 0.38,
                                patch_artist=True, manage_ticks=False,
                                medianprops=dict(color="black", lw=2.2),
                                whiskerprops=dict(color="#333333", lw=1.0),
                                capprops=dict(color="#333333", lw=1.0),
                                flierprops=dict(visible=False),
                                boxprops=dict(linewidth=0.6))
                for patch in bp["boxes"]:
                    patch.set_facecolor(mc); patch.set_alpha(0.40)
            elif len(vals) > 0:
                # Single country — just a horizontal line
                ax.hlines(vals[0], xc + xoff - 0.15, xc + xoff + 0.15,
                          color=mc, lw=2.0, zorder=4)

    ax.set_xticks([ri * gap for ri in range(n_reg)])
    ax.set_xticklabels(regions_present, fontsize=9)
    ax.set_ylabel("PR-AUC", labelpad=6)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#AAAAAA", lw=0.8, ls=":")

    legend_elems = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=MODEL_COLOURS["AR-Only"], ms=9, label="AR-Only"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor=MODEL_COLOURS["AR+News"], ms=9, label="AR+News"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig6b_region_prauc")


# ---------------------------------------------------------------------------
# FIGURE 7 — District-level correlates
# ---------------------------------------------------------------------------

def figure_7() -> None:
    print("\n[Fig 7] District-level scatter plots...")
    dm = pd.read_csv(RESULTS_DIR / "district_level_metrics.csv")

    # Filter to districts with >= 5 observations for reliable PR-AUC
    dm = dm[dm["n_obs"] >= 5].copy()
    n_dm = len(dm)
    print(f"  Districts with n_obs >= 5: {n_dm}")

    # Add country/region for colour encoding
    dm["country"] = dm["district_id"].apply(_country_from_district_id)
    dm["region"]  = dm["country"].map(COUNTRY_REGION).fillna("Other")

    def _scatter_panel(ax, x_arr, y_arr, region_arr, xlabel, ylabel,
                       x_log=False, add_hline=None):
        """Scatter coloured by region with regression overlay."""
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.5)
        for region in REGION_ORDER + ["Other"]:
            mask_r = (region_arr == region) & np.isfinite(x_arr) & np.isfinite(y_arr)
            if mask_r.sum() == 0:
                continue
            col = REGION_COLOURS.get(region, "#888888")
            ax.scatter(x_arr[mask_r], y_arr[mask_r],
                       color=col, alpha=0.65, s=28, edgecolors="white",
                       linewidths=0.4, label=region, zorder=3)
        if add_hline is not None:
            ax.axhline(add_hline, color="#888888", lw=0.9, ls=":", zorder=2)
        _regression_annotation(ax, x_arr, y_arr)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        _despine(ax)

    # ── Fig 7a — log₁₀(articles/month) vs ΔPR-AUC ───────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x7a = np.log10(dm["mean_articles_month"].clip(lower=1).values)
    y7a = dm["delta_prauc"].values
    _scatter_panel(ax, x7a, y7a, dm["region"].values,
                   r"$\log_{10}$(Mean articles month$^{-1}$)",
                   r"$\Delta$PR-AUC  =  PR-AUC(AR+News) $-$ PR-AUC(AR-Only)",
                   add_hline=0.0)
    ax.set_ylim(y7a[np.isfinite(y7a)].min() - 0.05,
                y7a[np.isfinite(y7a)].max() + 0.10)
    handles = [mpatches.Patch(color=REGION_COLOURS.get(r,"#888888"), label=r)
               for r in REGION_ORDER if r in dm["region"].values]
    ax.legend(handles=handles, fontsize=7.5, title="Region",
              title_fontsize=8, loc="upper left")
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig7a_articles_vs_delta")

    # ── Fig 7b — volatility vs AR-Only PR-AUC ────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x7b = dm["volatility"].values
    y7b = dm["prauc_ar"].values
    _scatter_panel(ax, x7b, y7b, dm["region"].values,
                   "Volatility  (fraction of periods with regime change)",
                   "PR-AUC — AR-Only")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    handles = [mpatches.Patch(color=REGION_COLOURS.get(r,"#888888"), label=r)
               for r in REGION_ORDER if r in dm["region"].values]
    ax.legend(handles=handles, fontsize=7.5, title="Region",
              title_fontsize=8, loc="lower left")
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig7b_volatility_vs_prauc")

    # ── Fig 7c — onset+chronic count vs AR-Only PR-AUC ───────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x7c = dm["onset_chronic_count"].values.astype(float)
    y7c = dm["prauc_ar"].values
    _scatter_panel(ax, x7c, y7c, dm["region"].values,
                   "Onset + Chronic observations per district",
                   "PR-AUC — AR-Only")
    ax.set_xlim(-0.3, x7c[np.isfinite(x7c)].max() + 0.5)
    ax.set_ylim(-0.02, 1.05)
    handles = [mpatches.Patch(color=REGION_COLOURS.get(r,"#888888"), label=r)
               for r in REGION_ORDER if r in dm["region"].values]
    ax.legend(handles=handles, fontsize=7.5, title="Region",
              title_fontsize=8, loc="lower right")
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig7c_onsetchronic_vs_prauc")


# ---------------------------------------------------------------------------
# FIGURE 6c — All-country prevalence + PR-AUC companion
# ---------------------------------------------------------------------------

def figure_6c() -> None:
    """Two-panel figure showing all 18 countries.

    Left panel  — horizontal stacked bar: crisis vs. non-crisis observations
                  (test set) for every country, ordered by region then crisis rate.
                  Countries where PR-AUC cannot be computed (all-crisis or
                  all-non-crisis) are shaded grey and annotated.

    Right panel — PR-AUC dot+arrow (AR-Only → AR+News) for countries that
                  have at least 2 positive AND 2 negative observations.
                  Countries excluded from the right panel are labelled
                  "insufficient class mix" in a footnote.
    """
    print("\n[Fig 6c] All-country prevalence + PR-AUC companion...")
    from sklearn.metrics import average_precision_score

    preds = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
    preds["ipc_country"] = preds["district_id"].apply(_country_from_district_id)

    # Build per-country stats (all 18)
    all_rows = []
    for country, grp in preds.groupby("ipc_country"):
        y      = grp["target_crisis_binary"].values.astype(int)
        n_pos  = int(y.sum())
        n_neg  = int((1 - y).sum())
        n_tot  = len(grp)
        prev   = n_pos / n_tot if n_tot > 0 else 0.0
        region = COUNTRY_REGION.get(country, "Other")

        can_score = (n_pos >= 2 and n_neg >= 2)
        pr_ar     = np.nan
        pr_full   = np.nan
        if can_score:
            try:
                pr_ar   = average_precision_score(y, grp["prob_ar"].values)
                pr_full = average_precision_score(y, grp["prob_combined"].values)
            except Exception:
                can_score = False

        all_rows.append({
            "country":    country,
            "region":     region,
            "n_pos":      n_pos,
            "n_neg":      n_neg,
            "n_tot":      n_tot,
            "prevalence": prev,
            "can_score":  can_score,
            "prauc_ar":   pr_ar,
            "prauc_full": pr_full,
        })

    adf = pd.DataFrame(all_rows)

    # Sort: region order, then descending crisis prevalence within region
    region_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    adf["region_rank"] = adf["region"].map(region_rank).fillna(99)
    adf = adf.sort_values(
        ["region_rank", "prevalence"], ascending=[True, False]
    ).reset_index(drop=True)
    n_all = len(adf)

    # Shorten DRC name for display
    adf["country_label"] = adf["country"].str.replace(
        "The Democratic Republic of the", "DR Congo", regex=False
    )

    # ── Layout: two panels side-by-side ──────────────────────────────────
    fig, (ax_prev, ax_prauc) = plt.subplots(
        1, 2, figsize=(13, max(7, n_all * 0.42)),
        gridspec_kw={"width_ratios": [1, 1.2]}
    )

    CRISIS_COL    = "#C0392B"
    NONCRISIS_COL = "#2980B9"
    NODATA_COL    = "#CCCCCC"

    # ── Left panel: stacked prevalence bars ──────────────────────────────
    ax_prev.grid(True, axis="x", alpha=0.18, linestyle="--", linewidth=0.5)
    for i, row in enumerate(adf.itertuples()):
        total = row.n_tot
        if total == 0:
            ax_prev.barh(i, 1.0, color=NODATA_COL, height=0.65)
            continue
        frac_pos = row.n_pos / total
        frac_neg = row.n_neg / total
        ax_prev.barh(i, frac_pos, color=CRISIS_COL,    height=0.65, zorder=3)
        ax_prev.barh(i, frac_neg, left=frac_pos,
                     color=NONCRISIS_COL, height=0.65, zorder=3)
        # prevalence % label inside or beside the crisis bar
        pct_str = f"{frac_pos*100:.0f}%"
        if frac_pos >= 0.12:
            ax_prev.text(frac_pos / 2, i, pct_str,
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold")
        else:
            ax_prev.text(frac_pos + 0.02, i, pct_str,
                         ha="left", va="center", fontsize=7,
                         color=CRISIS_COL, fontweight="bold")
        # n_tot on right margin
        ax_prev.text(1.02, i, f"n={total}",
                     transform=ax_prev.get_yaxis_transform(),
                     va="center", ha="left", fontsize=7, color="#555555")

    ax_prev.set_yticks(range(n_all))
    ax_prev.set_yticklabels(adf["country_label"].tolist(), fontsize=9)
    ax_prev.set_xlim(0, 1)
    ax_prev.set_xlabel("Fraction of test-set observations", labelpad=6)
    ax_prev.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")

    # Region separator lines on left panel
    prev_region = None
    for i, row in enumerate(adf.itertuples()):
        if row.region != prev_region and i > 0:
            ax_prev.axhline(i - 0.5, color="#555555", lw=0.7, ls="--", alpha=0.45)
        prev_region = row.region

    prev_legend = [
        mpatches.Patch(color=CRISIS_COL,    label="Crisis (IPC ≥ 3)"),
        mpatches.Patch(color=NONCRISIS_COL, label="Non-crisis"),
    ]
    ax_prev.legend(handles=prev_legend, fontsize=8, loc="lower right",
                   framealpha=0.95, edgecolor="#CCCCCC")
    _despine(ax_prev)

    # ── Right panel: PR-AUC dot+arrow ────────────────────────────────────
    ax_prauc.grid(True, axis="x", alpha=0.18, linestyle="--", linewidth=0.5)
    ax_prauc.set_yticks(range(n_all))
    ax_prauc.set_yticklabels([""] * n_all)  # labels on left panel only

    excluded = []
    for i, row in enumerate(adf.itertuples()):
        if not row.can_score:
            reason = ("all crisis" if row.n_neg < 2
                      else "no crisis" if row.n_pos < 2
                      else "too few obs")
            excluded.append(row.country_label)
            ax_prauc.text(0.5, i, f"—  {reason}",
                          ha="center", va="center", fontsize=8,
                          color="#999999", style="italic")
            continue

        col = REGION_COLOURS.get(row.region, "#888888")
        # Arrow AR-Only → AR+News
        ax_prauc.annotate("",
                          xy    =(row.prauc_full, i),
                          xytext=(row.prauc_ar,   i),
                          arrowprops=dict(arrowstyle="-|>", color=col,
                                          lw=1.8, mutation_scale=10))
        ax_prauc.plot(row.prauc_ar,   i, "o",
                      color=MODEL_COLOURS["AR-Only"], ms=8, zorder=5,
                      markeredgewidth=0.5, markeredgecolor="white")
        ax_prauc.plot(row.prauc_full, i, "s",
                      color=MODEL_COLOURS["AR+News"], ms=8, zorder=5,
                      markeredgewidth=0.5, markeredgecolor="white")
        delta = row.prauc_full - row.prauc_ar
        sign  = "+" if delta >= 0 else ""
        ax_prauc.text(1.02, i, f"{sign}{delta:.2f}",
                      transform=ax_prauc.get_yaxis_transform(),
                      va="center", ha="left", fontsize=7.5,
                      color=("#27AE60" if delta >= 0 else "#E74C3C"))

    # Region separator lines on right panel
    prev_region = None
    for i, row in enumerate(adf.itertuples()):
        if row.region != prev_region and i > 0:
            ax_prauc.axhline(i - 0.5, color="#555555", lw=0.7, ls="--", alpha=0.45)
        prev_region = row.region

    ax_prauc.set_xlim(0, 1)
    ax_prauc.set_xlabel("PR-AUC", labelpad=6)
    ax_prauc.axvline(0.5, color="#AAAAAA", lw=0.8, ls=":")

    # Region colour legend + model legend
    legend_elems = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=MODEL_COLOURS["AR-Only"], ms=8, label="AR-Only"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor=MODEL_COLOURS["AR+News"], ms=8, label="AR+News"),
    ]
    for r in REGION_ORDER:
        if r in adf["region"].values:
            legend_elems.append(
                mpatches.Patch(color=REGION_COLOURS[r], label=r, alpha=0.85))
    ax_prauc.legend(handles=legend_elems, fontsize=8, loc="lower right",
                    title="Model / Region", title_fontsize=8,
                    framealpha=0.95, edgecolor="#CCCCCC")

    # Right-margin header for delta column
    ax_prauc.text(1.02, n_all - 0.2, "ΔPR-AUC",
                  transform=ax_prauc.get_yaxis_transform(),
                  va="bottom", ha="left", fontsize=7.5,
                  color="#333333", fontweight="bold")

    _despine(ax_prauc)

    fig.tight_layout(pad=0.5, w_pad=0.8)
    save_pdf(fig, "fig6c_all_countries_prevalence_prauc")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Generating paper figures  (primary 2-year model only)")
    print("=" * 60)

    figure_1()   # choropleth maps
    figure_2()   # heatmaps
    figure_3()   # null distributions
    figure_4()   # regime analysis
    figure_5()   # temporal line plots
    figure_6()   # country / region performance
    figure_6c()  # all-country prevalence + PR-AUC companion
    figure_7()   # district-level correlates

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
