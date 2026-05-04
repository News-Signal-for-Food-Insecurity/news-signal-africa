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

SHAPEFILE    = DATA_DIR / "shapefiles" / "gadm" / "africa_adm2_combined.gpkg"
AFRICA_ADM0  = DATA_DIR / "shapefiles" / "gadm" / "africa_adm0_basemap.gpkg"
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
    "AR-Only": "#1f77b4",    # seaborn blue
    "AR+News": "#9467bd",    # seaborn purple
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
import seaborn as sns
sns.set_style("whitegrid")

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
    "lines.linewidth":      1.8,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "pdf.fonttype":         42,
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
    sns.despine(ax=ax)


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

def _build_district_gid2_bridge() -> "pd.DataFrame":
    """Return a DataFrame mapping district_id -> GID_2 via stage1 primary_gadm2."""
    import geopandas as gpd
    shp = gpd.read_file(SHAPEFILE)
    s1  = pd.read_parquet(DATA_DIR / "raw" / "stage1_features.parquet")

    shp["_dname"] = shp["district_name"].str.strip().str.lower()
    shp["_cname"] = shp["country_name"].str.strip().str.lower()

    s1["district_id"] = s1["ipc_geographic_unit_full"].str.strip()
    bridge = (s1[["district_id", "ipc_country", "primary_gadm2"]]
              .drop_duplicates()
              .dropna(subset=["primary_gadm2"]))
    bridge["_dname"] = bridge["primary_gadm2"].str.strip().str.lower()
    bridge["_cname"] = bridge["ipc_country"].str.strip().str.lower()

    bridge = bridge.merge(shp[["GID_2", "_dname", "_cname"]],
                          on=["_dname", "_cname"], how="left")
    bridge = (bridge.dropna(subset=["GID_2"])
              .drop_duplicates("district_id")[["district_id", "GID_2"]])
    return bridge


def figure_1() -> None:
    print("\n[Fig 1] Choropleth maps (ADM2)...")
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("  geopandas not installed — skipping Fig 1.")
        return
    if not SHAPEFILE.exists():
        print(f"  Shapefile not found at {SHAPEFILE} — skipping Fig 1.")
        return

    import warnings
    gdf = gpd.read_file(SHAPEFILE)

    # Full Africa ADM0 basemap (all 54 countries) — backdrop for the map
    if AFRICA_ADM0.exists():
        africa_base = gpd.read_file(AFRICA_ADM0)
    else:
        # Fallback: dissolve the ADM2 shapefile to country level (only 23 countries)
        africa_base = gdf.dissolve(by="country_name").reset_index()
    print(f"  Africa basemap: {len(africa_base)} countries")

    ds = pd.read_parquet(DATA_DIR / "dataset.parquet")
    ds["ipc_period_start"] = pd.to_datetime(ds["ipc_period_start"])
    s1 = pd.read_parquet(DATA_DIR / "raw" / "stage1_features.parquet")
    s1["district_id"] = s1["ipc_geographic_unit_full"].str.strip()

    # Study countries (normalise DRC name to match shapefile)
    CNAME_FIX = {"The Democratic Republic of the": "Democratic Republic of the Congo"}
    study_ctry = set(CNAME_FIX.get(c, c) for c in ds["ipc_country"].unique())
    study_shp  = gdf[gdf["country_name"].isin(study_ctry)].copy()

    # ── Build zone-centroid GeoDataFrame (avg_lat/lon from stage1) ───────
    s1_coords = (s1[["district_id", "avg_latitude", "avg_longitude"]]
                 .dropna()
                 .query("avg_latitude != 0 or avg_longitude != 0"))
    zone_coords = (s1_coords.groupby("district_id")[["avg_latitude", "avg_longitude"]]
                   .mean().reset_index())

    # ── IPC crisis prevalence per livelihood zone ─────────────────────────
    ipc_zone = (ds.groupby("district_id")["target_crisis_binary"]
                  .mean().mul(100).round(1)
                  .rename("crisis_prev_pct").reset_index())

    zone_ipc = zone_coords.merge(ipc_zone, on="district_id", how="inner")

    # ── Cumulative articles per livelihood zone ───────────────────────────
    if MONTHLY_PATH.exists():
        mg = pd.read_parquet(MONTHLY_PATH)
        art_zone = (mg.groupby("district_id")["article_count"]
                      .sum().rename("total_articles").reset_index())
    else:
        art_zone = pd.DataFrame(columns=["district_id", "total_articles"])

    zone_art = zone_coords.merge(art_zone, on="district_id", how="inner")

    # ── GeoDataFrame of zone centroids ───────────────────────────────────
    def _zone_gdf(df, val_col):
        return gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat)
                      for lat, lon in zip(df["avg_latitude"], df["avg_longitude"])],
            crs="EPSG:4326"
        )

    gdf_ipc_zones = _zone_gdf(zone_ipc, "crisis_prev_pct")
    gdf_art_zones = _zone_gdf(zone_art, "total_articles")

    # ── Nearest-neighbour join: assign each study ADM2 the value of its
    #    closest livelihood-zone centroid (fills all 2634 study ADM2s) ─────
    adm2_pts = study_shp.copy()
    adm2_pts["geometry"] = study_shp.geometry.centroid  # ADM2 centroid for matching

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipc_nn  = gpd.sjoin_nearest(
            adm2_pts[["GID_2", "geometry"]],
            gdf_ipc_zones[["crisis_prev_pct", "geometry"]],
            how="left"
        )[["GID_2", "crisis_prev_pct"]].drop_duplicates("GID_2")

        art_nn  = gpd.sjoin_nearest(
            adm2_pts[["GID_2", "geometry"]],
            gdf_art_zones[["total_articles", "geometry"]],
            how="left"
        )[["GID_2", "total_articles"]].drop_duplicates("GID_2")

    # Re-attach geometry for plotting
    gdf_ipc_plot = study_shp.merge(ipc_nn,  on="GID_2", how="left")
    gdf_art_plot = study_shp.merge(art_nn,  on="GID_2", how="left")

    print(f"  IPC: {gdf_ipc_plot['crisis_prev_pct'].notna().sum()} / {len(study_shp)} ADM2 polygons filled")
    print(f"  News: {gdf_art_plot['total_articles'].notna().sum()} / {len(study_shp)} ADM2 polygons filled")

    # ── Country boundaries for borders + labels ──────────────────────────
    # Use the full Africa ADM0 basemap (already loaded above)
    countries_gdf = africa_base.copy()

    # Manual label nudges for overlapping/small countries (lon_offset, lat_offset)
    LABEL_NUDGE = {
        "Burundi":                       ( 1.5,  0.0),
        "Rwanda":                        ( 1.5,  0.5),
        "Malawi":                        ( 1.2,  0.0),
        "Swaziland":                     ( 1.5,  0.0),
        "South Sudan":                   ( 0.0,  0.5),
        "Democratic Republic of the Congo": (-1.5, -1.5),
        "Central African Republic":      ( 0.0,  0.5),
        "Mauritania":                    ( 0.0,  0.5),
    }
    # Short display names for cramped countries
    LABEL_SHORT = {
        "Democratic Republic of the Congo": "DR Congo",
        "Central African Republic":         "C.A.R.",
    }

    # ADM0 name column is "COUNTRY" in the basemap file
    _name_col = "COUNTRY" if "COUNTRY" in countries_gdf.columns else "country_name"

    def _add_country_labels(ax):
        """Draw country name labels at centroid positions."""
        import warnings
        for _, row in countries_gdf.iterrows():
            name = row[_name_col]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                centroid = row.geometry.centroid
            cx, cy = centroid.x, centroid.y
            dx, dy = LABEL_NUDGE.get(name, (0.0, 0.0))
            label  = LABEL_SHORT.get(name, name)
            ax.text(cx + dx, cy + dy, label,
                    fontsize=5.2, ha="center", va="center",
                    color="#222222", fontweight="normal",
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              alpha=0.60, edgecolor="none"),
                    zorder=7, clip_on=True)

    # ── Cartographic helper ──────────────────────────────────────────────
    def _draw_map(ax, gdf_all, gdf_data, col, cmap_listed, norm,
                  legend_patches):
        """Plot one choropleth panel on ax.

        gdf_all  — full ADM2 shapefile (used for study-region context only)
        gdf_data — study-country ADM2 polygons, all filled via nearest-neighbour
        africa_base — full 54-country Africa ADM0 for continent backdrop
        """
        # Layer 1 — full Africa ADM0 backdrop (all 54 countries, pale grey)
        africa_base.plot(ax=ax, color="#E8E8E8", linewidth=0.3,
                         edgecolor="#AAAAAA", zorder=1)
        # Layer 2 — study ADM2 polygons with choropleth fill
        gdf_has = gdf_data[gdf_data[col].notna()].copy()
        gdf_nos = gdf_data[gdf_data[col].isna()].copy()
        if len(gdf_nos):
            gdf_nos.plot(ax=ax, color="#CCCCCC", linewidth=0.1,
                         edgecolor="#BBBBBB", zorder=2)
        gdf_has.plot(ax=ax, column=col, cmap=cmap_listed, norm=norm,
                     linewidth=0.1, edgecolor="white", zorder=3)
        # Layer 3 — country borders (full Africa, drawn over choropleth)
        africa_base.plot(ax=ax, color="none", linewidth=0.7,
                         edgecolor="#555555", zorder=4)

        ax.set_axis_off()
        ax.set_xlim(-20, 52)
        ax.set_ylim(-35, 23)

        # Country name labels
        _add_country_labels(ax)


        # North arrow
        ax.annotate("", xy=(0.08, 0.19), xytext=(0.08, 0.12),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#333333",
                                    lw=1.8, mutation_scale=13))
        ax.text(0.08, 0.10, "N", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, fontweight="bold",
                color="#333333")

        # Legend
        legend_patches.append(
            mpatches.Patch(facecolor="#D8D8D8", edgecolor="#CCCCCC",
                           label="Not in study"))
        ax.legend(handles=legend_patches, loc="lower left", fontsize=7.5,
                  framealpha=0.96, edgecolor="#CCCCCC",
                  handlelength=1.2, handleheight=1.0,
                  borderpad=0.7, labelspacing=0.4)

    # ── Fig 1a — IPC crisis prevalence (%) ──────────────────────────────
    ipc_bins   = [0, 10, 25, 50, 75, 100]
    ipc_colors = ["#FFF7BC", "#FED976", "#FC4E2A", "#BD0026", "#67000D"]
    ipc_cmap   = ListedColormap(ipc_colors)
    ipc_norm   = BoundaryNorm(ipc_bins, ipc_cmap.N)
    ipc_labels = ["0-10 %", "10-25 %", "25-50 %", "50-75 %", "75-100 %"]
    ipc_patches = [mpatches.Patch(facecolor=ipc_colors[i], edgecolor="#999999",
                                  linewidth=0.4, label=ipc_labels[i])
                   for i in range(len(ipc_labels))]
    ipc_patches.insert(0, mpatches.Patch(visible=False,
                                          label="Crisis prevalence\n(% of periods)"))

    fig1a, ax1a = plt.subplots(figsize=(7.5, 9))
    _draw_map(ax1a, gdf, gdf_ipc_plot, "crisis_prev_pct",
              ipc_cmap, ipc_norm, ipc_patches)
    fig1a.tight_layout(pad=0.2)
    save_pdf(fig1a, "fig1a_ipc_choropleth")

    # ── Fig 1b — cumulative news articles ────────────────────────────────
    if gdf_art_plot["total_articles"].notna().sum() > 0:
        news_vals = gdf_art_plot["total_articles"].dropna()
        raw_qs    = np.nanpercentile(news_vals, [0, 20, 40, 60, 80, 100])
        news_bins = sorted(set(int(v) for v in raw_qs))
        if len(news_bins) < 3:
            news_bins = [int(news_vals.min()), int(news_vals.median()),
                         int(news_vals.max()) + 1]
        n_bins    = len(news_bins) - 1
        blues     = ["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B"][:n_bins]
        news_cmap = ListedColormap(blues)
        news_norm = BoundaryNorm(news_bins, news_cmap.N)

        def _fmt_k(v):
            return f"{v//1000}k" if v >= 1000 else str(v)
        news_labels  = [f"{_fmt_k(news_bins[i])}-{_fmt_k(news_bins[i+1])}"
                        for i in range(n_bins)]
        news_patches = [mpatches.Patch(facecolor=blues[i], edgecolor="#999999",
                                       linewidth=0.4, label=news_labels[i])
                        for i in range(n_bins)]
        news_patches.insert(0, mpatches.Patch(visible=False,
                                               label="Cumulative articles\n(2021-2024)"))

        gdf_art_plot["total_articles_plot"] = gdf_art_plot["total_articles"].clip(
            upper=news_bins[-1] - 1)

        fig1b, ax1b = plt.subplots(figsize=(7.5, 9))
        _draw_map(ax1b, gdf, gdf_art_plot, "total_articles_plot",
                  news_cmap, news_norm, news_patches)
        fig1b.tight_layout(pad=0.2)
        save_pdf(fig1b, "fig1b_news_choropleth")
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

    themes       = _load_themes(ds)
    rel_cols     = [f"{t}_relative_coverage" for t in themes]
    theme_labels = [t.replace("_", " ").title() for t in themes]

    cmap5 = ListedColormap(["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B"])
    cmap5.set_bad(color="#D0D0D0")

    def _quintile_bin(arr2d):
        vals = arr2d[np.isfinite(arr2d) & (arr2d > 0)]
        qs   = np.nanpercentile(vals, [20, 40, 60, 80])
        return np.digitize(arr2d, qs, right=True)  # 0..4

    def _quintile_boundaries(arr2d):
        """Return the 4 boundary values used for labelling the colorbar."""
        vals = arr2d[np.isfinite(arr2d) & (arr2d > 0)]
        return np.nanpercentile(vals, [20, 40, 60, 80])

    def _fmt_val(v):
        """Format a coverage value for colorbar tick labels."""
        if v >= 0.01:
            return f"{v:.2f}"
        return f"{v:.4f}"

    def _colorbar_with_values(fig, im, ax, arr2d, shrink=0.85, pad=0.015):
        """Add a colorbar showing actual quintile boundary values."""
        boundaries = _quintile_boundaries(arr2d)
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4],
                          pad=pad, shrink=shrink)
        tick_labels = [
            f"Q1\n(<{_fmt_val(boundaries[0])})",
            f"Q2\n(<{_fmt_val(boundaries[1])})",
            f"Q3\n(<{_fmt_val(boundaries[2])})",
            f"Q4\n(<{_fmt_val(boundaries[3])})",
            f"Q5\n(≥{_fmt_val(boundaries[3])})",
        ]
        cb.ax.set_yticklabels(tick_labels, fontsize=7.5)
        cb.set_label("Relative coverage (quintile)", fontsize=9)
        return cb

    # Shared tick positions for both heatmaps
    pivot_a = (ds.groupby("period_label")[rel_cols].mean().sort_index())
    pivot_a.columns = theme_labels
    periods   = list(pivot_a.index)
    n_periods = len(periods)
    tick_step = 3
    tick_pos  = list(range(0, n_periods, tick_step))
    tick_lbl  = [periods[i] for i in tick_pos]

    # ── Fig 2a — theme × time-period ─────────────────────────────────────
    # Rows = themes, Columns = time periods  (themes on y, time on x)
    binned_a = _quintile_bin(pivot_a.values)

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
    _colorbar_with_values(fig, im, ax, pivot_a.values, shrink=0.85, pad=0.015)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig2a_topic_heatmap")

    # ── Fig 2b — geography (x-axis) × theme (y-axis), grouped by region ──
    # Each cell = mean relative coverage for that country averaged across all themes
    # Axes: x = countries (grouped by region), y = themes
    ds2 = ds.copy()
    country_order = []
    for r in REGION_ORDER:
        cs = sorted(c for c in ds2["ipc_country"].unique()
                    if COUNTRY_REGION.get(c) == r)
        country_order.extend(cs)
    other = [c for c in ds2["ipc_country"].unique() if c not in country_order]
    country_order.extend(sorted(other))

    # pivot_b: index=country, columns=theme, values=mean relative coverage
    pivot_b = (ds2.groupby("ipc_country")[rel_cols].mean())
    pivot_b.columns = theme_labels
    pivot_b = pivot_b.reindex([c for c in country_order if c in pivot_b.index])

    raw_b    = pivot_b.values.astype(float)   # shape: (n_countries, n_themes)
    binned_b = np.where(np.isfinite(raw_b), _quintile_bin(raw_b), np.nan)
    masked_b = np.ma.masked_invalid(binned_b)

    n_countries = len(pivot_b)
    region_sizes = []
    for r in REGION_ORDER:
        n = sum(1 for c in pivot_b.index if COUNTRY_REGION.get(c) == r)
        if n:
            region_sizes.append((r, n))

    # imshow: rows=countries, cols=themes → imshow(masked_b) has shape (n_countries, n_themes)
    # We want x=geography (countries), y=themes → transpose so rows=themes, cols=countries
    fig, ax = plt.subplots(figsize=(max(10, n_countries * 0.55), 5.5))
    ax.grid(False)
    im = ax.imshow(masked_b.T, aspect="auto", cmap=cmap5, vmin=0, vmax=4,
                   interpolation="nearest")

    # x-axis: countries
    ax.set_xticks(range(n_countries))
    ax.set_xticklabels(pivot_b.index.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Country (grouped by region)", labelpad=6)

    # y-axis: themes
    ax.set_yticks(range(len(theme_labels)))
    ax.set_yticklabels(theme_labels, fontsize=9)
    ax.set_ylabel("News theme", labelpad=6)
    ax.spines[:].set_visible(False)

    # Region separator lines on x-axis (vertical)
    cumulative = 0
    region_mid_x = []
    region_names = []
    for rname, rsize in region_sizes:
        boundary = cumulative + rsize
        if boundary < n_countries:
            ax.axvline(boundary - 0.5, color="black", lw=1.2, ls="-")
        region_mid_x.append(cumulative + rsize / 2 - 0.5)
        region_names.append(rname)
        cumulative = boundary

    # Region labels above the x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(region_mid_x)
    ax2.set_xticklabels(region_names, fontsize=8.5, color="#333333",
                        fontweight="bold", rotation=0)
    ax2.tick_params(top=False, labeltop=True)
    ax2.spines[:].set_visible(False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    boundaries_b = _quintile_boundaries(raw_b)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.25)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels([
        f"Q1\n(<{_fmt_val(boundaries_b[0])})",
        f"Q2\n(<{_fmt_val(boundaries_b[1])})",
        f"Q3\n(<{_fmt_val(boundaries_b[2])})",
        f"Q4\n(<{_fmt_val(boundaries_b[3])})",
        f"Q5\n(≥{_fmt_val(boundaries_b[3])})",
    ], fontsize=7.5)
    cb.set_label("Relative coverage (quintile)", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.22)
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
    # Column renamed from mean_pr_auc to mean_full_pr_auc in the updated shuffle test
    pr_col  = "mean_full_pr_auc" if "mean_full_pr_auc" in null_df.columns else "mean_pr_auc"
    null_pr = null_df[pr_col].values
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
            color="#AAAACC", edgecolor="white", lw=0.5)
    ax.axvline(real_ar,   color=MODEL_COLOURS["AR-Only"], lw=2.2, ls="--")
    ax.axvline(real_full, color=MODEL_COLOURS["AR+News"],  lw=2.2, ls="-")

    # Null mean ± 1 SD band
    ax.axvspan(null_mean - null_std, null_mean + null_std,
               alpha=0.12, color="#666688")
    ax.axvline(null_mean, color="#666688", lw=1.0, ls=":")

    # Effect size annotation
    eff_txt = (f"z(AR-Only) = {z_ar:.1f}\n"
               f"z(AR+News) = {z_full:.1f}\n"
               f"Null: {null_mean:.3f} ± {null_std:.3f}")
    _annotation_box(ax, eff_txt, x=0.02, y=0.97, ha="left", va="top", fs=8)

    ax.set_xlabel("Mean PR-AUC (across folds)", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
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
            color="#AAAACC", edgecolor="white", lw=0.5)
    ax.axvline(real_ar_roc,   color=MODEL_COLOURS["AR-Only"], lw=2.2, ls="--")
    ax.axvline(real_full_roc, color=MODEL_COLOURS["AR+News"],  lw=2.2, ls="-")
    ax.set_xlabel("Mean ROC-AUC (across folds)", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
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

        ax.plot(x, y_ar,   "o-", color=MODEL_COLOURS["AR-Only"],
                label="AR Only", lw=2.0, ms=8, zorder=4)
        ax.plot(x, y_full, "s-", color=MODEL_COLOURS["AR+News"],
                label="AR + News", lw=2.0, ms=8, zorder=4)

        # Shaded area between models (no legend entry)
        ax.fill_between(x, y_ar, y_full,
                        where=(y_full >= y_ar),
                        alpha=0.12, color=MODEL_COLOURS["AR+News"],
                        interpolate=True)
        ax.fill_between(x, y_ar, y_full,
                        where=(y_full < y_ar),
                        alpha=0.12, color="#888888",
                        interpolate=True)

        # n_test labels above the higher of the two lines
        n_vals = fold_df["n_test"].values
        for xi, ya, yf, ni in zip(x, y_ar, y_full, n_vals):
            y_top = max(ya, yf)
            ax.annotate(f"n={ni}", (xi, y_top),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=7, ha="center", color="#555555")

        ax.set_xlabel("Test period (start)", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8.5, loc="lower left")
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
        arrow_col = "green" if row.prauc_full >= row.prauc_ar else "red"
        ax.annotate("",
                    xy     =(row.prauc_full, i),
                    xytext =(row.prauc_ar,   i),
                    arrowprops=dict(arrowstyle="->", color=arrow_col,
                                    lw=2, mutation_scale=15))
        ax.scatter(row.prauc_ar,   i, color=MODEL_COLOURS["AR-Only"],
                   s=100, zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter(row.prauc_full, i, marker="s", color=MODEL_COLOURS["AR+News"],
                   s=100, zorder=5, edgecolors="white", linewidths=0.5)
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

    legend_elems = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=MODEL_COLOURS["AR-Only"], ms=10, label="AR Only"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor=MODEL_COLOURS["AR+News"],  ms=10, label="AR + News"),
        Line2D([0],[0], color="green", lw=2, label="Improvement"),
        Line2D([0],[0], color="red",   lw=2, label="Degradation"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right", frameon=True)
    _despine(ax)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig6a_country_prauc")



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

    def _scatter_panel(ax, x_arr, y_arr, xlabel, ylabel, add_hline=None):
        """Scatter with single neutral colour and regression overlay."""
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.5)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        ax.scatter(x_arr[mask], y_arr[mask],
                   color="#555555", alpha=0.6, s=60, edgecolors="white",
                   linewidths=0.5, zorder=3)
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
    _scatter_panel(ax, x7a, y7a,
                   r"$\log_{10}$(Mean articles month$^{-1}$)",
                   r"$\Delta$PR-AUC  =  PR-AUC(AR+News) $-$ PR-AUC(AR-Only)",
                   add_hline=0.0)
    ax.set_ylim(y7a[np.isfinite(y7a)].min() - 0.05,
                y7a[np.isfinite(y7a)].max() + 0.10)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig7a_articles_vs_delta")

    # ── Fig 7b — volatility vs AR-Only PR-AUC ────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x7b = dm["volatility"].values
    y7b = dm["prauc_ar"].values
    _scatter_panel(ax, x7b, y7b,
                   "Volatility  (fraction of periods with regime change)",
                   "PR-AUC — AR-Only")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, "fig7b_volatility_vs_prauc")

    # ── Fig 7c — onset+chronic count vs AR-Only PR-AUC ───────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x7c = dm["onset_chronic_count"].values.astype(float)
    y7c = dm["prauc_ar"].values
    _scatter_panel(ax, x7c, y7c,
                   "Onset + Chronic observations per district",
                   "PR-AUC — AR-Only")
    ax.set_xlim(-0.3, x7c[np.isfinite(x7c)].max() + 0.5)
    ax.set_ylim(-0.02, 1.05)
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

        arrow_col = "green" if row.prauc_full >= row.prauc_ar else "red"
        ax_prauc.annotate("",
                          xy    =(row.prauc_full, i),
                          xytext=(row.prauc_ar,   i),
                          arrowprops=dict(arrowstyle="->", color=arrow_col,
                                          lw=2, mutation_scale=15))
        ax_prauc.scatter(row.prauc_ar,   i, color=MODEL_COLOURS["AR-Only"],
                         s=100, zorder=5, edgecolors="white", linewidths=0.5)
        ax_prauc.scatter(row.prauc_full, i, marker="s", color=MODEL_COLOURS["AR+News"],
                         s=100, zorder=5, edgecolors="white", linewidths=0.5)
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

    legend_elems = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=MODEL_COLOURS["AR-Only"], ms=10, label="AR Only"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor=MODEL_COLOURS["AR+News"], ms=10, label="AR + News"),
        Line2D([0],[0], color="green", lw=2, label="Improvement"),
        Line2D([0],[0], color="red",   lw=2, label="Degradation"),
    ]
    ax_prauc.legend(handles=legend_elems, fontsize=8, loc="lower right", frameon=True)

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
