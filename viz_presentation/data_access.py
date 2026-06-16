"""
data_access.py — the single, canonical data layer for the visualization suite.

Every figure pulls its numbers through the functions here, which read only the
project's real artefacts:

  DATA/dataset.parquet                         modelling dataset (features + target)
  DATA/modelling/monthly_gdelt_features.parquet   monthly GDELT news counts
  DATA/raw/stage1_features.parquet             IPC reference + zone centroids
  results/window_2yr/fold_predictions.csv      AR / AR+News test predictions + regime
  results/window_2yr/news_only_predictions.csv News-only test predictions (generated)
  results/window_2yr/fold_results.csv          per-fold metrics
  results/shuffle_test_v3/null_distribution.csv   1,000 null-permutation scores
  results/shuffle_test_v3/config.json          null-test configuration / p-values
  DATA/shapefiles/gadm/africa_adm2_combined.gpkg   ADM2 district boundaries
  DATA/shapefiles/gadm/africa_adm0_basemap.gpkg    ADM0 country backdrop

There are no hardcoded results in this module. Every score, count, bin edge and
baseline is computed from the files above.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from viz_theme import COUNTRY_REGION

BASE_DIR = Path(__file__).resolve().parent.parent          # news-signal-africa/
DATA_DIR = BASE_DIR / "DATA"
RESULTS_DIR = BASE_DIR / "results" / "window_2yr"
SHUFFLE_DIR = BASE_DIR / "results" / "shuffle_test_v3"

DATASET_PATH = DATA_DIR / "dataset.parquet"
MONTHLY_PATH = DATA_DIR / "modelling" / "monthly_gdelt_features.parquet"
STAGE1_PATH = DATA_DIR / "raw" / "stage1_features.parquet"
FOLD_PRED_PATH = RESULTS_DIR / "fold_predictions.csv"
NEWS_PRED_PATH = RESULTS_DIR / "news_only_predictions.csv"
FOLD_RESULTS_PATH = RESULTS_DIR / "fold_results.csv"
NULL_DIST_PATH = SHUFFLE_DIR / "null_distribution.csv"
NULL_CONFIG_PATH = SHUFFLE_DIR / "config.json"
SHAPEFILE_ADM2 = DATA_DIR / "shapefiles" / "gadm" / "africa_adm2_combined.gpkg"
SHAPEFILE_ADM0 = DATA_DIR / "shapefiles" / "gadm" / "africa_adm0_basemap.gpkg"

THEMES = ["conflict", "displacement", "economic", "food_security", "governance",
          "health", "humanitarian", "weather", "other"]


# ---------------------------------------------------------------------------
# Core loaders (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])
    return df


@lru_cache(maxsize=1)
def load_monthly_gdelt() -> pd.DataFrame:
    df = pd.read_parquet(MONTHLY_PATH)
    df["month"] = pd.to_datetime(df["month"])
    return df


@lru_cache(maxsize=1)
def load_fold_results() -> pd.DataFrame:
    return pd.read_csv(FOLD_RESULTS_PATH)


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    """AR + AR+News test predictions, with prob_news joined if available."""
    fp = pd.read_csv(FOLD_PRED_PATH)
    fp["ipc_period_start"] = pd.to_datetime(fp["ipc_period_start"])
    if NEWS_PRED_PATH.exists():
        news = pd.read_csv(NEWS_PRED_PATH)
        news["ipc_period_start"] = pd.to_datetime(news["ipc_period_start"])
        fp = fp.merge(
            news[["district_id", "ipc_period_start", "fold_id", "prob_news"]],
            on=["district_id", "ipc_period_start", "fold_id"], how="left",
        )
    return fp


def has_news_only() -> bool:
    return NEWS_PRED_PATH.exists()


# ---------------------------------------------------------------------------
# Precision-Recall
#
# The study's evaluation protocol (01_train_models.py) is: compute PR-AUC
# (average_precision_score) on each fold's held-out test set, then report the
# mean ± SD across the 7 rolling-CV folds. Those per-fold PR-AUC values are
# persisted in the canonical result files, which are the SINGLE SOURCE OF TRUTH:
#   AR        -> results/window_2yr/fold_results.csv  ['ar_pr_auc']
#   AR+News   -> results/window_2yr/fold_results.csv  ['full_pr_auc']
#   News      -> results/window_2yr/news_only_fold_results.csv ['news_pr_auc']
# The figure reads its reported number straight from these; it never recomputes
# the metric. Only the curve GEOMETRY (the mean PR curve) is derived from the
# stored predictions, and it is the mean of the per-fold curves so that the
# plotted curve and the reported mean-of-folds metric are the same aggregation.
# ---------------------------------------------------------------------------
NEWS_FOLD_RESULTS_PATH = RESULTS_DIR / "news_only_fold_results.csv"

# model key -> (prediction column, per-fold PR-AUC source file, column)
_MODELS = {
    "news":     ("prob_news", NEWS_FOLD_RESULTS_PATH, "news_pr_auc"),
    "ar":       ("prob_ar", FOLD_RESULTS_PATH, "ar_pr_auc"),
    "combined": ("prob_combined", FOLD_RESULTS_PATH, "full_pr_auc"),
}


def prob_col(model_key: str) -> str:
    return _MODELS[model_key][0]


def fold_pr_auc(model_key: str) -> np.ndarray:
    """Per-fold PR-AUC for a model, read from its canonical result file."""
    _, path, col = _MODELS[model_key]
    return pd.read_csv(path)[col].to_numpy()


def reported_pr_auc(model_key: str) -> tuple[float, float]:
    """The study's reported PR-AUC as (mean, SD) across folds — from file.

    SD uses ddof=1, matching how the study reports it (e.g. AR+News
    0.8711 ± 0.0405).
    """
    vals = fold_pr_auc(model_key)
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), sd


def mean_fold_pr_curve(model_key: str, n_grid: int = 200):
    """Mean of the per-fold PR curves on a shared recall grid (geometry only).

    Returns (recall_grid, mean_precision, sd_precision). Each fold's curve is
    interpolated onto the grid, then averaged, so the drawn curve is the
    fold-mean curve — the same aggregation as the reported mean-of-folds PR-AUC.
    """
    col = prob_col(model_key)
    fp = load_predictions().dropna(subset=[col])
    grid = np.linspace(0.0, 1.0, n_grid)
    precisions = []
    for _, g in fp.groupby("fold_id"):
        y = g["target_crisis_binary"].to_numpy()
        p = g[col].to_numpy()
        if y.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        order = np.argsort(recall)                     # recall ascending
        precisions.append(np.interp(grid, recall[order], precision[order]))
    precisions = np.array(precisions)
    sd = precisions.std(axis=0, ddof=1) if len(precisions) > 1 else \
        np.zeros_like(grid)
    return grid, precisions.mean(axis=0), sd


def crisis_prevalence() -> float:
    """Baseline precision = crisis rate over all test observations."""
    fp = load_predictions()
    return float(fp["target_crisis_binary"].mean())


# ---------------------------------------------------------------------------
# Regimes (predictions scatter + delta-P violins)
# ---------------------------------------------------------------------------
def regime_counts() -> dict[str, int]:
    fp = load_predictions()
    return fp["regime"].value_counts().to_dict()


def delta_prob_by_regime() -> pd.DataFrame:
    """ΔP(crisis) = P(AR+News) − P(AR) with its regime label, per test row."""
    fp = load_predictions().copy()
    fp["delta"] = fp["prob_combined"] - fp["prob_ar"]
    return fp[["regime", "prob_ar", "prob_combined", "delta"]]


# ---------------------------------------------------------------------------
# Null distribution (slide 26)
# ---------------------------------------------------------------------------
def load_null_distribution() -> pd.DataFrame:
    return pd.read_csv(NULL_DIST_PATH)


def load_null_config() -> dict:
    with open(NULL_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def null_summary(metric: str = "pr_auc") -> dict:
    """Everything the histogram needs, all derived from files.

    The null distribution stores, for each permutation, the metric *averaged
    across the 7 folds*. To compare like-with-like, the AR / AR+News markers use
    the same mean-of-folds metric from fold_results.csv (this reproduces the
    talk's markers: AR 0.8346 / Median-NULL 0.8462 / AR+News 0.8711). The
    p-value is taken from the null-test config when present, else computed.
    """
    null = load_null_distribution()
    scores = null[metric].to_numpy()
    cfg = load_null_config()
    fr = load_fold_results()

    ar_col, full_col = f"ar_{metric}", f"full_{metric}"
    real_ar = float(fr[ar_col].mean())
    real_full = float(fr[full_col].mean())

    median_null = float(np.median(scores))
    p_key = "p_value_prauc" if metric == "pr_auc" else "p_value_rocauc"
    p_value = cfg.get(p_key)
    if p_value is None:
        p_value = float((np.sum(scores >= real_full) + 1) / (len(scores) + 1))
    return {
        "scores": scores,
        "n_models": int(len(scores)),
        "median_null": median_null,
        "real_ar": real_ar,
        "real_full": real_full,
        "p_value": float(p_value),
        "metric": metric,
    }


def _config_real(cfg: dict, keys: list[str]):
    """Search nested config dict for the first matching numeric key."""
    def _search(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in keys and isinstance(v, (int, float)):
                    return float(v)
                found = _search(v)
                if found is not None:
                    return found
        return None
    return _search(cfg)


# ---------------------------------------------------------------------------
# Heatmaps — mean relative coverage from the modelling dataset
# ---------------------------------------------------------------------------
def temporal_theme_matrix() -> pd.DataFrame:
    """Rows = themes, cols = IPC periods; values = mean relative coverage."""
    df = load_dataset()
    cols = [f"{t}_relative_coverage" for t in THEMES]
    mat = (df.groupby("ipc_period_start")[cols].mean().T)
    mat.index = [t.replace("_relative_coverage", "") for t in mat.index]
    mat.columns = [d.strftime("%Y-%m") for d in mat.columns]
    return mat


def spatial_theme_matrix() -> pd.DataFrame:
    """Rows = themes, cols = countries; values = mean relative coverage."""
    df = load_dataset()
    cols = [f"{t}_relative_coverage" for t in THEMES]
    mat = (df.groupby("ipc_country")[cols].mean().T)
    mat.index = [t.replace("_relative_coverage", "") for t in mat.index]
    return mat


def countries_by_region() -> list[tuple[str, list[str]]]:
    """Region -> [countries present in the data], in REGION_ORDER."""
    from viz_theme import REGION_ORDER
    present = set(load_dataset()["ipc_country"].unique())
    out = []
    for region in REGION_ORDER:
        members = sorted(c for c, r in COUNTRY_REGION.items()
                         if r == region and c in present)
        if members:
            out.append((region, members))
    return out


def quantile_bins(values: np.ndarray, q: list[float]) -> list[float]:
    """Strictly-increasing bin edges from data quantiles (no hardcoded edges)."""
    edges = np.nanquantile(values, q)
    uniq = sorted(set(np.round(edges, 6)))
    if len(uniq) < 2:
        uniq = [float(np.nanmin(values)), float(np.nanmax(values)) + 1e-9]
    return uniq


# ---------------------------------------------------------------------------
# Geospatial — reusable centroid -> ADM2 nearest-neighbour join (from Fig 1)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _zone_centroids() -> pd.DataFrame:
    """district_id -> mean (lat, lon) from stage1 zone coordinates."""
    s1 = pd.read_parquet(STAGE1_PATH)
    s1["district_id"] = s1["ipc_geographic_unit_full"].str.strip()
    coords = (s1[["district_id", "avg_latitude", "avg_longitude"]]
              .dropna()
              .query("avg_latitude != 0 or avg_longitude != 0"))
    return (coords.groupby("district_id")[["avg_latitude", "avg_longitude"]]
            .mean().reset_index())


def study_countries() -> set[str]:
    cname_fix = {"The Democratic Republic of the": "Democratic Republic of the Congo"}
    return {cname_fix.get(c, c) for c in load_dataset()["ipc_country"].unique()}


def choropleth_frame(value_by_district: pd.Series, value_name: str,
                     country: str | None = None):
    """Return (study_adm2_gdf_with_value, africa_adm0_gdf).

    ``value_by_district`` maps district_id -> value. Each study ADM2 polygon gets
    the value of its nearest livelihood-zone centroid (same method as Fig 1), so
    every polygon in the study area is filled. Optionally restrict to one country.
    """
    import warnings

    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(SHAPEFILE_ADM2)
    africa = gpd.read_file(SHAPEFILE_ADM0)

    study = study_countries()
    study_shp = gdf[gdf["country_name"].isin(study)].copy()
    if country is not None:
        study_shp = study_shp[study_shp["country_name"] == country].copy()

    cent = _zone_centroids().copy()
    vals = value_by_district.rename(value_name).reset_index()
    vals.columns = ["district_id", value_name]
    cent = cent.merge(vals, on="district_id", how="inner").dropna(subset=[value_name])

    zone_gdf = gpd.GeoDataFrame(
        cent,
        geometry=[Point(lon, lat)
                  for lat, lon in zip(cent["avg_latitude"], cent["avg_longitude"])],
        crs="EPSG:4326",
    )

    adm2_pts = study_shp.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm2_pts["geometry"] = study_shp.geometry.centroid
        nn = gpd.sjoin_nearest(
            adm2_pts[["GID_2", "geometry"]],
            zone_gdf[[value_name, "geometry"]], how="left",
        )[["GID_2", value_name]].drop_duplicates("GID_2")

    plot = study_shp.merge(nn, on="GID_2", how="left")
    return plot, africa


def district_article_totals() -> pd.Series:
    """Cumulative GDELT articles per district (news-volume choropleth)."""
    mg = load_monthly_gdelt()
    return mg.groupby("district_id")["article_count"].sum()


def district_crisis_prevalence() -> pd.Series:
    """% of periods a district is in crisis (IPC choropleth), 0-100."""
    df = load_dataset()
    return df.groupby("district_id")["target_crisis_binary"].mean().mul(100)


def country_adm2_with_nearest_district(country: str):
    """Return (study_adm2_gdf_for_country, africa_adm0_gdf) where each polygon
    carries the district_id of its nearest livelihood-zone centroid in a column
    ``nn_district``. The geometry->district map is independent of any value, so a
    caller can colour the same polygons by many different per-district series
    (e.g. one per quarter) without re-running the spatial join.
    """
    import warnings

    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(SHAPEFILE_ADM2)
    africa = gpd.read_file(SHAPEFILE_ADM0)
    study = gdf[gdf["country_name"] == country].copy()

    cent = _zone_centroids()
    zone_gdf = gpd.GeoDataFrame(
        cent,
        geometry=[Point(lon, lat)
                  for lat, lon in zip(cent["avg_latitude"], cent["avg_longitude"])],
        crs="EPSG:4326",
    )
    pts = study.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts["geometry"] = study.geometry.centroid
        nn = gpd.sjoin_nearest(
            pts[["GID_2", "geometry"]],
            zone_gdf[["district_id", "geometry"]], how="left",
        )[["GID_2", "district_id"]].drop_duplicates("GID_2")
    nn = nn.rename(columns={"district_id": "nn_district"})
    study = study.merge(nn, on="GID_2", how="left")
    africa_country = africa[africa["COUNTRY"] == country].copy()
    return study, africa_country


def nigeria_quarterly_food_security() -> pd.DataFrame:
    """Per-district Nigeria food-security relative coverage by calendar quarter.

    Columns are quarters (e.g. '2021-Q1'); index is district_id. Values are the
    food-security share of monthly news, averaged within the quarter.
    """
    mg = load_monthly_gdelt().copy()
    df = load_dataset()
    nigeria_ids = set(df.loc[df["ipc_country"] == "Nigeria", "district_id"])
    mg = mg[mg["district_id"].isin(nigeria_ids)].copy()
    mg["quarter"] = mg["month"].dt.to_period("Q").astype(str)
    out = (mg.groupby(["district_id", "quarter"])["food_security_relative_coverage"]
           .mean().unstack("quarter"))
    return out
