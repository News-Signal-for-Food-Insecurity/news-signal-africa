"""
build_dataset.py
================
Reconstructs the three pipeline inputs from the two raw source files:

  DATA/raw/stage1_features.parquet       (44,435 district-period rows; IPC phases 2020-2024,
                                          temporal lag Lt, spatial lag Ls, y_h8 target)
  DATA/raw/ml_dataset_monthly.parquet    (district-month rows; 9 news-theme
                                          category counts, article_count for filtering; 2021-2024)

Outputs (DATA/):
  filtering/strict_filtered_districts.csv  -- districts with >= 200 mean articles/month
  filtering/district_coverage_stats.csv    -- full coverage stats for all districts
  dataset.parquet                          -- final merged modelling dataset (22 features + target)
  modelling/monthly_gdelt_features.parquet -- monthly news counts used by fold-aware
                                             z-score recomputation inside 01_train_models.py

Paper features produced
-----------------------
AR features (4):
  ipc_lag_1          -- binary IPC-crisis lag 1 period (from ipc_binary_crisis in stage1)
  ipc_persistence_2yr -- 6-period rolling mean of lagged binary crisis
  spatial_lag        -- IDW-weighted neighbour IPC phase (Ls from stage1)
  ipc_period         -- quarter of ipc_period_start (categorical; Q1/Q2/Q3/Q4)

News features (18):
  {theme}_relative_coverage  (9) -- share of monthly news for each theme
  {theme}_zscore             (9) -- fold-NAIVE global rolling z-score (pre-computed
                                   here for reference; fold-aware version is recomputed
                                   inside 01_train_models.py using the monthly file)

Target:
  target_crisis_binary -- IPC Phase >= 3 at t+8 months (= y_h8 from stage1)

Run
---
  python data_preparation/build_dataset.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "DATA"
RAW_DIR  = DATA_DIR / "raw"

STRICT_THRESHOLD   = 75    # mean articles/month (computed over non-zero months)
THEMES = [
    "conflict", "displacement", "economic", "food_security",
    "governance", "health", "humanitarian", "weather", "other",
]
THEME_COLS = [f"{t}_category" for t in THEMES]
EPSILON    = 1e-10


# -----------------------------------------------------------------------------
# 1. DISTRICT FILTER
# -----------------------------------------------------------------------------
def build_district_filter(df_monthly: pd.DataFrame) -> list[str]:
    print("\n-- 1. District filter -----------------------------------------")
    stats = (
        df_monthly.groupby("ipc_geographic_unit_full")["article_count"]
        .agg(["mean", "median", "count", "sum"])
        .reset_index()
        .rename(columns={
            "ipc_geographic_unit_full": "district",
            "mean":   "mean_articles_per_month",
            "median": "median_articles_per_month",
            "count":  "n_months_observed",
            "sum":    "total_articles",
        })
        .sort_values("mean_articles_per_month", ascending=False)
        .reset_index(drop=True)
    )

    strict = stats[stats["mean_articles_per_month"] >= STRICT_THRESHOLD].copy()
    print(f"   Total districts with any news : {len(stats):,}")
    print(f"   Strict (>= {STRICT_THRESHOLD}/month)            : {len(strict):,}")

    out_dir = DATA_DIR / "filtering"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Coverage threshold sensitivity table
    sensitivity_rows = []
    for t in [50, 75, 100, 125, 150, 200]:
        n = int((stats["mean_articles_per_month"] >= t).sum())
        sensitivity_rows.append({"threshold": t, "n_districts": n})
    sens_df = pd.DataFrame(sensitivity_rows)
    print(f"\n   Coverage threshold sensitivity:")
    for _, r in sens_df.iterrows():
        marker = " <-- active" if r["threshold"] == STRICT_THRESHOLD else ""
        print(f"     >= {int(r['threshold']):3d}/month: {int(r['n_districts']):4d} districts{marker}")
    sens_df.to_csv(out_dir / "coverage_threshold_sensitivity.csv", index=False)
    print(f"   Saved  filtering/coverage_threshold_sensitivity.csv")

    strict[["district", "mean_articles_per_month", "n_months_observed"]].to_csv(
        out_dir / "strict_filtered_districts.csv", index=False
    )
    stats.to_csv(out_dir / "district_coverage_stats.csv", index=False)
    print(f"   Saved  filtering/strict_filtered_districts.csv")
    print(f"   Saved  filtering/district_coverage_stats.csv")

    return strict["district"].tolist()


# -----------------------------------------------------------------------------
# 2. AR FEATURES
# -----------------------------------------------------------------------------
def build_ar_features(df_s1: pd.DataFrame, strict_districts: list[str]) -> pd.DataFrame:
    """
    Derives the four AR features directly from stage1_features.parquet.

    stage1_features columns used:
      ipc_geographic_unit_full -- district identifier (matches filtering list)
      ipc_period_start         -- 4-month period start date
      ipc_binary_crisis        -- 1 if IPC Phase >= 3, else 0
      Ls                       -- pre-computed inverse-distance-weighted spatial lag
                                  (300km radius, already in stage1)
      y_h8                     -- binary target: crisis 8 months ahead
      quarter                  -- calendar quarter of the period (used as ipc_period)
    """
    print("\n-- 2. AR features ---------------------------------------------")

    df = df_s1.copy()
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])

    # Apply district filter
    df = df[df["ipc_geographic_unit_full"].isin(strict_districts)].copy()
    print(f"   After district filter: {len(df):,} rows, "
          f"{df['ipc_geographic_unit_full'].nunique():,} districts")

    # Sort chronologically within each district
    df = df.sort_values(["ipc_geographic_unit_full", "ipc_period_start"]).reset_index(drop=True)

    grp = df.groupby("ipc_geographic_unit_full")

    # ipc_lag_1: binary crisis one period prior
    df["ipc_lag_1"] = grp["ipc_binary_crisis"].shift(1)

    # ipc_persistence_2yr: 6-period rolling mean of the lagged binary (shifted 1)
    df["ipc_persistence_2yr"] = grp["ipc_binary_crisis"].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=3).mean()
    )

    # spatial_lag: Ls is the IDW-weighted IPC phase of neighbours within 300 km,
    # computed at the SAME period as the observation. This is not leakage: the model
    # predicts crisis 8 months ahead (y_h8), and Ls reflects currently observed
    # neighbour conditions — analogous to ipc_lag_1 but in the spatial dimension.
    df["spatial_lag"] = df["Ls"]

    # ipc_period: quarter as string categorical (Q1 / Q2 / Q3 / Q4)
    df["ipc_period"] = "Q" + df["ipc_period_start"].dt.quarter.astype(str)

    # target
    df["target_crisis_binary"] = df["y_h8"].astype(float)

    # Drop rows without the primary AR lag (first period per district has no lag)
    before = len(df)
    df = df.dropna(subset=["ipc_lag_1"]).reset_index(drop=True)
    print(f"   Dropped {before - len(df):,} rows missing ipc_lag_1 (first-period per district)")

    # Fill persistence NaN for districts with very few periods
    df["ipc_persistence_2yr"] = df["ipc_persistence_2yr"].fillna(df["ipc_lag_1"])

    # Drop rows without a target (no IPC record 8 months ahead)
    before = len(df)
    df = df.dropna(subset=["target_crisis_binary"]).reset_index(drop=True)
    print(f"   Dropped {before - len(df):,} rows missing y_h8 target")

    print(f"   Final AR dataset: {len(df):,} rows, "
          f"{df['ipc_geographic_unit_full'].nunique():,} districts")
    print(f"   Crisis prevalence: {df['target_crisis_binary'].mean()*100:.1f}%")

    keep = [
        "ipc_geographic_unit_full", "ipc_country", "ipc_district",
        "ipc_period_start",
        "ipc_lag_1", "ipc_persistence_2yr", "spatial_lag", "ipc_period",
        "target_crisis_binary",
    ]
    return df[keep].copy()


# -----------------------------------------------------------------------------
# 3. MONTHLY NEWS FEATURES  (for fold-aware recomputation in training)
# -----------------------------------------------------------------------------
def build_monthly_gdelt_features(df_monthly: pd.DataFrame, strict_districts: list[str]) -> pd.DataFrame:
    """
    Produces the monthly-granularity file used by 01_train_models.py for
    fold-aware z-score recomputation. Keeps only the 9 relative_coverage
    columns plus identifiers and article_count -- no pre-baked z-scores
    (those are computed inside each fold using the training window only).
    """
    print("\n-- 3. Monthly GDELT features (for fold-aware CV) --------------")

    df = df_monthly[df_monthly["ipc_geographic_unit_full"].isin(strict_districts)].copy()
    df["month"] = pd.to_datetime(df["year_month"])

    # Relative coverage (9 features) -- computed at the monthly level
    total = df[THEME_COLS].sum(axis=1).clip(lower=EPSILON)
    for theme, col in zip(THEMES, THEME_COLS):
        df[f"{theme}_relative_coverage"] = df[col] / total

    keep_cols = (
        ["ipc_geographic_unit_full", "district_id" if "district_id" in df.columns else "ipc_geographic_unit_full",
         "month", "article_count"]
        + THEME_COLS
        + [f"{t}_relative_coverage" for t in THEMES]
    )
    # district_id alias
    df["district_id"] = df["ipc_geographic_unit_full"]
    keep = ["district_id", "month", "article_count"] + THEME_COLS + [f"{t}_relative_coverage" for t in THEMES]

    out = df[keep].copy().reset_index(drop=True)
    print(f"   Monthly features: {len(out):,} rows, "
          f"{out['district_id'].nunique():,} districts")
    print(f"   Date range: {out['month'].min().strftime('%Y-%m')} -> "
          f"{out['month'].max().strftime('%Y-%m')}")
    return out


# -----------------------------------------------------------------------------
# 4. MERGE -> FINAL DATASET
# -----------------------------------------------------------------------------
def build_final_dataset(ar_df: pd.DataFrame, df_monthly: pd.DataFrame,
                        strict_districts: list[str]) -> pd.DataFrame:
    """
    Merges AR features with period-averaged news features to produce
    DATA/dataset.parquet -- the single file consumed by all training scripts.

    News features are computed at the 4-month IPC-period level by averaging
    the 3-4 monthly values that fall within each IPC period window
    (Feb-May, Jun-Sep, Oct-Jan).  The fold-aware per-fold z-score recomputation
    inside 01_train_models.py uses the raw monthly counts file instead, so
    the z-scores here are for reference/exploration only.
    """
    print("\n-- 4. Merge -> dataset.parquet ---------------------------------")

    # --- Map each monthly row to its IPC 4-month period ---
    df = df_monthly[df_monthly["ipc_geographic_unit_full"].isin(strict_districts)].copy()
    df["month_dt"] = pd.to_datetime(df["year_month"])

    def month_to_ipc_period(ts):
        m = ts.month
        y = ts.year
        if m in (2, 3, 4, 5):   return pd.Timestamp(y,  2, 1)
        if m in (6, 7, 8, 9):   return pd.Timestamp(y,  6, 1)
        if m >= 10:              return pd.Timestamp(y, 10, 1)
        return pd.Timestamp(y - 1, 10, 1)   # January -> previous Oct period

    df["ipc_period_start"] = df["month_dt"].apply(month_to_ipc_period)

    # Relative coverage per month
    total = df[THEME_COLS].sum(axis=1).clip(lower=EPSILON)
    for theme, col in zip(THEMES, THEME_COLS):
        df[f"{theme}_relative_coverage"] = df[col] / total

    # Log-transform + global 12-month rolling z-score (for reference column only)
    df = df.sort_values(["ipc_geographic_unit_full", "month_dt"])
    for theme, col in zip(THEMES, THEME_COLS):
        df[f"{theme}_log"] = np.log1p(df[col])
        grp = df.groupby("ipc_geographic_unit_full")[f"{theme}_log"]
        mu  = grp.transform(lambda x: x.rolling(12, min_periods=6).mean().shift(1))
        sig = grp.transform(lambda x: x.rolling(12, min_periods=6).std().shift(1))
        df[f"{theme}_zscore"] = ((df[f"{theme}_log"] - mu) / (sig + EPSILON)).fillna(0)

    # Aggregate monthly -> 4-month IPC period (mean for coverage/zscore, sum for article count)
    news_cols = (
        [f"{t}_relative_coverage" for t in THEMES] +
        [f"{t}_zscore" for t in THEMES]
    )
    news_period_coverage = (
        df.groupby(["ipc_geographic_unit_full", "ipc_period_start"])[news_cols]
        .mean()
        .reset_index()
    )
    article_count_period = (
        df.groupby(["ipc_geographic_unit_full", "ipc_period_start"])["article_count"]
        .sum()
        .reset_index()
        .rename(columns={"article_count": "article_count"})
    )
    news_period = news_period_coverage.merge(article_count_period,
                                             on=["ipc_geographic_unit_full", "ipc_period_start"],
                                             how="left")

    # Merge AR + news on district × period
    merged = ar_df.merge(
        news_period,
        left_on=["ipc_geographic_unit_full", "ipc_period_start"],
        right_on=["ipc_geographic_unit_full", "ipc_period_start"],
        how="inner",
    )

    print(f"   AR rows:   {len(ar_df):,}")
    print(f"   News rows: {len(news_period):,}")
    print(f"   Merged:    {len(merged):,} rows "
          f"({len(ar_df) - len(merged):,} AR rows had no matching news period)")

    # Rename for clarity
    merged = merged.rename(columns={"ipc_geographic_unit_full": "district_id"})

    # Final column order
    id_cols    = ["district_id", "ipc_country", "ipc_district", "ipc_period_start"]
    ar_feats   = ["ipc_lag_1", "ipc_persistence_2yr", "spatial_lag", "ipc_period"]
    news_feats = news_cols + ["article_count"]
    target_col = ["target_crisis_binary"]

    final = merged[id_cols + ar_feats + news_feats + target_col].copy()
    final = final.dropna(subset=ar_feats + target_col).reset_index(drop=True)

    print(f"   Final dataset: {len(final):,} rows, "
          f"{final['district_id'].nunique():,} districts")
    print(f"   Crisis prevalence: {final['target_crisis_binary'].mean()*100:.1f}%")
    print(f"   Columns: {len(final.columns)} "
          f"(4 AR + 18 news + 1 target + 4 identifiers)")

    return final


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BUILD DATASET -- news-signal-africa paper pipeline")
    print("=" * 70)

    # Load sources
    print("\nLoading raw sources …")
    df_s1 = pd.read_parquet(RAW_DIR / "stage1_features.parquet")
    print(f"  stage1_features:    {len(df_s1):,} rows")

    df_monthly = pd.read_parquet(RAW_DIR / "ml_dataset_monthly.parquet")
    print(f"  ml_dataset_monthly: {len(df_monthly):,} rows")

    # Step 1: District filter
    strict_districts = build_district_filter(df_monthly)

    # Step 2: AR features
    ar_df = build_ar_features(df_s1, strict_districts)

    # Step 3: Monthly features file (for fold-aware CV)
    monthly_feats = build_monthly_gdelt_features(df_monthly, strict_districts)
    modelling_dir = DATA_DIR / "modelling"
    modelling_dir.mkdir(exist_ok=True)
    monthly_feats.to_parquet(modelling_dir / "monthly_gdelt_features.parquet", index=False)
    print(f"   Saved  modelling/monthly_gdelt_features.parquet")

    # Step 4: Merge -> final dataset
    final_df = build_final_dataset(ar_df, df_monthly, strict_districts)
    final_df.to_parquet(DATA_DIR / "dataset.parquet", index=False)
    print(f"\n   Saved  DATA/dataset.parquet  ({len(final_df):,} rows × {len(final_df.columns)} cols)")

    # Summary JSON
    summary = {
        "n_rows": int(len(final_df)),
        "n_districts": int(final_df["district_id"].nunique()),
        "n_countries": int(final_df["ipc_country"].nunique()),
        "crisis_prevalence_pct": round(float(final_df["target_crisis_binary"].mean() * 100), 2),
        "period_start_min": str(final_df["ipc_period_start"].min().date()),
        "period_start_max": str(final_df["ipc_period_start"].max().date()),
        "strict_districts": len(strict_districts),
        "ar_features": ["ipc_lag_1", "ipc_persistence_2yr", "spatial_lag", "ipc_period"],
        "news_features": [f"{t}_relative_coverage" for t in THEMES] + [f"{t}_zscore" for t in THEMES],
        "target": "target_crisis_binary",
    }
    with open(DATA_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Dataset build complete.")
    print(f"  Rows              : {summary['n_rows']:,}")
    print(f"  Districts         : {summary['n_districts']:,}")
    print(f"  Countries         : {summary['n_countries']}")
    print(f"  Crisis prevalence : {summary['crisis_prevalence_pct']}%")
    print(f"  Period range      : {summary['period_start_min']} -> {summary['period_start_max']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
