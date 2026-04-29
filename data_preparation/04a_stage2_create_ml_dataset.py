"""
Stage 2: Create Monthly ML Dataset
Combines monthly articles and locations into unified dataset for z-score computation.

KEY DIFFERENCE FROM STAGE 1 (Script 04):
- Stage 1: Merges on IPC period keys (ipc_id, ipc_period_start, etc.)
- Stage 2: Merges on (ipc_geographic_unit_full, year_month)

This creates a monthly time series per district for rolling z-score computation.

Author: Victor Collins Oppon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths -- all relative to news-signal-africa repo root
BASE_DIR        = Path(__file__).parent.parent
INTERIM_DIR     = BASE_DIR / 'DATA' / 'interim'
STAGE2_DATA_DIR = INTERIM_DIR / 'stage2'

# Input files
ARTICLES_FILE = STAGE2_DATA_DIR / 'articles_aggregated_monthly.parquet'
LOCATIONS_FILE = STAGE2_DATA_DIR / 'locations_aggregated_monthly.parquet'

# Output files -- written to DATA/raw so build_dataset.py picks it up
OUTPUT_COMPLETE = BASE_DIR / 'DATA' / 'raw' / 'ml_dataset_monthly.parquet'
OUTPUT_CSV      = STAGE2_DATA_DIR / 'ml_dataset_monthly.csv'


def main():
    print("=" * 80)
    print("Stage 2: Create Monthly ML Dataset")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print("\nKEY DIFFERENCE from Stage 1:")
    print("  - Stage 1: Merges on IPC period keys")
    print("  - Stage 2: Merges on (ipc_geographic_unit_full, year_month)")

    # Load monthly aggregated articles
    print("\n1. Loading monthly aggregated articles...")
    if not ARTICLES_FILE.exists():
        raise FileNotFoundError(
            f"Articles file not found: {ARTICLES_FILE}\n"
            "Run 02a_stage2_aggregate_articles_monthly.py first."
        )

    articles = pd.read_parquet(ARTICLES_FILE)
    print(f"   Loaded {len(articles):,} article aggregations")
    print(f"   Unique districts: {articles['ipc_geographic_unit_full'].nunique():,}")
    print(f"   Unique months: {articles['year_month'].nunique()}")
    print(f"   Columns: {len(articles.columns)}")

    # Load monthly aggregated locations
    print("\n2. Loading monthly aggregated locations...")
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Locations file not found: {LOCATIONS_FILE}\n"
            "Run 03a_stage2_aggregate_locations_monthly.py first."
        )

    locations = pd.read_parquet(LOCATIONS_FILE)
    print(f"   Loaded {len(locations):,} location aggregations")
    print(f"   Unique districts: {locations['ipc_geographic_unit_full'].nunique():,}")
    print(f"   Unique months: {locations['year_month'].nunique()}")
    print(f"   Columns: {len(locations.columns)}")

    # Define merge keys for Stage 2 (district + month)
    merge_keys = [
        'ipc_country',
        'ipc_country_code',
        'ipc_fips_code',
        'ipc_district',
        'ipc_region',
        'ipc_geographic_unit',
        'ipc_geographic_unit_full',  # KEY: District identifier
        'ipc_fewsnet_region',
        'ipc_geographic_group',
        'year_month'  # KEY: Monthly time bucket
    ]

    # Verify merge keys exist in both datasets
    print("\n3. Verifying merge keys...")
    articles_keys = set(merge_keys) & set(articles.columns)
    locations_keys = set(merge_keys) & set(locations.columns)

    print(f"   Articles has {len(articles_keys)}/{len(merge_keys)} keys")
    print(f"   Locations has {len(locations_keys)}/{len(merge_keys)} keys")

    # Use common keys
    actual_merge_keys = list(articles_keys & locations_keys)
    print(f"   Using {len(actual_merge_keys)} merge keys")

    # Critical key validation
    critical_keys = ['ipc_geographic_unit_full', 'year_month']
    missing_critical = set(critical_keys) - set(actual_merge_keys)
    if missing_critical:
        raise ValueError(f"CRITICAL ERROR: Missing essential merge keys: {missing_critical}")

    # Create merged dataset (left join from articles — preserves only districts with news)
    print("\n4. Creating merged monthly dataset...")
    df_merged = articles.merge(
        locations,
        on=actual_merge_keys,
        how='left',
        suffixes=('_articles', '_locations')
    )

    # Fill missing values with 0 for count columns
    count_cols = [c for c in df_merged.columns if 'count' in c.lower() or 'article' in c.lower()]
    for col in count_cols:
        if df_merged[col].dtype in ['float64', 'int64']:
            df_merged[col] = df_merged[col].fillna(0)

    print(f"   Result: {len(df_merged):,} rows")
    print(f"   Unique districts: {df_merged['ipc_geographic_unit_full'].nunique():,}")
    print(f"   Unique months: {df_merged['year_month'].nunique()}")

    # Handle match_level columns from both sources
    if 'match_level_articles' in df_merged.columns and 'match_level_locations' in df_merged.columns:
        df_merged['match_level'] = df_merged['match_level_locations'].fillna(
            df_merged['match_level_articles']
        )
        df_merged = df_merged.drop(['match_level_articles', 'match_level_locations'], axis=1, errors='ignore')
    elif 'match_level_articles' in df_merged.columns:
        df_merged = df_merged.rename(columns={'match_level_articles': 'match_level'})
    elif 'match_level_locations' in df_merged.columns:
        df_merged = df_merged.rename(columns={'match_level_locations': 'match_level'})

    # Extract date components
    print("\n5. Adding date components...")
    df_merged['year_month_dt'] = pd.to_datetime(df_merged['year_month'].astype(str))
    df_merged['year'] = df_merged['year_month_dt'].dt.year
    df_merged['month'] = df_merged['year_month_dt'].dt.month

    # Sort by district and time for rolling computations
    df_merged = df_merged.sort_values(['ipc_geographic_unit_full', 'year_month']).reset_index(drop=True)

    # Summary statistics
    print("\n" + "=" * 80)
    print("Monthly Dataset Summary - Stage 2")
    print("=" * 80)

    print(f"\nTotal records: {len(df_merged):,}")
    print(f"Unique districts: {df_merged['ipc_geographic_unit_full'].nunique():,}")
    print(f"Unique months: {df_merged['year_month'].nunique()}")
    print(f"Month range: {df_merged['year_month'].min()} to {df_merged['year_month'].max()}")
    print(f"Countries: {df_merged['ipc_country'].nunique()}")

    print("\nRecords per country:")
    country_counts = df_merged.groupby('ipc_country').size().sort_values(ascending=False)
    for country, count in country_counts.head(10).items():
        districts = df_merged[df_merged['ipc_country'] == country]['ipc_geographic_unit_full'].nunique()
        print(f"   {country}: {count:,} records, {districts} districts")

    print("\nMatch level distribution:")
    if 'match_level' in df_merged.columns:
        print(df_merged['match_level'].value_counts())

    # Verify data quality
    print("\n" + "=" * 80)
    print("Data Quality Check")
    print("=" * 80)

    # Check for duplicates
    unique_check = df_merged.groupby(['ipc_geographic_unit_full', 'year_month']).size()
    duplicates = (unique_check > 1).sum()
    print(f"\n(district, month) duplicates: {duplicates}")
    print("(Expected: 0 - each observation should be unique)")

    if duplicates > 0:
        print("\nWARNING: Found duplicate observations. Deduplicating...")
        # Keep first occurrence
        df_merged = df_merged.drop_duplicates(subset=['ipc_geographic_unit_full', 'year_month'], keep='first')
        print(f"   After deduplication: {len(df_merged):,} rows")

    # Check months per district
    months_per_district = df_merged.groupby('ipc_geographic_unit_full')['year_month'].nunique()
    print(f"\nMonths per district:")
    print(f"   Min: {months_per_district.min()}")
    print(f"   Max: {months_per_district.max()}")
    print(f"   Mean: {months_per_district.mean():.1f}")
    print(f"   Median: {months_per_district.median():.1f}")

    # Check alignment with Stage 1 districts
    print("\n" + "=" * 80)
    print("Alignment with Stage 1")
    print("=" * 80)

    stage1_art_path = Path('C:/GDELT_Africa_Extract/data/district_level/stage1/articles_aggregated.parquet')
    if not stage1_art_path.exists():
        print("  Skipping alignment check: Stage 1 articles_aggregated.parquet not distributed with repo")
    else:
        stage1_articles = pd.read_parquet(stage1_art_path)
        stage1_districts = set(stage1_articles['ipc_geographic_unit_full'].unique())
        stage2_districts = set(df_merged['ipc_geographic_unit_full'].unique())

        overlap = stage1_districts & stage2_districts
        only_stage1 = stage1_districts - stage2_districts
        only_stage2 = stage2_districts - stage1_districts

        print(f"\nStage 1 districts: {len(stage1_districts):,}")
        print(f"Stage 2 districts: {len(stage2_districts):,}")
        print(f"Overlap: {len(overlap):,} ({100*len(overlap)/len(stage1_districts):.1f}% of Stage 1)")
        print(f"Only in Stage 1: {len(only_stage1):,}")
        print(f"Only in Stage 2: {len(only_stage2):,}")

    if len(only_stage1) > 0 and len(only_stage1) <= 10:
        print(f"\nDistricts only in Stage 1:")
        for d in list(only_stage1)[:10]:
            print(f"   {d}")

    # Save
    print(f"\n6. Saving monthly ML dataset...")
    print(f"   Parquet: {OUTPUT_COMPLETE}")
    df_merged.to_parquet(OUTPUT_COMPLETE, index=False)
    print("   [OK] Parquet saved")

    print(f"\n   CSV: {OUTPUT_CSV}")
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    # Print column summary
    print("\n" + "=" * 80)
    print("Column Summary")
    print("=" * 80)

    ipc_cols = [c for c in df_merged.columns if c.startswith('ipc_')]
    article_cols = [c for c in df_merged.columns if 'article' in c.lower() or 'category' in c.lower()]
    location_cols = [c for c in df_merged.columns if 'location' in c.lower() or 'latitude' in c.lower() or 'longitude' in c.lower()]
    time_cols = ['year', 'month', 'year_month', 'year_month_dt']

    print(f"\nIPC metadata columns: {len(ipc_cols)}")
    print(f"Article-derived columns: {len(article_cols)}")
    print(f"Location-derived columns: {len(location_cols)}")
    print(f"Total columns: {len(df_merged.columns)}")

    print("\n" + "=" * 80)
    print("Stage 2 Monthly ML Dataset Complete")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_COMPLETE}")
    print(f"\nNext step: Run 12_stage2_feature_engineering.py")
    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
