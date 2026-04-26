"""
Stage 2: Monthly Article Aggregation for Dynamic News Features
Aggregates GDELT articles by (district, year-month) using Stage 1 matching logic.

KEY DIFFERENCE FROM STAGE 1 (Script 02):
- Stage 1: Aggregates within IPC assessment periods (Feb, Jun, Oct - ~3/year)
- Stage 2: Aggregates by calendar month (all 12 months) for rolling z-score computation

CRITICAL: Uses the SAME district matching logic as Script 02 to ensure alignment
with Stage 1 predictions and AR failures.

Matching Priority: GADM3 → GADM2 → GADM1 → Country-level

Author: Victor Collins Oppon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from rapidfuzz import fuzz  # 10-100x faster than fuzzywuzzy
import pyarrow.parquet as pq
import pyarrow as pa
import unicodedata
import gc
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# Paths -- all relative to news-signal-africa repo root
BASE_DIR      = Path(__file__).parent.parent
RAW_DIR       = BASE_DIR / 'DATA' / 'raw'
INTERIM_DIR   = BASE_DIR / 'DATA' / 'interim'
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Input files
ARTICLES_FILE  = INTERIM_DIR / 'african_gkg_articles.csv'
LOCATIONS_FILE = INTERIM_DIR / 'african_gkg_locations_aligned.parquet'
IPC_REF_FILE   = RAW_DIR / 'ipc_reference.parquet'

# Stage 2 output
STAGE2_DATA_DIR = INTERIM_DIR / 'stage2'
STAGE2_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET  = STAGE2_DATA_DIR / 'articles_aggregated_monthly.parquet'
OUTPUT_CSV      = STAGE2_DATA_DIR / 'articles_aggregated_monthly.csv'

# Processing parameters
CHUNK_SIZE = 100000  # Reduced for 8GB RAM systems
FUZZY_THRESHOLD = 80  # Same as Stage 1
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core
SQLITE_BATCH_SIZE = 50000  # Batch size for SQLite inserts (memory-efficient)
DEDUP_BATCH_SIZE = 5000000  # 5M rows per dedup batch for memory safety
CONSOLIDATE_EVERY = 50  # Consolidate aggregations every 50 chunks
SQL_QUERY_BATCH_SIZE = 500  # Max SQL placeholders per query

# Global cache for fuzzy matches (populated at startup)
FUZZY_MATCH_CACHE = {}

# Countries with ONLY national-level IPC data (same as Stage 1)
COUNTRY_LEVEL_ONLY = {'AO', 'CG', 'CT', 'LT', 'MR', 'RW', 'TO'}


def normalize_text(text):
    """Normalize text for matching with accent removal. (Same as Stage 1)"""
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()
    # Remove accents (e.g., Kasaï -> kasai, Équateur -> equateur)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    text = ' '.join(text.split())
    return text


def find_fuzzy_match(loc_name_norm, country_candidates, country_code):
    """
    Find best fuzzy match for location name among IPC district candidates.
    Uses pre-grouped country_candidates for O(1) country lookup.
    Returns: (ipc_info_list, match_score) or (None, 0)
    (Same as Stage 1)
    """
    if not loc_name_norm or country_code not in country_candidates:
        return None, 0

    best_match = None
    best_score = 0

    # Only iterate through candidates for this country (much faster!)
    for district_norm, ipc_list in country_candidates[country_code]:
        # Calculate fuzzy match score
        score = fuzz.ratio(loc_name_norm, district_norm)

        if score >= FUZZY_THRESHOLD and score > best_score:
            best_score = score
            best_match = ipc_list

    return best_match, best_score


def build_country_candidates(combined_lookup):
    """Pre-group candidates by country for fast fuzzy matching."""
    country_candidates = defaultdict(list)
    for (fips, district_norm), ipc_list in combined_lookup.items():
        country_candidates[fips].append((district_norm, ipc_list))
    return dict(country_candidates)


def precompute_fuzzy_matches(country_candidates, all_gadm_names_by_country):
    """
    Pre-compute ALL fuzzy matches at startup to avoid repeated computation.
    Returns: dict of {(country_code, gadm_norm): matched_district_list}
    """
    print("   Pre-computing fuzzy matches (one-time cost)...", flush=True)
    cache = {}
    total_computed = 0

    for country_code, gadm_names in all_gadm_names_by_country.items():
        if country_code not in country_candidates:
            continue

        candidates = country_candidates[country_code]
        for gadm_norm in gadm_names:
            if not gadm_norm:
                continue

            best_match = None
            best_score = 0

            for district_norm, district_list in candidates:
                score = fuzz.ratio(gadm_norm, district_norm)
                if score >= FUZZY_THRESHOLD and score > best_score:
                    best_score = score
                    best_match = district_list

            if best_match:
                cache[(country_code, gadm_norm)] = best_match
                total_computed += 1

    print(f"   Pre-computed {total_computed:,} fuzzy matches", flush=True)
    return cache


def get_fuzzy_match_cached(gadm_norm, country_code, fuzzy_cache, country_candidates):
    """Get fuzzy match from cache, or compute if not cached."""
    key = (country_code, gadm_norm)
    if key in fuzzy_cache:
        return fuzzy_cache[key], FUZZY_THRESHOLD
    # Fallback to computation (should be rare after pre-computation)
    return find_fuzzy_match(gadm_norm, country_candidates, country_code)


def build_district_lookup(ipc_ref):
    """
    Build lookup dictionary for district matching.

    KEY: This builds the SAME lookup as Stage 1 Script 02, but stores only
    district metadata (not period-specific info) for monthly aggregation.

    Returns: (ipc_lookup, word_lookup, unique_districts)
    """
    print("\n2. Building district lookup dictionary...", flush=True)

    # Get unique districts from IPC reference
    # Use the same columns as Stage 1 for matching
    unique_districts = ipc_ref.drop_duplicates(subset=['geographic_unit_full_name'])[
        ['country', 'country_code', 'fips_code', 'district', 'region',
         'geographic_unit_name', 'geographic_unit_full_name', 'district_normalized',
         'full_name_normalized', 'fewsnet_region', 'geographic_group']
    ].copy()

    print(f"   Unique districts: {len(unique_districts):,}", flush=True)

    # Build primary lookup: (country_fips, district_normalized) -> district_info
    ipc_lookup = defaultdict(list)

    for idx, row in unique_districts.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row['district_normalized']):
            key = (row['fips_code'], row['district_normalized'])
            district_info = {
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
            }
            ipc_lookup[key].append(district_info)

    print(f"   Primary lookup: {len(ipc_lookup):,} (country, district) keys", flush=True)

    # Build word-based lookup from full_name_normalized (for livelihood zones)
    print("   Building word-based lookup from full_name_normalized...", flush=True)
    word_lookup = defaultdict(list)

    for idx, row in unique_districts.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row.get('full_name_normalized')):
            country = row['fips_code']
            full_name_norm = normalize_text(row['full_name_normalized'])
            words = full_name_norm.split()

            district_info = {
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
            }

            for word in words:
                if len(word) > 2:  # Skip short words
                    key = (country, word)
                    word_lookup[key].append(district_info)

    print(f"   Word-based lookup: {len(word_lookup):,} keys", flush=True)

    # Combine lookups (ipc_lookup takes precedence)
    combined_lookup = defaultdict(list)
    for k, v in word_lookup.items():
        combined_lookup[k].extend(v)
    for k, v in ipc_lookup.items():
        combined_lookup[k].extend(v)

    print(f"   Combined lookup: {len(combined_lookup):,} keys", flush=True)

    # Pre-group candidates by country for fast fuzzy matching
    country_candidates = build_country_candidates(combined_lookup)
    print(f"   Pre-grouped by country: {len(country_candidates)} countries", flush=True)

    return combined_lookup, unique_districts, country_candidates


def cleanup_temp_files(temp_parquets, lookup_file=None):
    """Clean up temporary files on success or failure"""
    cleaned = 0
    for temp_file in temp_parquets:
        try:
            if temp_file.exists():
                temp_file.unlink()
                cleaned += 1
        except Exception as e:
            print(f"   Warning: Could not delete {temp_file}: {e}")

    if lookup_file and lookup_file.exists():
        try:
            lookup_file.unlink()
            cleaned += 1
        except Exception as e:
            print(f"   Warning: Could not delete {lookup_file}: {e}")

    return cleaned


def process_batch_monthly(args):
    """
    Process a single batch of locations in parallel for monthly aggregation.
    Workers read their own batch from file to avoid memory duplication.
    Returns: (batch_num, results_list, match_stats)
    """
    batch_num, row_start, row_end, locations_path, ipc_ref_path = args

    # Each worker builds its own lookup to avoid passing large dicts
    ipc_ref = pd.read_parquet(ipc_ref_path)

    # Build lookup in worker
    unique_districts = ipc_ref.drop_duplicates(subset=['geographic_unit_full_name'])[
        ['country', 'country_code', 'fips_code', 'district', 'region',
         'geographic_unit_name', 'geographic_unit_full_name', 'district_normalized',
         'full_name_normalized', 'fewsnet_region', 'geographic_group']
    ].copy()

    ipc_lookup = defaultdict(list)
    for idx, row in unique_districts.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row['district_normalized']):
            key = (row['fips_code'], row['district_normalized'])
            district_info = {
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
            }
            ipc_lookup[key].append(district_info)

    word_lookup = defaultdict(list)
    for idx, row in unique_districts.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row.get('full_name_normalized')):
            country = row['fips_code']
            full_name_norm = normalize_text(row['full_name_normalized'])
            words = full_name_norm.split()
            district_info = {
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
            }
            for word in words:
                if len(word) > 2:
                    key = (country, word)
                    word_lookup[key].append(district_info)

    combined_lookup = defaultdict(list)
    for k, v in word_lookup.items():
        combined_lookup[k].extend(v)
    for k, v in ipc_lookup.items():
        combined_lookup[k].extend(v)

    country_candidates = build_country_candidates(combined_lookup)

    # Read just this batch from the parquet file
    parquet_file = pq.ParquetFile(locations_path)
    batch_df = parquet_file.read_row_group(batch_num).to_pandas()

    results = []
    match_stats = {
        'GADM3_exact': 0, 'GADM3_fuzzy': 0,
        'GADM2_exact': 0, 'GADM2_fuzzy': 0,
        'GADM1_exact': 0, 'GADM1_fuzzy': 0,
        'Country_level': 0
    }

    locations_chunk = batch_df
    locations_chunk['date_extracted'] = pd.to_datetime(locations_chunk['date_extracted'])
    locations_chunk['year_month'] = locations_chunk['date_extracted'].dt.to_period('M')

    # Normalize geographic fields
    locations_chunk['gadm1_norm'] = locations_chunk['gadm1_name'].apply(normalize_text)
    locations_chunk['gadm2_norm'] = locations_chunk['gadm2_name'].apply(normalize_text)
    locations_chunk['gadm3_norm'] = locations_chunk['gadm3_name'].apply(normalize_text)

    # Group by country for efficiency
    for country_code, country_group in locations_chunk.groupby('african_country_code'):

        # PRIORITY 1: GADM3 Matching
        for gadm3_norm, gadm3_group in country_group.groupby('gadm3_norm'):
            if not gadm3_norm:
                continue

            key = (country_code, gadm3_norm)
            matched_district_list = None
            match_type = None

            if key in combined_lookup:
                matched_district_list = combined_lookup[key]
                match_type = 'GADM3_exact'
            else:
                matched_district_list, _ = find_fuzzy_match(gadm3_norm, country_candidates, country_code)
                if matched_district_list:
                    match_type = 'GADM3_fuzzy'

            if matched_district_list:
                n_rows = len(gadm3_group)
                gkgids = gadm3_group['GKGRECORDID'].values
                year_months = gadm3_group['year_month'].astype(str).values
                dates = gadm3_group['date_extracted'].astype(str).values

                for district_info in matched_district_list:
                    batch_tuples = list(zip(
                        gkgids,
                        year_months,
                        dates,
                        [match_type] * n_rows,
                        [district_info['ipc_country']] * n_rows,
                        [district_info['ipc_country_code']] * n_rows,
                        [district_info['ipc_fips_code']] * n_rows,
                        [district_info['ipc_district']] * n_rows,
                        [district_info['ipc_region']] * n_rows,
                        [district_info['ipc_geographic_unit']] * n_rows,
                        [district_info['ipc_geographic_unit_full']] * n_rows,
                        [district_info['ipc_fewsnet_region']] * n_rows,
                        [district_info['ipc_geographic_group']] * n_rows
                    ))
                    results.extend(batch_tuples)
                    match_stats[match_type] += n_rows

        # PRIORITY 2: GADM2 Matching
        for gadm2_norm, gadm2_group in country_group.groupby('gadm2_norm'):
            if not gadm2_norm:
                continue

            key = (country_code, gadm2_norm)
            matched_district_list = None
            match_type = None

            if key in combined_lookup:
                matched_district_list = combined_lookup[key]
                match_type = 'GADM2_exact'
            else:
                matched_district_list, _ = find_fuzzy_match(gadm2_norm, country_candidates, country_code)
                if matched_district_list:
                    match_type = 'GADM2_fuzzy'

            if matched_district_list:
                n_rows = len(gadm2_group)
                gkgids = gadm2_group['GKGRECORDID'].values
                year_months = gadm2_group['year_month'].astype(str).values
                dates = gadm2_group['date_extracted'].astype(str).values

                for district_info in matched_district_list:
                    batch_tuples = list(zip(
                        gkgids,
                        year_months,
                        dates,
                        [match_type] * n_rows,
                        [district_info['ipc_country']] * n_rows,
                        [district_info['ipc_country_code']] * n_rows,
                        [district_info['ipc_fips_code']] * n_rows,
                        [district_info['ipc_district']] * n_rows,
                        [district_info['ipc_region']] * n_rows,
                        [district_info['ipc_geographic_unit']] * n_rows,
                        [district_info['ipc_geographic_unit_full']] * n_rows,
                        [district_info['ipc_fewsnet_region']] * n_rows,
                        [district_info['ipc_geographic_group']] * n_rows
                    ))
                    results.extend(batch_tuples)
                    match_stats[match_type] += n_rows

        # PRIORITY 3: GADM1 Matching
        for gadm1_norm, gadm1_group in country_group.groupby('gadm1_norm'):
            if not gadm1_norm:
                continue

            key = (country_code, gadm1_norm)
            matched_district_list = None
            match_type = None

            if key in combined_lookup:
                matched_district_list = combined_lookup[key]
                match_type = 'GADM1_exact'
            else:
                matched_district_list, _ = find_fuzzy_match(gadm1_norm, country_candidates, country_code)
                if matched_district_list:
                    match_type = 'GADM1_fuzzy'

            if matched_district_list:
                n_rows = len(gadm1_group)
                gkgids = gadm1_group['GKGRECORDID'].values
                year_months = gadm1_group['year_month'].astype(str).values
                dates = gadm1_group['date_extracted'].astype(str).values

                for district_info in matched_district_list:
                    batch_tuples = list(zip(
                        gkgids,
                        year_months,
                        dates,
                        [match_type] * n_rows,
                        [district_info['ipc_country']] * n_rows,
                        [district_info['ipc_country_code']] * n_rows,
                        [district_info['ipc_fips_code']] * n_rows,
                        [district_info['ipc_district']] * n_rows,
                        [district_info['ipc_region']] * n_rows,
                        [district_info['ipc_geographic_unit']] * n_rows,
                        [district_info['ipc_geographic_unit_full']] * n_rows,
                        [district_info['ipc_fewsnet_region']] * n_rows,
                        [district_info['ipc_geographic_group']] * n_rows
                    ))
                    results.extend(batch_tuples)
                    match_stats[match_type] += n_rows

        # PRIORITY 4: Country-level matching
        if country_code in COUNTRY_LEVEL_ONLY:
            n_rows = len(country_group)
            gkgids = country_group['GKGRECORDID'].values
            year_months = country_group['year_month'].astype(str).values
            dates = country_group['date_extracted'].astype(str).values

            for (fips, district), district_list in combined_lookup.items():
                if fips == country_code:
                    for district_info in district_list:
                        batch_tuples = list(zip(
                            gkgids,
                            year_months,
                            dates,
                            ['Country_level'] * n_rows,
                            [district_info['ipc_country']] * n_rows,
                            [district_info['ipc_country_code']] * n_rows,
                            [district_info['ipc_fips_code']] * n_rows,
                            [district_info['ipc_district']] * n_rows,
                            [district_info['ipc_region']] * n_rows,
                            [district_info['ipc_geographic_unit']] * n_rows,
                            [district_info['ipc_geographic_unit_full']] * n_rows,
                            [district_info['ipc_fewsnet_region']] * n_rows,
                            [district_info['ipc_geographic_group']] * n_rows
                        ))
                        results.extend(batch_tuples)
                        match_stats['Country_level'] += n_rows

    return batch_num, results, match_stats


def main():
    print("=" * 80)
    print("Stage 2: Monthly Article Aggregation")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Fuzzy matching threshold: {FUZZY_THRESHOLD}")
    print("\nKEY DIFFERENCE from Stage 1:")
    print("  - Stage 1: Aggregates within IPC assessment periods (Feb, Jun, Oct)")
    print("  - Stage 2: Aggregates by calendar month (all 12 months)")
    print("  - Uses SAME district matching logic as Stage 1 for alignment")

    # Ensure output directory exists
    STAGE2_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load IPC reference (same as Stage 1)
    print("\n1. Loading IPC district reference...")
    ipc_ref = pd.read_parquet(IPC_REF_FILE)
    print(f"   Loaded {len(ipc_ref):,} IPC periods")
    print(f"   Unique districts: {ipc_ref['district'].nunique():,}")
    print(f"   Unique full_names: {ipc_ref['geographic_unit_full_name'].nunique():,}")

    # Build district lookup (same matching logic as Stage 1)
    combined_lookup, unique_districts, country_candidates = build_district_lookup(ipc_ref)

    # Process locations to build GKGRECORDID -> district mapping
    # KEY CHANGE: Instead of filtering by IPC period, we capture the article date
    print("\n3. Processing locations to build GKGRECORDID -> district mapping...", flush=True)
    print("   Matching priority: GADM3 -> GADM2 -> GADM1 -> Country-level", flush=True)
    print(f"   Using {NUM_WORKERS} parallel workers", flush=True)

    parquet_file = pq.ParquetFile(LOCATIONS_FILE)

    # Use SQLite for streaming - eliminates memory issues
    # Use unique timestamped filename to avoid conflicts from old processes
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    sqlite_file = STAGE2_DATA_DIR / f'temp_gkg_district_lookup_{timestamp_str}.db'
    # Clean up old temp files (ignore errors if locked)
    for old_file in STAGE2_DATA_DIR.glob('temp_gkg_district_lookup_*.db'):
        try:
            old_file.unlink()
        except:
            pass  # Ignore locked files

    conn = sqlite3.connect(str(sqlite_file))
    cursor = conn.cursor()

    # SQLite optimizations for bulk inserts
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.execute("PRAGMA cache_size=100000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    conn.commit()

    # Create table for GKG-district mappings
    cursor.execute('''
        CREATE TABLE gkg_district (
            GKGRECORDID TEXT,
            year_month TEXT,
            date_extracted TEXT,
            match_level TEXT,
            ipc_country TEXT,
            ipc_country_code TEXT,
            ipc_fips_code TEXT,
            ipc_district TEXT,
            ipc_region TEXT,
            ipc_geographic_unit TEXT,
            ipc_geographic_unit_full TEXT,
            ipc_fewsnet_region TEXT,
            ipc_geographic_group TEXT
        )
    ''')
    conn.commit()

    # Get number of row groups in parquet file
    num_row_groups = parquet_file.metadata.num_row_groups
    print(f"   Total row groups to process: {num_row_groups}", flush=True)

    # Use SEQUENTIAL processing to avoid memory issues with multiprocessing
    print(f"\n   Processing {num_row_groups} batches sequentially (SQLite streaming)...", flush=True)
    print("   (Sequential mode avoids memory duplication in subprocess pickling)", flush=True)

    # Pre-compute unique GADM names from locations file for fuzzy cache
    print("   Scanning locations for unique GADM names (for fuzzy cache)...", flush=True)
    all_gadm_names_by_country = defaultdict(set)
    for rg_idx in range(num_row_groups):
        sample_batch = parquet_file.read_row_group(rg_idx, columns=['african_country_code', 'gadm1_name', 'gadm2_name', 'gadm3_name']).to_pandas()
        for country_code in sample_batch['african_country_code'].unique():
            country_data = sample_batch[sample_batch['african_country_code'] == country_code]
            all_gadm_names_by_country[country_code].update(country_data['gadm1_name'].dropna().apply(normalize_text).unique())
            all_gadm_names_by_country[country_code].update(country_data['gadm2_name'].dropna().apply(normalize_text).unique())
            all_gadm_names_by_country[country_code].update(country_data['gadm3_name'].dropna().apply(normalize_text).unique())
        del sample_batch
        if (rg_idx + 1) % 20 == 0:
            print(f"      Scanned {rg_idx + 1}/{num_row_groups} row groups...", flush=True)
    print(f"   Found {sum(len(v) for v in all_gadm_names_by_country.values()):,} unique GADM names across {len(all_gadm_names_by_country)} countries", flush=True)

    # Pre-compute fuzzy matches
    fuzzy_cache = precompute_fuzzy_matches(country_candidates, all_gadm_names_by_country)
    del all_gadm_names_by_country  # Free memory
    gc.collect()

    total_match_stats = {
        'GADM3_exact': 0, 'GADM3_fuzzy': 0,
        'GADM2_exact': 0, 'GADM2_fuzzy': 0,
        'GADM1_exact': 0, 'GADM1_fuzzy': 0,
        'Country_level': 0
    }

    total_rows = 0
    start_time = datetime.now()

    # Process batches sequentially - write directly to SQLite
    for batch_num in range(num_row_groups):
        batch_start = datetime.now()

        # Read this batch from parquet
        batch_df = parquet_file.read_row_group(batch_num).to_pandas()
        batch_df['date_extracted'] = pd.to_datetime(batch_df['date_extracted'])
        batch_df['year_month'] = batch_df['date_extracted'].dt.to_period('M').astype(str)

        # Normalize geographic fields
        batch_df['gadm1_norm'] = batch_df['gadm1_name'].apply(normalize_text)
        batch_df['gadm2_norm'] = batch_df['gadm2_name'].apply(normalize_text)
        batch_df['gadm3_norm'] = batch_df['gadm3_name'].apply(normalize_text)

        batch_matches = 0
        batch_stats = {'GADM3_exact': 0, 'GADM3_fuzzy': 0, 'GADM2_exact': 0, 'GADM2_fuzzy': 0,
                       'GADM1_exact': 0, 'GADM1_fuzzy': 0, 'Country_level': 0}

        # Buffer for batched SQLite inserts
        insert_buffer = []

        # Track matched GKGRECORDIDs to avoid redundant lower-priority matching
        matched_gkgids = set()

        # Process by country for efficiency
        for country_code, country_group in batch_df.groupby('african_country_code'):

            # PRIORITY 1: GADM3 Matching
            for gadm3_norm, gadm3_group in country_group.groupby('gadm3_norm'):
                if not gadm3_norm:
                    continue

                key = (country_code, gadm3_norm)
                matched_district_list = None
                match_type = None

                if key in combined_lookup:
                    matched_district_list = combined_lookup[key]
                    match_type = 'GADM3_exact'
                else:
                    # Use cached fuzzy match
                    matched_district_list, _ = get_fuzzy_match_cached(gadm3_norm, country_code, fuzzy_cache, country_candidates)
                    if matched_district_list:
                        match_type = 'GADM3_fuzzy'

                if matched_district_list:
                    for district_info in matched_district_list:
                        # VECTORIZED: Use .values arrays instead of iterrows()
                        gkgids = gadm3_group['GKGRECORDID'].values
                        year_months = gadm3_group['year_month'].values
                        dates = gadm3_group['date_extracted'].astype(str).values
                        n_rows = len(gkgids)
                        tuples = list(zip(
                            gkgids,
                            year_months,
                            dates,
                            [match_type] * n_rows,
                            [district_info['ipc_country']] * n_rows,
                            [district_info['ipc_country_code']] * n_rows,
                            [district_info['ipc_fips_code']] * n_rows,
                            [district_info['ipc_district']] * n_rows,
                            [district_info['ipc_region']] * n_rows,
                            [district_info['ipc_geographic_unit']] * n_rows,
                            [district_info['ipc_geographic_unit_full']] * n_rows,
                            [district_info['ipc_fewsnet_region']] * n_rows,
                            [district_info['ipc_geographic_group']] * n_rows
                        ))
                        insert_buffer.extend(tuples)
                        batch_matches += n_rows
                        batch_stats[match_type] += n_rows
                        matched_gkgids.update(gkgids)

                        # Flush buffer when it gets large
                        if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                            cursor.executemany('INSERT INTO gkg_district VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                            insert_buffer.clear()

            # PRIORITY 2: GADM2 Matching
            for gadm2_norm, gadm2_group in country_group.groupby('gadm2_norm'):
                if not gadm2_norm:
                    continue

                # Skip already matched GKGRECORDIDs
                gadm2_group = gadm2_group[~gadm2_group['GKGRECORDID'].isin(matched_gkgids)]
                if len(gadm2_group) == 0:
                    continue

                key = (country_code, gadm2_norm)
                matched_district_list = None
                match_type = None

                if key in combined_lookup:
                    matched_district_list = combined_lookup[key]
                    match_type = 'GADM2_exact'
                else:
                    # Use cached fuzzy match
                    matched_district_list, _ = get_fuzzy_match_cached(gadm2_norm, country_code, fuzzy_cache, country_candidates)
                    if matched_district_list:
                        match_type = 'GADM2_fuzzy'

                if matched_district_list:
                    for district_info in matched_district_list:
                        # VECTORIZED: Use .values arrays instead of iterrows()
                        gkgids = gadm2_group['GKGRECORDID'].values
                        year_months = gadm2_group['year_month'].values
                        dates = gadm2_group['date_extracted'].astype(str).values
                        n_rows = len(gkgids)
                        tuples = list(zip(
                            gkgids,
                            year_months,
                            dates,
                            [match_type] * n_rows,
                            [district_info['ipc_country']] * n_rows,
                            [district_info['ipc_country_code']] * n_rows,
                            [district_info['ipc_fips_code']] * n_rows,
                            [district_info['ipc_district']] * n_rows,
                            [district_info['ipc_region']] * n_rows,
                            [district_info['ipc_geographic_unit']] * n_rows,
                            [district_info['ipc_geographic_unit_full']] * n_rows,
                            [district_info['ipc_fewsnet_region']] * n_rows,
                            [district_info['ipc_geographic_group']] * n_rows
                        ))
                        insert_buffer.extend(tuples)
                        batch_matches += n_rows
                        batch_stats[match_type] += n_rows
                        matched_gkgids.update(gkgids)

                        # Flush buffer when it gets large
                        if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                            cursor.executemany('INSERT INTO gkg_district VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                            insert_buffer.clear()

            # PRIORITY 3: GADM1 Matching
            for gadm1_norm, gadm1_group in country_group.groupby('gadm1_norm'):
                if not gadm1_norm:
                    continue

                # Skip already matched GKGRECORDIDs
                gadm1_group = gadm1_group[~gadm1_group['GKGRECORDID'].isin(matched_gkgids)]
                if len(gadm1_group) == 0:
                    continue

                key = (country_code, gadm1_norm)
                matched_district_list = None
                match_type = None

                if key in combined_lookup:
                    matched_district_list = combined_lookup[key]
                    match_type = 'GADM1_exact'
                else:
                    # Use cached fuzzy match
                    matched_district_list, _ = get_fuzzy_match_cached(gadm1_norm, country_code, fuzzy_cache, country_candidates)
                    if matched_district_list:
                        match_type = 'GADM1_fuzzy'

                if matched_district_list:
                    for district_info in matched_district_list:
                        # VECTORIZED: Use .values arrays instead of iterrows()
                        gkgids = gadm1_group['GKGRECORDID'].values
                        year_months = gadm1_group['year_month'].values
                        dates = gadm1_group['date_extracted'].astype(str).values
                        n_rows = len(gkgids)
                        tuples = list(zip(
                            gkgids,
                            year_months,
                            dates,
                            [match_type] * n_rows,
                            [district_info['ipc_country']] * n_rows,
                            [district_info['ipc_country_code']] * n_rows,
                            [district_info['ipc_fips_code']] * n_rows,
                            [district_info['ipc_district']] * n_rows,
                            [district_info['ipc_region']] * n_rows,
                            [district_info['ipc_geographic_unit']] * n_rows,
                            [district_info['ipc_geographic_unit_full']] * n_rows,
                            [district_info['ipc_fewsnet_region']] * n_rows,
                            [district_info['ipc_geographic_group']] * n_rows
                        ))
                        insert_buffer.extend(tuples)
                        batch_matches += n_rows
                        batch_stats[match_type] += n_rows
                        matched_gkgids.update(gkgids)

                        # Flush buffer when it gets large
                        if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                            cursor.executemany('INSERT INTO gkg_district VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                            insert_buffer.clear()

            # PRIORITY 4: Country-level matching
            if country_code in COUNTRY_LEVEL_ONLY:
                # Skip already matched GKGRECORDIDs
                unmatched_group = country_group[~country_group['GKGRECORDID'].isin(matched_gkgids)]
                if len(unmatched_group) == 0:
                    continue

                for (fips, district), district_list in combined_lookup.items():
                    if fips == country_code:
                        for district_info in district_list:
                            # VECTORIZED: Use .values arrays instead of iterrows()
                            gkgids = unmatched_group['GKGRECORDID'].values
                            year_months = unmatched_group['year_month'].values
                            dates = unmatched_group['date_extracted'].astype(str).values
                            n_rows = len(gkgids)
                            tuples = list(zip(
                                gkgids,
                                year_months,
                                dates,
                                ['Country_level'] * n_rows,
                                [district_info['ipc_country']] * n_rows,
                                [district_info['ipc_country_code']] * n_rows,
                                [district_info['ipc_fips_code']] * n_rows,
                                [district_info['ipc_district']] * n_rows,
                                [district_info['ipc_region']] * n_rows,
                                [district_info['ipc_geographic_unit']] * n_rows,
                                [district_info['ipc_geographic_unit_full']] * n_rows,
                                [district_info['ipc_fewsnet_region']] * n_rows,
                                [district_info['ipc_geographic_group']] * n_rows
                            ))
                            insert_buffer.extend(tuples)
                            batch_matches += n_rows
                            batch_stats['Country_level'] += n_rows

                            # Flush buffer when it gets large
                            if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                                cursor.executemany('INSERT INTO gkg_district VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                                insert_buffer.clear()

        # Flush any remaining buffer and commit after each batch
        if insert_buffer:
            cursor.executemany('INSERT INTO gkg_district VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
            insert_buffer.clear()
        conn.commit()
        total_rows += batch_matches

        # Update aggregate stats
        for key, count in batch_stats.items():
            total_match_stats[key] += count

        # Progress reporting
        batch_time = (datetime.now() - batch_start).total_seconds()
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (batch_num + 1) / elapsed if elapsed > 0 else 0
        eta = (num_row_groups - batch_num - 1) / rate if rate > 0 else 0
        print(f"      Batch {batch_num + 1}/{num_row_groups}: {batch_matches:,} matches ({batch_time:.1f}s) | Total: {total_rows:,} | ETA: {eta/60:.1f}min", flush=True)

        # Free memory
        del batch_df, matched_gkgids
        gc.collect()

    # Free fuzzy cache before deduplication
    del fuzzy_cache
    gc.collect()

    # SKIP index on large table - causes OOM on 181M+ rows
    # Index will be created on deduplicated table instead (much smaller)
    print("   Skipping index on main table (will index after dedup)...", flush=True)

    match_stats = total_match_stats
    gc.collect()

    # Print match statistics
    print(f"\n   Match statistics:")
    total_matches = sum(match_stats.values())
    for method, count in sorted(match_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_matches * 100) if total_matches > 0 else 0
        print(f"     {method}: {count:,} ({pct:.1f}%)")

    # Check if we have data
    cursor.execute('SELECT COUNT(*) FROM gkg_district')
    total_rows = cursor.fetchone()[0]
    if total_rows == 0:
        print("\n   WARNING: No matched locations found!")
        conn.close()
        return

    # Deduplicate in SQLite using BATCHED approach (memory-safe for 8GB systems)
    print(f"\n4. Deduplicating {total_rows:,} records in SQLite (batched for memory safety)...")

    # Checkpoint WAL to reduce memory footprint before heavy operation
    print("   Checkpointing WAL to free memory...", flush=True)
    cursor.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    gc.collect()

    # Get max rowid for batching
    cursor.execute('SELECT MAX(rowid) FROM gkg_district')
    max_rowid = cursor.fetchone()[0] or 0

    # Create intermediate dedup table for batched processing
    print(f"   Creating batched dedup table (processing {max_rowid:,} rows in {DEDUP_BATCH_SIZE:,}-row batches)...", flush=True)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gkg_district_dedup_temp (
            GKGRECORDID TEXT,
            year_month TEXT,
            date_extracted TEXT,
            match_level TEXT,
            ipc_country TEXT,
            ipc_country_code TEXT,
            ipc_fips_code TEXT,
            ipc_district TEXT,
            ipc_region TEXT,
            ipc_geographic_unit TEXT,
            ipc_geographic_unit_full TEXT,
            ipc_fewsnet_region TEXT,
            ipc_geographic_group TEXT
        )
    ''')
    conn.commit()

    # Process deduplication in batches
    num_batches = (max_rowid + DEDUP_BATCH_SIZE - 1) // DEDUP_BATCH_SIZE
    batch_start_time = datetime.now()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * DEDUP_BATCH_SIZE
        batch_end = min((batch_idx + 1) * DEDUP_BATCH_SIZE, max_rowid)

        cursor.execute('''
            INSERT INTO gkg_district_dedup_temp
            SELECT * FROM gkg_district
            WHERE rowid > ? AND rowid <= ?
            GROUP BY GKGRECORDID, ipc_geographic_unit_full, year_month
        ''', (batch_start, batch_end))
        conn.commit()
        gc.collect()

        elapsed = (datetime.now() - batch_start_time).total_seconds()
        rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
        print(f"      Dedup batch {batch_idx + 1}/{num_batches} (rowid {batch_start:,}-{batch_end:,}) | ETA: {eta/60:.1f}min", flush=True)

    # Final cross-batch deduplication (now on much smaller intermediate table)
    cursor.execute('SELECT COUNT(*) FROM gkg_district_dedup_temp')
    temp_rows = cursor.fetchone()[0]
    print(f"   Intermediate table: {temp_rows:,} rows (now doing final cross-batch dedup)...", flush=True)

    cursor.execute('''
        CREATE TABLE gkg_district_dedup AS
        SELECT * FROM gkg_district_dedup_temp
        GROUP BY GKGRECORDID, ipc_geographic_unit_full, year_month
    ''')
    conn.commit()

    # Drop intermediate table to free space
    cursor.execute('DROP TABLE gkg_district_dedup_temp')
    conn.commit()
    gc.collect()

    # Now create index on the SMALLER deduplicated table
    print("   Creating index on deduplicated table...", flush=True)
    cursor.execute('CREATE INDEX idx_dedup_gkgrecordid ON gkg_district_dedup(GKGRECORDID)')
    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM gkg_district_dedup')
    dedup_rows = cursor.fetchone()[0]
    print(f"   Deduplicated: {total_rows:,} -> {dedup_rows:,} unique matches")

    # Load articles and aggregate BY MONTH
    print("\n5. Loading and aggregating articles BY MONTH...")
    articles_sample = pd.read_csv(ARTICLES_FILE, nrows=1)
    all_article_cols = articles_sample.columns.tolist()

    # Define numeric columns for aggregation (same as Stage 1)
    numeric_cols = [col for col in all_article_cols if col not in [
        'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
        'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
        'Dates', 'Persons', 'V2Persons', 'Organizations', 'V2Organizations',
        'V2AllNames', 'Locations', 'V2Locations', 'ADM1', 'ADM2', 'ADM3',
        'V2Quotations', 'V2Amounts', 'V2RelatedImages', 'V2DateTimeFields',
        'V2Extras', 'V2ExtendedField', 'V2TranslationInfo', 'OutletType',
        'all_countries_mentioned', 'all_african_countries', 'date_extracted',
        'V2Tone'
    ]]

    aggregated_data = []

    # Define group columns once for reuse
    group_cols = [
        'ipc_country', 'ipc_country_code', 'ipc_fips_code',
        'ipc_district', 'ipc_region',
        'ipc_geographic_unit', 'ipc_geographic_unit_full',
        'ipc_fewsnet_region', 'ipc_geographic_group',
        'year_month', 'match_level'
    ]

    for chunk_num, articles_chunk in enumerate(pd.read_csv(ARTICLES_FILE, chunksize=CHUNK_SIZE)):
        print(f"   Articles chunk {chunk_num + 1}: {len(articles_chunk):,} articles...", flush=True)

        article_ids = list(articles_chunk['GKGRECORDID'].unique())

        # Query SQLite for matching GKGRECORDIDs in BATCHES to avoid too many placeholders
        gkg_chunks = []
        for i in range(0, len(article_ids), SQL_QUERY_BATCH_SIZE):
            batch_ids = article_ids[i:i + SQL_QUERY_BATCH_SIZE]
            placeholders = ','.join(['?'] * len(batch_ids))
            query = f'SELECT * FROM gkg_district_dedup WHERE GKGRECORDID IN ({placeholders})'
            gkg_chunk = pd.read_sql_query(query, conn, params=batch_ids)
            if len(gkg_chunk) > 0:
                gkg_chunks.append(gkg_chunk)

        if not gkg_chunks:
            del articles_chunk, article_ids
            continue

        gkg_lookup_chunk = pd.concat(gkg_chunks, ignore_index=True)
        del gkg_chunks

        merged = articles_chunk.merge(gkg_lookup_chunk, on='GKGRECORDID', how='inner')
        print(f"      Matched: {len(merged):,} article-month pairs")

        del articles_chunk, gkg_lookup_chunk, article_ids

        if len(merged) == 0:
            continue

        # Aggregate
        agg_dict = {col: 'sum' for col in numeric_cols if col in merged.columns}
        agg_dict['GKGRECORDID'] = 'count'
        if 'SourceCommonName' in merged.columns:
            agg_dict['SourceCommonName'] = 'nunique'

        agg_result = merged.groupby(group_cols).agg(agg_dict).reset_index()
        agg_result = agg_result.rename(columns={
            'GKGRECORDID': 'article_count',
            'SourceCommonName': 'unique_sources'
        })

        aggregated_data.append(agg_result)

        del merged, agg_result
        gc.collect()

        # PERIODIC CONSOLIDATION: Prevent memory accumulation (memory-safe for 8GB)
        if len(aggregated_data) >= CONSOLIDATE_EVERY:
            print(f"      Consolidating {len(aggregated_data)} partial aggregations...", flush=True)
            combined = pd.concat(aggregated_data, ignore_index=True)

            # Re-aggregate to reduce memory footprint
            numeric_to_sum = [c for c in combined.columns if c not in group_cols]
            agg_dict_consol = {col: 'sum' for col in numeric_to_sum if col in combined.columns}
            if 'match_level' in combined.columns and 'match_level' not in group_cols:
                agg_dict_consol['match_level'] = 'first'

            consolidated = combined.groupby(group_cols).agg(agg_dict_consol).reset_index()
            aggregated_data = [consolidated]
            del combined, consolidated
            gc.collect()
            print(f"      Consolidated to {len(aggregated_data[0]):,} aggregated rows", flush=True)

    # Combine and final aggregation
    print("\n6. Combining and final aggregation...")
    if not aggregated_data:
        print("   WARNING: No data to aggregate!")
        return

    final_df = pd.concat(aggregated_data, ignore_index=True)
    del aggregated_data
    gc.collect()

    # Final aggregation by (district, month) - note: excludes match_level to merge across match types
    final_group_cols = [
        'ipc_country', 'ipc_country_code', 'ipc_fips_code',
        'ipc_district', 'ipc_region',
        'ipc_geographic_unit', 'ipc_geographic_unit_full',
        'ipc_fewsnet_region', 'ipc_geographic_group',
        'year_month'
    ]

    # Sum numeric columns, keep first match_level
    numeric_to_sum = [c for c in final_df.columns if c not in final_group_cols and c != 'match_level']
    agg_dict = {col: 'sum' for col in numeric_to_sum if col in final_df.columns}
    agg_dict['match_level'] = 'first'

    final_agg = final_df.groupby(final_group_cols).agg(agg_dict).reset_index()

    # Summary
    print("\n" + "=" * 80)
    print("Monthly Aggregation Summary - Stage 2")
    print("=" * 80)
    print(f"\nTotal monthly aggregations: {len(final_agg):,}")
    print(f"Unique districts (ipc_geographic_unit_full): {final_agg['ipc_geographic_unit_full'].nunique():,}")
    print(f"Unique months: {final_agg['year_month'].nunique()}")
    print(f"Month range: {final_agg['year_month'].min()} to {final_agg['year_month'].max()}")
    print(f"Countries: {final_agg['ipc_country'].nunique()}")

    print(f"\nMatch level distribution:")
    print(final_agg['match_level'].value_counts())

    print(f"\nRecords by country:")
    print(final_agg['ipc_country'].value_counts().head(10))

    # Verify alignment with Stage 1 districts
    print("\n" + "=" * 80)
    print("District Alignment Check")
    print("=" * 80)

    stage1_articles = pd.read_parquet(DISTRICT_DATA_DIR / 'articles_aggregated.parquet')
    stage1_districts = set(stage1_articles['ipc_geographic_unit_full'].unique())
    stage2_districts = set(final_agg['ipc_geographic_unit_full'].unique())

    overlap = stage1_districts & stage2_districts
    only_stage1 = stage1_districts - stage2_districts
    only_stage2 = stage2_districts - stage1_districts

    print(f"\nStage 1 districts: {len(stage1_districts):,}")
    print(f"Stage 2 districts: {len(stage2_districts):,}")
    print(f"Overlap: {len(overlap):,} ({100*len(overlap)/len(stage1_districts):.1f}% of Stage 1)")
    print(f"Only in Stage 1: {len(only_stage1):,}")
    print(f"Only in Stage 2: {len(only_stage2):,}")

    if len(only_stage1) > 0:
        print(f"\nSample districts only in Stage 1:")
        for d in list(only_stage1)[:5]:
            print(f"   {d}")

    # Save
    print(f"\n7. Saving to {OUTPUT_PARQUET}...")
    final_agg.to_parquet(OUTPUT_PARQUET, index=False)
    print("   [OK] Parquet saved")

    print(f"\n8. Saving to {OUTPUT_CSV}...")
    final_agg.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    # Cleanup SQLite database
    print("\n   Cleaning up temporary SQLite database...")
    conn.close()
    if sqlite_file.exists():
        sqlite_file.unlink()
        print("   Cleaned up SQLite database")

    print("\n" + "=" * 80)
    print("Stage 2 Monthly Aggregation Complete")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"\nOutput: {OUTPUT_PARQUET}")
    print(f"Next step: Run 03a_stage2_aggregate_locations_monthly.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Attempting to clean up temporary SQLite database...")
        # Try to clean up SQLite file
        sqlite_file = STAGE2_DATA_DIR / 'temp_gkg_district_lookup.db'
        if sqlite_file.exists():
            try:
                sqlite_file.unlink()
                print("Cleaned up SQLite database after error")
            except Exception:
                print("Could not clean up SQLite database")
        raise
