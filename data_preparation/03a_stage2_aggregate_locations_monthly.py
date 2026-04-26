"""
Stage 2: Monthly Location Aggregation for Dynamic News Features
Aggregates GDELT location mentions by (district, year-month) using Stage 1 matching logic.

KEY DIFFERENCE FROM STAGE 1 (Script 03):
- Stage 1: Aggregates within IPC assessment periods (Feb, Jun, Oct - ~3/year)
- Stage 2: Aggregates by calendar month (all 12 months)

CRITICAL: Uses the SAME district matching logic as Script 03 to ensure alignment
with Stage 1 predictions and AR failures.

Matching Priority: GADM3 → GADM2 → GADM1 → Country-level

Author: Victor Collins Oppon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fuzzywuzzy import fuzz
from collections import defaultdict
import unicodedata
import gc
import pyarrow.parquet as pq

# Paths -- all relative to news-signal-africa repo root
BASE_DIR      = Path(__file__).parent.parent
RAW_DIR       = BASE_DIR / 'DATA' / 'raw'
INTERIM_DIR   = BASE_DIR / 'DATA' / 'interim'
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Input files
LOCATIONS_FILE  = INTERIM_DIR / 'african_gkg_locations_aligned.parquet'
IPC_REF_FILE    = RAW_DIR / 'ipc_reference.parquet'

# Stage 2 output
STAGE2_DATA_DIR = INTERIM_DIR / 'stage2'
STAGE2_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET  = STAGE2_DATA_DIR / 'locations_aggregated_monthly.parquet'
OUTPUT_CSV      = STAGE2_DATA_DIR / 'locations_aggregated_monthly.csv'

# Processing parameters
CHUNK_SIZE = 500000  # Reduced for 8GB RAM systems
FUZZY_THRESHOLD = 80  # Same as Stage 1
CONSOLIDATE_EVERY = 20  # Consolidate aggregations every N batches for memory safety

# Countries with ONLY national-level IPC data (same as Stage 1)
COUNTRY_LEVEL_ONLY = {'AO', 'CG', 'CT', 'LT', 'MR', 'RW', 'TO'}


def normalize_text(text):
    """Normalize text for matching with accent removal. (Same as Stage 1)"""
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    text = ' '.join(text.split())
    return text


def find_fuzzy_match(loc_name_norm, ipc_candidates, country_code):
    """
    Find best fuzzy match for location name among IPC district candidates.
    Returns: (district_info_list, match_score) or (None, 0)
    (Same as Stage 1)
    """
    if not loc_name_norm or not ipc_candidates:
        return None, 0

    best_match = None
    best_score = 0

    for (fips, district_norm), district_list in ipc_candidates.items():
        if fips != country_code:
            continue

        score = fuzz.ratio(loc_name_norm, district_norm)

        if score >= FUZZY_THRESHOLD and score > best_score:
            best_score = score
            best_match = district_list

    return best_match, best_score


def build_district_lookup(ipc_ref):
    """
    Build lookup dictionary for district matching.

    Returns: combined_lookup (dict) with district metadata (not period-specific)
    """
    print("\n2. Building district lookup dictionary...", flush=True)

    # Get unique districts from IPC reference
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

    # Build word-based lookup from full_name_normalized
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
                if len(word) > 2:
                    key = (country, word)
                    word_lookup[key].append(district_info)

    print(f"   Word-based lookup: {len(word_lookup):,} keys", flush=True)

    # Combine lookups
    combined_lookup = defaultdict(list)
    for k, v in word_lookup.items():
        combined_lookup[k].extend(v)
    for k, v in ipc_lookup.items():
        combined_lookup[k].extend(v)

    print(f"   Combined lookup: {len(combined_lookup):,} keys", flush=True)

    return combined_lookup, unique_districts


def main():
    print("=" * 80)
    print("Stage 2: Monthly Location Aggregation")
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
    print("\n1. Loading IPC district reference data...")
    ipc_ref = pd.read_parquet(IPC_REF_FILE)
    print(f"   Loaded {len(ipc_ref):,} IPC periods")
    print(f"   Unique districts: {ipc_ref['district'].nunique():,}")
    print(f"   Unique full_names: {ipc_ref['geographic_unit_full_name'].nunique():,}")

    # Build district lookup (same matching logic as Stage 1)
    combined_lookup, unique_districts = build_district_lookup(ipc_ref)

    # Process locations in chunks
    print("\n3. Processing locations with DISTRICT-level matching...", flush=True)
    print("   Priority: GADM3 -> GADM2 -> GADM1 -> Country (aligned with Stage 1)", flush=True)
    print("   Aggregating by MONTH instead of IPC period", flush=True)

    parquet_file = pq.ParquetFile(LOCATIONS_FILE)

    all_aggregations = []
    total_processed = 0
    match_stats = {
        'GADM3_exact': 0, 'GADM3_fuzzy': 0,
        'GADM2_exact': 0, 'GADM2_fuzzy': 0,
        'GADM1_exact': 0, 'GADM1_fuzzy': 0,
        'Country_level': 0,
        'no_match': 0
    }

    for batch_num, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
        chunk_start = datetime.now()
        locations_chunk = batch.to_pandas()

        print(f"\n   Batch {batch_num + 1}: Processing {len(locations_chunk):,} locations...", flush=True)

        locations_chunk['date_extracted'] = pd.to_datetime(locations_chunk['date_extracted'])

        # KEY CHANGE: Extract year-month from location date
        locations_chunk['year_month'] = locations_chunk['date_extracted'].dt.to_period('M')

        # Normalize geographic fields (same as Stage 1)
        locations_chunk['gadm1_norm'] = locations_chunk['gadm1_name'].apply(normalize_text)
        locations_chunk['gadm2_norm'] = locations_chunk['gadm2_name'].apply(normalize_text)
        locations_chunk['gadm3_norm'] = locations_chunk['gadm3_name'].apply(normalize_text)

        matched_records = []
        # KEY CHANGE: Track unique (district, month) combinations
        matched_district_months = set()

        # Group by country for efficiency
        for country_code, country_locs in locations_chunk.groupby('african_country_code'):

            # ================================================================
            # PRIORITY 1: GADM3 Matching (Exact → Fuzzy)
            # ================================================================
            for gadm3_norm, gadm3_locs in country_locs.groupby('gadm3_norm'):
                if not gadm3_norm:
                    continue

                key = (country_code, gadm3_norm)
                matched_district_list = None
                match_type = None
                match_score = 0

                if key in combined_lookup:
                    matched_district_list = combined_lookup[key]
                    match_type = 'GADM3_exact'
                    match_score = 100
                else:
                    matched_district_list, match_score = find_fuzzy_match(gadm3_norm, combined_lookup, country_code)
                    if matched_district_list:
                        match_type = 'GADM3_fuzzy'

                if matched_district_list:
                    for district_info in matched_district_list:
                        # KEY CHANGE: Group by year_month instead of IPC period
                        for year_month, month_locs in gadm3_locs.groupby('year_month'):
                            district_month_key = (district_info['ipc_geographic_unit_full'], str(year_month))
                            if district_month_key in matched_district_months:
                                continue

                            if len(month_locs) > 0:
                                agg_data = {
                                    'year_month': str(year_month),
                                    'location_mention_count': len(month_locs),
                                    'unique_location_names': month_locs['location_fullname'].nunique(),
                                    'unique_cities': month_locs['city_name'].nunique() if 'city_name' in month_locs.columns else 0,
                                    'unique_days': month_locs['date_extracted'].nunique(),
                                    'avg_latitude': month_locs['latitude'].mean(),
                                    'avg_longitude': month_locs['longitude'].mean(),
                                    'latitude_std': month_locs['latitude'].std(),
                                    'longitude_std': month_locs['longitude'].std(),
                                    'primary_gadm2': month_locs['gadm2_name'].mode()[0] if not month_locs['gadm2_name'].mode().empty else None,
                                    'primary_gadm3': month_locs['gadm3_name'].mode()[0] if not month_locs['gadm3_name'].mode().empty else None,
                                    'match_level': match_type,
                                    'match_score': match_score,
                                    **district_info
                                }
                                matched_records.append(agg_data)
                                matched_district_months.add(district_month_key)
                                match_stats[match_type] += 1

            # ================================================================
            # PRIORITY 2: GADM2 Matching (Exact → Fuzzy)
            # ================================================================
            for gadm2_norm, gadm2_locs in country_locs.groupby('gadm2_norm'):
                if not gadm2_norm:
                    continue

                key = (country_code, gadm2_norm)
                matched_district_list = None
                match_type = None
                match_score = 0

                if key in combined_lookup:
                    matched_district_list = combined_lookup[key]
                    match_type = 'GADM2_exact'
                    match_score = 100
                else:
                    matched_district_list, match_score = find_fuzzy_match(gadm2_norm, combined_lookup, country_code)
                    if matched_district_list:
                        match_type = 'GADM2_fuzzy'

                if matched_district_list:
                    for district_info in matched_district_list:
                        for year_month, month_locs in gadm2_locs.groupby('year_month'):
                            district_month_key = (district_info['ipc_geographic_unit_full'], str(year_month))
                            if district_month_key in matched_district_months:
                                continue

                            if len(month_locs) > 0:
                                agg_data = {
                                    'year_month': str(year_month),
                                    'location_mention_count': len(month_locs),
                                    'unique_location_names': month_locs['location_fullname'].nunique(),
                                    'unique_cities': month_locs['city_name'].nunique() if 'city_name' in month_locs.columns else 0,
                                    'unique_days': month_locs['date_extracted'].nunique(),
                                    'avg_latitude': month_locs['latitude'].mean(),
                                    'avg_longitude': month_locs['longitude'].mean(),
                                    'latitude_std': month_locs['latitude'].std(),
                                    'longitude_std': month_locs['longitude'].std(),
                                    'primary_gadm2': month_locs['gadm2_name'].mode()[0] if not month_locs['gadm2_name'].mode().empty else None,
                                    'primary_gadm3': month_locs['gadm3_name'].mode()[0] if not month_locs['gadm3_name'].mode().empty else None,
                                    'match_level': match_type,
                                    'match_score': match_score,
                                    **district_info
                                }
                                matched_records.append(agg_data)
                                matched_district_months.add(district_month_key)
                                match_stats[match_type] += 1

            # ================================================================
            # PRIORITY 3: GADM1 Matching (State/Region level)
            # ================================================================
            for gadm1_norm, gadm1_locs in country_locs.groupby('gadm1_norm'):
                if not gadm1_norm:
                    continue

                key = (country_code, gadm1_norm)
                matched_district_list = None
                match_type = None
                match_score = 0

                if key in combined_lookup:
                    matched_district_list = combined_lookup[key]
                    match_type = 'GADM1_exact'
                    match_score = 100
                else:
                    matched_district_list, match_score = find_fuzzy_match(gadm1_norm, combined_lookup, country_code)
                    if matched_district_list:
                        match_type = 'GADM1_fuzzy'

                if matched_district_list:
                    for district_info in matched_district_list:
                        for year_month, month_locs in gadm1_locs.groupby('year_month'):
                            district_month_key = (district_info['ipc_geographic_unit_full'], str(year_month))
                            if district_month_key in matched_district_months:
                                continue

                            if len(month_locs) > 0:
                                agg_data = {
                                    'year_month': str(year_month),
                                    'location_mention_count': len(month_locs),
                                    'unique_location_names': month_locs['location_fullname'].nunique(),
                                    'unique_cities': month_locs['city_name'].nunique() if 'city_name' in month_locs.columns else 0,
                                    'unique_days': month_locs['date_extracted'].nunique(),
                                    'avg_latitude': month_locs['latitude'].mean(),
                                    'avg_longitude': month_locs['longitude'].mean(),
                                    'latitude_std': month_locs['latitude'].std(),
                                    'longitude_std': month_locs['longitude'].std(),
                                    'primary_gadm2': month_locs['gadm2_name'].mode()[0] if not month_locs['gadm2_name'].mode().empty else None,
                                    'primary_gadm3': month_locs['gadm3_name'].mode()[0] if not month_locs['gadm3_name'].mode().empty else None,
                                    'match_level': match_type,
                                    'match_score': match_score,
                                    **district_info
                                }
                                matched_records.append(agg_data)
                                matched_district_months.add(district_month_key)
                                match_stats[match_type] += 1

            # ================================================================
            # PRIORITY 4: Country-level matching
            # ================================================================
            if country_code in COUNTRY_LEVEL_ONLY:
                for (fips, district), district_list in combined_lookup.items():
                    if fips == country_code:
                        for district_info in district_list:
                            for year_month, month_locs in country_locs.groupby('year_month'):
                                district_month_key = (district_info['ipc_geographic_unit_full'], str(year_month))
                                if district_month_key in matched_district_months:
                                    continue

                                if len(month_locs) > 0:
                                    agg_data = {
                                        'year_month': str(year_month),
                                        'location_mention_count': len(month_locs),
                                        'unique_location_names': month_locs['location_fullname'].nunique(),
                                        'unique_cities': month_locs['city_name'].nunique() if 'city_name' in month_locs.columns else 0,
                                        'unique_days': month_locs['date_extracted'].nunique(),
                                        'avg_latitude': month_locs['latitude'].mean(),
                                        'avg_longitude': month_locs['longitude'].mean(),
                                        'latitude_std': month_locs['latitude'].std(),
                                        'longitude_std': month_locs['longitude'].std(),
                                        'primary_gadm2': month_locs['gadm2_name'].mode()[0] if not month_locs['gadm2_name'].mode().empty else None,
                                        'primary_gadm3': month_locs['gadm3_name'].mode()[0] if not month_locs['gadm3_name'].mode().empty else None,
                                        'match_level': 'Country_level',
                                        'match_score': 100,
                                        **district_info
                                    }
                                    matched_records.append(agg_data)
                                    matched_district_months.add(district_month_key)
                                    match_stats['Country_level'] += 1

        if matched_records:
            chunk_df = pd.DataFrame(matched_records)
            all_aggregations.append(chunk_df)

        total_processed += len(locations_chunk)
        chunk_time = (datetime.now() - chunk_start).total_seconds()

        print(f"      Matched: {len(matched_records):,} district-month aggregations", flush=True)
        print(f"      Time: {chunk_time:.1f}s", flush=True)

        del locations_chunk
        gc.collect()

        # PERIODIC CONSOLIDATION: Prevent memory accumulation (memory-safe for 8GB)
        if len(all_aggregations) >= CONSOLIDATE_EVERY:
            print(f"      Consolidating {len(all_aggregations)} partial aggregations...", flush=True)
            combined = pd.concat(all_aggregations, ignore_index=True)

            # Re-aggregate to reduce memory footprint
            consol_group_cols = [
                'ipc_country', 'ipc_country_code', 'ipc_fips_code',
                'ipc_district', 'ipc_region',
                'ipc_geographic_unit', 'ipc_geographic_unit_full',
                'ipc_fewsnet_region', 'ipc_geographic_group',
                'year_month'
            ]
            consol_agg_dict = {
                'location_mention_count': 'sum',
                'unique_location_names': 'sum',
                'unique_cities': 'sum',
                'unique_days': 'max',
                'avg_latitude': 'mean',
                'avg_longitude': 'mean',
                'latitude_std': 'mean',
                'longitude_std': 'mean',
                'primary_gadm2': 'first',
                'primary_gadm3': 'first',
                'match_level': 'first',
                'match_score': 'mean'
            }
            consolidated = combined.groupby(consol_group_cols).agg(consol_agg_dict).reset_index()
            all_aggregations = [consolidated]
            del combined, consolidated
            gc.collect()
            print(f"      Consolidated to {len(all_aggregations[0]):,} aggregated rows", flush=True)

    # Combine all aggregations
    print(f"\n4. Combining {len(all_aggregations)} chunks...")
    if all_aggregations:
        final_df = pd.concat(all_aggregations, ignore_index=True)

        # Final aggregation by (district, month) to combine across chunks
        group_cols = [
            'ipc_country', 'ipc_country_code', 'ipc_fips_code',
            'ipc_district', 'ipc_region',
            'ipc_geographic_unit', 'ipc_geographic_unit_full',
            'ipc_fewsnet_region', 'ipc_geographic_group',
            'year_month'
        ]

        agg_dict = {
            'location_mention_count': 'sum',
            'unique_location_names': 'sum',
            'unique_cities': 'sum',
            'unique_days': 'max',
            'avg_latitude': 'mean',
            'avg_longitude': 'mean',
            'latitude_std': 'mean',
            'longitude_std': 'mean',
            'primary_gadm2': 'first',
            'primary_gadm3': 'first',
            'match_level': 'first',
            'match_score': 'mean'
        }

        final_agg = final_df.groupby(group_cols).agg(agg_dict).reset_index()

        # Summary
        print("\n" + "=" * 80)
        print("Monthly Aggregation Summary - Stage 2")
        print("=" * 80)
        print(f"\nTotal locations processed: {total_processed:,}")
        print(f"Total monthly aggregations: {len(final_agg):,}")
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

        stage1_locations = pd.read_parquet(DISTRICT_DATA_DIR / 'locations_aggregated.parquet')
        stage1_districts = set(stage1_locations['ipc_geographic_unit_full'].unique())
        stage2_districts = set(final_agg['ipc_geographic_unit_full'].unique())

        overlap = stage1_districts & stage2_districts
        only_stage1 = stage1_districts - stage2_districts
        only_stage2 = stage2_districts - stage1_districts

        print(f"\nStage 1 districts: {len(stage1_districts):,}")
        print(f"Stage 2 districts: {len(stage2_districts):,}")
        print(f"Overlap: {len(overlap):,} ({100*len(overlap)/len(stage1_districts):.1f}% of Stage 1)")
        print(f"Only in Stage 1: {len(only_stage1):,}")
        print(f"Only in Stage 2: {len(only_stage2):,}")

        # Save
        print(f"\n5. Saving to {OUTPUT_PARQUET}...")
        final_agg.to_parquet(OUTPUT_PARQUET, index=False)
        print("   [OK] Parquet saved")

        print(f"\n6. Saving to {OUTPUT_CSV}...")
        final_agg.to_csv(OUTPUT_CSV, index=False)
        print("   [OK] CSV saved")

        print("\n" + "=" * 80)
        print("Stage 2 Monthly Location Aggregation Complete")
        print("=" * 80)
    else:
        print("\n   WARNING: No matched records found!")

    print(f"\nEnd time: {datetime.now()}")
    print(f"\nOutput: {OUTPUT_PARQUET}")
    print(f"Next step: Run 04a_stage2_create_ml_dataset.py")


if __name__ == "__main__":
    main()
