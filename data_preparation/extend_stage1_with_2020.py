"""
extend_stage1_with_2020.py
==========================
Extends DATA/raw/stage1_features.parquet (2021-02 to 2024-11) with
2020 IPC data rows (2020-02, 2020-06, 2020-10) from the FEWS NET CSV.

All paths are relative to the news-signal-africa repo root (BASE_DIR).

Run from repo root:
  python data_preparation/extend_stage1_with_2020.py

Inputs (DATA/raw/):
  stage1_features.parquet    -- existing stage1 (2021-2024)
  spatial_weights.parquet    -- IDW weight matrix
  ipcFic_fews_2020_2024.csv  -- FEWS NET IPC phases 2020-2024

Output:
  DATA/raw/stage1_features.parquet  -- overwritten in place (now 2020-2024)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).parent.parent   # news-signal-africa root

# All inputs from DATA/raw/
RAW_DIR     = BASE_DIR / 'DATA' / 'raw'
NEW_IPC_CSV = RAW_DIR / 'ipcFic_fews_2020_2024.csv'
EXISTING_S1 = RAW_DIR / 'stage1_features.parquet'
SPATIAL_WTS = RAW_DIR / 'spatial_weights.parquet'

# Output overwrites stage1_features in place
OUTPUT_FILE = EXISTING_S1

# IPC period boundaries (Feb, Jun, Oct)
IPC_PERIOD_MONTHS = [2, 6, 10]
TARGET_HORIZON_MONTHS = 8


def month_to_ipc_period_start(ts):
    """Map a monthly timestamp to the 4-month IPC period it belongs to."""
    m = ts.month
    y = ts.year
    if m in (2, 3, 4, 5):
        return pd.Timestamp(y, 2, 1)
    if m in (6, 7, 8, 9):
        return pd.Timestamp(y, 6, 1)
    if m >= 10:
        return pd.Timestamp(y, 10, 1)
    return pd.Timestamp(y - 1, 10, 1)  # January -> previous Oct period


def compute_ls_for_new_rows(new_df, existing_df, W_df):
    """
    Compute Ls for 2020 IPC rows using spatial weights matrix.
    Ls_it = IDW-weighted IPC of neighbours at same period t.
    """
    print('  Computing Ls for 2020 rows...')

    # Build IPC lookup from both datasets
    ipc_lookup = {}

    # From existing data (2021+)
    for _, row in existing_df.iterrows():
        key = (row['ipc_geographic_unit_full'], row['ipc_period_start'])
        ipc_lookup[key] = row['ipc_value']

    # From new 2020 data
    for _, row in new_df.iterrows():
        key = (row['ipc_geographic_unit_full'], row['ipc_period_start'])
        ipc_lookup[key] = row['ipc_value']

    print(f'  IPC lookup: {len(ipc_lookup):,} entries')

    # Build neighbor lookup from spatial weights
    meta_cols = [c for c in ['country', 'district'] if c in W_df.columns]
    district_neighbors = {}
    district_weights_dict = {}

    for district_full in W_df.index:
        weights = W_df.loc[district_full].drop(meta_cols, errors='ignore')
        neighbors = weights[weights > 0]
        if len(neighbors) > 0:
            district_neighbors[district_full] = list(neighbors.index)
            district_weights_dict[district_full] = neighbors.to_dict()

    ls_values = []
    for _, row in new_df.iterrows():
        district_full = row['ipc_geographic_unit_full']
        period_start  = row['ipc_period_start']

        if district_full not in district_neighbors:
            ls_values.append(np.nan)
            continue

        neighbors = district_neighbors[district_full]
        weights   = district_weights_dict[district_full]

        weighted_sum = 0.0
        weight_sum   = 0.0
        for neighbor in neighbors:
            key = (neighbor, period_start)
            if key in ipc_lookup:
                w = weights[neighbor]
                weighted_sum += w * ipc_lookup[key]
                weight_sum   += w

        ls_values.append(weighted_sum / weight_sum if weight_sum > 0 else np.nan)

    return ls_values


def main():
    print('=' * 70)
    print('Extending stage1_features with 2020 IPC data')
    print('=' * 70)

    # --- Load existing stage1_features ---
    print('\n1. Loading existing stage1_features...')
    s1_path = EXISTING_S1
    sw_path = SPATIAL_WTS
    print(f'   Using: {s1_path}')
    existing = pd.read_parquet(s1_path)
    existing['ipc_period_start'] = pd.to_datetime(existing['ipc_period_start'])
    print(f'   Existing rows: {len(existing):,}')
    print(f'   Period range : {existing["ipc_period_start"].min().date()} -> {existing["ipc_period_start"].max().date()}')
    print(f'   Columns      : {existing.columns.tolist()[:15]}')

    # --- Load new IPC data ---
    print('\n2. Loading new IPC data (2020-2024)...')
    ipc_new = pd.read_csv(NEW_IPC_CSV)
    ipc_new['projection_start'] = pd.to_datetime(ipc_new['projection_start'])
    ipc_new = ipc_new[ipc_new['scenario_name'] == 'Current Situation'].copy()

    # Filter to IPC period start months only (Feb, Jun, Oct)
    ipc_new = ipc_new[ipc_new['projection_start'].dt.month.isin(IPC_PERIOD_MONTHS)].copy()

    # Filter to 2020 periods only (2020-02, 2020-06, 2020-10)
    ipc_2020 = ipc_new[ipc_new['projection_start'].dt.year == 2020].copy()

    # Only keep districts that appear in existing stage1
    known_districts = set(existing['ipc_geographic_unit_full'].unique())
    ipc_2020 = ipc_2020[ipc_2020['geographic_unit_full_name'].isin(known_districts)].copy()

    print(f'   2020 IPC rows (known districts): {len(ipc_2020):,}')
    print(f'   Period dates: {sorted(ipc_2020["projection_start"].unique())}')
    print(f'   Districts   : {ipc_2020["geographic_unit_full_name"].nunique()}')

    if len(ipc_2020) == 0:
        print('ERROR: No 2020 IPC rows found for known districts!')
        return

    # --- Build 2020 rows matching existing schema ---
    print('\n3. Building 2020 rows matching stage1 schema...')

    # Convert id to string BEFORE rename so ipc_id matches existing object dtype
    ipc_2020['id'] = ipc_2020['id'].astype(str)

    # Map column names from IPC CSV to existing stage1 schema
    ipc_2020 = ipc_2020.rename(columns={
        'geographic_unit_full_name': 'ipc_geographic_unit_full',
        'geographic_unit_name':      'ipc_geographic_unit',
        'projection_start':          'ipc_period_start',
        'projection_end':            'ipc_period_end',
        'country':                   'ipc_country',
        'country_code':              'ipc_country_code',
        'fewsnet_region':            'ipc_fewsnet_region',
        'geographic_group':          'ipc_geographic_group',
        'scenario':                  'ipc_scenario',
        'scenario_name':             'ipc_scenario_name',
        'classification_scale':      'ipc_classification_scale',
        'value':                     'ipc_value',
        'description':               'ipc_description',
        'reporting_date':            'ipc_reporting_date',
        'id':                        'ipc_id',
    }).copy()

    # Add ipc_binary_crisis
    ipc_2020['ipc_binary_crisis'] = (ipc_2020['ipc_value'] >= 3.0).astype(int)

    # Add quarter
    ipc_2020['quarter'] = ipc_2020['ipc_period_start'].dt.quarter

    # Fill columns that exist in existing but not in 2020 data
    # (GDELT-derived columns like article_count, tone_score, etc.)
    gdelt_cols = [c for c in existing.columns if c not in ipc_2020.columns]
    print(f'   Cols to fill with 0/NaN: {len(gdelt_cols)}')

    # Numeric GDELT cols -> 0, others -> NaN
    numeric_existing = existing.select_dtypes(include=[float, int]).columns
    for col in gdelt_cols:
        if col in numeric_existing:
            ipc_2020[col] = 0.0
        else:
            ipc_2020[col] = np.nan

    # --- Load spatial weights ---
    print('\n4. Loading spatial weights...')
    W_df = pd.read_parquet(sw_path)
    print(f'   Spatial weights shape: {W_df.shape}')

    # --- Compute Ls ---
    ls_vals = compute_ls_for_new_rows(ipc_2020, existing, W_df)
    ipc_2020['Ls'] = ls_vals
    print(f'   Ls computed: {sum(v is not None and not np.isnan(v) for v in ls_vals)} valid')

    # --- Compute y_h8 for 2020 rows ---
    print('\n5. Computing y_h8 targets for 2020 rows...')
    # 2020-02 + 8mo = 2020-10 -> crisis available from 2020 data
    # 2020-06 + 8mo = 2021-02 -> crisis available from existing
    # 2020-10 + 8mo = 2021-06 -> crisis available from existing

    # Build future IPC lookup
    all_ipc_lookup = {}
    for _, row in existing.iterrows():
        key = (row['ipc_geographic_unit_full'], row['ipc_period_start'])
        all_ipc_lookup[key] = row['ipc_value']
    for _, row in ipc_2020.iterrows():
        key = (row['ipc_geographic_unit_full'], row['ipc_period_start'])
        all_ipc_lookup[key] = row['ipc_value']

    y_h8_vals = []
    for _, row in ipc_2020.iterrows():
        district = row['ipc_geographic_unit_full']
        current  = row['ipc_period_start']

        target_min = current + relativedelta(months=TARGET_HORIZON_MONTHS)
        target_max = current + relativedelta(months=TARGET_HORIZON_MONTHS + 2)

        # Find IPC period at target date (Feb/Jun/Oct aligned)
        future_val = None
        for period_ts, ipc_val in all_ipc_lookup.items():
            if period_ts[0] == district:
                if target_min <= period_ts[1] <= target_max:
                    future_val = ipc_val
                    break

        if future_val is not None:
            y_h8_vals.append(1.0 if future_val >= 3 else 0.0)
        else:
            y_h8_vals.append(np.nan)

    # Better approach: iterate by period
    y_h8_vals = []
    for _, row in ipc_2020.iterrows():
        district = row['ipc_geographic_unit_full']
        current  = row['ipc_period_start']
        target   = current + relativedelta(months=TARGET_HORIZON_MONTHS)

        # IPC periods: Feb, Jun, Oct - find closest
        key = (district, target)
        future_val = all_ipc_lookup.get(key)

        # Try +2mo if exact period not found
        if future_val is None:
            key2 = (district, target + relativedelta(months=2))
            future_val = all_ipc_lookup.get(key2)

        if future_val is not None:
            y_h8_vals.append(1.0 if future_val >= 3 else 0.0)
        else:
            y_h8_vals.append(np.nan)

    ipc_2020['y_h8'] = y_h8_vals
    valid_targets = sum(1 for v in y_h8_vals if v is not None and not np.isnan(v))
    print(f'   y_h8 valid: {valid_targets} / {len(y_h8_vals)}')

    # Also compute y_h4 and y_h12
    for h in [4, 12]:
        vals = []
        for _, row in ipc_2020.iterrows():
            district = row['ipc_geographic_unit_full']
            current  = row['ipc_period_start']
            target   = current + relativedelta(months=h)
            key = (district, target)
            future_val = all_ipc_lookup.get(key)
            if future_val is None:
                key2 = (district, target + relativedelta(months=2))
                future_val = all_ipc_lookup.get(key2)
            if future_val is not None:
                vals.append(1.0 if future_val >= 3 else 0.0)
            else:
                vals.append(np.nan)
        ipc_2020[f'y_h{h}'] = vals

    # --- Ensure column alignment ---
    print('\n6. Aligning columns with existing schema...')
    for col in existing.columns:
        if col not in ipc_2020.columns:
            ipc_2020[col] = np.nan

    ipc_2020 = ipc_2020[existing.columns].copy()

    # --- Combine ---
    print('\n7. Combining 2020 rows with existing stage1_features...')
    combined = pd.concat([ipc_2020, existing], ignore_index=True)
    combined['ipc_period_start'] = pd.to_datetime(combined['ipc_period_start'])
    combined = combined.sort_values(['ipc_geographic_unit_full', 'ipc_period_start']).reset_index(drop=True)

    print(f'   Combined rows: {len(combined):,}')
    print(f'   Period range : {combined["ipc_period_start"].min().date()} -> {combined["ipc_period_start"].max().date()}')
    print(f'   Districts    : {combined["ipc_geographic_unit_full"].nunique():,}')

    # --- Fix mixed types ---
    print('\n8. Fixing mixed-type columns...')
    # Enforce dtypes from existing schema
    for col in existing.columns:
        if col not in combined.columns:
            continue
        target_dtype = existing[col].dtype
        if pd.api.types.is_datetime64_any_dtype(target_dtype):
            combined[col] = pd.to_datetime(combined[col], errors='coerce')
        elif pd.api.types.is_float_dtype(target_dtype):
            combined[col] = pd.to_numeric(combined[col], errors='coerce').astype(float)
        elif pd.api.types.is_integer_dtype(target_dtype):
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    # --- Save ---
    print(f'\n9. Saving to {OUTPUT_FILE}...')
    combined.to_parquet(OUTPUT_FILE, index=False)
    # ipc_id: force to string in both parts before concat (avoid int64/str conflict)
    # This is already handled but make absolutely sure
    combined['ipc_id'] = combined['ipc_id'].astype(str)

    # Enforce dtypes from existing - be careful with object columns
    for col in combined.columns:
        if col not in existing.columns:
            continue
        src_dtype = existing[col].dtype
        if pd.api.types.is_datetime64_any_dtype(src_dtype):
            combined[col] = pd.to_datetime(combined[col], errors='coerce')
        elif pd.api.types.is_float_dtype(src_dtype):
            combined[col] = pd.to_numeric(combined[col], errors='coerce').astype(float)
        elif pd.api.types.is_integer_dtype(src_dtype):
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
        elif src_dtype == object:
            combined[col] = combined[col].astype(str).replace('nan', np.nan)

    print(f'   Saved {len(combined):,} rows')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    periods = sorted(combined['ipc_period_start'].unique())
    print(f'IPC period starts: {[str(p.date()) for p in periods]}')
    y8_valid = combined['y_h8'].notna().sum()
    y8_crisis = (combined['y_h8'] == 1).sum()
    print(f'y_h8: {y8_valid:,} valid, {y8_crisis:,} crisis ({y8_crisis/y8_valid*100:.1f}%)')
    print(f'Ls valid: {combined["Ls"].notna().sum():,}')
    print('Done.')


if __name__ == '__main__':
    main()
