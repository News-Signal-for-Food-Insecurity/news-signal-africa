"""
IPC Reference Preparation Script - DISTRICT LEVEL CORRECTED
Creates a standardized IPC reference dataset with proper district extraction.

KEY CORRECTION:
- Uses geographic_unit_full_name as the unique geographic identifier (NOT geographic_unit_name)
- Extracts district name from the hierarchical full_name structure
- Each (geographic_unit_full_name, period) is one unique observation

Author: Corrected for district-level analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Paths -- all relative to news-signal-africa repo root
BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / 'DATA' / 'raw'
IPC_FILE   = RAW_DIR / 'ipcFic_fews_2020_2024.csv'

OUTPUT_DIR  = RAW_DIR
OUTPUT_FILE = RAW_DIR / 'ipc_reference.parquet'
OUTPUT_CSV  = RAW_DIR / 'ipc_reference.csv'

# FIPS to Country name mapping (GDELT uses FIPS codes)
FIPS_TO_COUNTRY = {
    'AO': 'Angola',
    'UV': 'Burkina Faso',
    'BY': 'Burundi',
    'CM': 'Cameroon',
    'CT': 'Central African Republic',
    'CD': 'Chad',
    'CG': 'Democratic Republic of the Congo',
    'ET': 'Ethiopia',
    'KE': 'Kenya',
    'LT': 'Lesotho',
    'MA': 'Madagascar',
    'MI': 'Malawi',
    'ML': 'Mali',
    'MR': 'Mauritania',
    'MZ': 'Mozambique',
    'NG': 'Niger',
    'NI': 'Nigeria',
    'RW': 'Rwanda',
    'SO': 'Somalia',
    'OD': 'South Sudan',
    'SU': 'Sudan',
    'TO': 'Togo',
    'UG': 'Uganda',
    'ZI': 'Zimbabwe',
}

# Country names for parsing (including variations)
COUNTRY_NAMES = [
    'Ethiopia', 'Kenya', 'Nigeria', 'Democratic Republic of the Congo',
    'Madagascar', 'Uganda', 'Somalia', 'Sudan', 'South Sudan', 'Chad',
    'Mali', 'Niger', 'Burkina Faso', 'Cameroon', 'Mozambique', 'Zimbabwe',
    'Malawi', 'Angola', 'Burundi', 'Lesotho', 'Mauritania', 'Rwanda', 'Togo',
    'Central African Republic', 'Congo', 'The Democratic Republic of the'
]


def normalize_text(text):
    """Normalize text for matching: lowercase, strip, remove special chars"""
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()
    # Remove special characters but keep spaces
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    # Collapse multiple spaces
    text = ' '.join(text.split())
    return text


def extract_district_from_full_name(full_name, unit_name, country):
    """
    Extract district name from geographic_unit_full_name.

    Structure varies by country but generally:
    - Admin records: [District], [Zone], [Region], [Country]
    - LHZ records: [LHZ_name], [District], [Zone], [Country]

    The key insight: geographic_unit_full_name contains the district,
    even for LHZ-labeled records.
    """
    if pd.isna(full_name):
        return unit_name if pd.notna(unit_name) else ''

    full_name = str(full_name).strip()
    unit_name = str(unit_name).strip() if pd.notna(unit_name) else ''

    # Split by comma
    parts = [p.strip() for p in full_name.split(',')]

    # Remove empty parts
    parts = [p for p in parts if p]

    if len(parts) <= 1:
        return unit_name if unit_name else full_name

    # Remove country names from the end
    while parts and parts[-1] in COUNTRY_NAMES:
        parts = parts[:-1]

    if not parts:
        return unit_name if unit_name else full_name

    # Determine if this is an LHZ or admin record
    word_count = len(unit_name.split()) if unit_name else 0

    if word_count >= 4:
        # LHZ record: unit_name is LHZ, district follows
        # Try to find where unit_name ends in full_name
        if full_name.startswith(unit_name):
            # Remove the LHZ name and get the next part
            remainder = full_name[len(unit_name):].lstrip(', ')
            remainder_parts = [p.strip() for p in remainder.split(',')]
            remainder_parts = [p for p in remainder_parts if p and p not in COUNTRY_NAMES]
            if remainder_parts:
                return remainder_parts[0]

        # Fallback: district is typically 2nd from last (before region, country)
        if len(parts) >= 2:
            return parts[-2] if parts[-2] not in COUNTRY_NAMES else parts[-1]
    else:
        # Admin record: unit_name IS the district
        return unit_name if unit_name else parts[0]

    return unit_name if unit_name else parts[0]


def extract_region_from_full_name(full_name):
    """Extract region/province from full_name (typically 2nd-to-last before country)"""
    if pd.isna(full_name):
        return ''

    parts = [p.strip() for p in str(full_name).split(',')]
    parts = [p for p in parts if p]

    # Remove country names
    while parts and parts[-1] in COUNTRY_NAMES:
        parts = parts[:-1]

    if len(parts) >= 2:
        return parts[-1]  # Region is last after removing country

    return ''


def main():
    print("=" * 80)
    print("IPC Reference Preparation - DISTRICT LEVEL")
    print("=" * 80)

    # Load IPC data
    print("\n1. Loading IPC data...")
    ipc = pd.read_csv(IPC_FILE)
    print(f"   Loaded {len(ipc):,} IPC records")

    # Convert dates
    print("\n2. Converting dates...")
    ipc['projection_start'] = pd.to_datetime(ipc['projection_start'])
    ipc['projection_end'] = pd.to_datetime(ipc['projection_end'])
    ipc['reporting_date'] = pd.to_datetime(ipc['reporting_date'])

    # Add country FIPS codes for easier GDELT matching
    print("\n3. Adding FIPS country codes...")
    country_to_fips = {v: k for k, v in FIPS_TO_COUNTRY.items()}
    ipc['fips_code'] = ipc['country'].map(country_to_fips)

    # Calculate period length in days
    ipc['period_length_days'] = (ipc['projection_end'] - ipc['projection_start']).dt.days + 1

    # CRITICAL: Extract district from full_name
    print("\n4. Extracting district names from geographic_unit_full_name...")
    ipc['district'] = ipc.apply(
        lambda row: extract_district_from_full_name(
            row['geographic_unit_full_name'],
            row['geographic_unit_name'],
            row['country']
        ),
        axis=1
    )

    # Extract region
    ipc['region'] = ipc['geographic_unit_full_name'].apply(extract_region_from_full_name)

    # Normalize for matching
    print("\n5. Normalizing geographic names for matching...")
    ipc['district_normalized'] = ipc['district'].apply(normalize_text)
    ipc['full_name_normalized'] = ipc['geographic_unit_full_name'].apply(normalize_text)
    ipc['unit_name_normalized'] = ipc['geographic_unit_name'].apply(normalize_text)

    # Create binary classification
    print("\n6. Creating binary IPC classification...")
    ipc['ipc_binary_crisis'] = (ipc['value'] >= 3.0).astype(int)

    # Create unique observation identifier
    # Each (geographic_unit_full_name, period) is one observation
    ipc['observation_id'] = (
        ipc['geographic_unit_full_name'].astype(str) + '_' +
        ipc['projection_start'].astype(str)
    )

    # Check for duplicates
    n_duplicates = len(ipc) - ipc['observation_id'].nunique()
    print(f"\n   Duplicate (full_name, period) combinations: {n_duplicates}")

    # Select and rename columns for clarity
    print("\n7. Selecting columns...")
    ipc_reference = ipc[[
        'id',  # Original IPC ID
        'country', 'country_code', 'fips_code',
        # Geographic hierarchy
        'geographic_unit_name',  # Original unit name (may be LHZ or admin)
        'geographic_unit_full_name',  # FULL hierarchical name (THE KEY IDENTIFIER)
        'district',  # Extracted district name
        'region',  # Extracted region/province
        # Normalized versions for matching
        'district_normalized',
        'full_name_normalized',
        'unit_name_normalized',
        # FEWS NET metadata
        'fewsnet_region', 'geographic_group',
        # Time period
        'projection_start', 'projection_end', 'period_length_days',
        # IPC values
        'value', 'description', 'ipc_binary_crisis',
        'is_allowing_for_assistance',
        # Additional metadata
        'scenario', 'scenario_name', 'classification_scale',
        'reporting_date', 'collection_schedule', 'collection_status',
        'source_organization', 'source_document',
        # Observation ID
        'observation_id'
    ]].copy()

    # Rename for clarity in downstream scripts
    ipc_reference = ipc_reference.rename(columns={
        'id': 'ipc_id',
        'value': 'ipc_value',
        'description': 'ipc_description'
    })

    # Sort by country and date
    ipc_reference = ipc_reference.sort_values(
        ['country', 'district', 'projection_start']
    ).reset_index(drop=True)

    # Summary statistics
    print("\n" + "=" * 80)
    print("IPC Reference Summary - DISTRICT LEVEL")
    print("=" * 80)
    print(f"\nTotal records: {len(ipc_reference):,}")
    print(f"Unique observations (full_name, period): {ipc_reference['observation_id'].nunique():,}")
    print(f"Date range: {ipc_reference['projection_start'].min()} to {ipc_reference['projection_end'].max()}")
    print(f"Countries: {ipc_reference['country'].nunique()}")
    print(f"Unique geographic_unit_full_name: {ipc_reference['geographic_unit_full_name'].nunique():,}")
    print(f"Unique districts (extracted): {ipc_reference['district'].nunique():,}")

    print(f"\nDistricts per country:")
    district_counts = ipc_reference.groupby('country')['district'].nunique().sort_values(ascending=False)
    for country, count in district_counts.items():
        records = len(ipc_reference[ipc_reference['country'] == country])
        print(f"   {country}: {count} districts, {records:,} records")

    print(f"\nIPC Value distribution:")
    print(ipc_reference['ipc_value'].value_counts().sort_index())

    print(f"\nIPC Binary Crisis distribution:")
    print(ipc_reference['ipc_binary_crisis'].value_counts())

    # Verify district extraction quality
    print("\n" + "=" * 80)
    print("District Extraction Verification")
    print("=" * 80)

    # Sample from different countries
    for country in ['Ethiopia', 'Nigeria', 'Kenya', 'Democratic Republic of the Congo']:
        print(f"\n{country} samples:")
        sample = ipc_reference[ipc_reference['country'] == country].head(3)
        for _, row in sample.iterrows():
            print(f"   Unit: {row['geographic_unit_name'][:50]}...")
            print(f"   Full: {row['geographic_unit_full_name'][:60]}...")
            print(f"   District: {row['district']}")
            print()

    # Save
    print(f"\n8. Saving to {OUTPUT_FILE}...")
    ipc_reference.to_parquet(OUTPUT_FILE, index=False)
    print("   [OK] Parquet saved")

    # Also save as CSV for inspection
    print(f"\n9. Saving CSV to {OUTPUT_CSV}...")
    ipc_reference.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    print("\n" + "=" * 80)
    print("IPC Reference Preparation Complete!")
    print("=" * 80)
    print("\nKey columns for downstream scripts:")
    print("   - ipc_id: Unique IPC assessment ID")
    print("   - geographic_unit_full_name: THE PRIMARY GEOGRAPHIC IDENTIFIER")
    print("   - district: Extracted district name for spatial matching")
    print("   - district_normalized: Normalized district for fuzzy matching")


if __name__ == "__main__":
    main()
