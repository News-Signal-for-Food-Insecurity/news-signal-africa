"""
fix_2020_gadm_names.py
======================
The african_gkg_locations_aligned.parquet has null gadm1_name / gadm2_name / gadm3_name
for all 2020 rows (~7.2M rows, row groups 0-14). The GID columns (gadm1_gid, gadm2_gid,
gadm3_gid) are populated. This script builds a GID->name lookup from the GADM reference
CSV and backfills the name columns, then rewrites the parquet.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

LOCATIONS_FILE = Path(r'C:\GDELT_Africa_Extract\Scripts\district_pipeline\FINAL_PIPELINE - StratifiedSpatialCV\Paper_Global_Model\news-signal-africa\DATA\interim\african_gkg_locations_aligned.parquet')
GADM_LOOKUP    = Path(r'C:\GDELT_Africa_Extract\spatial_alignment\gadm_name_lookup.csv')
OUTPUT_FILE    = LOCATIONS_FILE  # overwrite in place

print("Loading GADM name lookup...")
gadm = pd.read_csv(GADM_LOOKUP, encoding='latin-1')
print(f"  {len(gadm):,} rows, cols: {gadm.columns.tolist()}")

# Build GID -> name dicts for levels 1, 2, 3
gid_to_name1 = gadm.dropna(subset=['name_1']).set_index('gid')['name_1'].to_dict()
gid_to_name2 = gadm.dropna(subset=['name_2']).set_index('gid')['name_2'].to_dict()
gid_to_name3 = gadm.dropna(subset=['name_3']).set_index('gid')['name_3'].to_dict() if 'name_3' in gadm.columns else {}
print(f"  gadm1 entries: {len(gid_to_name1):,}")
print(f"  gadm2 entries: {len(gid_to_name2):,}")
print(f"  gadm3 entries: {len(gid_to_name3):,}")

pf = pq.ParquetFile(LOCATIONS_FILE)
schema = pf.schema_arrow
num_rg = pf.metadata.num_row_groups
print(f"\nProcessing {num_rg} row groups...")

fixed_tables = []
n_fixed = 0

for rg in range(num_rg):
    tbl = pf.read_row_group(rg)
    df = tbl.to_pandas()

    # Only fix rows where gadm2_name is null but gadm2_gid exists
    needs_fix = df['gadm2_name'].isna() & df['gadm2_gid'].notna()
    n_this = needs_fix.sum()

    if n_this > 0:
        df.loc[needs_fix, 'gadm1_name'] = df.loc[needs_fix, 'gadm1_gid'].map(gid_to_name1)
        df.loc[needs_fix, 'gadm2_name'] = df.loc[needs_fix, 'gadm2_gid'].map(gid_to_name2)
        if 'gadm3_name' in df.columns:
            df.loc[needs_fix, 'gadm3_name'] = df.loc[needs_fix, 'gadm3_gid'].map(gid_to_name3)
        n_fixed += n_this
        print(f"  RG {rg:2d}: fixed {n_this:,} rows (gadm2_name now {df['gadm2_name'].notna().sum():,} non-null)", flush=True)
    else:
        print(f"  RG {rg:2d}: no fix needed ({df['gadm2_name'].notna().sum():,} gadm2_name non-null)", flush=True)

    fixed_tables.append(pa.Table.from_pandas(df, schema=schema, preserve_index=False))

print(f"\nTotal rows fixed: {n_fixed:,}")
print(f"Writing {num_rg} row groups back to {OUTPUT_FILE}...")

writer = pq.ParquetWriter(OUTPUT_FILE, schema)
for tbl in fixed_tables:
    writer.write_table(tbl)
writer.close()

print("Done.")
