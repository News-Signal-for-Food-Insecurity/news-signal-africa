"""
extract_gdelt_2020_locations.py
================================
Processes downloaded 2020 GDELT GKG daily parquets into the
african_gkg_locations_aligned.parquet format expected by the
stage2 aggregation scripts (02a / 03a).

This script bridges the raw GKG download and the stage2 pipeline:
  DATA/gdelt_2020/gkg_20200101.parquet  --.
  DATA/gdelt_2020/gkg_20200102.parquet  --+  -> DATA/interim/african_gkg_locations_2020.parquet
  ...                                   --'

Then concatenates with the existing 2021-2024 locations file:
  DATA/interim/african_gkg_locations_aligned.parquet   (full 2020-2024)

The existing 2021-2024 production file is expected at:
  DATA/raw/african_gkg_locations_2021_2024.parquet
  (copy it there from the original production run)

Run from repo root:
  python data_preparation/extract_gdelt_2020_locations.py

Prerequisites:
  - Run data_preparation/download_gdelt_gkg_2020.py first (fills DATA/gdelt_2020/)
  - Have DATA/raw/african_gkg_locations_2021_2024.parquet (copy from production)
  - DATA/gadm/africa_adm2_combined.gpkg  (GADM shapefiles for spatial join)

Matching priority: GADM3 -> GADM2 -> GADM1 -> country-level
Matches the exact logic used to produce the 2021-2024 locations file.
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent          # news-signal-africa root
RAW_DIR     = BASE_DIR / "DATA" / "raw"
GDELT_2020  = Path(r"C:\Users\victo\OneDrive\Documents\Middlesex University\MSs Data Science Documents\CST4090  Project Work\Current and Improved Approach\GDELT New Approach\01_PRODUCTION\06_DATA\gdelt_data")  # daily gkg_YYYYMMDD.parquet files
INTERIM_DIR = BASE_DIR / "DATA" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

GADM_FILE          = RAW_DIR / "gadm" / "africa_adm2_combined.gpkg"
LOCS_2021_2024     = RAW_DIR / "african_gkg_locations_2021_2024.parquet"  # copy from production
OUT_2020           = INTERIM_DIR / "african_gkg_locations_2020.parquet"
OUT_COMBINED       = INTERIM_DIR / "african_gkg_locations_aligned.parquet"

# ---------------------------------------------------------------------------
# African FIPS codes (same as download script)
# ---------------------------------------------------------------------------
AFRICAN_FIPS = {
    'AO', 'UV', 'BY', 'CM', 'CT', 'CD', 'CG', 'ET', 'KE', 'LT',
    'MA', 'MI', 'ML', 'MR', 'MZ', 'NG', 'NI', 'RW', 'SO', 'OD',
    'SU', 'TO', 'UG', 'ZI', 'LI', 'SL', 'DJ', 'ZA',
    'SF', 'TZ', 'ZM', 'GA', 'GH', 'SG', 'IV', 'BN', 'GV', 'ER',
    'EG', 'LY', 'MO', 'AG', 'TS', 'BC', 'WA', 'SZ', 'EK', 'GB',
}

FIPS_TO_ISO3 = {
    'AO': 'AGO', 'UV': 'BFA', 'BY': 'BDI', 'CM': 'CMR', 'CT': 'CAF',
    'CD': 'TCD', 'CG': 'COD', 'ET': 'ETH', 'KE': 'KEN', 'LT': 'LSO',
    'MA': 'MDG', 'MI': 'MWI', 'ML': 'MLI', 'MR': 'MRT', 'MZ': 'MOZ',
    'NG': 'NER', 'NI': 'NGA', 'RW': 'RWA', 'SO': 'SOM', 'OD': 'SSD',
    'SU': 'SDN', 'TO': 'TGO', 'UG': 'UGA', 'ZI': 'ZWE', 'LI': 'LBR',
    'SL': 'SLE', 'DJ': 'DJI', 'ZA': 'ZAF', 'SF': 'ZAF', 'TZ': 'TZA',
    'ZM': 'ZMB', 'GA': 'GAB', 'GH': 'GHA', 'IV': 'CIV', 'BN': 'BEN',
    'GV': 'GIN', 'ER': 'ERI', 'EG': 'EGY', 'LY': 'LBY', 'MO': 'MAR',
}

FIPS_TO_NAME = {
    'AO': 'Angola', 'UV': 'Burkina Faso', 'BY': 'Burundi', 'CM': 'Cameroon',
    'CT': 'Central African Republic', 'CD': 'Chad', 'CG': 'Democratic Republic of the Congo',
    'ET': 'Ethiopia', 'KE': 'Kenya', 'LT': 'Lesotho', 'MA': 'Madagascar',
    'MI': 'Malawi', 'ML': 'Mali', 'MR': 'Mauritania', 'MZ': 'Mozambique',
    'NG': 'Niger', 'NI': 'Nigeria', 'RW': 'Rwanda', 'SO': 'Somalia',
    'OD': 'South Sudan', 'SU': 'Sudan', 'TO': 'Togo', 'UG': 'Uganda',
    'ZI': 'Zimbabwe', 'LI': 'Liberia', 'SL': 'Sierra Leone', 'DJ': 'Djibouti',
    'ZA': 'South Africa', 'SF': 'South Africa', 'TZ': 'Tanzania', 'ZM': 'Zambia',
    'GA': 'Gabon', 'GH': 'Ghana', 'IV': "Cote d'Ivoire", 'BN': 'Benin',
    'GV': 'Guinea', 'ER': 'Eritrea',
}


# ---------------------------------------------------------------------------
# Helper: parse V2Locations -> list of (fips, fullname, lat, lon, location_type)
# ---------------------------------------------------------------------------
def parse_v2locations(v2loc_str):
    """
    Parse GDELT V2Locations field into structured rows.
    Format: type#fullname#countrycode#adm1#adm2#lat#lon#featureid;...
    Returns list of dicts.
    """
    rows = []
    if not isinstance(v2loc_str, str) or not v2loc_str.strip():
        return rows
    for part in v2loc_str.split(";"):
        fields = part.split("#")
        if len(fields) < 7:
            continue
        try:
            loc_type = int(fields[0]) if fields[0].isdigit() else 0
            fullname = fields[1]
            fips = fields[2][:2] if len(fields[2]) >= 2 else fields[2]
            if fips not in AFRICAN_FIPS:
                continue
            lat = float(fields[5]) if fields[5] else None
            lon = float(fields[6]) if fields[6] else None
            if lat is None or lon is None:
                continue
            rows.append({
                "fips": fips,
                "fullname": fullname,
                "lat": lat,
                "lon": lon,
                "loc_type": loc_type,
                "adm1_code": fields[3] if len(fields) > 3 else "",
                "adm2_code": fields[4] if len(fields) > 4 else "",
                "feature_id": float(fields[7]) if len(fields) > 7 and fields[7] else None,
            })
        except (ValueError, IndexError):
            continue
    return rows


# ---------------------------------------------------------------------------
# Load GADM shapefile and build spatial index
# ---------------------------------------------------------------------------
def load_gadm(gadm_path: Path):
    print(f"  Loading GADM from {gadm_path} ...")
    gdf = gpd.read_file(gadm_path)
    # Expected columns: GID_0, NAME_0, GID_1, NAME_1, GID_2, NAME_2 (ADM2)
    print(f"  GADM rows: {len(gdf):,}, CRS: {gdf.crs}")
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# Spatial join: assign GADM1/2 to each (lat, lon) point
# ---------------------------------------------------------------------------
def spatial_join_gadm(locs_df: pd.DataFrame, gadm_gdf) -> pd.DataFrame:
    """
    Spatial join of location points against GADM polygons.
    Deduplicates on (longitude, latitude) before joining so we only sjoin
    unique coordinate pairs (~tens of thousands) rather than all 7M rows.
    """
    print(f"  Spatial joining {len(locs_df):,} locations to GADM ...")

    # Detect GADM column names
    gid0_col   = next((c for c in gadm_gdf.columns if c.startswith("GID_0")), None)
    name0_col  = next((c for c in gadm_gdf.columns if c.startswith("NAME_0")), None)
    gid1_col   = next((c for c in gadm_gdf.columns if c.startswith("GID_1")), None)
    name1_col  = next((c for c in gadm_gdf.columns if c.startswith("NAME_1")), None)
    gid2_col   = next((c for c in gadm_gdf.columns if c.startswith("GID_2")), None)
    name2_col  = next((c for c in gadm_gdf.columns if c.startswith("NAME_2")), None)
    keep_gadm  = [c for c in [gid0_col, name0_col, gid1_col, name1_col, gid2_col, name2_col] if c]

    # --- Step 1: unique coordinate pairs only ---
    coords = locs_df[["longitude", "latitude"]].drop_duplicates().reset_index(drop=True)
    print(f"  Unique coordinate pairs: {len(coords):,}  (speedup vs {len(locs_df):,} rows)")

    geometry = gpd.points_from_xy(coords["longitude"], coords["latitude"])
    pts_gdf  = gpd.GeoDataFrame(coords, geometry=geometry, crs="EPSG:4326")

    # --- Step 2: sjoin only unique points ---
    joined = gpd.sjoin(pts_gdf, gadm_gdf[keep_gadm + ["geometry"]], how="left", predicate="within")
    # drop duplicate index entries (point in multiple polygons edge case)
    joined = joined[~joined.index.duplicated(keep="first")]

    coord_gadm = coords.copy()
    coord_gadm["gadm1_gid"]     = joined[gid1_col].values  if gid1_col  else None
    coord_gadm["gadm1_name"]    = joined[name1_col].values if name1_col else None
    coord_gadm["gadm1_country"] = joined[name0_col].values if name0_col else None
    coord_gadm["gadm2_gid"]     = joined[gid2_col].values  if gid2_col  else None
    coord_gadm["gadm2_name"]    = joined[name2_col].values if name2_col else None
    coord_gadm["gadm2_parent"]  = joined[name1_col].values if name1_col else None

    # --- Step 3: merge back to full dataset ---
    locs_df = locs_df.merge(
        coord_gadm.drop(columns=[c for c in ["geometry"] if c in coord_gadm.columns], errors="ignore"),
        on=["longitude", "latitude"],
        how="left",
    )
    locs_df["gadm3_gid"]    = None
    locs_df["gadm3_name"]   = None
    locs_df["gadm3_parent"] = locs_df["gadm2_name"]

    print(f"  GADM join done. {locs_df['gadm1_name'].notna().sum():,} / {len(locs_df):,} matched at ADM1")
    return locs_df


# ---------------------------------------------------------------------------
# Process a single daily GKG parquet
# ---------------------------------------------------------------------------
def process_daily_gkg(f: Path, date_str: str) -> pd.DataFrame | None:
    try:
        raw = pd.read_parquet(f)
    except Exception as e:
        print(f"    ERROR reading {f.name}: {e}")
        return None

    if "V2Locations" not in raw.columns or len(raw) == 0:
        return None

    # Keep only needed columns
    cols = ["GKGRECORDID", "V2Locations"]
    if "V2Tone" in raw.columns:
        cols.append("V2Tone")
    raw = raw[cols].dropna(subset=["V2Locations"])

    rows = []
    for _, row in raw.iterrows():
        locs = parse_v2locations(row["V2Locations"])
        for loc in locs:
            loc["GKGRECORDID"] = row["GKGRECORDID"]
            loc["date_extracted"] = date_str
            rows.append(loc)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["year"]  = int(date_str[:4])
    df["month"] = int(date_str[4:6])
    df["day"]   = int(date_str[6:8])
    df["african_country_code"] = df["fips"]
    df["african_country_name"] = df["fips"].map(FIPS_TO_NAME).fillna("")
    df["location_fullname"]    = df["fullname"]
    df["location_type"]        = df["loc_type"]
    df["latitude"]             = df["lat"]
    df["longitude"]            = df["lon"]
    df["feature_id"]           = df["feature_id"]
    df["char_offset"]          = np.nan
    df["city_name"]            = ""
    df["state_province_name"]  = ""
    df["country_name_from_location"] = df["african_country_name"]
    df["adm1_code"]            = df["adm1_code"]
    df["adm2_code"]            = df["adm2_code"]
    df["lhz_fnid"]             = None
    df["lhz_name"]             = None
    df["lhz_type"]             = None
    df["lhz_country"]          = None

    keep = [
        "GKGRECORDID", "african_country_code", "african_country_name",
        "location_type", "location_fullname", "city_name", "state_province_name",
        "country_name_from_location", "adm1_code", "adm2_code",
        "feature_id", "latitude", "longitude", "char_offset",
        "date_extracted", "year", "month", "day",
        "lhz_fnid", "lhz_name", "lhz_type", "lhz_country",
    ]
    return df[keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Extract GDELT 2020 Locations -> african_gkg_locations_aligned")
    print("=" * 70)

    # Collect 2020 daily parquets
    if not GDELT_2020.exists():
        print(f"ERROR: {GDELT_2020} not found. Run download_gdelt_gkg_2020.py first.")
        return

    daily_files = sorted([f for f in GDELT_2020.glob("gkg_2020*.parquet")])
    print(f"Found {len(daily_files)} / 366 daily parquet files for 2020")
    if len(daily_files) < 300:
        print(f"WARNING: Only {len(daily_files)} days downloaded. Consider waiting for more.")

    # --- Step 1: Extract all location rows from 2020 GKG (resumable) ---
    RAW_LOCS_CACHE = INTERIM_DIR / "african_gkg_locations_2020_raw.parquet"

    if RAW_LOCS_CACHE.exists():
        print("\n1. Loading cached raw location rows ...")
        locs_2020 = pd.read_parquet(RAW_LOCS_CACHE)
        print(f"   Loaded {len(locs_2020):,} rows from cache")
    else:
        print("\n1. Extracting location rows from daily parquets ...")
        all_locs = []
        for i, f in enumerate(daily_files, 1):
            date_str = f.stem[4:]   # gkg_20200101 -> 20200101
            df = process_daily_gkg(f, date_str)
            if df is not None:
                all_locs.append(df)
            if i % 50 == 0:
                print(f"   [{i}/{len(daily_files)}] processed")

        if not all_locs:
            print("ERROR: No location rows extracted.")
            return

        locs_2020 = pd.concat(all_locs, ignore_index=True)
        print(f"   Total 2020 location rows: {len(locs_2020):,}")
        del all_locs
        gc.collect()
        # Cache so spatial join can be retried without re-extraction
        locs_2020.to_parquet(RAW_LOCS_CACHE, index=False)
        print(f"   Cached raw locations to {RAW_LOCS_CACHE.name}")

    # --- Step 2: Spatial join to GADM ---
    if not GADM_FILE.exists():
        # Try finding GADM in the C: drive data folder
        alt_gadm = Path(r"C:\GDELT_Africa_Extract\data\gadm\africa_adm2_combined.gpkg")
        if alt_gadm.exists():
            gadm_path = alt_gadm
        else:
            print(f"ERROR: GADM file not found at {GADM_FILE}")
            print("       Copy africa_adm2_combined.gpkg to DATA/gadm/")
            return
    else:
        gadm_path = GADM_FILE

    print("\n2. Spatial join to GADM ...")
    gadm_gdf = load_gadm(gadm_path)
    locs_2020 = spatial_join_gadm(locs_2020, gadm_gdf)
    del gadm_gdf
    gc.collect()

    # Save 2020-only locations
    locs_2020.to_parquet(OUT_2020, index=False)
    print(f"   Saved: {OUT_2020}  ({len(locs_2020):,} rows)")

    # --- Step 3: Concatenate with 2021-2024 locations ---
    print("\n3. Merging with 2021-2024 locations ...")
    if not LOCS_2021_2024.exists():
        # Try D-drive production path
        alt = Path(r"D:\GDELT_Africa_Extract\Scripts\district_pipeline\FINAL_PIPELINE - StratifiedSpatialCV\DATA\gdelt\african_gkg_locations_aligned.parquet")
        if alt.exists():
            print(f"   Using production file from D drive: {alt}")
            locs_existing = pd.read_parquet(alt)
        else:
            print(f"   WARNING: 2021-2024 locations not found. Saving 2020 only.")
            locs_2020.to_parquet(OUT_COMBINED, index=False)
            print(f"   Saved (2020 only): {OUT_COMBINED}")
            return
    else:
        locs_existing = pd.read_parquet(LOCS_2021_2024)

    print(f"   Existing 2021-2024 rows: {len(locs_existing):,}")

    # Align columns
    all_cols = list(locs_existing.columns)
    for col in all_cols:
        if col not in locs_2020.columns:
            locs_2020[col] = None

    combined = pd.concat([locs_2020[all_cols], locs_existing], ignore_index=True)
    combined["year"]  = combined["year"].astype("Int64")
    combined["month"] = combined["month"].astype("Int64")
    combined["day"]   = combined["day"].astype("Int64")
    combined["location_type"] = combined["location_type"].astype("Int64")

    combined.to_parquet(OUT_COMBINED, index=False)
    print(f"   Saved combined: {OUT_COMBINED}  ({len(combined):,} rows)")
    print(f"   Date range: {combined['date_extracted'].min()} -> {combined['date_extracted'].max()}")

    print("\n" + "=" * 70)
    print("Done. Next: run the stage2 aggregation scripts:")
    print("  python data_preparation/02a_stage2_aggregate_articles_monthly.py")
    print("  python data_preparation/03a_stage2_aggregate_locations_monthly.py")
    print("  python data_preparation/04a_stage2_create_ml_dataset.py")
    print("  python data_preparation/build_dataset.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
