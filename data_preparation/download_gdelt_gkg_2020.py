"""
download_gdelt_gkg_2020.py
==========================
Downloads GDELT GKG 2.0 daily files for 2020 and saves them as parquet
to DATA/gdelt_2020/ (excluded from git via .gitignore — ~40GB raw data).

GDELT GKG 2.0 files are available at:
  http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip

Uses the master file list to get all 2020 GKG URLs, downloads and
processes each daily batch (15-min intervals merged to daily parquets).

Run from repo root:
  python data_preparation/download_gdelt_gkg_2020.py

Run time: ~2-3 hours for all of 2020 on a fast connection.
After completion, run:
  python data_preparation/aggregate_gdelt_2020_monthly.py
"""

import io
import gc
import os
import sys
import time
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================================
# Configuration
# ============================================================
GDELT_DIR = Path(__file__).parent.parent / "DATA" / "gdelt_2020"
GDELT_DIR.mkdir(parents=True, exist_ok=True)

GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_MASTER_LAST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"

# GKG 2.0 columns (full 27 columns)
GKG_COLS = [
    'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
    'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
    'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
    'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
    'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
    'AllNames', 'Amounts', 'TranslationInfo', 'Extras'
]

# African countries FIPS codes to filter
AFRICAN_FIPS = {
    'AO', 'UV', 'BY', 'CM', 'CT', 'CD', 'CG', 'ET', 'KE', 'LT',
    'MA', 'MI', 'ML', 'MR', 'MZ', 'NG', 'NI', 'RW', 'SO', 'OD',
    'SU', 'TO', 'UG', 'ZI', 'LI', 'SL', 'DJ', 'ZA',
    # Extra African countries
    'SF', 'TZ', 'ZM', 'GA', 'GH', 'SG', 'IV', 'BN', 'GV', 'ER',
    'EG', 'LY', 'MO', 'AG', 'TS', 'BC', 'WA', 'SZ', 'EK', 'GB',
}

YEAR = 2020
MAX_WORKERS = 4
RETRY_COUNT = 3
REQUEST_TIMEOUT = 60

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def get_2020_gkg_urls():
    """Get all 2020 GKG file URLs from GDELT master list."""
    safe_print("Fetching GDELT master file list...")
    try:
        r = requests.get(GDELT_MASTER_URL, timeout=120)
        r.raise_for_status()
        lines = r.text.strip().split('\n')
    except Exception as e:
        safe_print(f"  Error fetching master list: {e}")
        return []

    urls = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            size, md5, url = parts[0], parts[1], parts[2]
            # Filter: 2020 GKG files only
            fname = url.split('/')[-1]
            if fname.startswith('2020') and fname.endswith('.gkg.csv.zip'):
                urls.append(url)

    safe_print(f"Found {len(urls)} GKG files for 2020")
    return urls


def download_gkg_file(url):
    """Download a single GDELT GKG file and return as DataFrame."""
    for attempt in range(RETRY_COUNT):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue

            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                fname = z.namelist()[0]
                with z.open(fname) as f:
                    df = pd.read_csv(
                        f,
                        sep='\t',
                        header=None,
                        names=GKG_COLS,
                        dtype=str,
                        encoding='utf-8',
                        on_bad_lines='skip',
                        low_memory=False,
                    )
            return df
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


def is_african(v2locations_str):
    """Check if V2Locations contains any African FIPS code."""
    if not isinstance(v2locations_str, str) or not v2locations_str:
        return False
    # V2Locations: type#fullname#countrycode#adm1#adm2#lat#lon#featureid;...
    parts = v2locations_str.split(';')
    for part in parts:
        fields = part.split('#')
        if len(fields) >= 3:
            cc = fields[2][:2] if len(fields[2]) >= 2 else fields[2]
            if cc in AFRICAN_FIPS:
                return True
    return False


def filter_african(df):
    """Keep only rows mentioning African countries."""
    if df is None or len(df) == 0:
        return None
    mask = df['V2Locations'].apply(is_african)
    filtered = df[mask].copy()
    return filtered if len(filtered) > 0 else None


def process_day_files(date_str, urls_for_date):
    """
    Download and merge all 15-minute GKG files for a given date,
    keep only Africa-relevant rows, save as parquet.

    Returns: (date_str, n_rows) or (date_str, 0) on failure
    """
    output_path = GDELT_DIR / f"gkg_{date_str}.parquet"

    if output_path.exists():
        # Already downloaded
        try:
            existing = pd.read_parquet(output_path)
            return date_str, len(existing)
        except Exception:
            pass  # Re-download if corrupted

    day_dfs = []
    for url in urls_for_date:
        df = download_gkg_file(url)
        if df is not None:
            filtered = filter_african(df)
            if filtered is not None:
                day_dfs.append(filtered)

    if not day_dfs:
        # Save empty parquet to mark as processed
        empty_df = pd.DataFrame(columns=GKG_COLS)
        empty_df.to_parquet(output_path, index=False)
        return date_str, 0

    combined = pd.concat(day_dfs, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    return date_str, len(combined)


def group_urls_by_date(urls):
    """Group 15-minute GKG file URLs by date (YYYYMMDD)."""
    from collections import defaultdict
    by_date = defaultdict(list)
    for url in urls:
        fname = url.split('/')[-1]
        # fname: YYYYMMDDHHMMSS.gkg.csv.zip
        date_str = fname[:8]
        by_date[date_str].append(url)
    return dict(by_date)


def main():
    print("=" * 70)
    print(f"GDELT GKG Download: {YEAR}")
    print("=" * 70)

    # Ensure output directory exists
    GDELT_DIR.mkdir(parents=True, exist_ok=True)

    # Check which dates already have parquet files
    existing_dates = set()
    for f in GDELT_DIR.iterdir():
        if f.name.startswith(f'gkg_{YEAR}') and f.suffix == '.parquet':
            date_str = f.name[4:12]
            existing_dates.add(date_str)

    print(f"Already downloaded: {len(existing_dates)} days for {YEAR}")

    # Get all 2020 URLs
    all_urls = get_2020_gkg_urls()
    if not all_urls:
        print("ERROR: No URLs found. Exiting.")
        return

    # Group by date
    by_date = group_urls_by_date(all_urls)
    total_dates = len(by_date)

    # Filter out already-downloaded dates
    dates_to_download = {d: urls for d, urls in by_date.items() if d not in existing_dates}
    print(f"Dates to download: {len(dates_to_download)} / {total_dates}")

    if not dates_to_download:
        print("All 2020 GKG data already downloaded!")
        return

    # Process dates
    completed = 0
    failed = []
    start_time = time.time()

    sorted_dates = sorted(dates_to_download.items())
    total = len(sorted_dates)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_day_files, date_str, urls): date_str
            for date_str, urls in sorted_dates
        }

        for future in as_completed(futures):
            date_str = futures[future]
            try:
                result_date, n_rows = future.result()
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed * 60
                eta_min = (total - completed) / rate if rate > 0 else 0
                safe_print(
                    f"  [{completed:3d}/{total}] {result_date}: {n_rows:,} Africa rows "
                    f"| ETA: {eta_min:.0f} min"
                )
            except Exception as e:
                failed.append(date_str)
                safe_print(f"  FAILED {date_str}: {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted {completed}/{total} dates in {elapsed/60:.1f} min")
    if failed:
        print(f"Failed: {failed}")

    print("\nVerifying output...")
    downloaded = sorted([f.name for f in GDELT_DIR.iterdir() if f.name.startswith(f'gkg_{YEAR}')])
    print(f"Downloaded {YEAR} parquet files: {len(downloaded)}")
    print(f"  First: {downloaded[0] if downloaded else 'none'}")
    print(f"  Last:  {downloaded[-1] if downloaded else 'none'}")
    print("\nDone. Run the stage2 monthly aggregation next.")


if __name__ == "__main__":
    main()
