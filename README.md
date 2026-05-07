# Does News Contain Meaningful Incremental Signal Beyond Autoregressive Structure?
## Evidence from Food Crisis Prediction in Sub-Saharan Africa

This repository contains the complete replication code and pre-processed data for the paper. The pipeline predicts district-level IPC Phase ≥ 3 food crises 8 months ahead across Sub-Saharan African countries using GDELT news features evaluated against a pure autoregressive baseline under rolling temporal cross-validation.

---

## Repository Structure

```
news-signal-africa/
├── data_preparation/
│   ├── build_dataset.py                      # Reconstructs DATA/dataset.parquet from raw sources
│   ├── 01_prepare_ipc_reference.py           # Prepares IPC reference data
│   ├── 02a_stage2_aggregate_articles_monthly.py  # GDELT article aggregation (requires raw GKG)
│   ├── 03a_stage2_aggregate_locations_monthly.py # GDELT location aggregation (requires raw GKG)
│   ├── 04a_stage2_create_ml_dataset.py       # Merges aggregations → monthly parquet
│   ├── download_gdelt_gkg_2020.py            # Downloads raw GDELT GKG data
│   ├── extract_gdelt_2020_locations.py       # Extracts location mentions from GKG
│   ├── fix_2020_gadm_names.py                # Harmonises GADM district names
│   ├── concat_locations.py                   # Concatenates location files
│   └── extend_stage1_with_2020.py            # Extends IPC dataset with 2020 data
├── DATA/
│   ├── dataset.parquet                       # Final modelling dataset (all scripts read from here)
│   ├── dataset_summary.json                  # Dataset metadata
│   ├── raw/                                  # Pre-processed source data
│   ├── filtering/                            # Coverage threshold analysis outputs
│   ├── modelling/                            # Fold-aware monthly news features
│   └── shapefiles/gadm/                      # GADM v4.1 ADM2 geometry (for Fig 1 maps)
├── 01_train_models.py            # Primary 2-year rolling CV: AR-only vs AR+News
├── 02_rolling_cv_train.py        # Window sensitivity check (28-month window)
├── 04_temporal_shuffle_test.py   # Rolling CV null test (1,000 permutations)
├── 04_operational_impact_analysis.py  # Operational impact analysis
├── 06_paper_figures.py           # All paper figures (Fig 1–7, PDFs)
├── 08_onset_chronic_report.py    # Fig 8: onset vs. chronic crisis analysis
├── generate_fig7_new.py          # Fig 7a/7b: temporal and spatial delta panels
├── generate_summary_pdf.py       # One-page implementation and results summary PDF
├── results/
│   ├── window_2yr/               # Primary results: fold metrics, predictions, feature importance
│   ├── window_sensitivity/       # 28-month sensitivity window results
│   └── shuffle_test_v3/          # Null distribution (1,000-model rolling CV null test)
├── figures/                      # All output figures (PDF)
├── requirements.txt
└── README.md
```

---

## What Is Included

Everything needed to reproduce all results and figures is in this repository:

| File | Size | Purpose |
|------|------|---------|
| `DATA/dataset.parquet` | ~0.4 MB | Final modelling dataset — all training scripts read from here |
| `DATA/modelling/monthly_gdelt_features.parquet` | ~1.3 MB | Monthly news counts for fold-aware z-score recomputation |
| `DATA/filtering/strict_filtered_districts.csv` | ~0.03 MB | Districts passing the ≥75 articles/month filter |
| `DATA/raw/stage1_features.parquet` | ~6 MB | IPC phases, spatial lag, targets (2020–2024) |
| `DATA/raw/ml_dataset_monthly.parquet` | ~35 MB | Monthly GDELT news counts by district and theme |
| `DATA/raw/spatial_weights.parquet` | ~11 MB | IDW spatial weight matrix |
| `DATA/shapefiles/gadm/africa_adm2_combined.gpkg` | ~62 MB | GADM v4.1 ADM2 boundaries (Fig 1 maps; Git LFS) |
| `DATA/shapefiles/gadm/africa_adm0_basemap.gpkg` | ~37 MB | Africa ADM0 country boundaries (Fig 1 backdrop; Git LFS) |
| `results/window_2yr/` | — | Pre-computed primary results (fold metrics, predictions, feature importance) |
| `results/shuffle_test_v3/` | — | Pre-computed null distribution (1,000 permutations) |
| `figures/` | — | All paper figures as PDFs |

**Not included** (raw ingestion only — not needed to reproduce results):
- Raw GDELT GKG articles (~47 GB)
- Raw daily GDELT parquet files

These are only needed to re-run the raw ingestion scripts in `data_preparation/` (`02a_*`, `03a_*`).

---

## Research Design

**Question:** Does GDELT news coverage provide statistically meaningful predictive signal for food crises beyond what is already captured by the autoregressive structure of IPC data?

**Prediction task:**
- Target: IPC Phase ≥ 3 (crisis) at district level, 8 months ahead (L = 2 IPC periods of 4 months each)
- CV: Rolling temporal cross-validation, 7 folds, ~2-year training window per fold
- Primary metric: PR-AUC (appropriate for imbalanced binary classification)

**Models — identical CatBoost configuration:**

| Model | Features | CatBoost params |
|-------|----------|-----------------|
| AR-only | 5 autoregressive features | iterations=1000, depth=6, lr=0.03, Logloss |
| AR+News | 5 AR + 18 GDELT news features | identical params |
| Null | AR+News with shuffled news | identical params |

Both real models and each null draw use the same CatBoost hyperparameters and training protocol — no early stopping, no class weights — ensuring a fair null comparison.

**AR features (5):**
- `ipc_lag_1` — binary IPC crisis indicator lagged one period (current state)
- `ipc_persistence_2yr` — 6-period rolling mean of lagged binary crisis (2-year persistence)
- `spatial_lag` — IDW-weighted IPC phase of district neighbours within 300 km
- `ipc_period` — calendar quarter of period start (categorical)
- `ipc_country` — country identifier (categorical)

**GDELT news features (18):**
Nine thematic categories (conflict, displacement, economic, food\_security, governance, health, humanitarian, weather, other), each as:
- `{theme}_relative_coverage` — theme share of total monthly news (controls for media volume)
- `{theme}_zscore` — fold-aware z-score computed within the training window only (prevents leakage)

**Coverage threshold:** Districts with ≥ 75 mean articles/month included. See `DATA/filtering/coverage_threshold_sensitivity.csv`.

**Africa regions (used in figures):**
| Region | Countries |
|--------|-----------|
| East Africa | South Sudan, Somalia, Kenya, Ethiopia, Uganda, Burundi |
| West Africa | Niger, Nigeria, Burkina Faso, Mali |
| Central Africa | Cameroon, DRC, Chad |
| North Africa | Sudan |
| Southern Africa | Madagascar, Zimbabwe, Mozambique, Malawi |

**Statistical null test:**
Rolling CV null test (1,000 permutations): within each permutation, all 18 news feature columns are scrambled across both districts and time periods (within-column cell swap, 10,000 swaps per fold), breaking all temporal and spatial alignment with the target. Both AR-only and AR+News are then fully retrained across all 7 folds under this scrambled news, and the mean PR-AUC across folds is recorded as one null draw. The real AR+News mean PR-AUC is ranked against this 1,000-draw null distribution.

---

## Results

### Primary results (7-fold rolling CV, 2-year training window)

| Model | Mean PR-AUC | Mean ROC-AUC |
|-------|-------------|--------------|
| AR-only | 0.8346 | 0.9069 |
| AR+News | 0.8711 | 0.9337 |
| Delta | +0.0365 | +0.0268 |

**Test set:** 2,148 observations across 7 folds spanning Feb 2022 – May 2024.

### Null test (1,000-permutation rolling CV null)

| Metric | Null mean ± std | Real score | p-value |
|--------|-----------------|------------|---------|
| PR-AUC | 0.8452 ± 0.0120 | 0.8711 | **0.008** |
| ROC-AUC | 0.9244 ± 0.0061 | 0.9337 | 0.028 |

AR+News significantly outperforms the null distribution (p = 0.008, one-sided empirical). The AR-only model (0.8346) falls *below* the null median (0.8452), confirming that news features — not the CatBoost architecture — drive the above-chance performance of the combined model.

### Fold-level breakdown (PR-AUC)

| Fold | Test period | AR-only | AR+News | Delta |
|------|-------------|---------|---------|-------|
| 1 | Feb 2022 | 0.889 | 0.899 | +0.009 |
| 2 | Jun 2022 | 0.907 | 0.904 | −0.003 |
| 3 | Oct 2022 | 0.909 | 0.886 | −0.024 |
| 4 | Feb 2023 | 0.862 | 0.885 | +0.023 |
| 5 | Jun 2023 | 0.714 | 0.887 | +0.174 |
| 6 | Oct 2023 | 0.851 | 0.849 | −0.003 |
| 7 | Feb 2024 | 0.710 | 0.788 | +0.079 |

News improves PR-AUC in 4 of 7 folds, with the largest gain in Fold 5 (Jun 2023) where the AR model collapses to 0.714 and news features rescue performance to 0.887.

---

## Quickstart

### Requirements

- **Python 3.9 or later** (uses built-in `list[str]` type hints)
- **Git LFS** — large data files (`DATA/raw/ml_dataset_monthly.parquet`, `DATA/shapefiles/gadm/*.gpkg`) are stored via Git Large File Storage. Install Git LFS before cloning:

```bash
# macOS
brew install git-lfs

# Linux
sudo apt install git-lfs   # or: sudo yum install git-lfs

# Windows
# Download from https://git-lfs.com and run the installer
```

Then enable LFS in your Git installation once:

```bash
git lfs install
```

### 1. Clone the repository

```bash
git clone https://github.com/News-Signal-for-Food-Insecurity/news-signal-africa.git
cd news-signal-africa
```

Git LFS will automatically download the large data files during clone.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Geospatial packages (`geopandas`, `shapely`) are required for Fig 1 choropleth maps only. All other figures work without them.

### 3. Rebuild the modelling dataset (optional)

The final modelling dataset (`DATA/dataset.parquet`) is included. If you want to rebuild it from the included raw sources:

```bash
python data_preparation/build_dataset.py
```

This takes ~1 minute and regenerates `DATA/dataset.parquet`, `DATA/filtering/`, and `DATA/modelling/`.

### 4. Run the pipeline

```bash
# Primary results (2-year rolling window, ~5 min)
python 01_train_models.py

# Window sensitivity check (28-month window, ~3 min)
python 02_rolling_cv_train.py

# Null shuffle test (1,000 permutations; ~24 h on 16 cores, scales with CPU count)
python 04_temporal_shuffle_test.py

# All paper figures (Fig 1–7)
python 06_paper_figures.py
python generate_fig7_new.py

# Fig 8: onset vs. chronic crisis analysis
python 08_onset_chronic_report.py
```

Results → `results/` | Figures → `figures/`

Pre-computed results and all figures are included in the repository so figures can be regenerated without retraining.

---

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| FEWSNET IPC | Food security phase classifications | Sub-Saharan Africa, 2020–2024 |
| GDELT GKG | Global Knowledge Graph news event data | Africa, 2021–2024 |
| GADM v4.1 | Administrative boundaries (ADM2) | Africa |

Raw GDELT articles (47 GB) are not included. All pre-processed district-level aggregations in `DATA/raw/` are sufficient to reproduce all results.

---

## Citation

```
[To be added upon publication]
```

---

## License

Code: MIT License
Data: Subject to FEWSNET and GDELT terms of use.
