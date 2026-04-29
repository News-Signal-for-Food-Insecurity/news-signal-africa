# Does News Contain Meaningful Incremental Signal Beyond Autoregressive Structure?
## Evidence from Food Crisis Prediction in Sub-Saharan Africa

This repository contains the complete, self-contained replication code and data for the paper. The pipeline predicts district-level IPC Phase ≥ 3 food crises 8 months ahead across Sub-Saharan African countries using GDELT news features and autoregressive baselines under rolling stratified spatial cross-validation.

---

## Repository Structure

```
news-signal-africa/
├── data_preparation/
│   ├── build_dataset.py                 # Reconstructs DATA/dataset.parquet from raw sources
│   ├── 04a_stage2_create_ml_dataset.py  # Merges GDELT articles + locations → monthly parquet
│   └── [other ingestion scripts]        # Raw GDELT ingestion (require external data; see below)
├── DATA/
│   ├── raw/
│   │   ├── stage1_features.parquet        # IPC + spatial lag (district-periods; 2020-2024)
│   │   └── ml_dataset_monthly.parquet     # GDELT monthly news counts (2021-2024)
│   ├── interim/stage2/                    # Intermediate aggregation outputs
│   ├── filtering/
│   │   ├── strict_filtered_districts.csv  # Districts >= 100 articles/month
│   │   ├── district_coverage_stats.csv    # Full coverage statistics
│   │   └── coverage_threshold_sensitivity.csv  # District counts at 50-200 thresholds
│   ├── modelling/
│   │   └── monthly_gdelt_features.parquet # Monthly news for fold-aware z-score CV
│   ├── shapefiles/gadm/
│   │   └── africa_adm2_combined.gpkg      # Admin level 2 geometry (GADM v4.1)
│   ├── dataset.parquet                    # Final modelling dataset
│   └── dataset_summary.json              # Dataset metadata
├── 01_train_models.py            # Primary 2-year rolling CV (AR + Combined)
├── 02_rolling_cv_train.py        # Window sensitivity check (28-month window)
├── 04_temporal_shuffle_test.py   # Temporal permutation null test (100 permutations)
├── 04b_delta_permutation_test.py # Delta permutation test: news incremental value beyond AR
├── 04_operational_impact_analysis.py  # Net saves / operational impact analysis
├── 06_paper_figures.py           # All 7 paper figures (13 panels, PDFs, no captions)
├── results/
│   ├── window_2yr/               # Primary results: fold metrics, predictions, models
│   │   └── district_level_metrics.csv   # Per-district delta PR-AUC, volatility, etc.
│   ├── window_sensitivity/       # Sensitivity window results
│   ├── shuffle_test/             # Null distribution from temporal permutation test
│   └── delta_permutation_test/   # Null distribution for news-given-AR incremental value
├── figures/                      # All output figures (PDF)
├── requirements.txt
└── README.md
```

---

## What Is Self-Contained

The following can be reproduced directly from the files in this repository with no external data:

- **All model training and evaluation**: `01_train_models.py`, `02_rolling_cv_train.py`
- **All statistical tests**: `04_temporal_shuffle_test.py`, `04b_delta_permutation_test.py`
- **All paper figures**: `06_paper_figures.py` (including maps using `DATA/shapefiles/gadm/africa_adm2_combined.gpkg`)
- **Dataset rebuilding**: `data_preparation/build_dataset.py` reads from `DATA/raw/` which is included

**Not distributed** (external ingestion only):
- Raw GDELT GKG articles (~47 GB CSV archive)
- Personal GDELT daily parquet files (referenced in `02a_stage2_aggregate_articles_monthly.py`)

These are only needed to re-run the raw ingestion scripts (`02a_*`, `03a_*`). All pre-processed intermediate outputs are included.

---

## Research Design

**Question:** Does GDELT news coverage provide statistically meaningful predictive signal for food crises beyond what is already captured by the autoregressive structure of IPC data?

**Prediction task:**
- Target: IPC Phase ≥ 3 (crisis) at district level, 8 months ahead (L = 2 IPC periods of 4 months)
- Features: 4 AR features + 18 GDELT news features (9 themes × relative coverage + z-score)
- CV: Rolling stratified spatial cross-validation, primary metric = PR-AUC

**Models:**
| Model | Features |
|-------|----------|
| AR-only | `ipc_lag_1`, `ipc_persistence_2yr`, `spatial_lag`, `ipc_period` |
| Combined | AR features + 18 GDELT news features |

**AR features (4):**
- `ipc_lag_1` — binary IPC crisis indicator lagged one 4-month period (current state)
- `ipc_persistence_2yr` — 6-period rolling mean of lagged binary crisis (2-year persistence window)
- `spatial_lag` — IDW-weighted IPC phase of neighbours within 300 km (contemporaneous; not a future leak since prediction horizon is 8 months ahead)
- `ipc_period` — calendar quarter of period start (categorical: Q1/Q2/Q3/Q4)

**GDELT news features (18):**  
Nine thematic categories (conflict, displacement, economic, food\_security, governance, health, humanitarian, weather, other), each as:
- `{theme}_relative_coverage` — theme share of monthly news (controls for media volume bias)
- `{theme}_zscore` — fold-aware z-score: recomputed within each CV fold using training-window baseline only (prevents data leakage)

**Coverage threshold:** Districts with ≥ 75 mean articles/month included (387 qualifying; 353 in final dataset after IPC merge). See `DATA/filtering/coverage_threshold_sensitivity.csv` for sensitivity at 10–200 thresholds.

**Africa regions (used in figures):**
| Region | Countries |
|--------|-----------|
| East Africa | South Sudan, Somalia, Kenya, Ethiopia, Uganda, Burundi |
| West Africa | Niger, Nigeria, Burkina Faso, Mali |
| Central Africa | Cameroon, DRC, Chad |
| North Africa | Sudan |
| Southern Africa | Madagascar, Zimbabwe, Mozambique, Malawi |

**Volatility definition (Figure 7b):** Fraction of consecutive 4-month IPC periods in which a district's crisis regime changed (transition rate = number of regime changes / (number of periods − 1)). Crisis regimes: Onset (crisis now, no crisis prior), Chronic (crisis in both periods), Recovery (no crisis now, crisis prior), Stable (no crisis in either period).

**Statistical tests:**
- Temporal shuffle null test (100 permutations): shuffles news feature time ordering within each district, tests whether news temporal signal is non-random
- Delta permutation test (100 permutations): shuffles news features and refits both AR and Combined, tests whether news provides incremental value **beyond AR**

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For maps (Fig 1): pip install geopandas
```

### 2. Rebuild dataset from included raw sources

```bash
python data_preparation/build_dataset.py
```

Takes ~1 minute. Outputs to `DATA/dataset.parquet`, `DATA/filtering/`, `DATA/modelling/`.

### 3. Run the full pipeline

```bash
# Primary results (2-year rolling window)
python 01_train_models.py

# Window sensitivity check (28-month window)
python 02_rolling_cv_train.py

# Temporal shuffle null test (100 permutations, ~20 min)
python 04_temporal_shuffle_test.py

# Delta permutation test: news given AR (100 permutations, ~40 min)
python 04b_delta_permutation_test.py

# Operational impact analysis
python 04_operational_impact_analysis.py

# All paper figures (PDFs, no captions)
python 06_paper_figures.py
```

Results → `results/` | Figures → `figures/`

---

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| FEWSNET IPC | Food security phase classifications | Sub-Saharan Africa, 2020–2024 |
| GDELT GKG | Global Knowledge Graph news event data | Africa, 2021–2024 |
| GADM v4.1 | Administrative boundaries (ADM2) | Africa |

Raw GDELT articles (47 GB) are not included. Pre-processed district-level aggregations in `DATA/raw/` are sufficient to reproduce all results.

---

## Current Results (2021–2024, ≥75 articles/month threshold, 7-fold rolling CV)

| Model | Mean PR-AUC | Mean ROC-AUC | Mean F1 |
|-------|-------------|--------------|---------|
| AR-only | 0.800 ± 0.095 | — | — |
| Combined | 0.823 ± 0.073 | — | — |
| Delta | +0.023 | — | — |

**Window sensitivity (28-month window, 5 folds):** AR 0.775 ± 0.102, Combined 0.796 ± 0.084, Delta +0.021

Combined model outperforms AR-only in 5/7 primary folds and 4/5 sensitivity folds.

Dataset: 3,245 district-period observations, 353 districts, 18 countries, 2021–2024.

**Model improvements vs. baseline:**
- `auto_class_weights="Balanced"` on Combined model: corrects 28% crisis class imbalance
- `eval_metric="PRAUC"` for Combined early stopping: directly targets paper's primary metric
- `ipc_country` as categorical feature: country-level baseline crisis rates and media patterns
- `article_count_zscore`: fold-aware volume anomaly feature (total article spike signal)
- Full training-window z-score baseline (vs. last-12-months only): more stable per-district stats
- Z-score fallback changed from corrupted std=1.0 to neutral z=0 for low-coverage districts

---

## Citation

```
[Citation to be added upon publication]
```

---

## License

Code: MIT License  
Data: Subject to FEWSNET and GDELT terms of use.
