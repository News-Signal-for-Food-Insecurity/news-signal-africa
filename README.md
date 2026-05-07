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

## What Is Self-Contained

The following can be reproduced directly from files in this repository with no external data:

- **All model training and evaluation**: `01_train_models.py`, `02_rolling_cv_train.py`
- **All statistical tests**: `04_temporal_shuffle_test.py`
- **All paper figures**: `06_paper_figures.py`, `generate_fig7_new.py`, `08_onset_chronic_report.py`
- **Dataset rebuilding**: `data_preparation/build_dataset.py` reads from `DATA/raw/`

**Not distributed** (external ingestion only):
- Raw GDELT GKG articles (~47 GB CSV archive)
- Raw daily GDELT parquet files

These are only needed to re-run the raw ingestion scripts (`02a_*`, `03a_*`). All pre-processed data in `DATA/raw/` is included and sufficient to reproduce all results.

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

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For map figures (Fig 1): `pip install geopandas`

### 2. Rebuild the modelling dataset (optional — already included)

```bash
python data_preparation/build_dataset.py
```

Outputs to `DATA/dataset.parquet`.

### 3. Run the full pipeline

```bash
# Primary results (2-year rolling window, ~5 min)
python 01_train_models.py

# Window sensitivity check (28-month window, ~3 min)
python 02_rolling_cv_train.py

# Rolling CV null test (1,000 permutations, ~24 h on 16 cores)
python 04_temporal_shuffle_test.py

# All paper figures (Fig 1–7)
python 06_paper_figures.py

# Fig 7: temporal and spatial delta panels
python generate_fig7_new.py

# Fig 8: onset vs. chronic crisis analysis
python 08_onset_chronic_report.py
```

Results → `results/` | Figures → `figures/`

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
