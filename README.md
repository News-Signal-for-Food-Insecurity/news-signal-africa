# Does News Contain Meaningful Incremental Signal Beyond Autoregressive Structure?
## Evidence from Food Crisis Prediction in Sub-Saharan Africa

This repository contains the complete, self-contained replication code and data for the paper. The pipeline predicts district-level IPC Phase ≥ 3 food crises 8 months ahead across 24 Sub-Saharan African countries using GDELT news features and autoregressive baselines under rolling stratified spatial cross-validation.

---

## Repository Structure

```
news-signal-africa/
├── data_preparation/
│   └── build_dataset.py          # Reconstructs DATA/dataset.parquet from raw sources
├── DATA/
│   ├── raw/
│   │   ├── stage1_features.parquet        # IPC + spatial lag (44,435 district-periods; 2020-2024)
│   │   └── ml_dataset_monthly.parquet     # GDELT monthly news counts (234,405 rows; 2021-2024)
│   ├── filtering/
│   │   ├── strict_filtered_districts.csv  # 586 districts, >= 200 articles/month
│   │   └── district_coverage_stats.csv    # Full coverage statistics
│   ├── modelling/
│   │   └── monthly_gdelt_features.parquet # Monthly news for fold-aware z-score CV
│   ├── dataset.parquet                    # Final modelling dataset (3,905 rows, 533 districts)
│   └── dataset_summary.json               # Dataset metadata
├── 01_train_models.py            # Primary 2-year rolling CV (AR + Combined)
├── 02_rolling_cv_train.py        # Window sensitivity check
├── 04_temporal_shuffle_test.py   # Temporal permutation null test (100 permutations)
├── 04_operational_impact_analysis.py  # Net saves / operational impact (Figure 6)
├── 05_paper_figures.py           # Supplementary figures B-F + model comparison table
├── results/
│   ├── window_2yr/               # Primary results: fold metrics, predictions, models
│   ├── window_sensitivity/       # Sensitivity window results
│   └── shuffle_test/             # Null distribution from permutation test
├── figures/                      # All output figures (PNG)
├── requirements.txt
└── README.md
```

---

## Research Design

**Question:** Does GDELT news coverage provide statistically meaningful predictive signal for food crises beyond what is already captured by the autoregressive structure of IPC food security data?

**Prediction task:**
- Target: IPC Phase ≥ 3 (crisis) at district level, 8 months ahead (L = 2 IPC periods)
- Features: 4 AR features + 18 GDELT news features (9 themes × relative coverage + z-score)
- Evaluation: Rolling stratified spatial cross-validation, primary metric = PR-AUC

**Models:**
| Model | Features | Notes |
|-------|----------|-------|
| AR-only | `ipc_lag_1`, `ipc_persistence_2yr`, `spatial_lag`, `ipc_period` | Autoregressive baseline |
| Combined | AR features + 18 GDELT news features | Full model |

**AR features (4):**
- `ipc_lag_1` — binary IPC crisis indicator lagged one 4-month period
- `ipc_persistence_2yr` — 6-period rolling mean of lagged binary crisis (2-year window)
- `spatial_lag` — IDW-weighted IPC phase of neighbours within 300 km
- `ipc_period` — calendar quarter of period start (categorical)

**GDELT news features (18):**  
Nine thematic categories (conflict, displacement, economic, food\_security, governance, health, humanitarian, weather, other), each represented as:
- `{theme}_relative_coverage` — theme's share of all monthly news (controls for media volume bias)
- `{theme}_zscore` — abnormality relative to district-specific 12-month rolling baseline (fold-aware: recomputed within each CV fold using training data only to prevent leakage)

**Null test:** Within-district temporal shuffle of news feature time ordering (100 permutations); p-value = fraction of null PR-AUC ≥ observed PR-AUC.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the modelling dataset (from raw sources)

```bash
python data_preparation/build_dataset.py
```

This takes ~2 minutes. Outputs to `DATA/dataset.parquet`, `DATA/filtering/`, `DATA/modelling/`.

### 3. Run the full pipeline

```bash
# Primary 2-year rolling CV
python 01_train_models.py

# Window sensitivity
python 02_rolling_cv_train.py

# Temporal shuffle null test (100 permutations, ~20 min)
python 04_temporal_shuffle_test.py

# Operational impact analysis + Figure 6
python 04_operational_impact_analysis.py

# Supplementary figures B-F
python 05_paper_figures.py
```

Results are written to `results/` and figures to `figures/`.

---

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| FEWSNET IPC | Food security phase classifications | 24 countries, 2015–2024 |
| GDELT GKG | Global Knowledge Graph news event data | Africa, 2020–2024 |
| GADM v4.1 | Administrative boundaries (ADM2) | Africa |

Raw GDELT articles (47 GB CSV) are not included. The pre-processed district-level aggregations in `DATA/raw/` are sufficient to reproduce all results.

---

## Results Summary

| Model | Mean PR-AUC | Mean ROC-AUC | Mean F1 |
|-------|-------------|--------------|---------|
| AR-only | 0.816 ± 0.077 | 0.920 ± 0.035 | 0.756 ± 0.073 |
| Combined | 0.835 ± 0.075 | 0.928 ± 0.032 | 0.755 ± 0.070 |
| Delta | +0.019 | — | — |

**Null test:** permutation p-value = 0.00 (100 permutations; null mean PR-AUC = 0.758 ± 0.009).

Dataset: 3,905 district-period observations, 533 districts, 18 countries, 2020–2024 (crisis prevalence 21.2%).

---

## Citation

```
[Citation to be added upon publication]
```

---

## License

Code: MIT License  
Data: Subject to FEWSNET and GDELT terms of use.
