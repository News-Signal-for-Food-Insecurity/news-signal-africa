# Presentation visualization suite

Publication-grade reproductions of the GSSI talk's data graphics, rebuilt from the
project's **real datasets, model outputs and results** — no hardcoded figures. Each
graphic is written to `../figures/presentation/` as a vector **PDF** and a 600-DPI
**PNG**.

## Run

```bash
cd news-signal-africa/viz_presentation
python build_all.py            # render the whole suite
python build_all.py --force-news   # also regenerate the News-only predictions
```

`build_all.py` first ensures the News-only model's predictions exist (training it
via the real pipeline if absent), then renders every figure **twice**:

- `../figures/presentation/` — titled versions (heading, subtitle, source caption)
  for slides / standalone use.
- `../figures/presentation_untitled/` — **paper-ready** versions: figure body and
  legends only, with no title / subtitle / caption (the manuscript supplies the
  caption). Axis labels, tick labels, legends, colour bars, small-multiple panel
  labels and data annotations are kept. Same file names in both folders.

## Design system

`viz_theme.py` is the single source of design tokens (serif display type, the
teal/coral/violet model palette, blue/sage/IPC scales, crimson accent) and the
`save_fig` helper (PDF + 600-DPI PNG, embedded fonts). `data_access.py` is the only
place that reads data; every score, count and bin edge is computed there from the
files below.

## Figure → data source map

| Output file (`figures/presentation/`) | Talk slide | Data source(s) |
|---|---|---|
| `study_scope_datasources_map` | 25 | GADM ADM2/ADM0 (`DATA/shapefiles/gadm/*.gpkg`) |
| `news_volume_choropleth` | 16L | `DATA/modelling/monthly_gdelt_features.parquet` + GADM |
| `ipc_crisis_prevalence_choropleth` | 16R | `DATA/dataset.parquet` (crisis rate/district) + GADM |
| `nigeria_quarterly_food_security` | 17 | monthly GDELT food-security share, Nigeria ADM2 |
| `temporal_heterogeneity_heatmap` | 18 | `DATA/dataset.parquet` relative coverage by IPC period |
| `spatial_heterogeneity_heatmap` | 19 | `DATA/dataset.parquet` relative coverage by country |
| `pr_1_news`, `pr_2_news_ar`, `pr_3_all` | 15/22/23 | curve = mean of per-fold PR curves (predictions); PR-AUC ± SD read from `fold_results.csv` / `news_only_fold_results.csv` |
| `null_distribution_pr_auc`, `_roc_auc` | 26 | `results/shuffle_test_v3/null_distribution.csv` + `fold_results.csv` |
| `predictions_scatter_by_regime` | 29 | `results/window_2yr/fold_predictions.csv` |
| `deltap_violin_by_regime` | 30 | `fold_predictions.csv` (`prob_combined − prob_ar`) |

## Faithful but honest

The figures match the talk's visual language but report the *current pipeline's*
real numbers. Two deliberate differences from the slides:

- The null histogram title shows the real **N = 1,000** null models (the slide
  rounded it to "100"); markers are the mean-of-folds metrics (AR 0.8346 /
  Median-NULL 0.8462 / AR+News 0.8711), consistent with how the null scores are
  built.
- The Precision-Recall figure follows the study's own evaluation protocol: PR-AUC
  is the **mean ± SD of the per-fold average precision**, read straight from the
  canonical result files (News 0.62 ± 0.17, AR 0.83 ± 0.09, AR+News 0.87 ± 0.04) —
  not recomputed in the plotting code, and not the talk's older AdaBoost numbers.
  The plotted curve is the **mean of the per-fold PR curves** with a ±1 SD band, so
  the curve and the label are the same fold-wise aggregation (the standard
  cross-validation curve representation).

## Generated artefacts (News-only model)

`make_news_only_predictions.py` reproduces the study's third model by reusing the
exact rolling-CV, fold-aware z-scoring and CatBoost settings of
`01_train_models.py`, restricted to `Config.NEWS_FEATURES`. It writes two
first-class pipeline artefacts to `results/window_2yr/`:

- `news_only_predictions.csv` — per-observation out-of-sample predictions (`prob_news`).
- `news_only_fold_results.csv` — per-fold metrics (`news_pr_auc`, `news_roc_auc`,
  precision/recall/F1), computed identically to the AR / AR+News `fold_results.csv`.
  This file is the single source of truth for the News-only PR-AUC the figure reports.
