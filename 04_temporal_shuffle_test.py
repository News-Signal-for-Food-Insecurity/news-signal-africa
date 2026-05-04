"""
Temporal Shuffle Null Test
==========================

Constructs a null distribution for the combined model's PR-AUC by breaking
the temporal alignment between news features and food insecurity outcomes.

METHODOLOGY:
  Null = real model with EXACTLY ONE intervention: news features shuffled.
  Everything else (CatBoost params, early stopping, validation split,
  fold-aware z-score recomputation, categorical encoding, use_best_model)
  is identical to 01_train_models.py.

  For each of N permutations:
    1. For each fold, recompute fold-aware z-scores (identical logic to
       01_train_models.py — full training window baseline, log-scale article
       count, district-level mean/std, no leakage).
    2. Apply a double shuffle to all 19 news feature columns:
         a) Row shuffle (temporal): for each district, permute values across
            time. Each district's marginal totals are preserved.
         b) Column shuffle (spatial): for each time period, permute values
            across districts. Each period's marginal totals are preserved.
       Net effect: global quantity of each news feature is preserved exactly,
       but values are scattered across BOTH time and space — no district-
       level or period-level structure remains aligned with outcomes.
       AR features and the target are NOT shuffled.
    3. Train BOTH AR-only and Combined (AR + shuffled news) models using the
       SAME CatBoost params, same early-stopping config, same validation split,
       and same use_best_model=True logic as 01_train_models.py.
    4. Record PR-AUC on the test set for both models.
    5. Mean PR-AUC across folds = one null draw.

  The real Combined model's mean PR-AUC is then compared directly to the null
  Combined distribution. Because protocol is identical, any gap is attributable
  to the shuffle alone — i.e., to the temporal signal in news features.

KEY DESIGN DECISIONS:
  - Protocol parity: identical CATBOOST_PARAMS_AR / CATBOOST_PARAMS_FULL,
    identical VALIDATION_FRACTION, identical eval_set + use_best_model=True
    as 01_train_models.py. This is the ONLY way the comparison is valid —
    differing protocol confounds the test.
  - Fold-aware z-scores recomputed from raw monthly GDELT (same baseline
    logic as 01_train_models.py). Shuffling globally-precomputed z-scores
    would mix normalization standards.
  - Double shuffle (row then column). Preserves the global quantity of each
    news feature in the dataset exactly, while scattering values across both
    time and space. This tests whether news features carry ANY signal aligned
    with outcomes — temporal, spatial, or district-baseline — beyond AR.
  - random_state = permutation index for full reproducibility.

RUNTIME:
  Approximately the same as the real run (01_train_models.py) per permutation,
  multiplied by N_PERMUTATIONS. With early stopping enabled, expect ~10-20h
  for 100 permutations on 6 folds.

OUTPUTS:
  results/shuffle_test/
    null_distribution.csv    -- N rows: permutation_id, mean_ar_pr_auc,
                                mean_full_pr_auc, mean_delta, fold-level cols
    config.json              -- run configuration + final p-values

REFERENCE:
  Real combined model: results/window_2yr/fold_results.csv
  p-value (PR-AUC) = proportion of null mean_full_pr_auc >= real mean_full_pr_auc
  p-value (delta)  = proportion of null mean_delta       >= real mean_delta
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

class ShuffleTestConfig:
    BASE_DIR     = Path(__file__).parent
    DATA_DIR     = BASE_DIR / "DATA"
    DATA_FILE    = DATA_DIR / "dataset.parquet"
    MONTHLY_FILE = DATA_DIR / "modelling" / "monthly_gdelt_features.parquet"
    FILTER_FILE  = DATA_DIR / "filtering" / "strict_filtered_districts.csv"
    RESULTS_DIR  = BASE_DIR / "results" / "shuffle_test"

    # Reference results for p-value computation
    REAL_RESULTS_FILE = BASE_DIR / "results" / "window_2yr" / "fold_results.csv"

    # Number of permutations: 20 for feasibility check, 100 for paper
    N_PERMUTATIONS = 100

    # Rolling CV parameters (identical to 01_train_models.py)
    TRAIN_WINDOW_MONTHS  = 20
    IPC_PERIOD_MONTHS    = 4
    MONTHLY_DATA_START   = pd.Timestamp("2020-02-01")

    # Features (identical to 01_train_models.py)
    AR_FEATURES = ["ipc_lag_1", "ipc_persistence_2yr", "spatial_lag", "ipc_period", "ipc_country"]

    NEWS_THEMES = [
        "conflict", "displacement", "economic", "food_security",
        "governance", "health", "humanitarian", "weather", "other",
    ]
    RELATIVE_COVERAGE_FEATURES = [f"{t}_relative_coverage" for t in NEWS_THEMES]
    ZSCORE_FEATURES            = [f"{t}_zscore" for t in NEWS_THEMES]
    NEWS_FEATURES              = RELATIVE_COVERAGE_FEATURES + ZSCORE_FEATURES + ["article_count_zscore"]  # 19
    COMBINED_FEATURES          = AR_FEATURES + NEWS_FEATURES                   # 24

    TARGET = "target_crisis_binary"

    EPSILON = 1e-10

    # CatBoost params -- IDENTICAL to 01_train_models.py.
    # Protocol parity is mandatory: any difference confounds the null test.
    CATBOOST_PARAMS_AR = dict(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=30,
    )
    CATBOOST_PARAMS_FULL = dict(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    # Identical to 01_train_models.py
    VALIDATION_FRACTION = 0.40
    THRESHOLD = 0.5


# ============================================================================
# FOLD GENERATION
# ============================================================================

def generate_rolling_folds(df: pd.DataFrame, cfg: ShuffleTestConfig):
    all_starts = sorted(df["ipc_period_start"].unique())
    folds = []
    fold_id = 0

    for test_start in all_starts:
        test_start_ts  = pd.Timestamp(test_start)
        train_end_ts   = test_start_ts - pd.DateOffset(months=cfg.IPC_PERIOD_MONTHS)
        train_start_ts = train_end_ts - pd.DateOffset(months=cfg.TRAIN_WINDOW_MONTHS)

        if train_start_ts < cfg.MONTHLY_DATA_START:
            continue

        train_mask = (
            (df["ipc_period_start"] >= train_start_ts)
            & (df["ipc_period_start"] <= train_end_ts)
        )
        test_mask = df["ipc_period_start"] == test_start_ts

        train_idx = df.index[train_mask].tolist()
        test_idx  = df.index[test_mask].tolist()

        if len(train_idx) < 50 or len(test_idx) < 5:
            continue

        fold_id += 1
        folds.append({
            "fold_id":    fold_id,
            "train_start": train_start_ts,
            "train_end":   train_end_ts,
            "test_start":  test_start_ts,
            "train_idx":   train_idx,
            "test_idx":    test_idx,
        })

    return folds


# ============================================================================
# FOLD-AWARE Z-SCORE RECOMPUTATION
# ============================================================================

def compute_fold_news_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    monthly_gdelt: pd.DataFrame,
    train_end: pd.Timestamp,
    cfg: ShuffleTestConfig,
) -> tuple:
    """
    Fold-aware z-score recomputation using full training window as baseline.
    Identical logic to 01_train_models.py — full window, log-scale article count.
    """
    baseline_gdelt = monthly_gdelt[monthly_gdelt["month"] <= train_end].copy()
    baseline_gdelt["log_article_count"] = np.log1p(baseline_gdelt["article_count"].clip(lower=0))

    cov_cols = [f"{t}_relative_coverage" for t in cfg.NEWS_THEMES]
    all_stat_cols = cov_cols + ["log_article_count"]
    district_stats = (
        baseline_gdelt.groupby("district_id")[all_stat_cols]
        .agg(["mean", "std"])
    )
    district_stats.columns = [f"{c[0]}__{c[1]}" for c in district_stats.columns]

    def apply_zscore(df_split: pd.DataFrame) -> pd.DataFrame:
        df_out = df_split.copy()
        stats = district_stats.reindex(df_out["district_id"].values)
        for theme in cfg.NEWS_THEMES:
            cov_col    = f"{theme}_relative_coverage"
            zscore_col = f"{theme}_zscore"
            means = stats[f"{cov_col}__mean"].values
            stds  = stats[f"{cov_col}__std"].values
            valid = np.isfinite(means) & np.isfinite(stds) & (stds > cfg.EPSILON)
            raw   = df_out[cov_col].values.astype(float)
            df_out[zscore_col] = np.where(valid, (raw - means) / stds, 0.0)
        # article_count_zscore: log-scale z-score (numerator and denominator both log1p)
        log_count    = np.log1p(df_out["article_count"].values.astype(float)) if "article_count" in df_out.columns else np.zeros(len(df_out))
        ac_log_means = stats["log_article_count__mean"].values
        ac_log_stds  = stats["log_article_count__std"].values
        valid_ac = np.isfinite(ac_log_means) & np.isfinite(ac_log_stds) & (ac_log_stds > cfg.EPSILON)
        df_out["article_count_zscore"] = np.where(valid_ac, (log_count - ac_log_means) / ac_log_stds, 0.0)
        return df_out

    return apply_zscore(df_train), apply_zscore(df_test)


# ============================================================================
# TEMPORAL SHUFFLE
# ============================================================================

def shuffle_news_rows_and_columns(df: pd.DataFrame, news_features: list, rng) -> pd.DataFrame:
    """
    Double shuffle: scatters news feature values across BOTH time and space
    while preserving the global quantity of each news feature in the dataset.

    Step 1 — Row shuffle (temporal): for each district independently, permute
      the time ordering of all news feature columns. After this step, each
      district's marginal totals are preserved exactly; temporal alignment
      with outcomes within districts is destroyed.

    Step 2 — Column shuffle (spatial): for each time period independently,
      permute the (already time-shuffled) values across districts. After this
      step, each period's marginal totals are preserved exactly; spatial
      alignment with outcomes is destroyed.

    Net effect: the global quantity of each news feature (sum across the full
    dataset) is preserved exactly, and row/column totals are approximately
    preserved — but the values are scattered across both time and space, so
    no district-level or period-level structure remains aligned with outcomes.

    AR features and the target are unchanged.
    """
    df_shuffled = df.copy().reset_index(drop=True)

    # Step 1: row shuffle within each district (temporal)
    for district_id, group_idx in df_shuffled.groupby("district_id").groups.items():
        row_positions = list(group_idx)
        if len(row_positions) < 2:
            continue
        perm = rng.permutation(len(row_positions))
        original_values = df_shuffled.loc[row_positions, news_features].values
        df_shuffled.loc[row_positions, news_features] = original_values[perm]

    # Step 2: column shuffle within each time period (spatial)
    for period, period_idx in df_shuffled.groupby("ipc_period_start").groups.items():
        row_positions = list(period_idx)
        if len(row_positions) < 2:
            continue
        perm = rng.permutation(len(row_positions))
        original_values = df_shuffled.loc[row_positions, news_features].values
        df_shuffled.loc[row_positions, news_features] = original_values[perm]

    return df_shuffled


# ============================================================================
# SINGLE FOLD -- NULL TEST VERSION
# ============================================================================

def run_fold_shuffled(
    fold_info: dict,
    df: pd.DataFrame,
    monthly_gdelt: pd.DataFrame,
    strict_districts: set,
    rng,
    cfg: ShuffleTestConfig,
) -> tuple:
    """
    Mirrors run_single_fold() in 01_train_models.py EXACTLY.
    Only difference: news features are double-shuffled (row-then-column,
    preserving global quantity while scattering across time and space)
    on the combined train+test rows BEFORE the validation split.

    Steps:
      1. Restrict to strict districts, recompute fold-aware z-scores
      2. Apply row+column double shuffle to news features
      3. Last 40% of training periods = validation set (same as real model)
      4. Train AR-only with CATBOOST_PARAMS_AR + early stopping + use_best_model
         (AR features are NOT shuffled, but trained under identical protocol
         to the real AR model — gives a same-protocol AR null reference)
      5. Train Combined with CATBOOST_PARAMS_FULL + early stopping + use_best_model
      6. Return (ar_pr_auc, full_pr_auc, ar_roc_auc, full_roc_auc)

    Both models train under IDENTICAL protocol to 01_train_models.py.
    The null PR-AUC and ROC-AUC are therefore directly comparable to the real values.
    """
    df_train = df.loc[fold_info["train_idx"]].copy()
    df_test  = df.loc[fold_info["test_idx"]].copy()

    df_train = df_train[df_train["district_id"].isin(strict_districts)].copy()
    df_test  = df_test[df_test["district_id"].isin(strict_districts)].copy()

    if len(df_train) == 0 or len(df_test) == 0:
        return np.nan, np.nan, np.nan, np.nan

    y_test = df_test[cfg.TARGET].values
    if len(np.unique(y_test)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # 1. Fold-aware z-score recomputation (identical to real model)
    df_train, df_test = compute_fold_news_features(
        df_train, df_test, monthly_gdelt, fold_info["train_end"], cfg
    )

    # 2. Apply within-district temporal shuffle to news features.
    #    Shuffle on combined train+test rows so a district's full timeline is
    #    permuted as one unit, then split back into train/test on period boundary.
    train_periods = set(df_train["ipc_period_start"].unique())
    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined = shuffle_news_rows_and_columns(combined, cfg.NEWS_FEATURES, rng)
    df_train = combined[combined["ipc_period_start"].isin(train_periods)].copy()
    df_test  = combined[~combined["ipc_period_start"].isin(train_periods)].copy()

    y_test = df_test[cfg.TARGET].values

    # 3. Validation split: last VALIDATION_FRACTION of training periods
    #    (identical to 01_train_models.py)
    train_period_list = sorted(df_train["ipc_period_start"].unique())
    n_val     = max(1, int(len(train_period_list) * cfg.VALIDATION_FRACTION))
    val_perds = set(train_period_list[-n_val:])
    val_mask  = df_train["ipc_period_start"].isin(val_perds)

    df_fit = df_train[~val_mask]
    df_val = df_train[val_mask]

    y_fit = df_fit[cfg.TARGET].values
    y_val = df_val[cfg.TARGET].values

    # Identical prep_X to 01_train_models.py
    def prep_X(df_subset, features):
        X = df_subset[features].copy()
        X["ipc_period"] = X["ipc_period"].astype(str)
        if "ipc_country" in X.columns:
            X["ipc_country"] = X["ipc_country"].astype(str)
        return X

    # 4. AR-only model — identical protocol to real model
    cat_idx_ar = [cfg.AR_FEATURES.index("ipc_period"),
                  cfg.AR_FEATURES.index("ipc_country")]
    model_ar = CatBoostClassifier(**cfg.CATBOOST_PARAMS_AR, cat_features=cat_idx_ar)
    model_ar.fit(
        prep_X(df_fit, cfg.AR_FEATURES), y_fit,
        eval_set=(prep_X(df_val, cfg.AR_FEATURES), y_val),
        use_best_model=True,
    )
    prob_ar    = model_ar.predict_proba(prep_X(df_test, cfg.AR_FEATURES))[:, 1]
    prauc_ar   = float(average_precision_score(y_test, prob_ar))
    rocauc_ar  = float(roc_auc_score(y_test, prob_ar))

    # 5. Combined model — identical protocol to real model, on AR + shuffled news
    cat_idx_full = [cfg.COMBINED_FEATURES.index("ipc_period"),
                    cfg.COMBINED_FEATURES.index("ipc_country")]
    model_full = CatBoostClassifier(**cfg.CATBOOST_PARAMS_FULL, cat_features=cat_idx_full)
    model_full.fit(
        prep_X(df_fit, cfg.COMBINED_FEATURES), y_fit,
        eval_set=(prep_X(df_val, cfg.COMBINED_FEATURES), y_val),
        use_best_model=True,
    )
    prob_full   = model_full.predict_proba(prep_X(df_test, cfg.COMBINED_FEATURES))[:, 1]
    prauc_full  = float(average_precision_score(y_test, prob_full))
    rocauc_full = float(roc_auc_score(y_test, prob_full))

    return prauc_ar, prauc_full, rocauc_ar, rocauc_full


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = ShuffleTestConfig()

    print("=" * 70)
    print("TEMPORAL SHUFFLE NULL TEST")
    print("=" * 70)
    print(f"N_PERMUTATIONS : {cfg.N_PERMUTATIONS}")
    print(f"News features  : {len(cfg.NEWS_FEATURES)} columns shuffled (19 = 9 rel_cov + 9 zscore + article_count_zscore)")

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(cfg.DATA_FILE)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])

    monthly_gdelt = pd.read_parquet(cfg.MONTHLY_FILE)
    monthly_gdelt["month"] = pd.to_datetime(monthly_gdelt["month"])

    strict_districts = set(pd.read_csv(cfg.FILTER_FILE)["district"].tolist())
    df = df[df["district_id"].isin(strict_districts)].copy()
    print(f"  Dataset: {len(df):,} rows, {df['district_id'].nunique()} districts")

    # Real model reference (both AR and Combined, PR-AUC and ROC-AUC)
    real_mean_pr_auc      = None
    real_mean_ar_auc      = None
    real_mean_delta       = None
    real_mean_roc_full    = None
    real_mean_roc_ar      = None
    real_mean_roc_delta   = None
    if cfg.REAL_RESULTS_FILE.exists():
        real_df = pd.read_csv(cfg.REAL_RESULTS_FILE)
        real_mean_pr_auc    = real_df["full_pr_auc"].mean()
        real_mean_ar_auc    = real_df["ar_pr_auc"].mean()
        real_mean_delta     = real_mean_pr_auc - real_mean_ar_auc
        real_mean_roc_full  = real_df["full_roc_auc"].mean()
        real_mean_roc_ar    = real_df["ar_roc_auc"].mean()
        real_mean_roc_delta = real_mean_roc_full - real_mean_roc_ar
        print(f"  Real AR-only  mean PR-AUC : {real_mean_ar_auc:.4f}  ROC-AUC: {real_mean_roc_ar:.4f}")
        print(f"  Real combined mean PR-AUC : {real_mean_pr_auc:.4f}  ROC-AUC: {real_mean_roc_full:.4f}")
        print(f"  Real observed delta PR-AUC: {real_mean_delta:+.4f}  delta ROC-AUC: {real_mean_roc_delta:+.4f}")

    # Generate folds
    folds = generate_rolling_folds(df, cfg)
    n_folds = len(folds)
    print(f"  Folds: {n_folds}")

    # Output setup
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_config = {
        "n_permutations": cfg.N_PERMUTATIONS,
        "n_folds": n_folds,
        "news_features_shuffled": cfg.NEWS_FEATURES,
        "shuffle_strategy": "row_then_column_double_shuffle",
        "validation_fraction": cfg.VALIDATION_FRACTION,
        "catboost_params_ar": cfg.CATBOOST_PARAMS_AR,
        "catboost_params_full": cfg.CATBOOST_PARAMS_FULL,
        "real_mean_ar_pr_auc": real_mean_ar_auc,
        "real_mean_pr_auc": real_mean_pr_auc,
        "real_mean_ar_roc_auc": real_mean_roc_ar,
        "real_mean_full_roc_auc": real_mean_roc_full,
    }
    with open(cfg.RESULTS_DIR / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Resume support
    results_path = cfg.RESULTS_DIR / "null_distribution.csv"
    completed_perms = set()
    all_rows = []

    if results_path.exists():
        existing = pd.read_csv(results_path)
        completed_perms = set(existing["permutation_id"].tolist())
        all_rows = existing.to_dict("records")
        print(f"  Resuming: {len(completed_perms)} permutations already done.")

    fold_ar_col_names       = [f"fold_{i+1}_ar_pr_auc"    for i in range(n_folds)]
    fold_full_col_names     = [f"fold_{i+1}_full_pr_auc"  for i in range(n_folds)]
    fold_ar_roc_col_names   = [f"fold_{i+1}_ar_roc_auc"   for i in range(n_folds)]
    fold_full_roc_col_names = [f"fold_{i+1}_full_roc_auc" for i in range(n_folds)]

    print(f"\nRunning {cfg.N_PERMUTATIONS} permutations...\n")

    for perm_id in range(cfg.N_PERMUTATIONS):
        if perm_id in completed_perms:
            continue

        rng = np.random.default_rng(seed=perm_id)

        fold_ar_aucs       = []
        fold_full_aucs     = []
        fold_ar_roc_aucs   = []
        fold_full_roc_aucs = []
        for fold_info in folds:
            prauc_ar, prauc_full, rocauc_ar, rocauc_full = run_fold_shuffled(
                fold_info, df, monthly_gdelt, strict_districts, rng, cfg
            )
            fold_ar_aucs.append(prauc_ar)
            fold_full_aucs.append(prauc_full)
            fold_ar_roc_aucs.append(rocauc_ar)
            fold_full_roc_aucs.append(rocauc_full)

        mean_ar       = float(np.nanmean(fold_ar_aucs))
        mean_full     = float(np.nanmean(fold_full_aucs))
        mean_delta    = mean_full - mean_ar
        mean_ar_roc   = float(np.nanmean(fold_ar_roc_aucs))
        mean_full_roc = float(np.nanmean(fold_full_roc_aucs))
        mean_roc_delta = mean_full_roc - mean_ar_roc

        row = {"permutation_id": perm_id,
               "mean_ar_pr_auc": mean_ar,    "mean_full_pr_auc": mean_full,    "mean_delta": mean_delta,
               "mean_ar_roc_auc": mean_ar_roc, "mean_full_roc_auc": mean_full_roc, "mean_roc_delta": mean_roc_delta}
        for j, val in enumerate(fold_ar_aucs):
            row[fold_ar_col_names[j]] = val
        for j, val in enumerate(fold_full_aucs):
            row[fold_full_col_names[j]] = val
        for j, val in enumerate(fold_ar_roc_aucs):
            row[fold_ar_roc_col_names[j]] = val
        for j, val in enumerate(fold_full_roc_aucs):
            row[fold_full_roc_col_names[j]] = val
        all_rows.append(row)

        pd.DataFrame(all_rows).to_csv(results_path, index=False)

        null_deltas = np.array([r["mean_delta"] for r in all_rows], dtype=float)
        print(f"  Perm {perm_id:3d}/{cfg.N_PERMUTATIONS-1} | "
              f"AR={mean_ar:.4f} full={mean_full:.4f} dPR={mean_delta:+.4f} | "
              f"AR_roc={mean_ar_roc:.4f} full_roc={mean_full_roc:.4f} dROC={mean_roc_delta:+.4f} | "
              f"null dPR={np.nanmean(null_deltas):+.4f} +/- {np.nanstd(null_deltas):.4f} "
              f"(n={len(all_rows)})")

    # Final summary — test null distributions against real observed values
    results_df    = pd.DataFrame(all_rows)
    null_deltas   = results_df["mean_delta"].dropna()
    null_ar       = results_df["mean_ar_pr_auc"].dropna()
    null_full     = results_df["mean_full_pr_auc"].dropna()
    null_ar_roc   = results_df["mean_ar_roc_auc"].dropna()   if "mean_ar_roc_auc"   in results_df.columns else pd.Series(dtype=float)
    null_full_roc = results_df["mean_full_roc_auc"].dropna() if "mean_full_roc_auc" in results_df.columns else pd.Series(dtype=float)
    null_roc_deltas = results_df["mean_roc_delta"].dropna()  if "mean_roc_delta"    in results_df.columns else pd.Series(dtype=float)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"  Null AR  PR-AUC  : {null_ar.mean():.4f} +/- {null_ar.std():.4f}")
    print(f"  Null full PR-AUC : {null_full.mean():.4f} +/- {null_full.std():.4f}")
    print(f"  Null delta PR    : {null_deltas.mean():+.4f} +/- {null_deltas.std():.4f}")
    if len(null_full_roc) > 0:
        print(f"  Null AR  ROC-AUC : {null_ar_roc.mean():.4f} +/- {null_ar_roc.std():.4f}")
        print(f"  Null full ROC-AUC: {null_full_roc.mean():.4f} +/- {null_full_roc.std():.4f}")
        print(f"  Null delta ROC   : {null_roc_deltas.mean():+.4f} +/- {null_roc_deltas.std():.4f}")

    p_value_delta     = None
    p_value_prauc     = None
    p_value_rocauc    = None
    p_value_roc_delta = None
    if real_mean_delta is not None:
        # PR-AUC tests
        p_value_prauc = float((null_full >= real_mean_pr_auc).mean())
        p_value_delta = float((null_deltas >= real_mean_delta).mean())

        print(f"  Real full PR-AUC : {real_mean_pr_auc:.4f}")
        print(f"  p-value (PR-AUC) : {p_value_prauc:.4f}")
        print(f"  Real delta PR    : {real_mean_delta:+.4f}")
        print(f"  p-value (dPR)    : {p_value_delta:.4f}")

        # ROC-AUC tests (if available)
        if len(null_full_roc) > 0 and real_mean_roc_full is not None:
            p_value_rocauc    = float((null_full_roc >= real_mean_roc_full).mean())
            p_value_roc_delta = float((null_roc_deltas >= real_mean_roc_delta).mean())
            print(f"  Real full ROC-AUC: {real_mean_roc_full:.4f}")
            print(f"  p-value (ROC-AUC): {p_value_rocauc:.4f}")
            print(f"  Real delta ROC   : {real_mean_roc_delta:+.4f}")
            print(f"  p-value (dROC)   : {p_value_roc_delta:.4f}")

        interp = "NOT significant (p > 0.05)" if p_value_prauc > 0.05 else "SIGNIFICANT (p <= 0.05)"
        print(f"  News temporal signal (PR-AUC): {interp}")

        run_config["real_mean_ar_pr_auc"]    = real_mean_ar_auc
        run_config["real_mean_pr_auc"]       = real_mean_pr_auc
        run_config["real_mean_delta"]        = real_mean_delta
        run_config["null_mean_ar_pr_auc"]    = float(null_ar.mean())
        run_config["null_std_ar_pr_auc"]     = float(null_ar.std())
        run_config["null_mean_full_pr_auc"]  = float(null_full.mean())
        run_config["null_std_full_pr_auc"]   = float(null_full.std())
        run_config["null_mean_delta"]        = float(null_deltas.mean())
        run_config["null_std_delta"]         = float(null_deltas.std())
        run_config["p_value_prauc"]          = p_value_prauc
        run_config["p_value_delta"]          = p_value_delta

        if len(null_full_roc) > 0:
            run_config["real_mean_ar_roc_auc"]    = real_mean_roc_ar
            run_config["real_mean_full_roc_auc"]  = real_mean_roc_full
            run_config["real_mean_roc_delta"]     = real_mean_roc_delta
            run_config["null_mean_ar_roc_auc"]    = float(null_ar_roc.mean())
            run_config["null_std_ar_roc_auc"]     = float(null_ar_roc.std())
            run_config["null_mean_full_roc_auc"]  = float(null_full_roc.mean())
            run_config["null_std_full_roc_auc"]   = float(null_full_roc.std())
            run_config["null_mean_roc_delta"]     = float(null_roc_deltas.mean())
            run_config["null_std_roc_delta"]      = float(null_roc_deltas.std())
            run_config["p_value_rocauc"]          = p_value_rocauc
            run_config["p_value_roc_delta"]       = p_value_roc_delta

        with open(cfg.RESULTS_DIR / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
