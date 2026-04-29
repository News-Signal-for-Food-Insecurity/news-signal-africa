"""
Temporal Shuffle Null Test
==========================

Constructs a null distribution for the combined model's PR-AUC by breaking
the temporal alignment between news features and food insecurity outcomes.

METHODOLOGY:
  For each of N permutations:
    1. For each fold, recompute fold-aware z-scores (same as 01_train_models.py)
       using only the training window baseline -- no leakage.
    2. For each district independently, randomly permute the time ordering of
       all 18 news feature columns (9 relative_coverage + 9 fold-aware zscore).
       AR features and the target are NOT shuffled.
       This preserves each district's news distribution and volume, but destroys
       temporal alignment between news signals and IPC outcomes.
    3. Train the combined model (AR + shuffled news) with fixed CatBoost params.
    4. Record PR-AUC on the test set.
    5. Mean PR-AUC across 6 folds = one null draw.

  The real combined model's mean PR-AUC (~0.8181) is then compared to this
  null distribution. If the real model performs only marginally better than the
  shuffled null, this provides strong evidence that news features do not carry
  meaningful temporal signal -- their apparent contribution comes from structural
  patterns (persistence, location effects) rather than timely crisis information.

KEY DESIGN DECISIONS:
  - Fold-aware z-scores are recomputed per fold from raw monthly GDELT data
    (same compute_fold_news_features logic as 01_train_models.py).
    This ensures the null test and the real model use z-scores on the same basis --
    both free from future-period leakage. Shuffling globally-precomputed z-scores
    (dataset.parquet) would mix two different normalization standards and make
    the comparison invalid.
  - Shuffle is applied AFTER fold-aware recomputation, WITHIN each fold's
    combined train+test rows, within each district. This preserves the marginal
    distribution of clean z-scores while destroying their temporal ordering.
  - Fixed hyperparameters (no early stopping) to reduce runtime. Note: the real
    combined model uses early stopping + a 20% validation split, so null models
    are slightly under-optimised. This biases the null distribution downward,
    making the p-value conservative (if anything, harder to reject H0).
  - random_state = permutation index for full reproducibility.

RUNTIME:
  ~100 permutations x 6 folds x ~60-90s/fold ~ 10-15 hours. Use N_PERMUTATIONS=20
  first to verify runtime. Set N_PERMUTATIONS=100 for final paper results.

OUTPUTS:
  results_rolling_cv/shuffle_test/
    null_distribution.csv    -- N rows: permutation_id, mean_pr_auc, fold_1..6_pr_auc
    config.json              -- run configuration + final p-value

REFERENCE:
  Real combined model: results_rolling_cv/window_2yr/fold_results.csv
  mean full_pr_auc = 0.8181 (6 folds, 2-year window, fold-aware z-scores)

  p-value = proportion of null mean_pr_auc >= real mean_pr_auc
  If p > 0.05: news temporal signal not statistically significant.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

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
    TRAIN_WINDOW_MONTHS  = 24
    IPC_PERIOD_MONTHS    = 4
    MONTHLY_DATA_START   = pd.Timestamp("2020-02-01")

    # Features (identical to 01_train_models.py)
    AR_FEATURES = ["ipc_lag_1", "ipc_persistence_2yr", "spatial_lag", "ipc_period"]

    NEWS_THEMES = [
        "conflict", "displacement", "economic", "food_security",
        "governance", "health", "humanitarian", "weather", "other",
    ]
    RELATIVE_COVERAGE_FEATURES = [f"{t}_relative_coverage" for t in NEWS_THEMES]
    ZSCORE_FEATURES            = [f"{t}_zscore" for t in NEWS_THEMES]
    NEWS_FEATURES              = RELATIVE_COVERAGE_FEATURES + ZSCORE_FEATURES  # 18
    COMBINED_FEATURES          = AR_FEATURES + NEWS_FEATURES                   # 22

    TARGET = "target_crisis_binary"

    EPSILON = 1e-10

    # CatBoost -- fixed params, NO early stopping (speed)
    CATBOOST_PARAMS_SHUFFLE = dict(
        iterations=300,
        depth=7,
        learning_rate=0.03,
        random_seed=42,
        verbose=False,
        loss_function="Logloss",
        eval_metric="PRAUC",
    )


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
    Fold-aware z-score recomputation using 12-month rolling baseline from
    training months only. Test rows receive frozen (train-end) statistics.
    Identical logic to 01_train_models.py.
    """
    baseline_start = train_end - pd.DateOffset(months=12)
    baseline_gdelt = monthly_gdelt[
        (monthly_gdelt["month"] >= baseline_start)
        & (monthly_gdelt["month"] <= train_end)
    ].copy()

    cov_cols = [f"{t}_relative_coverage" for t in cfg.NEWS_THEMES]
    district_stats = (
        baseline_gdelt.groupby("district_id")[cov_cols]
        .agg(["mean", "std"])
    )
    district_stats.columns = [f"{c[0]}__{c[1]}" for c in district_stats.columns]

    def apply_zscore(df_split: pd.DataFrame) -> pd.DataFrame:
        df_out = df_split.copy()
        for theme in cfg.NEWS_THEMES:
            cov_col    = f"{theme}_relative_coverage"
            zscore_col = f"{theme}_zscore"
            stats = district_stats.reindex(df_out["district_id"])
            means = stats[f"{cov_col}__mean"].values
            stds  = stats[f"{cov_col}__std"].values
            stds  = np.where(stds < cfg.EPSILON, 1.0, stds)
            df_out[zscore_col] = (df_out[cov_col].values - means) / stds
        return df_out

    return apply_zscore(df_train), apply_zscore(df_test)


# ============================================================================
# TEMPORAL SHUFFLE
# ============================================================================

def shuffle_news_within_districts(df: pd.DataFrame, news_features: list, rng) -> pd.DataFrame:
    """
    For each district independently, randomly permute the time ordering of
    all news feature columns. AR features and target are unchanged.

    Shuffle is applied within each district's rows only, preserving the
    marginal distribution while destroying temporal alignment.
    """
    df_shuffled = df.copy().reset_index(drop=True)

    for district_id, group_idx in df_shuffled.groupby("district_id").groups.items():
        row_positions = list(group_idx)
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
) -> float:
    """
    1. Split fold
    2. Recompute fold-aware z-scores (leakage-free)
    3. Concatenate train+test, shuffle news within districts, re-split
    4. Train combined model (fixed iterations, no early stopping)
    5. Return PR-AUC on test set
    """
    df_train = df.loc[fold_info["train_idx"]].copy()
    df_test  = df.loc[fold_info["test_idx"]].copy()

    # Filter to strict districts
    df_train = df_train[df_train["district_id"].isin(strict_districts)].copy()
    df_test  = df_test[df_test["district_id"].isin(strict_districts)].copy()

    if len(df_train) == 0 or len(df_test) == 0:
        return np.nan

    y_test = df_test[cfg.TARGET].values
    if len(np.unique(y_test)) < 2:
        return np.nan

    # Fold-aware z-score recomputation
    df_train, df_test = compute_fold_news_features(
        df_train, df_test, monthly_gdelt, fold_info["train_end"], cfg
    )

    # Concatenate, shuffle news, re-split
    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined = shuffle_news_within_districts(combined, cfg.NEWS_FEATURES, rng)

    train_periods = df_train["ipc_period_start"].unique()
    df_train_s = combined[combined["ipc_period_start"].isin(train_periods)].copy()
    df_test_s  = combined[~combined["ipc_period_start"].isin(train_periods)].copy()

    # Encode ipc_period
    for d in [df_train_s, df_test_s]:
        d["ipc_period"] = d["ipc_period"].astype(str)

    cat_idx = [cfg.COMBINED_FEATURES.index("ipc_period")]

    X_train = df_train_s[cfg.COMBINED_FEATURES]
    y_train = df_train_s[cfg.TARGET].values
    X_test  = df_test_s[cfg.COMBINED_FEATURES]

    model = CatBoostClassifier(**cfg.CATBOOST_PARAMS_SHUFFLE, cat_features=cat_idx)
    model.fit(Pool(X_train, y_train, cat_features=cat_idx))

    y_proba = model.predict_proba(X_test)[:, 1]
    return float(average_precision_score(y_test, y_proba))


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = ShuffleTestConfig()

    print("=" * 70)
    print("TEMPORAL SHUFFLE NULL TEST")
    print("=" * 70)
    print(f"N_PERMUTATIONS : {cfg.N_PERMUTATIONS}")
    print(f"News features  : {len(cfg.NEWS_FEATURES)} columns shuffled within each district")

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(cfg.DATA_FILE)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])

    monthly_gdelt = pd.read_parquet(cfg.MONTHLY_FILE)
    monthly_gdelt["month"] = pd.to_datetime(monthly_gdelt["month"])

    strict_districts = set(pd.read_csv(cfg.FILTER_FILE)["district"].tolist())
    df = df[df["district_id"].isin(strict_districts)].copy()
    print(f"  Dataset: {len(df):,} rows, {df['district_id'].nunique()} districts")

    # Real model reference
    real_mean_pr_auc = None
    if cfg.REAL_RESULTS_FILE.exists():
        real_df = pd.read_csv(cfg.REAL_RESULTS_FILE)
        real_mean_pr_auc = real_df["full_pr_auc"].mean()
        print(f"  Real combined model mean PR-AUC: {real_mean_pr_auc:.4f}")

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
        "catboost_params": cfg.CATBOOST_PARAMS_SHUFFLE,
        "real_mean_pr_auc": real_mean_pr_auc,
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

    fold_col_names = [f"fold_{i+1}_pr_auc" for i in range(n_folds)]

    print(f"\nRunning {cfg.N_PERMUTATIONS} permutations...\n")

    for perm_id in range(cfg.N_PERMUTATIONS):
        if perm_id in completed_perms:
            continue

        rng = np.random.default_rng(seed=perm_id)

        fold_pr_aucs = []
        for fold_info in folds:
            pr_auc = run_fold_shuffled(
                fold_info, df, monthly_gdelt, strict_districts, rng, cfg
            )
            fold_pr_aucs.append(pr_auc)

        mean_pr_auc = float(np.nanmean(fold_pr_aucs))

        row = {"permutation_id": perm_id, "mean_pr_auc": mean_pr_auc}
        for j, val in enumerate(fold_pr_aucs):
            row[fold_col_names[j]] = val
        all_rows.append(row)

        pd.DataFrame(all_rows).to_csv(results_path, index=False)

        completed_vals = np.array([r["mean_pr_auc"] for r in all_rows], dtype=float)
        print(f"  Perm {perm_id:3d}/{cfg.N_PERMUTATIONS-1} | "
              f"PR-AUC={mean_pr_auc:.4f} | "
              f"Running null={np.nanmean(completed_vals):.4f} +/- {np.nanstd(completed_vals):.4f} "
              f"(n={len(all_rows)})")

    # Final summary
    results_df = pd.DataFrame(all_rows)
    null_pr_aucs = results_df["mean_pr_auc"].dropna()

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"  Null mean   : {null_pr_aucs.mean():.4f}")
    print(f"  Null std    : {null_pr_aucs.std():.4f}")
    print(f"  Null [min, max]: [{null_pr_aucs.min():.4f}, {null_pr_aucs.max():.4f}]")

    if real_mean_pr_auc is not None:
        p_value = float((null_pr_aucs >= real_mean_pr_auc).mean())
        print(f"  Real PR-AUC : {real_mean_pr_auc:.4f}")
        print(f"  p-value     : {p_value:.4f}")
        interp = "NOT significant (p > 0.05)" if p_value > 0.05 else "SIGNIFICANT (p <= 0.05)"
        print(f"  News temporal signal: {interp}")

        run_config["null_mean_pr_auc"] = float(null_pr_aucs.mean())
        run_config["null_std_pr_auc"]  = float(null_pr_aucs.std())
        run_config["p_value"]          = p_value
        with open(cfg.RESULTS_DIR / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
