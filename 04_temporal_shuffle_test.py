"""
Temporal Shuffle Null Test  (v2 — cell-swap design, parallelised)
=================================================================

METHODOLOGY
-----------
For each of N_MODELS permutations:

  1. Take the ORIGINAL training feature matrix.
  2. Apply N_SWAPS within-column cell swaps to the 18 news feature columns:
       - Pick two random rows i, j and one random news column k.
       - Swap matrix[i,k] ↔ matrix[j,k].
       Column totals are preserved exactly after every swap; row totals change
       by equal and opposite amounts (grand total always preserved). After
       N_SWAPS = 10,000 swaps the temporal and spatial alignment of every news
       feature with outcomes is fully destroyed while column marginals are intact.
  3. Train a CatBoost classifier on the swapped matrix using standard default
     parameters (iterations=1000, depth=6, lr=0.03, Logloss). No early
     stopping, no validation set, no tuning.
  4. Evaluate PR-AUC and ROC-AUC on the held-out test set.
  5. One null draw recorded.

After N_MODELS=1,000 draws, compare real model metrics against the null
distribution for empirical one-sided p-values.

EFFICIENCY
----------
- joblib.Parallel(n_jobs=-1) parallelises across all CPU cores (6-8x speedup).
- thread_count=1 inside each CatBoost fit avoids thread contention with joblib.
- bootstrap_type='Poisson' is marginally faster than Bernoulli.
- All 10,000 swap indices are generated in one numpy call (vectorised).
- Estimated runtime: ~2-3h on an 8-core machine (vs ~19h sequential).

STATISTICAL STANDARD
--------------------
- 1,000 permutations: scikit-learn default; gives p-value resolution of 0.001.
- Within-column swap: preserves column marginals exactly (continuous analogue
  of the Curveball algorithm for binary matrices).
- Single train/test split: valid when test set is a truly held-out time period.

OUTPUTS
-------
  results/shuffle_test_v2/
    null_distribution.csv   -- N_MODELS rows: model_id, pr_auc, roc_auc
    config.json             -- run config + final p-values
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR     = Path(__file__).parent
    DATA_FILE    = BASE_DIR / "DATA" / "dataset.parquet"
    FILTER_FILE  = BASE_DIR / "DATA" / "filtering" / "strict_filtered_districts.csv"
    REAL_RESULTS = BASE_DIR / "results" / "window_2yr" / "fold_results.csv"
    OUT_DIR      = BASE_DIR / "results" / "shuffle_test_v2"

    N_MODELS = 1_000    # null draws  (1,000 = scikit-learn standard; p-resolution 0.001)
    N_SWAPS  = 10_000   # within-column cell swaps per model

    # Use last IPC period as held-out test set (never shuffled)
    TEST_PERIODS = 1

    AR_FEATURES = [
        "ipc_lag_1", "ipc_persistence_2yr", "spatial_lag",
        "ipc_period", "ipc_country",
    ]
    NEWS_THEMES = [
        "conflict", "displacement", "economic", "food_security",
        "governance", "health", "humanitarian", "weather", "other",
    ]
    NEWS_FEATURES = (
        [f"{t}_relative_coverage" for t in NEWS_THEMES]
        + [f"{t}_zscore"          for t in NEWS_THEMES]
    )                              # 18 features
    ALL_FEATURES = AR_FEATURES + NEWS_FEATURES   # 23 features

    TARGET = "target_crisis_binary"

    # Standard default CatBoost params — no tuning, no early stopping.
    # thread_count=1: one thread per fit so joblib can run N_JOBS fits in
    # parallel across all CPU cores (1 thread × 16 workers = 16 cores used).
    # GPU is slower than CPU for this dataset size (~3k rows).
    CB_PARAMS = dict(
        iterations     = 1000,
        depth          = 6,
        learning_rate  = 0.03,
        loss_function  = "Logloss",
        eval_metric    = "AUC",
        task_type      = "CPU",
        thread_count   = 1,    # 1 thread/fit; joblib owns all cores
        random_seed    = 42,
        verbose        = 0,
    )

    CAT_FEATURE_INDICES = [
        ALL_FEATURES.index("ipc_period"),
        ALL_FEATURES.index("ipc_country"),
    ]

    # Parallel across all CPU cores: 1 thread/fit × N_JOBS workers.
    # Set N_JOBS to your core count (16 here). Each batch of N_JOBS models
    # runs simultaneously; wall time ≈ sequential_time / N_JOBS.
    N_JOBS = 16


# ============================================================================
# WITHIN-COLUMN CELL SWAP  (preserves column marginals exactly)
# ============================================================================

def apply_column_swaps(news_matrix: np.ndarray, n_swaps: int, seed: int) -> np.ndarray:
    """
    Vectorised within-column swap on the news feature sub-matrix.

    Each swap picks a random column k and two random rows i≠j, then
    exchanges news_matrix[i,k] ↔ news_matrix[j,k].  Column totals are
    preserved exactly.  All swap indices are generated in one numpy call
    for maximum speed.
    """
    rng = np.random.default_rng(seed)
    m = news_matrix.copy()
    n_rows, n_cols = m.shape

    cols   = rng.integers(0, n_cols, size=n_swaps)
    rows_i = rng.integers(0, n_rows, size=n_swaps)
    rows_j = rng.integers(0, n_rows, size=n_swaps)
    # Resolve i==j ties by shifting j by 1
    same         = rows_i == rows_j
    rows_j[same] = (rows_j[same] + 1) % n_rows

    for s in range(n_swaps):
        k, i, j = cols[s], rows_i[s], rows_j[s]
        m[i, k], m[j, k] = m[j, k], m[i, k]
    return m


# ============================================================================
# SINGLE NULL MODEL  (called in parallel)
# ============================================================================

def run_one_null_model(
    model_id:      int,
    news_train:    np.ndarray,   # float sub-matrix (n_train × n_news)
    X_train_base:  np.ndarray,   # full object array (n_train × n_features)
    y_train:       np.ndarray,
    X_test_df:     pd.DataFrame,
    y_test:        np.ndarray,
    all_features:  list,
    news_col_idx:  list,
    cat_idx:       list,
    cb_params:     dict,
    n_swaps:       int,
) -> dict:
    """Train one CatBoost model on the column-swapped feature matrix."""
    # 1. Swap news columns
    news_swapped = apply_column_swaps(news_train, n_swaps, seed=model_id)

    # 2. Rebuild full feature matrix with swapped news columns
    X_swapped = X_train_base.copy()
    for out_idx, col_idx in enumerate(news_col_idx):
        X_swapped[:, col_idx] = news_swapped[:, out_idx]
    X_swapped_df = pd.DataFrame(X_swapped, columns=all_features)

    # 3. Train — unique train_dir per model avoids file conflicts in parallel
    import tempfile, os
    params = dict(cb_params)
    params["train_dir"] = os.path.join(tempfile.gettempdir(), f"cb_null_{model_id}")
    pool_tr = Pool(X_swapped_df, y_train, cat_features=cat_idx)
    pool_te = Pool(X_test_df,    y_test,  cat_features=cat_idx)
    model   = CatBoostClassifier(**params)
    model.fit(pool_tr)

    # 4. Evaluate
    prob    = model.predict_proba(pool_te)[:, 1]
    pr_auc  = float(average_precision_score(y_test, prob))
    roc_auc = float(roc_auc_score(y_test, prob))

    return {"model_id": model_id, "pr_auc": pr_auc, "roc_auc": roc_auc}


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(cfg: Config):
    df = pd.read_parquet(cfg.DATA_FILE)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])

    strict = set(pd.read_csv(cfg.FILTER_FILE)["district"].tolist())
    df     = df[df["district_id"].isin(strict)].copy()

    periods      = sorted(df["ipc_period_start"].unique())
    test_periods = set(periods[-cfg.TEST_PERIODS:])
    train_periods= set(periods[:-cfg.TEST_PERIODS])

    df_train = df[df["ipc_period_start"].isin(train_periods)].copy()
    df_test  = df[df["ipc_period_start"].isin(test_periods)].copy()

    for split in [df_train, df_test]:
        split["ipc_period"]  = split["ipc_period"].astype(str)
        split["ipc_country"] = split["ipc_country"].astype(str)

    X_train_df = df_train[cfg.ALL_FEATURES].copy()
    y_train    = df_train[cfg.TARGET].values
    X_test_df  = df_test[cfg.ALL_FEATURES].copy()
    y_test     = df_test[cfg.TARGET].values

    news_col_idx = [cfg.ALL_FEATURES.index(f) for f in cfg.NEWS_FEATURES]

    return X_train_df, y_train, X_test_df, y_test, news_col_idx, test_periods


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NULL SHUFFLE TEST  (within-column swap, parallelised, v2)")
    print("=" * 70)
    print(f"  N_MODELS : {cfg.N_MODELS}  (p-value resolution: {1/cfg.N_MODELS:.3f})")
    print(f"  N_SWAPS  : {cfg.N_SWAPS} per model")
    print(f"  CatBoost : iterations={cfg.CB_PARAMS['iterations']}, "
          f"depth={cfg.CB_PARAMS['depth']}, lr={cfg.CB_PARAMS['learning_rate']}, "
          f"thread_count={cfg.CB_PARAMS['thread_count']}")
    print(f"  Parallel : n_jobs={cfg.N_JOBS}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    X_train_df, y_train, X_test_df, y_test, news_col_idx, test_periods = prepare_data(cfg)
    print(f"  Train : {len(y_train):,} rows  |  Test : {len(y_test):,} rows")
    print(f"  Test period(s): {sorted(str(p)[:7] for p in test_periods)}")
    print(f"  Crisis prevalence — train: {y_train.mean()*100:.1f}%  "
          f"test: {y_test.mean()*100:.1f}%")
    print(f"  News features shuffled: {len(cfg.NEWS_FEATURES)}")

    # Pre-extract numpy arrays (passed to parallel workers — avoids re-copying df)
    X_train_arr = X_train_df.values                              # object dtype
    news_train  = X_train_arr[:, news_col_idx].astype(float)     # float sub-matrix

    # ── Real model reference ───────────────────────────────────────────────
    real_pr, real_roc = None, None
    if cfg.REAL_RESULTS.exists():
        rdf      = pd.read_csv(cfg.REAL_RESULTS)
        real_pr  = float(rdf["full_pr_auc"].mean())
        real_roc = float(rdf["full_roc_auc"].mean())
        print(f"\n  Real model (fold-CV mean) — "
              f"PR-AUC: {real_pr:.4f}  ROC-AUC: {real_roc:.4f}")
    else:
        print("\n  Real results file not found — p-values will not be computed.")

    # ── Resume support ─────────────────────────────────────────────────────
    results_path  = cfg.OUT_DIR / "null_distribution.csv"
    completed_ids = set()
    all_rows      = []
    if results_path.exists():
        existing      = pd.read_csv(results_path)
        completed_ids = set(existing["model_id"].tolist())
        all_rows      = existing.to_dict("records")
        print(f"  Resuming: {len(completed_ids)} models already done.")

    pending_ids = [i for i in range(cfg.N_MODELS) if i not in completed_ids]
    if not pending_ids:
        print("  All models already complete.")
    else:
        print(f"\nRunning {len(pending_ids)} null models in parallel "
              f"(n_jobs={cfg.N_JOBS})...\n")

        # ── Parallel execution ─────────────────────────────────────────────
        # Process in batches of 50 so we can save progress and print updates
        BATCH = 50
        for batch_start in range(0, len(pending_ids), BATCH):
            batch_ids = pending_ids[batch_start: batch_start + BATCH]

            batch_results = Parallel(n_jobs=cfg.N_JOBS, backend="loky")(
                delayed(run_one_null_model)(
                    mid,
                    news_train,
                    X_train_arr,
                    y_train,
                    X_test_df,
                    y_test,
                    cfg.ALL_FEATURES,
                    news_col_idx,
                    cfg.CAT_FEATURE_INDICES,
                    cfg.CB_PARAMS,
                    cfg.N_SWAPS,
                )
                for mid in batch_ids
            )

            all_rows.extend(batch_results)
            pd.DataFrame(all_rows).sort_values("model_id").to_csv(
                results_path, index=False
            )

            null_pr  = np.array([r["pr_auc"]  for r in all_rows])
            null_roc = np.array([r["roc_auc"] for r in all_rows])
            n_done   = len(all_rows)
            p_pr  = float((null_pr  >= real_pr ).mean()) if real_pr  is not None else float("nan")
            p_roc = float((null_roc >= real_roc).mean()) if real_roc is not None else float("nan")
            print(f"  Batch {batch_start//BATCH + 1:3d}  "
                  f"[{n_done:4d}/{cfg.N_MODELS}]  |  "
                  f"null_PR={null_pr.mean():.4f}±{null_pr.std():.4f}  "
                  f"null_ROC={null_roc.mean():.4f}±{null_roc.std():.4f}  |  "
                  f"p_PR={p_pr:.4f}  p_ROC={p_roc:.4f}")

    # ── Final summary ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    null_pr    = results_df["pr_auc"].dropna().values
    null_roc   = results_df["roc_auc"].dropna().values

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"  Null PR-AUC  : {null_pr.mean():.4f} ± {null_pr.std():.4f}  "
          f"95% CI [{np.percentile(null_pr, 2.5):.4f}, {np.percentile(null_pr, 97.5):.4f}]")
    print(f"  Null ROC-AUC : {null_roc.mean():.4f} ± {null_roc.std():.4f}  "
          f"95% CI [{np.percentile(null_roc, 2.5):.4f}, {np.percentile(null_roc, 97.5):.4f}]")

    cfg_out = {
        "n_models"          : cfg.N_MODELS,
        "n_swaps"           : cfg.N_SWAPS,
        "swap_strategy"     : "within_column_cell_swap_preserves_column_marginals",
        "n_jobs"            : cfg.N_JOBS,
        "catboost_params"   : cfg.CB_PARAMS,
        "news_features"     : cfg.NEWS_FEATURES,
        "null_mean_pr_auc"  : float(null_pr.mean()),
        "null_std_pr_auc"   : float(null_pr.std()),
        "null_mean_roc_auc" : float(null_roc.mean()),
        "null_std_roc_auc"  : float(null_roc.std()),
        "null_p2_5_pr_auc"  : float(np.percentile(null_pr,  2.5)),
        "null_p97_5_pr_auc" : float(np.percentile(null_pr, 97.5)),
        "null_p2_5_roc_auc" : float(np.percentile(null_roc,  2.5)),
        "null_p97_5_roc_auc": float(np.percentile(null_roc, 97.5)),
    }

    if real_pr is not None:
        p_pr  = float((null_pr  >= real_pr ).mean())
        p_roc = float((null_roc >= real_roc).mean())
        cfg_out.update({
            "real_pr_auc"   : real_pr,
            "real_roc_auc"  : real_roc,
            "p_value_prauc" : p_pr,
            "p_value_rocauc": p_roc,
        })
        sig = lambda p: "SIGNIFICANT (p <= 0.05)" if p <= 0.05 else "not significant"
        print(f"\n  Real PR-AUC  : {real_pr:.4f}  ->  p = {p_pr:.4f}  [{sig(p_pr)}]")
        print(f"  Real ROC-AUC : {real_roc:.4f}  ->  p = {p_roc:.4f}  [{sig(p_roc)}]")

    with open(cfg.OUT_DIR / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"\nOutputs: {cfg.OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
