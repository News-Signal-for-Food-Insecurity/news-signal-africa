"""
Temporal Shuffle Null Test  (v3 — rolling CV, cell-swap, parallelised)
=======================================================================

METHODOLOGY
-----------
For each of N_MODELS permutations:

  1. Replicate the EXACT same rolling CV folds used in 01_train_models.py
     (same training windows, same test periods, same feature set).
  2. For each fold, apply N_SWAPS within-column cell swaps to the 18 news
     feature columns in that fold's training matrix:
       - Pick two random rows i≠j and one random column k.
       - Swap train_matrix[i,k] ↔ train_matrix[j,k].
       Column totals preserved exactly; temporal/spatial alignment destroyed.
  3. Train a CatBoost classifier on the swapped training data using the
     IDENTICAL params as 01_train_models.py:
       iterations=1000, depth=6, lr=0.03, Logloss, CPU, no early stopping.
  4. Evaluate PR-AUC and ROC-AUC on that fold's test set.
  5. Average PR-AUC and ROC-AUC across all folds → one null draw.

This ensures the null distribution is generated under IDENTICAL conditions
to the real models — same folds, same params, same features — with the
only difference being that news features are randomly scrambled.

EFFICIENCY
----------
- Each null draw = one full CV run (6 folds × 1 model fit).
- joblib.Parallel(n_jobs=N_JOBS) runs N_JOBS draws simultaneously.
- thread_count=1 inside CatBoost avoids thread contention with joblib.
- All swap indices generated in one numpy call (vectorised).
- Batches of 50 with progress saves for resume support.

STATISTICAL STANDARD
--------------------
- 1,000 permutations: scikit-learn standard; p-value resolution = 0.001.
- Within-column swap: preserves column marginals (continuous Curveball).
- Rolling CV: same fold structure as real pipeline → fair comparison.

OUTPUTS
-------
  results/shuffle_test_v3/
    null_distribution.csv   -- N_MODELS rows: model_id, pr_auc, roc_auc
    config.json             -- run config + final p-values
"""

import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION  — must mirror 01_train_models.py exactly
# ============================================================================

class Config:
    BASE_DIR     = Path(__file__).parent
    DATA_FILE    = BASE_DIR / "DATA" / "dataset.parquet"
    FILTER_FILE  = BASE_DIR / "DATA" / "filtering" / "strict_filtered_districts.csv"
    REAL_RESULTS = BASE_DIR / "results" / "window_2yr" / "fold_results.csv"
    OUT_DIR      = BASE_DIR / "results" / "shuffle_test_v3"

    N_MODELS = 1_000    # null draws  (p-value resolution: 0.001)
    N_SWAPS  = 10_000   # within-column cell swaps per fold per model

    # Rolling CV params — must match 01_train_models.py
    TRAIN_WINDOW_MONTHS  = 20
    TEST_HORIZON_PERIODS = 2        # not used in fold gen but kept for clarity
    IPC_PERIOD_MONTHS    = 4
    MONTHLY_DATA_START   = pd.Timestamp("2020-02-01")

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
    )                              # 18 features — identical to 01_train_models.py
    ALL_FEATURES = AR_FEATURES + NEWS_FEATURES   # 23 total

    TARGET = "target_crisis_binary"

    # Standard default CatBoost params — identical to 01_train_models.py.
    # thread_count=1: one thread per fit; joblib owns all cores.
    CB_PARAMS = dict(
        iterations    = 1000,
        depth         = 6,
        learning_rate = 0.03,
        loss_function = "Logloss",
        eval_metric   = "AUC",
        task_type     = "CPU",
        thread_count  = 1,
        random_seed   = 42,
        verbose       = 0,
    )

    # Indices computed after class body — see module-level lines below class

    N_JOBS = max(1, (os.cpu_count() or 4) - 1)  # leave one core free


# Compute index lists after class definition (can't self-reference inside class body)
Config.CAT_FEATURE_INDICES = [
    Config.ALL_FEATURES.index("ipc_period"),
    Config.ALL_FEATURES.index("ipc_country"),
]
Config.NEWS_COL_INDICES = [Config.ALL_FEATURES.index(f) for f in Config.NEWS_FEATURES]


# ============================================================================
# FOLD GENERATION  — mirrors generate_rolling_folds() in 01_train_models.py
# ============================================================================

def generate_rolling_folds(df: pd.DataFrame, cfg: Config) -> list:
    all_starts = sorted(df["ipc_period_start"].unique())
    folds = []
    fold_id = 0
    for test_start in all_starts:
        test_start_ts  = pd.Timestamp(test_start)
        train_end_ts   = test_start_ts - pd.DateOffset(months=cfg.IPC_PERIOD_MONTHS)
        train_start_ts = train_end_ts  - pd.DateOffset(months=cfg.TRAIN_WINDOW_MONTHS)
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
            "fold_id":   fold_id,
            "train_idx": train_idx,
            "test_idx":  test_idx,
        })
    return folds


# ============================================================================
# WITHIN-COLUMN CELL SWAP  (preserves column marginals exactly)
# ============================================================================

def apply_column_swaps(news_matrix: np.ndarray, n_swaps: int, seed: int) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    m      = news_matrix.copy()
    n_rows, n_cols = m.shape
    cols   = rng.integers(0, n_cols, size=n_swaps)
    rows_i = rng.integers(0, n_rows, size=n_swaps)
    rows_j = rng.integers(0, n_rows, size=n_swaps)
    same         = rows_i == rows_j
    rows_j[same] = (rows_j[same] + 1) % n_rows
    for s in range(n_swaps):
        k, i, j = cols[s], rows_i[s], rows_j[s]
        m[i, k], m[j, k] = m[j, k], m[i, k]
    return m


# ============================================================================
# SINGLE NULL DRAW  — one full CV run with scrambled news (called in parallel)
# ============================================================================

def run_one_null_model(
    model_id:   int,
    fold_data:  list,    # list of (X_train_arr, y_train, X_test_df, y_test)
    cat_idx:    list,
    news_idx:   list,
    all_feats:  list,
    cb_params:  dict,
    n_swaps:    int,
) -> dict:
    """
    Run one full CV with scrambled news features.
    Seed = model_id * 1000 + fold_id so each fold gets a different scramble
    but results are fully reproducible.
    """
    pr_aucs  = []
    roc_aucs = []

    for fold_id, (X_train_arr, y_train, X_test_df, y_test) in enumerate(fold_data):
        # Scramble news columns in this fold's training matrix
        news_train   = X_train_arr[:, news_idx].astype(float)
        news_swapped = apply_column_swaps(news_train, n_swaps,
                                          seed=model_id * 1000 + fold_id)

        X_swapped = X_train_arr.copy()
        for out_pos, col_idx in enumerate(news_idx):
            X_swapped[:, col_idx] = news_swapped[:, out_pos]
        X_swapped_df = pd.DataFrame(X_swapped, columns=all_feats)

        # Unique train_dir per worker avoids CatBoost file conflicts
        params = dict(cb_params)
        params["train_dir"] = os.path.join(
            tempfile.gettempdir(), f"cb_null_{model_id}_f{fold_id}"
        )

        pool_tr = Pool(X_swapped_df, y_train, cat_features=cat_idx)
        pool_te = Pool(X_test_df,    y_test,  cat_features=cat_idx)
        model   = CatBoostClassifier(**params)
        model.fit(pool_tr)

        prob    = model.predict_proba(pool_te)[:, 1]
        if y_test.sum() > 0:
            pr_aucs.append(float(average_precision_score(y_test, prob)))
            roc_aucs.append(float(roc_auc_score(y_test, prob)))

    return {
        "model_id": model_id,
        "pr_auc":   float(np.mean(pr_aucs))  if pr_aucs  else float("nan"),
        "roc_auc":  float(np.mean(roc_aucs)) if roc_aucs else float("nan"),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NULL SHUFFLE TEST  (rolling CV, within-column swap, v3)")
    print("=" * 70)
    print(f"  N_MODELS : {cfg.N_MODELS}  (p-resolution: {1/cfg.N_MODELS:.3f})")
    print(f"  N_SWAPS  : {cfg.N_SWAPS} per fold per model")
    print(f"  CatBoost : iterations={cfg.CB_PARAMS['iterations']}, "
          f"depth={cfg.CB_PARAMS['depth']}, lr={cfg.CB_PARAMS['learning_rate']}")
    print(f"  Parallel : n_jobs={cfg.N_JOBS}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_parquet(cfg.DATA_FILE)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])
    strict = set(pd.read_csv(cfg.FILTER_FILE)["district"].tolist())
    df = df[df["district_id"].isin(strict)].copy()
    df["ipc_period"]  = df["ipc_period"].astype(str)
    df["ipc_country"] = df["ipc_country"].astype(str)
    print(f"  Dataset: {len(df):,} rows, {df['district_id'].nunique()} districts")

    # ── Generate folds (identical to 01_train_models.py) ──────────────────
    folds = generate_rolling_folds(df, cfg)
    print(f"  Folds: {len(folds)}")

    # Pre-build fold data arrays once — reused across all null models
    fold_data = []
    for fold in folds:
        df_tr = df.loc[fold["train_idx"]][cfg.ALL_FEATURES + [cfg.TARGET]].copy()
        df_te = df.loc[fold["test_idx"]][cfg.ALL_FEATURES + [cfg.TARGET]].copy()
        X_train_arr = df_tr[cfg.ALL_FEATURES].values          # object dtype
        y_train     = df_tr[cfg.TARGET].values
        X_test_df   = df_te[cfg.ALL_FEATURES].copy()
        y_test      = df_te[cfg.TARGET].values
        fold_data.append((X_train_arr, y_train, X_test_df, y_test))

    total_train = sum(len(fd[1]) for fd in fold_data)
    total_test  = sum(len(fd[3]) for fd in fold_data)
    print(f"  Total rows across folds — train: {total_train:,}  test: {total_test:,}")
    print(f"  News features shuffled: {len(cfg.NEWS_FEATURES)}")

    # ── Real model reference ───────────────────────────────────────────────
    real_pr, real_roc = None, None
    if cfg.REAL_RESULTS.exists():
        rdf      = pd.read_csv(cfg.REAL_RESULTS)
        real_pr  = float(rdf["full_pr_auc"].mean())
        real_roc = float(rdf["full_roc_auc"].mean())
        print(f"\n  Real AR+News (fold-CV mean) — "
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
        print(f"\nRunning {len(pending_ids)} null models "
              f"({len(folds)} folds each) in parallel (n_jobs={cfg.N_JOBS})...\n")

        BATCH = 50
        for batch_start in range(0, len(pending_ids), BATCH):
            batch_ids = pending_ids[batch_start: batch_start + BATCH]

            batch_results = Parallel(n_jobs=cfg.N_JOBS, backend="loky")(
                delayed(run_one_null_model)(
                    mid,
                    fold_data,
                    cfg.CAT_FEATURE_INDICES,
                    cfg.NEWS_COL_INDICES,
                    cfg.ALL_FEATURES,
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
        "version"           : "v3_rolling_cv",
        "n_models"          : cfg.N_MODELS,
        "n_swaps"           : cfg.N_SWAPS,
        "n_folds"           : len(folds),
        "swap_strategy"     : "within_column_cell_swap_per_fold",
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
        print(f"\n  Real AR+News PR-AUC  : {real_pr:.4f}  ->  p = {p_pr:.4f}  [{sig(p_pr)}]")
        print(f"  Real AR+News ROC-AUC : {real_roc:.4f}  ->  p = {p_roc:.4f}  [{sig(p_roc)}]")

    with open(cfg.OUT_DIR / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"\nOutputs: {cfg.OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
