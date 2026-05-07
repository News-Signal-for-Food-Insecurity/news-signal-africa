"""
02_rolling_cv_train.py
======================
Window-sensitivity check for the primary rolling CV pipeline.

Uses a 28-month training window (vs. the primary 20-month window in
01_train_models.py) to verify that results are not sensitive to the
choice of training window length.

All model parameters, features, and training protocol are identical to
01_train_models.py — only TRAIN_WINDOW_MONTHS and RESULTS_DIR differ.

Outputs (results/window_sensitivity/):
  - fold_results.csv
  - fold_predictions.csv
  - feature_importance.csv
  - metrics_summary.json
  - models/ar_fold_{k}.cbm
  - models/combined_fold_{k}.cbm
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration — only TRAIN_WINDOW_MONTHS and RESULTS_DIR differ from 01_
# ---------------------------------------------------------------------------

class Config:
    BASE_DIR    = Path(__file__).parent
    DATA_DIR    = BASE_DIR / "DATA"
    RESULTS_DIR = BASE_DIR / "results" / "window_sensitivity"

    DATASET_PATH         = DATA_DIR / "dataset.parquet"
    MONTHLY_GDELT_PATH   = DATA_DIR / "modelling" / "monthly_gdelt_features.parquet"

    TRAIN_WINDOW_MONTHS  = 28   # longer window for sensitivity check
    IPC_PERIOD_MONTHS    = 4
    MONTHLY_DATA_START   = pd.Timestamp("2020-02-01")

    STRICT_DISTRICTS_PATH = DATA_DIR / "filtering" / "strict_filtered_districts.csv"

    AR_FEATURES = [
        "ipc_lag_1",
        "ipc_persistence_2yr",
        "spatial_lag",
        "ipc_period",
        "ipc_country",
    ]
    NEWS_THEMES = [
        "conflict", "displacement", "economic", "food_security",
        "governance", "health", "humanitarian", "weather", "other",
    ]
    NEWS_FEATURES = (
        [f"{t}_relative_coverage" for t in NEWS_THEMES]
        + [f"{t}_zscore" for t in NEWS_THEMES]
    )                                # 18 features — identical to 01_train_models.py
    COMBINED_FEATURES = AR_FEATURES + NEWS_FEATURES
    TARGET = "target_crisis_binary"

    # Identical CatBoost params as 01_train_models.py — no early stopping, no class weights.
    CATBOOST_PARAMS = dict(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        random_seed=42,
        verbose=0,
    )
    CATBOOST_PARAMS_AR   = CATBOOST_PARAMS
    CATBOOST_PARAMS_FULL = CATBOOST_PARAMS
    THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers — identical to 01_train_models.py
# ---------------------------------------------------------------------------

def generate_rolling_folds(df: pd.DataFrame, cfg: Config) -> list:
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
            "fold_id":     fold_id,
            "train_start": train_start_ts,
            "train_end":   train_end_ts,
            "test_start":  test_start_ts,
            "train_idx":   train_idx,
            "test_idx":    test_idx,
        })
    return folds


def compute_fold_news_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    monthly_gdelt: pd.DataFrame,
    train_end: pd.Timestamp,
    cfg: Config,
) -> tuple:
    baseline_gdelt = monthly_gdelt[monthly_gdelt["month"] <= train_end].copy()
    cov_cols = [f"{t}_relative_coverage" for t in cfg.NEWS_THEMES]
    baseline_gdelt["log_article_count"] = np.log1p(
        baseline_gdelt["article_count"].clip(lower=0)
    )
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
            valid = np.isfinite(means) & np.isfinite(stds) & (stds > 1e-6)
            raw   = df_out[cov_col].values.astype(float)
            df_out[zscore_col] = np.where(valid, (raw - means) / stds, 0.0)
        return df_out

    return apply_zscore(df_train), apply_zscore(df_test)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred  = (y_prob >= threshold).astype(int)
    has_pos = y_true.sum() > 0
    return {
        "pr_auc":    float(average_precision_score(y_true, y_prob)) if has_pos else np.nan,
        "roc_auc":   float(roc_auc_score(y_true, y_prob))           if has_pos else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "n_pos":     int(y_true.sum()),
        "n_total":   int(len(y_true)),
    }


def run_single_fold(fold_info: dict, df: pd.DataFrame, monthly_gdelt: pd.DataFrame, cfg: Config):
    fold_id   = fold_info["fold_id"]
    train_end = fold_info["train_end"]

    df_train = df.loc[fold_info["train_idx"]].copy()
    df_test  = df.loc[fold_info["test_idx"]].copy()

    df_train, df_test = compute_fold_news_features(df_train, df_test, monthly_gdelt, train_end, cfg)

    y_train = df_train[cfg.TARGET].values
    y_test  = df_test[cfg.TARGET].values

    def prep_X(df_subset: pd.DataFrame, features: list) -> pd.DataFrame:
        X = df_subset[features].copy()
        X["ipc_period"]  = X["ipc_period"].astype(str)
        X["ipc_country"] = X["ipc_country"].astype(str)
        return X

    cat_idx_ar = [cfg.AR_FEATURES.index("ipc_period"),
                  cfg.AR_FEATURES.index("ipc_country")]
    model_ar = CatBoostClassifier(**cfg.CATBOOST_PARAMS_AR, cat_features=cat_idx_ar)
    model_ar.fit(prep_X(df_train, cfg.AR_FEATURES), y_train)
    prob_ar = model_ar.predict_proba(prep_X(df_test, cfg.AR_FEATURES))[:, 1]
    m_ar    = compute_metrics(y_test, prob_ar, cfg.THRESHOLD)

    cat_idx_full = [cfg.COMBINED_FEATURES.index("ipc_period"),
                    cfg.COMBINED_FEATURES.index("ipc_country")]
    model_full = CatBoostClassifier(**cfg.CATBOOST_PARAMS_FULL, cat_features=cat_idx_full)
    model_full.fit(prep_X(df_train, cfg.COMBINED_FEATURES), y_train)
    prob_full = model_full.predict_proba(prep_X(df_test, cfg.COMBINED_FEATURES))[:, 1]
    m_full    = compute_metrics(y_test, prob_full, cfg.THRESHOLD)

    fi_df = pd.DataFrame({
        "feature":    cfg.COMBINED_FEATURES,
        "importance": model_full.get_feature_importance(),
        "fold_id":    fold_id,
    })

    pred_df = df_test[["district_id", "ipc_period_start", cfg.TARGET]].copy()
    pred_df["fold_id"]       = fold_id
    pred_df["prob_ar"]       = prob_ar
    pred_df["prob_combined"] = prob_full

    fold_result = {
        "fold_id":        fold_id,
        "train_start":    str(fold_info["train_start"].date()),
        "train_end":      str(fold_info["train_end"].date()),
        "test_start":     str(fold_info["test_start"].date()),
        "n_train":        len(df_train),
        "n_test":         len(df_test),
        "ar_pr_auc":      m_ar["pr_auc"],
        "ar_roc_auc":     m_ar["roc_auc"],
        "ar_precision":   m_ar["precision"],
        "ar_recall":      m_ar["recall"],
        "ar_f1":          m_ar["f1"],
        "ar_n_pos":       m_ar["n_pos"],
        "full_pr_auc":    m_full["pr_auc"],
        "full_roc_auc":   m_full["roc_auc"],
        "full_precision": m_full["precision"],
        "full_recall":    m_full["recall"],
        "full_f1":        m_full["f1"],
        "full_n_pos":     m_full["n_pos"],
        "delta_pr_auc":   m_full["pr_auc"] - m_ar["pr_auc"],
    }
    return fold_result, pred_df, fi_df, model_ar, model_full


def save_results(fold_records: list, all_preds: list, all_fi: list, cfg: Config):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(cfg.RESULTS_DIR / "fold_results.csv", index=False)
    print(f"  Saved fold_results.csv  ({len(fold_df)} folds)")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(cfg.RESULTS_DIR / "fold_predictions.csv", index=False)
    print(f"  Saved fold_predictions.csv  ({len(pred_df)} rows)")

    fi_df   = pd.concat(all_fi, ignore_index=True)
    fi_mean = (
        fi_df.groupby("feature")["importance"].mean()
        .reset_index()
        .rename(columns={"importance": "mean_importance"})
        .sort_values("mean_importance", ascending=False)
    )
    fi_mean.to_csv(cfg.RESULTS_DIR / "feature_importance.csv", index=False)

    numeric_cols = fold_df.select_dtypes(include=[float, np.float64]).columns.tolist()
    summary = {
        col: {"mean": float(fold_df[col].dropna().mean()),
              "std":  float(fold_df[col].dropna().std())}
        for col in numeric_cols
    }
    with open(cfg.RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved metrics_summary.json")
    return fold_df, pred_df, fi_mean


def save_models(model_pairs: list, models_dir: Path):
    models_dir.mkdir(exist_ok=True)
    for fold_id, model_ar, model_full in model_pairs:
        model_ar.save_model(str(models_dir / f"ar_fold_{fold_id}.cbm"))
        model_full.save_model(str(models_dir / f"combined_fold_{fold_id}.cbm"))
    print(f"  Saved {len(model_pairs)*2} model files (.cbm)")


def main():
    cfg = Config()
    print("=" * 60)
    print("Rolling CV: window sensitivity check  (28-month window)")
    print("=" * 60)

    print("\nLoading data...")
    df = pd.read_parquet(cfg.DATASET_PATH)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])

    monthly_gdelt = pd.read_parquet(cfg.MONTHLY_GDELT_PATH)
    monthly_gdelt["month"] = pd.to_datetime(monthly_gdelt["month"])

    strict_districts = pd.read_csv(cfg.STRICT_DISTRICTS_PATH)["district"].tolist()
    df = df[df["district_id"].isin(strict_districts)].copy()
    print(f"  Dataset: {len(df):,} rows, {df['district_id'].nunique()} districts")

    folds = generate_rolling_folds(df, cfg)
    print(f"  Folds: {len(folds)}")

    fold_records, all_preds, all_fi, model_pairs = [], [], [], []

    for fold_info in folds:
        fid = fold_info["fold_id"]
        print(f"\nFold {fid}: train {fold_info['train_start'].date()} "
              f"-> {fold_info['train_end'].date()} | "
              f"test {fold_info['test_start'].date()} "
              f"({len(fold_info['train_idx'])} train, {len(fold_info['test_idx'])} test)")

        result, pred_df, fi_df, m_ar, m_full = run_single_fold(
            fold_info, df, monthly_gdelt, cfg
        )
        fold_records.append(result)
        all_preds.append(pred_df)
        all_fi.append(fi_df)
        model_pairs.append((fid, m_ar, m_full))

        print(f"  AR PR-AUC: {result['ar_pr_auc']:.4f} | "
              f"Full PR-AUC: {result['full_pr_auc']:.4f} | "
              f"Delta: {result['delta_pr_auc']:+.4f}")

    print("\nSaving results...")
    fold_df, _, fi_mean = save_results(fold_records, all_preds, all_fi, cfg)
    save_models(model_pairs, cfg.RESULTS_DIR / "models")

    print("\n" + "=" * 60)
    print("Summary (28-month sensitivity window):")
    print(f"  AR    mean PR-AUC : {fold_df['ar_pr_auc'].mean():.4f} "
          f"+/- {fold_df['ar_pr_auc'].std():.4f}")
    print(f"  Full  mean PR-AUC : {fold_df['full_pr_auc'].mean():.4f} "
          f"+/- {fold_df['full_pr_auc'].std():.4f}")
    print(f"  Delta mean        : {fold_df['delta_pr_auc'].mean():+.4f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
