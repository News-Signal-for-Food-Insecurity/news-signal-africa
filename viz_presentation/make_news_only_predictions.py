"""
make_news_only_predictions.py — regenerate the News-only model's test predictions.

The study compares three models that differ ONLY in their feature set: AR (5
autoregressive features), News-only (the 18 GDELT news features), and AR+News
(all 23). The repository ships serialized predictions for AR (``prob_ar``) and
AR+News (``prob_combined``) but not for the News-only model, so the talk's
three-curve Precision-Recall build cannot be reproduced from disk alone.

This script fills that gap WITHOUT inventing anything: it imports the real
training module (``01_train_models.py``) and reuses its exact rolling-fold logic,
fold-aware z-score recomputation, CatBoost hyper-parameters and district filter —
changing only the feature set to ``Config.NEWS_FEATURES``. It writes one row per
test observation to ``results/window_2yr/news_only_predictions.csv``:

    district_id, ipc_period_start, target_crisis_binary, fold_id, prob_news

Idempotent: if the output already exists it is left untouched unless --force.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent.parent          # news-signal-africa/
TRAIN_SCRIPT = BASE_DIR / "01_train_models.py"
OUT_PATH = BASE_DIR / "results" / "window_2yr" / "news_only_predictions.csv"
FOLD_RESULTS_PATH = BASE_DIR / "results" / "window_2yr" / "news_only_fold_results.csv"


def _load_training_module():
    """Import the numerically-named 01_train_models.py as a module."""
    spec = importlib.util.spec_from_file_location("train_models", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def generate(force: bool = False) -> Path:
    if OUT_PATH.exists() and FOLD_RESULTS_PATH.exists() and not force:
        print(f"[news-only] predictions + fold metrics already present — "
              f"skipping (use --force to regenerate).")
        return OUT_PATH

    tm = _load_training_module()
    cfg = tm.Config()

    print("[news-only] Loading data (same filter as the primary run)...")
    df = pd.read_parquet(cfg.DATASET_PATH)
    df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])
    monthly_gdelt = pd.read_parquet(cfg.MONTHLY_GDELT_PATH)
    monthly_gdelt["month"] = pd.to_datetime(monthly_gdelt["month"])
    strict = pd.read_csv(cfg.STRICT_DISTRICTS_PATH)["district"].tolist()
    df = df[df["district_id"].isin(strict)].copy()

    folds = tm.generate_rolling_folds(df, cfg)
    print(f"[news-only] {len(folds)} folds | features = "
          f"{len(cfg.NEWS_FEATURES)} news features (no AR, no categoricals)")

    preds, fold_metrics = [], []
    for fold in folds:
        fid = fold["fold_id"]
        df_train = df.loc[fold["train_idx"]].copy()
        df_test = df.loc[fold["test_idx"]].copy()
        # Identical fold-aware z-score recomputation used by AR+News.
        df_train, df_test = tm.compute_fold_news_features(
            df_train, df_test, monthly_gdelt, fold["train_end"], cfg)

        y_train = df_train[cfg.TARGET].values
        y_test = df_test[cfg.TARGET].values
        X_train = df_train[cfg.NEWS_FEATURES].copy()
        X_test = df_test[cfg.NEWS_FEATURES].copy()

        # News features are all numeric (relative coverage + z-scores); no
        # categorical indices. Same CatBoost params as AR / AR+News.
        model = CatBoostClassifier(**cfg.CATBOOST_PARAMS)
        model.fit(X_train, y_train)
        prob_news = model.predict_proba(X_test)[:, 1]

        out = df_test[["district_id", "ipc_period_start", cfg.TARGET]].copy()
        out["fold_id"] = fid
        out["prob_news"] = prob_news
        preds.append(out)

        # Per-fold metrics computed exactly as the pipeline computes them
        # (average_precision_score / roc_auc_score on the fold's test set).
        m = tm.compute_metrics(y_test, prob_news, cfg.THRESHOLD)
        fold_metrics.append({
            "fold_id": fid,
            "test_start": str(fold["test_start"].date()),
            "n_test": len(df_test),
            "news_pr_auc": m["pr_auc"],
            "news_roc_auc": m["roc_auc"],
            "news_precision": m["precision"],
            "news_recall": m["recall"],
            "news_f1": m["f1"],
            "news_n_pos": m["n_pos"],
        })
        print(f"  fold {fid}: News-only PR-AUC {m['pr_auc']:.4f} "
              f"({len(df_test)} test rows)")

    result = pd.concat(preds, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    # Persist the News-only per-fold metrics as a first-class pipeline artifact,
    # mirroring results/window_2yr/fold_results.csv for AR / AR+News.
    fr = pd.DataFrame(fold_metrics)
    fr.to_csv(FOLD_RESULTS_PATH, index=False)

    mean_pr = fr["news_pr_auc"].mean()
    sd_pr = fr["news_pr_auc"].std()       # ddof=1, as the study reports
    print(f"[news-only] Saved {OUT_PATH.name}  ({len(result)} rows) and "
          f"{FOLD_RESULTS_PATH.name}")
    print(f"[news-only] News-only PR-AUC (mean of folds): "
          f"{mean_pr:.4f} +/- {sd_pr:.4f}")
    assert result["prob_news"].between(0, 1).all(), "prob_news out of [0,1]"
    return OUT_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the output already exists")
    generate(force=ap.parse_args().force)
