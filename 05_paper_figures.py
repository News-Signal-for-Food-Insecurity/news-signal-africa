"""
Paper Figures -- Publication-Ready Analysis
==========================================

Generates six supplementary figures and one summary table for the paper:
  "Does news contain meaningful incremental signal beyond autoregressive structure?"

PREREQUISITES (run these scripts first):
  1. scripts/01_train_models.py       -> results_rolling_cv/window_2yr/
  2. scripts/02_rolling_cv_train.py   -> results_rolling_cv/window_3yr/
  3. scripts/03_news_only_model.py    -> results_rolling_cv/news_only/
  4. scripts/04_temporal_shuffle_test.py -> results_rolling_cv/shuffle_test/

OUTPUTS (all saved to figures_paper/):
  table_model_comparison.csv     -- Model comparison: PR-AUC, precision, recall, F1
  fig_B_ar_vs_news_scatter.png   -- AR vs news-only prediction correlation
  fig_C_null_distribution.png    -- Temporal shuffle null distribution
  fig_D_strategic_type_probdist.png -- Predicted probability by strategic type
  fig_E_regional_breakdown.png   -- PR-AUC by region: AR-only vs combined
  fig_F_temporal_breakdown.png   -- PR-AUC over time: AR-only vs combined
  summary_stats.json             -- All key numbers for paper text

NOTE: For the full figure set (fig_3a through fig_7c), use 05_paper_figures_full.py.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

class FigConfig:
    BASE_DIR        = Path(__file__).parent
    RESULTS_DIR     = BASE_DIR / "results"
    OUTPUT_DIR      = BASE_DIR / "figures"

    AR_FULL_RESULTS  = RESULTS_DIR / "window_2yr" / "fold_results.csv"
    AR_FULL_PREDS    = RESULTS_DIR / "window_2yr" / "fold_predictions.csv"
    NEWS_RESULTS     = RESULTS_DIR / "news_only"   / "fold_results.csv"
    NEWS_PREDS       = RESULTS_DIR / "news_only"   / "fold_predictions.csv"
    SHUFFLE_RESULTS  = RESULTS_DIR / "shuffle_test" / "null_distribution.csv"
    REGIONAL_RESULTS = RESULTS_DIR / "regional_temporal" / "by_region.csv"
    TEMPORAL_RESULTS = RESULTS_DIR / "regional_temporal" / "by_time.csv"

    # Fold alignment: drop folds with no monthly GDELT coverage
    ALIGN_FOLDS_DROP = [1]

    FIGSIZE_SCATTER  = (7, 6)
    FIGSIZE_HIST     = (7, 5)
    FIGSIZE_BAR      = (8, 5)
    DPI              = 150
    FONT_SIZE        = 11

    COLOR_AR   = "#1f77b4"
    COLOR_FULL = "#2ca02c"
    COLOR_NEWS = "#ff7f0e"


# ============================================================================
# HELPERS
# ============================================================================

def load_and_check(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: {label} not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {len(df)} rows from {path.name}")
    return df


# ============================================================================
# TABLE A: MODEL COMPARISON
# ============================================================================

def make_table_model_comparison(ar_results, news_results, output_dir):
    """3-row table: news-only | AR-only | Combined x metrics."""
    print("\n--- Table: Model Comparison ---")
    rows = []

    if not news_results.empty and "news_pr_auc" in news_results.columns:
        rows.append({
            "model":     "News-only",
            "pr_auc":    round(news_results["news_pr_auc"].mean(), 4),
            "roc_auc":   round(news_results.get("news_roc_auc", pd.Series([np.nan])).mean(), 4),
            "precision": round(news_results.get("news_precision", pd.Series([np.nan])).mean(), 4),
            "recall":    round(news_results.get("news_recall", pd.Series([np.nan])).mean(), 4),
            "f1":        round(news_results.get("news_f1", pd.Series([np.nan])).mean(), 4),
            "n_folds":   len(news_results),
        })

    if not ar_results.empty:
        rows.append({
            "model":     "AR-only",
            "pr_auc":    round(ar_results["ar_pr_auc"].mean(), 4),
            "roc_auc":   round(ar_results["ar_roc_auc"].mean(), 4),
            "precision": round(ar_results["ar_precision"].mean(), 4),
            "recall":    round(ar_results["ar_recall"].mean(), 4),
            "f1":        round(ar_results["ar_f1"].mean(), 4),
            "n_folds":   len(ar_results),
        })
        rows.append({
            "model":     "Combined",
            "pr_auc":    round(ar_results["full_pr_auc"].mean(), 4),
            "roc_auc":   round(ar_results["full_roc_auc"].mean(), 4),
            "precision": round(ar_results["full_precision"].mean(), 4),
            "recall":    round(ar_results["full_recall"].mean(), 4),
            "f1":        round(ar_results["full_f1"].mean(), 4),
            "n_folds":   len(ar_results),
        })

    if rows:
        table_df = pd.DataFrame(rows)
        out_path = output_dir / "table_model_comparison.csv"
        table_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name}")
        print(table_df.to_string(index=False))
    return rows


# ============================================================================
# FIGURE B: AR vs NEWS-ONLY SCATTER
# ============================================================================

def make_figure_b(ar_preds, news_preds, output_dir):
    """Scatter plot of AR vs news-only predicted probabilities."""
    print("\n--- Figure B: AR vs News-Only Scatter ---")
    cfg = FigConfig

    if ar_preds.empty or news_preds.empty:
        print("  Skipped: missing predictions.")
        return {}

    merge_keys = ["fold_id", "district_id", "ipc_period_start"]
    # news model may save column as prob_news or y_pred_proba_news — detect dynamically
    news_prob_col = next((c for c in ["prob_news", "y_pred_proba_news", "prob_combined"]
                          if c in news_preds.columns), None)
    if news_prob_col is None:
        print("  Skipped: no recognised probability column in news predictions.")
        return {}
    merged = ar_preds.merge(
        news_preds[merge_keys + [news_prob_col]].rename(columns={news_prob_col: "prob_news"}),
        on=merge_keys, how="inner"
    )
    if merged.empty:
        print("  Skipped: no overlapping predictions.")
        return {}

    prob_ar   = merged["prob_ar"].values
    prob_news = merged["prob_news"].values

    r_pearson, _ = stats.pearsonr(prob_ar, prob_news)
    r_spearman, _ = stats.spearmanr(prob_ar, prob_news)

    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_SCATTER, dpi=cfg.DPI)
    ax.scatter(prob_ar, prob_news, alpha=0.3, s=10, color=cfg.COLOR_NEWS, rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("AR-only predicted probability", fontsize=cfg.FONT_SIZE)
    ax.set_ylabel("News-only predicted probability", fontsize=cfg.FONT_SIZE)
    ax.set_title("AR-only vs News-only model predictions", fontsize=cfg.FONT_SIZE)
    ax.text(0.05, 0.92, f"Pearson r = {r_pearson:.3f}\nSpearman r = {r_spearman:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    out_path = output_dir / "fig_B_ar_vs_news_scatter.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")
    return {"pearson_r": round(r_pearson, 4), "spearman_r": round(r_spearman, 4)}


# ============================================================================
# FIGURE C: NULL DISTRIBUTION
# ============================================================================

def make_figure_c(shuffle_results, ar_results, output_dir):
    """Histogram of null PR-AUC distribution with real model line."""
    print("\n--- Figure C: Temporal Shuffle Null Distribution ---")
    cfg = FigConfig

    if shuffle_results.empty:
        print("  Skipped: no shuffle results.")
        return {}

    null_vals = shuffle_results["mean_pr_auc"].dropna().values
    real_mean_pr_auc = ar_results["full_pr_auc"].mean() if not ar_results.empty else None

    fig, ax = plt.subplots(figsize=cfg.FIGSIZE_HIST, dpi=cfg.DPI)
    ax.hist(null_vals, bins=25, color=cfg.COLOR_NEWS, alpha=0.7, edgecolor="white",
            label=f"Null distribution (n={len(null_vals)})")

    stats_out = {
        "null_mean": round(float(null_vals.mean()), 4),
        "null_std":  round(float(null_vals.std()), 4),
        "null_min":  round(float(null_vals.min()), 4),
        "null_max":  round(float(null_vals.max()), 4),
    }

    if real_mean_pr_auc is not None:
        p_value = float((null_vals >= real_mean_pr_auc).mean())
        ax.axvline(real_mean_pr_auc, color=cfg.COLOR_FULL, linewidth=2.5, linestyle="--",
                   label=f"Real combined PR-AUC = {real_mean_pr_auc:.4f}")
        ax.text(0.97, 0.95, f"p = {p_value:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        stats_out.update({"real_mean_pr_auc": round(real_mean_pr_auc, 4), "p_value": round(p_value, 4)})

    ax.set_xlabel("Mean PR-AUC (null model, shuffled news)", fontsize=cfg.FONT_SIZE)
    ax.set_ylabel("Count", fontsize=cfg.FONT_SIZE)
    ax.set_title("Temporal shuffle null distribution\n"
                 "(within-district permutation of news feature time ordering)",
                 fontsize=cfg.FONT_SIZE)
    ax.legend(fontsize=9)

    out_path = output_dir / "fig_C_null_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")
    return stats_out


# ============================================================================
# FIGURE D: STRATEGIC TYPE PROBABILITY DISTRIBUTIONS
# ============================================================================

def make_figure_d(ar_preds, news_preds, output_dir):
    """Violin plot of predicted probabilities by strategic type."""
    print("\n--- Figure D: Predicted Probability by Strategic Type ---")
    cfg = FigConfig

    if ar_preds.empty or "strategic_type" not in ar_preds.columns:
        print("  Skipped: strategic_type column missing.")
        return {}

    type_order  = ["N_STABLE", "N_RECOVERY", "P_ONSET", "P_CHRONIC"]
    type_labels = {"N_STABLE": "Stable", "N_RECOVERY": "Recovery",
                   "P_ONSET": "Onset", "P_CHRONIC": "Chronic"}
    present_types = [t for t in type_order if t in ar_preds["strategic_type"].unique()]

    models = {
        "AR-only":   ("prob_ar",       cfg.COLOR_AR),
        "Combined":  ("prob_combined", cfg.COLOR_FULL),
    }
    if not news_preds.empty and "prob_news" in news_preds.columns:
        models["News-only"] = ("prob_news", cfg.COLOR_NEWS)
        # Merge news probs
        keys = ["fold_id", "district_id", "ipc_period_start"]
        ar_preds = ar_preds.merge(news_preds[keys + ["prob_news"]], on=keys, how="left")

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5),
                             dpi=cfg.DPI, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (model_name, (col, color)) in zip(axes, models.items()):
        if col not in ar_preds.columns:
            continue
        data = [ar_preds.loc[ar_preds["strategic_type"] == t, col].dropna().values
                for t in present_types]
        vp = ax.violinplot(data, positions=range(len(present_types)),
                           showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        ax.set_xticks(range(len(present_types)))
        ax.set_xticklabels([type_labels.get(t, t) for t in present_types], fontsize=cfg.FONT_SIZE)
        ax.set_title(model_name, fontsize=cfg.FONT_SIZE)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)

    axes[0].set_ylabel("Predicted crisis probability", fontsize=cfg.FONT_SIZE)
    fig.suptitle("Predicted probability distributions by district type",
                 fontsize=cfg.FONT_SIZE, y=1.02)

    out_path = output_dir / "fig_D_strategic_type_probdist.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")
    return {}


# ============================================================================
# FIGURE E: REGIONAL BREAKDOWN
# ============================================================================

def make_figure_e(regional_df, output_dir):
    """Cleveland dot plot: AR-only vs Combined PR-AUC by region."""
    print("\n--- Figure E: Regional PR-AUC Breakdown ---")
    cfg = FigConfig

    if regional_df.empty:
        print("  Skipped: no regional data.")
        return {}

    required = ["region", "pr_auc_ar_only_mean", "pr_auc_baseline_mean"]
    missing  = [c for c in required if c not in regional_df.columns]
    if missing:
        print(f"  WARNING: Missing columns {missing}. Skipping.")
        return {}

    df = regional_df[required].dropna().sort_values("pr_auc_ar_only_mean").reset_index(drop=True)
    n  = len(df)
    y  = np.arange(n)

    deltas     = df["pr_auc_baseline_mean"] - df["pr_auc_ar_only_mean"]
    bar_colors = [cfg.COLOR_FULL if d >= 0 else "#d62728" for d in deltas]

    fig, (ax_dot, ax_bar) = plt.subplots(
        1, 2, figsize=(10, max(4.0, n * 1.2 + 1.5)), dpi=cfg.DPI,
        gridspec_kw={"width_ratios": [3, 1.2]}
    )

    for i, row in df.iterrows():
        ar_v, cb_v = row["pr_auc_ar_only_mean"], row["pr_auc_baseline_mean"]
        color = cfg.COLOR_FULL if cb_v >= ar_v else "#d62728"
        ax_dot.plot([ar_v, cb_v], [i, i], color=color, linewidth=1.8, alpha=0.55)

    ax_dot.scatter(df["pr_auc_ar_only_mean"],  y, marker="o", s=90, color=cfg.COLOR_AR,   label="AR-only",         zorder=3)
    ax_dot.scatter(df["pr_auc_baseline_mean"], y, marker="s", s=90, color=cfg.COLOR_FULL, label="Combined (AR+news)", zorder=3)
    ax_dot.set_yticks(y)
    ax_dot.set_yticklabels(df["region"], fontsize=cfg.FONT_SIZE)
    ax_dot.set_xlabel("PR-AUC (mean across folds)", fontsize=cfg.FONT_SIZE)
    ax_dot.set_title("Model performance by region", fontsize=cfg.FONT_SIZE)
    ax_dot.legend(fontsize=9, loc="lower right")
    ax_dot.xaxis.grid(True, linestyle=":", alpha=0.4)

    bars = ax_bar.barh(y, deltas, color=bar_colors, height=0.55, alpha=0.85)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    for bar, d in zip(bars, deltas):
        ha   = "left" if d >= 0 else "right"
        xpos = d + (0.002 if d >= 0 else -0.002)
        ax_bar.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{d:+.3f}", ha=ha, va="center", fontsize=8.5)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel("Delta (Combined - AR)", fontsize=cfg.FONT_SIZE)
    ax_bar.set_title("News delta", fontsize=cfg.FONT_SIZE)

    plt.suptitle("PR-AUC by region: AR-only vs Combined model",
                 fontsize=cfg.FONT_SIZE + 1, y=1.01)
    out_path = output_dir / "fig_E_regional_breakdown.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")
    return {}


# ============================================================================
# FIGURE F: TEMPORAL BREAKDOWN
# ============================================================================

def make_figure_f(temporal_df, output_dir):
    """Line chart: PR-AUC over test windows (AR-only vs combined)."""
    print("\n--- Figure F: Temporal PR-AUC Breakdown ---")
    cfg = FigConfig

    if temporal_df.empty:
        print("  Skipped: no temporal data.")
        return {}

    required = ["ipc_period", "pr_auc_ar_only_mean", "pr_auc_baseline_mean"]
    missing  = [c for c in required if c not in temporal_df.columns]
    if missing:
        print(f"  WARNING: Missing columns {missing}. Skipping.")
        return {}

    df = temporal_df.dropna(subset=["pr_auc_ar_only_mean", "pr_auc_baseline_mean"]).copy()
    df = df.sort_values("ipc_period").reset_index(drop=True)
    n  = len(df)
    x  = np.arange(n)

    xlabels   = [str(p)[:7] for p in df["ipc_period"]]
    ar_mean   = df["pr_auc_ar_only_mean"].values
    comb_mean = df["pr_auc_baseline_mean"].values
    ar_std    = df.get("pr_auc_ar_only_std",   pd.Series(np.zeros(n))).values
    comb_std  = df.get("pr_auc_baseline_std",  pd.Series(np.zeros(n))).values

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5), dpi=cfg.DPI)

    ax.plot(x, ar_mean,   "o-", color=cfg.COLOR_AR,   linewidth=2, markersize=6, label="AR-only",           zorder=3)
    ax.fill_between(x, ar_mean - ar_std, ar_mean + ar_std, color=cfg.COLOR_AR,   alpha=0.15)

    ax.plot(x, comb_mean, "s--", color=cfg.COLOR_FULL, linewidth=2, markersize=6, label="Combined (AR+news)", zorder=3)
    ax.fill_between(x, comb_mean - comb_std, comb_mean + comb_std, color=cfg.COLOR_FULL, alpha=0.15)

    for i, (ar, cb) in enumerate(zip(ar_mean, comb_mean)):
        delta = cb - ar
        color = "#d62728" if delta < -0.01 else (cfg.COLOR_FULL if delta > 0.01 else "gray")
        ax.annotate(f"{delta:+.3f}", xy=(i, max(ar, cb)),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("PR-AUC (mean +/- SD)", fontsize=cfg.FONT_SIZE)
    ax.set_title("PR-AUC over test windows: AR-only vs Combined", fontsize=cfg.FONT_SIZE)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(max(0, min(ar_mean.min(), comb_mean.min()) - 0.08),
                min(1.0, max(ar_mean.max(), comb_mean.max()) + 0.12))

    out_path = output_dir / "fig_F_temporal_breakdown.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")
    return {}


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = FigConfig()
    print("=" * 70)
    print("PAPER FIGURES (Base Set: B-F)")
    print("=" * 70)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading input files...")
    ar_results    = load_and_check(cfg.AR_FULL_RESULTS, "AR+Full fold_results")
    ar_preds      = load_and_check(cfg.AR_FULL_PREDS,   "AR+Full fold_predictions")
    news_results  = load_and_check(cfg.NEWS_RESULTS,    "News-only fold_results")
    news_preds    = load_and_check(cfg.NEWS_PREDS,      "News-only fold_predictions")
    shuffle_res   = load_and_check(cfg.SHUFFLE_RESULTS, "Shuffle null distribution")
    regional_df   = load_and_check(cfg.REGIONAL_RESULTS,"Regional breakdown")
    temporal_df   = load_and_check(cfg.TEMPORAL_RESULTS,"Temporal breakdown")

    # Align folds
    if not ar_results.empty:
        ar_results = ar_results[~ar_results["fold_id"].isin(cfg.ALIGN_FOLDS_DROP)].copy()
    if not ar_preds.empty:
        ar_preds = ar_preds[~ar_preds["fold_id"].isin(cfg.ALIGN_FOLDS_DROP)].copy()

    for df in [ar_preds, news_preds]:
        if not df.empty and "ipc_period_start" in df.columns:
            df["ipc_period_start"] = pd.to_datetime(df["ipc_period_start"])

    # Generate all outputs
    table_rows = make_table_model_comparison(ar_results, news_results, cfg.OUTPUT_DIR)
    stats_b    = make_figure_b(ar_preds, news_preds, cfg.OUTPUT_DIR)
    stats_c    = make_figure_c(shuffle_res, ar_results, cfg.OUTPUT_DIR)
    make_figure_d(ar_preds, news_preds, cfg.OUTPUT_DIR)
    make_figure_e(regional_df, cfg.OUTPUT_DIR)
    make_figure_f(temporal_df, cfg.OUTPUT_DIR)

    # Summary JSON
    summary = {
        "ar_only_mean_pr_auc":   round(ar_results["ar_pr_auc"].mean(), 4)   if not ar_results.empty else None,
        "combined_mean_pr_auc":  round(ar_results["full_pr_auc"].mean(), 4) if not ar_results.empty else None,
        "delta_pr_auc":          round((ar_results["full_pr_auc"] - ar_results["ar_pr_auc"]).mean(), 4)
                                 if not ar_results.empty else None,
        "figure_b":              stats_b,
        "figure_c":              stats_c,
    }
    with open(cfg.OUTPUT_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary_stats.json")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
