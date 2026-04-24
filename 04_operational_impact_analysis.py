"""
04_operational_impact_analysis.py
==================================
Computes operational impact metrics from rolling CV fold predictions and
generates FIGURE_6 (6-panel operational impact figure).

Operational impact framing:
  For each crisis observation in the test set, classify as one of:
    - both_detect   : both AR and combined model predict crisis (prob >= threshold)
    - ar_only       : AR detects, combined misses
    - full_only     : combined detects, AR misses  <- net saves (news value)
    - neither       : neither model detects

  net_saves = full_only (crises AR missed but combined caught)
            - ar_only  (crises combined missed but AR caught)

  Positive net_saves => combined model adds operational value beyond AR baseline.

Strategic typology breakdown:
  P_ONSET   : new crisis (no history in prior 24 months)
  P_CHRONIC : persistent crisis (history present)
  N_STABLE  : no current crisis, no history
  N_RECOVERY: no current crisis, but history present

Outputs (results_rolling_cv/):
  - operational_impact_2yr.csv         : per-fold impact (2-year window)
  - operational_impact_3yr.csv         : per-fold impact (3-year window)
  - operational_impact_summary.json    : aggregate summary (both windows)
  - operational_type_breakdown_2yr.csv : breakdown by strategic type
  - operational_type_breakdown_3yr.csv

Figures (figures_rolling_cv/):
  - FIGURE_6_operational_impact.png   : 6-panel composite figure
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    BASE_DIR    = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "figures"

    PRED_2YR = RESULTS_DIR / "window_2yr" / "fold_predictions.csv"
    PRED_3YR = RESULTS_DIR / "window_sensitivity" / "fold_predictions.csv"

    DATASET_PATH = BASE_DIR / "DATA" / "dataset.parquet"

    THRESHOLD = 0.5
    TARGET    = "target_crisis_binary"

    COLOR_AR   = "#1f77b4"
    COLOR_FULL = "#2ca02c"
    COLOR_NET  = "#d62728"

    STRATEGIC_TYPES = ["P_ONSET", "P_CHRONIC", "N_STABLE", "N_RECOVERY"]


# ---------------------------------------------------------------------------
# Compute operational impact
# ---------------------------------------------------------------------------

def compute_operational_impact(pred_df: pd.DataFrame, threshold: float, target_col: str) -> pd.DataFrame:
    """
    For each fold, compute detection agreement matrix and net_saves.
    Returns per-fold impact DataFrame.
    """
    # Only actual crisis rows (target == 1) for saves computation
    crisis_df = pred_df[pred_df[target_col] == 1].copy()

    records = []
    for fold_id, group in crisis_df.groupby("fold_id"):
        ar_detect   = (group["prob_ar"] >= threshold).astype(int)
        full_detect = (group["prob_combined"] >= threshold).astype(int)

        both_detect = int(((ar_detect == 1) & (full_detect == 1)).sum())
        ar_only     = int(((ar_detect == 1) & (full_detect == 0)).sum())
        full_only   = int(((ar_detect == 0) & (full_detect == 1)).sum())
        neither     = int(((ar_detect == 0) & (full_detect == 0)).sum())
        n_crises    = len(group)

        net_saves       = full_only - ar_only
        pct_net_saves   = net_saves / n_crises * 100 if n_crises > 0 else 0.0

        records.append({
            "fold_id":        fold_id,
            "n_crises":       n_crises,
            "both_detect":    both_detect,
            "ar_only":        ar_only,
            "full_only":      full_only,
            "neither":        neither,
            "net_saves":      net_saves,
            "pct_net_saves":  pct_net_saves,
        })

    return pd.DataFrame(records)


def compute_type_breakdown(pred_df: pd.DataFrame, dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Compute net_saves breakdown by strategic typology.
    Merges on district_id + ipc_period_start to get typology labels.
    """
    cols = ["district_id", "ipc_period_start", "strategic_type"]
    type_map = dataset[cols].drop_duplicates()

    merged = pred_df.merge(type_map, on=["district_id", "ipc_period_start"], how="left")
    crisis_df = merged[merged["target_crisis_binary"] == 1].copy()

    crisis_df["ar_detect"]   = (crisis_df["prob_ar"] >= threshold).astype(int)
    crisis_df["full_detect"] = (crisis_df["prob_combined"] >= threshold).astype(int)
    crisis_df["ar_only"]     = ((crisis_df["ar_detect"] == 1) & (crisis_df["full_detect"] == 0)).astype(int)
    crisis_df["full_only"]   = ((crisis_df["ar_detect"] == 0) & (crisis_df["full_detect"] == 1)).astype(int)

    breakdown = (
        crisis_df.groupby("strategic_type")
        .agg(
            n_crises=("target_crisis_binary", "sum"),
            ar_only=("ar_only", "sum"),
            full_only=("full_only", "sum"),
        )
        .reset_index()
    )
    breakdown["net_saves"] = breakdown["full_only"] - breakdown["ar_only"]
    return breakdown


# ---------------------------------------------------------------------------
# Figure 6: 6-panel operational impact
# ---------------------------------------------------------------------------

def plot_figure_6(fold_2yr, fold_3yr, summary, type_2yr, type_3yr, country_stats, cfg: Config):
    """
    6-panel composite operational impact figure.

    Panel A (top-left)  : per-fold net saves, 2yr window
    Panel B (top-right) : per-fold net saves, 3yr window
    Panel C (mid-left)  : crisis fate stacked bar by fold, 2yr
    Panel D (mid-right) : crisis fate stacked bar by fold, 3yr
    Panel E (bot-left)  : strategic type breakdown (both windows)
    Panel F (bot-right) : country-level net saves (2yr), horizontal bars
    """
    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[2, 0])
    ax_F = fig.add_subplot(gs[2, 1])

    # --- Panel A: per-fold net saves (2yr) ---
    colors_A = [cfg.COLOR_NET if v >= 0 else cfg.COLOR_AR for v in fold_2yr["net_saves"]]
    ax_A.bar(fold_2yr["fold_id"], fold_2yr["net_saves"], color=colors_A, edgecolor="white")
    ax_A.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_A.set_title("A. Net saves per fold (2-year window)", fontsize=11, fontweight="bold")
    ax_A.set_xlabel("Fold")
    ax_A.set_ylabel("Net saves (crises)")
    ax_A.set_xticks(fold_2yr["fold_id"])

    # --- Panel B: per-fold net saves (3yr) ---
    colors_B = [cfg.COLOR_NET if v >= 0 else cfg.COLOR_AR for v in fold_3yr["net_saves"]]
    ax_B.bar(fold_3yr["fold_id"], fold_3yr["net_saves"], color=colors_B, edgecolor="white")
    ax_B.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_B.set_title("B. Net saves per fold (3-year window)", fontsize=11, fontweight="bold")
    ax_B.set_xlabel("Fold")
    ax_B.set_ylabel("Net saves (crises)")
    ax_B.set_xticks(fold_3yr["fold_id"])

    # --- Panel C: stacked bar, crisis fate (2yr) ---
    folds_c = fold_2yr["fold_id"].values
    ax_C.bar(folds_c, fold_2yr["both_detect"], label="Both detect", color="#4c72b0")
    ax_C.bar(folds_c, fold_2yr["full_only"],   bottom=fold_2yr["both_detect"],
             label="Full only (net gain)", color=cfg.COLOR_FULL)
    ax_C.bar(folds_c, fold_2yr["ar_only"],
             bottom=fold_2yr["both_detect"] + fold_2yr["full_only"],
             label="AR only (net loss)", color=cfg.COLOR_AR)
    ax_C.bar(folds_c, fold_2yr["neither"],
             bottom=fold_2yr["both_detect"] + fold_2yr["full_only"] + fold_2yr["ar_only"],
             label="Neither", color="#d3d3d3")
    ax_C.set_title("C. Crisis fate breakdown (2-year window)", fontsize=11, fontweight="bold")
    ax_C.set_xlabel("Fold")
    ax_C.set_ylabel("Crisis count")
    ax_C.set_xticks(folds_c)
    ax_C.legend(fontsize=8, loc="upper right")

    # --- Panel D: stacked bar, crisis fate (3yr) ---
    folds_d = fold_3yr["fold_id"].values
    ax_D.bar(folds_d, fold_3yr["both_detect"], label="Both detect", color="#4c72b0")
    ax_D.bar(folds_d, fold_3yr["full_only"],   bottom=fold_3yr["both_detect"],
             label="Full only (net gain)", color=cfg.COLOR_FULL)
    ax_D.bar(folds_d, fold_3yr["ar_only"],
             bottom=fold_3yr["both_detect"] + fold_3yr["full_only"],
             label="AR only (net loss)", color=cfg.COLOR_AR)
    ax_D.bar(folds_d, fold_3yr["neither"],
             bottom=fold_3yr["both_detect"] + fold_3yr["full_only"] + fold_3yr["ar_only"],
             label="Neither", color="#d3d3d3")
    ax_D.set_title("D. Crisis fate breakdown (3-year window)", fontsize=11, fontweight="bold")
    ax_D.set_xlabel("Fold")
    ax_D.set_ylabel("Crisis count")
    ax_D.set_xticks(folds_d)
    ax_D.legend(fontsize=8, loc="upper right")

    # --- Panel E: strategic type breakdown ---
    x_pos   = np.arange(len(cfg.STRATEGIC_TYPES))
    w       = 0.35
    _empty  = pd.Series(0, index=cfg.STRATEGIC_TYPES)
    type_2yr_dict = (type_2yr.set_index("strategic_type")["net_saves"].reindex(cfg.STRATEGIC_TYPES, fill_value=0)
                     if not type_2yr.empty else _empty)
    type_3yr_dict = (type_3yr.set_index("strategic_type")["net_saves"].reindex(cfg.STRATEGIC_TYPES, fill_value=0)
                     if not type_3yr.empty else _empty)

    ax_E.bar(x_pos - w/2, type_2yr_dict.values, width=w, label="2-year window", color=cfg.COLOR_FULL)
    ax_E.bar(x_pos + w/2, type_3yr_dict.values, width=w, label="3-year window", color=cfg.COLOR_AR, alpha=0.8)
    ax_E.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_E.set_xticks(x_pos)
    ax_E.set_xticklabels(cfg.STRATEGIC_TYPES, rotation=15, fontsize=9)
    ax_E.set_title("E. Net saves by strategic type", fontsize=11, fontweight="bold")
    ax_E.set_ylabel("Net saves (crises)")
    ax_E.legend(fontsize=9)

    # --- Panel F: country-level net saves (2yr) ---
    if country_stats is not None and len(country_stats) > 0:
        country_stats_sorted = country_stats.sort_values("net_saves", ascending=True)
        colors_F = [cfg.COLOR_NET if v >= 0 else cfg.COLOR_AR for v in country_stats_sorted["net_saves"]]
        ax_F.barh(country_stats_sorted["country"], country_stats_sorted["net_saves"],
                  color=colors_F, edgecolor="white")
        ax_F.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_F.set_title("F. Country-level net saves (2-year window)", fontsize=11, fontweight="bold")
        ax_F.set_xlabel("Net saves (crises)")
    else:
        ax_F.text(0.5, 0.5, "Country data\nnot available",
                  ha="center", va="center", transform=ax_F.transAxes, fontsize=12)
        ax_F.set_title("F. Country-level net saves (2-year window)", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Operational Impact: Combined vs AR-Only Model\n"
        f"2yr window: net saves = {int(fold_2yr['net_saves'].sum())} "
        f"({fold_2yr['pct_net_saves'].mean():.1f}% of crises) | "
        f"3yr window: net saves = {int(fold_3yr['net_saves'].sum())} "
        f"({fold_3yr['pct_net_saves'].mean():.1f}% of crises)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.FIGURES_DIR / "FIGURE_6_operational_impact.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Country-level breakdown (optional, from dataset)
# ---------------------------------------------------------------------------

def compute_country_impact(pred_df: pd.DataFrame, dataset: pd.DataFrame, threshold: float):
    """Compute net_saves per country using country column from dataset."""
    if "country" not in dataset.columns:
        return None

    country_map = dataset[["district_id", "country"]].drop_duplicates()
    merged = pred_df.merge(country_map, on="district_id", how="left")
    crisis_df = merged[merged["target_crisis_binary"] == 1].copy()

    crisis_df["ar_detect"]   = (crisis_df["prob_ar"] >= threshold).astype(int)
    crisis_df["full_detect"] = (crisis_df["prob_combined"] >= threshold).astype(int)
    crisis_df["ar_only"]     = ((crisis_df["ar_detect"] == 1) & (crisis_df["full_detect"] == 0)).astype(int)
    crisis_df["full_only"]   = ((crisis_df["ar_detect"] == 0) & (crisis_df["full_detect"] == 1)).astype(int)

    country_stats = (
        crisis_df.groupby("country")
        .agg(
            n_crises=("target_crisis_binary", "sum"),
            ar_only=("ar_only", "sum"),
            full_only=("full_only", "sum"),
        )
        .reset_index()
    )
    country_stats["net_saves"] = country_stats["full_only"] - country_stats["ar_only"]
    return country_stats.sort_values("net_saves", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    print("=" * 60)
    print("Operational Impact Analysis")
    print("=" * 60)

    # Load dataset for typology and country info
    dataset = pd.read_parquet(cfg.DATASET_PATH)
    dataset["ipc_period_start"] = pd.to_datetime(dataset["ipc_period_start"])

    # Load predictions
    if not cfg.PRED_2YR.exists():
        print(f"ERROR: {cfg.PRED_2YR} not found. Run 01_train_models.py first.")
        return
    if not cfg.PRED_3YR.exists():
        print(f"WARNING: {cfg.PRED_3YR} not found. Run 02_rolling_cv_train.py first.")

    pred_2yr = pd.read_csv(cfg.PRED_2YR)
    pred_2yr["ipc_period_start"] = pd.to_datetime(pred_2yr["ipc_period_start"])
    print(f"  2yr predictions: {len(pred_2yr):,} rows, {pred_2yr['fold_id'].nunique()} folds")

    # Compute operational impact
    print("\nComputing operational impact...")
    fold_2yr = compute_operational_impact(pred_2yr, cfg.THRESHOLD, cfg.TARGET)
    print(f"  2yr net saves: {fold_2yr['net_saves'].sum()} total "
          f"({fold_2yr['net_saves'].sum() / fold_2yr['n_crises'].sum() * 100:.2f}% of crises)")

    pred_3yr = None
    fold_3yr = pd.DataFrame(columns=fold_2yr.columns)
    if cfg.PRED_3YR.exists():
        pred_3yr = pd.read_csv(cfg.PRED_3YR)
        pred_3yr["ipc_period_start"] = pd.to_datetime(pred_3yr["ipc_period_start"])
        fold_3yr = compute_operational_impact(pred_3yr, cfg.THRESHOLD, cfg.TARGET)
        print(f"  3yr net saves: {fold_3yr['net_saves'].sum()} total "
              f"({fold_3yr['net_saves'].sum() / fold_3yr['n_crises'].sum() * 100:.2f}% of crises)")

    # Strategic type breakdown
    print("\nComputing strategic type breakdown...")
    type_2yr = pd.DataFrame()
    type_3yr = pd.DataFrame()

    if "strategic_type" in dataset.columns:
        type_2yr = compute_type_breakdown(pred_2yr, dataset, cfg.THRESHOLD)
        if pred_3yr is not None:
            type_3yr = compute_type_breakdown(pred_3yr, dataset, cfg.THRESHOLD)

    # Country-level
    country_stats = compute_country_impact(pred_2yr, dataset, cfg.THRESHOLD)

    # Save CSVs
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_2yr.to_csv(cfg.RESULTS_DIR / "operational_impact_2yr.csv", index=False)
    fold_3yr.to_csv(cfg.RESULTS_DIR / "operational_impact_3yr.csv", index=False)
    if not type_2yr.empty:
        type_2yr.to_csv(cfg.RESULTS_DIR / "operational_type_breakdown_2yr.csv", index=False)
    if not type_3yr.empty:
        type_3yr.to_csv(cfg.RESULTS_DIR / "operational_type_breakdown_3yr.csv", index=False)
    if country_stats is not None:
        country_stats.to_csv(cfg.RESULTS_DIR / "country_impact_2yr.csv", index=False)

    # Summary JSON
    summary = {
        "window_2yr": {
            "total_net_saves":     int(fold_2yr["net_saves"].sum()),
            "total_crises":        int(fold_2yr["n_crises"].sum()),
            "pct_net_saves":       float(fold_2yr["net_saves"].sum() / fold_2yr["n_crises"].sum() * 100),
            "total_full_only":     int(fold_2yr["full_only"].sum()),
            "total_ar_only":       int(fold_2yr["ar_only"].sum()),
            "total_both_detect":   int(fold_2yr["both_detect"].sum()),
            "total_neither":       int(fold_2yr["neither"].sum()),
        },
        "window_sensitivity": {
            "total_net_saves":     int(fold_3yr["net_saves"].sum()) if not fold_3yr.empty else None,
            "total_crises":        int(fold_3yr["n_crises"].sum())  if not fold_3yr.empty else None,
            "pct_net_saves":       float(fold_3yr["net_saves"].sum() / fold_3yr["n_crises"].sum() * 100)
                                   if not fold_3yr.empty and fold_3yr["n_crises"].sum() > 0 else None,
        },
    }
    with open(cfg.RESULTS_DIR / "operational_impact_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved operational_impact_summary.json")

    # Generate Figure 6
    print("\nGenerating Figure 6...")
    plot_figure_6(fold_2yr, fold_3yr, summary, type_2yr, type_3yr, country_stats, cfg)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  2yr window: {summary['window_2yr']['total_net_saves']} net saves "
          f"({summary['window_2yr']['pct_net_saves']:.2f}% of crises)")
    if summary["window_sensitivity"]["total_net_saves"] is not None:
        print(f"  3yr window: {summary['window_3yr']['total_net_saves']} net saves "
              f"({summary['window_3yr']['pct_net_saves']:.2f}% of crises)")
    print("Done.")


if __name__ == "__main__":
    main()
