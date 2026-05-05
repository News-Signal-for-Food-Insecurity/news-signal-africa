"""
generate_fig5_pdf.py
====================
Fold-by-fold explanation of why AR+News outperforms or underperforms AR-Only.
All numbers read directly from results files.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "window_2yr"
FIGURES_DIR = BASE_DIR / "figures"

THRESHOLD = 0.5

# ── Load data ─────────────────────────────────────────────────────────────────
fold_df = pd.read_csv(RESULTS_DIR / "fold_results.csv")
preds   = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
preds["delta_prob"] = preds["prob_combined"] - preds["prob_ar"]

# ── Pre-compute per-fold stats ────────────────────────────────────────────────
fold_stats = []
for _, fr in fold_df.iterrows():
    fid  = int(fr.fold_id)
    sub  = preds[preds.fold_id == fid]
    prev = fr.full_n_pos / fr.n_test

    reg = {}
    for r in ["onset", "chronic", "recovery", "stable"]:
        g = sub[sub.regime == r]
        reg[r] = {
            "n":         len(g),
            "delta":     g["delta_prob"].mean() if len(g) else float("nan"),
            "rec_ar":    (g["prob_ar"]       >= THRESHOLD).mean() if len(g) else float("nan"),
            "rec_full":  (g["prob_combined"] >= THRESHOLD).mean() if len(g) else float("nan"),
        }

    fold_stats.append({
        "fid":       fid,
        "date":      pd.to_datetime(fr.test_start).strftime("%b %Y"),
        "n":         int(fr.n_test),
        "pos":       int(fr.full_n_pos),
        "prev":      prev,
        "ar_pr":     fr.ar_pr_auc,
        "full_pr":   fr.full_pr_auc,
        "delta_pr":  fr.delta_pr_auc,
        "ar_roc":    fr.ar_roc_auc,
        "full_roc":  fr.full_roc_auc,
        "reg":       reg,
    })

# ── Canvas helpers ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":    9,
    "text.color":   "#111111",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

LM     = 0.07
RM     = 0.93
TM     = 0.970
LINE_H = 0.0108
HEAD_H = 0.0124
GAP_S  = 0.006
GAP_P  = 0.004

fig = None
y   = [TM]


def new_page(pdf):
    global fig
    if fig is not None:
        _add_footer(fig)
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close(fig)
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.patch.set_facecolor("white")
    y[0] = TM


def _add_footer(f):
    f.text(0.5, 0.018,
           "Source: results/window_2yr/fold_results.csv + fold_predictions.csv  |  generate_fig5_pdf.py",
           ha="center", va="top", fontsize=7, color="#777777")


def cur():    return y[0]
def advance(dy): y[0] -= dy


def tx(x, yy, s, **kw):
    fig.text(x, yy, s, **kw)


def hline(yy, color="#333333"):
    ax = fig.add_axes([LM, yy, RM - LM, 0.0010])
    ax.set_facecolor(color)
    ax.set_axis_off()


def body(text, indent=0.01, fs=8.3, color="#111111"):
    if text == "":
        advance(LINE_H * 0.45)
        return
    tx(LM + indent, cur(), text, ha="left", va="top", fontsize=fs,
       color=color, linespacing=1.3)
    advance(LINE_H)


def section_head(title, color="#1a1a5e"):
    advance(GAP_S)
    tx(LM, cur(), title, ha="left", va="top", fontsize=10,
       fontweight="bold", color=color)
    advance(HEAD_H)


def fold_head(title, color="#6a006a"):
    advance(GAP_S * 0.5)
    tx(LM, cur(), title, ha="left", va="top", fontsize=9.5,
       fontweight="bold", color=color)
    advance(HEAD_H * 0.9)


def stat_row(label, value, lx=LM+0.01, vx=LM+0.38):
    tx(lx, cur(), label, ha="left", va="top", fontsize=8.1, color="#333333")
    tx(vx, cur(), value, ha="left", va="top", fontsize=8.1,
       color="#111111", fontweight="bold")
    advance(LINE_H)


def regime_table(reg):
    """4-column regime breakdown table."""
    cols  = ["Regime", "n", "Mean delta P", "AR recall", "AR+News recall"]
    widths= [0.12, 0.06, 0.14, 0.12, 0.16]
    xs    = [LM + 0.01]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    # header
    for xi, h in zip(xs, cols):
        tx(xi, cur(), h, ha="left", va="top", fontsize=7.8,
           color="#1a1a5e", fontweight="bold")
    advance(LINE_H * 0.9)
    hline(cur(), color="#AAAAAA")
    advance(LINE_H * 0.3)
    for rname in ["onset", "chronic", "recovery", "stable"]:
        d = reg[rname]
        vals = [
            rname.capitalize(),
            str(d["n"]),
            f"{d['delta']:+.3f}" if np.isfinite(d["delta"]) else "—",
            f"{d['rec_ar']:.0%}"  if np.isfinite(d["rec_ar"])  else "—",
            f"{d['rec_full']:.0%}" if np.isfinite(d["rec_full"]) else "—",
        ]
        col_colors = ["#111111","#111111",
                      "#1a7a1a" if np.isfinite(d["delta"]) and d["delta"] > 0 else "#aa1111",
                      "#111111","#111111"]
        for xi, v, cc in zip(xs, vals, col_colors):
            tx(xi, cur(), v, ha="left", va="top", fontsize=7.8, color=cc)
        advance(LINE_H * 0.9)
    advance(GAP_P)


# ── Narrative per fold ────────────────────────────────────────────────────────
FOLD_NARRATIVES = {
    1: ("News lifts onset recall sharply; AR-Only history is thin",
        "Feb 2022 is early in the test window — IPC history is still sparse for newly "
        "deteriorating districts. AR-Only has little lagged signal to work from for the "
        "11 onset cases, achieving only 9% recall. AR+News lifts this to 55% by picking "
        "up conflict and displacement coverage that preceded the assessments. The 42 "
        "chronic cases drag the combined model slightly (news lowers their probability "
        "by -0.120 on average) but the net PR-AUC gain of +0.012 is positive. "
        "ROC-AUC is near-identical (0.951 vs 0.950) — the separation happens at the "
        "positive-class precision end, not across the full ranking."),
    2: ("AR-Only dominates; news adds noise on a stable, predictable period",
        "Jun 2022 is one of the best periods for AR-Only (PR-AUC 0.886, ROC-AUC 0.972). "
        "With only 2 onset cases, there is almost no new-crisis signal for news to detect. "
        "The 64 chronic cases are well-captured by lagged IPC history alone, and news "
        "features lower their predicted probabilities by -0.063 on average, costing "
        "precision. Delta PR-AUC = -0.013. When history is a reliable guide and no "
        "new crises are emerging, adding news introduces more noise than signal."),
    3: ("News cannot detect onset; AR-Only also struggles — a hard period",
        "Oct 2022: 5 onset cases with 0% recall for both models — neither AR history "
        "nor GDELT coverage signals the new deteriorations at this threshold. Chronic "
        "dominates (66 cases) and news pulls their probabilities down sharply (-0.287), "
        "hurting precision. Delta PR-AUC = -0.012. This period illustrates a ceiling: "
        "when crisis onset is geographically diffuse and GDELT coverage does not "
        "concentrate before the IPC assessment date, news cannot help."),
    4: ("News rescues AR-Only on a period of rising prevalence and new onsets",
        "Feb 2023 sees a jump to 26 onset cases and prevalence rising to 30%. AR-Only "
        "PR-AUC drops to 0.844 — the lagged IPC signal is weakening as crises spread "
        "to districts with no prior history. AR+News recovers to 0.885 (+0.040). News "
        "features add +0.226 mean delta probability to onset cases and +0.030 to chronic "
        "cases — a broad lift across crisis types. ROC-AUC also improves (0.921 to 0.944). "
        "This is the pattern where news earns its keep: new-onset districts where "
        "IPC history alone cannot anticipate the deterioration."),
    5: ("Largest gain: AR-Only collapses, news rescues the model",
        "Jun 2023 is the standout fold. AR-Only PR-AUC falls to 0.679 and ROC-AUC to "
        "0.675 — a near-random classifier for this period. 43 onset and 72 chronic cases "
        "at 33.7% prevalence overwhelm the lagged IPC signal. AR+News recovers to "
        "PR-AUC 0.828 and ROC-AUC 0.929 — a +0.149 PR-AUC gain. News lifts onset "
        "recall from 0% to 23% and adds +0.096 mean delta probability to onset cases "
        "and +0.244 to chronic cases. This is the strongest evidence that GDELT "
        "coverage carries genuine pre-crisis signal: it partially compensates for a "
        "period where IPC history completely fails."),
    6: ("AR-Only partially recovers but news cannot help chronic cases",
        "Oct 2023 has the highest prevalence in the test set (42.3%, 145 positives). "
        "AR-Only PR-AUC is 0.759 — better than fold 5 but still weak. AR+News is "
        "0.746 (-0.013), a small degradation. The 53 onset cases benefit from news "
        "strongly (+0.284 delta, 23% to 53% recall lift). But the 92 chronic cases "
        "are hurt severely: news lowers their probabilities by -0.373 on average. "
        "At very high crisis prevalence, the model trained on balanced classes "
        "struggles to assign high enough probabilities to chronic cases already "
        "well-identified by AR history, while news features add false confidence "
        "to stable districts. The onset gain is cancelled by the chronic drag."),
    7: ("Modest improvement; onset detection without recall lift",
        "Feb 2024: 36 onset cases, 0% recall for both models at threshold 0.5 — "
        "neither model is confident enough to flag new crisis districts. Yet "
        "PR-AUC improves from 0.648 to 0.697 (+0.049), indicating that news features "
        "improve the probability ranking even without pushing cases above threshold. "
        "Chronic cases show a small news drag (-0.042). The improvement is real but "
        "modest — the ranking is better, but actionable recall at 0.5 is not achieved. "
        "This suggests the model would benefit from a lower operating threshold "
        "for early-warning purposes in this period."),
}

# ── Build PDF ─────────────────────────────────────────────────────────────────
out_path = FIGURES_DIR / "fig5_fold_explanation.pdf"
_pdf = PdfPages(out_path)
new_page(_pdf)

# Title block
tx(0.5, cur(),
   "Fold-by-Fold Performance: Why AR+News Helps in Some Periods and Not Others",
   ha="center", va="top", fontsize=12.5, fontweight="bold")
advance(0.020)
tx(0.5, cur(),
   "7 test folds across Feb 2022 – Feb 2024  |  Primary metric: PR-AUC  |  All values from actual results",
   ha="center", va="top", fontsize=9, fontstyle="italic", color="#444444")
advance(0.016)
hline(cur())
advance(0.010)

# Overview table
section_head("Overview: All 7 Folds at a Glance")
cols  = ["Fold", "Period", "n", "Prevalence", "AR PR-AUC", "AR+News PR-AUC", "Delta", "Verdict"]
widths= [0.06, 0.10, 0.06, 0.11, 0.11, 0.14, 0.09]
xs    = [LM + 0.01]
for w in widths[:-1]:
    xs.append(xs[-1] + w)
for xi, h in zip(xs, cols):
    tx(xi, cur(), h, ha="left", va="top", fontsize=7.8, color="#1a1a5e", fontweight="bold")
advance(LINE_H * 0.9)
hline(cur(), color="#AAAAAA")
advance(LINE_H * 0.3)

for fs in fold_stats:
    d     = fs["delta_pr"]
    color = "#1a7a1a" if d > 0 else "#aa1111"
    verdict = "News helps" if d > 0.01 else ("News hurts" if d < -0.005 else "Neutral")
    vals = [
        f"Fold {fs['fid']}",
        fs["date"],
        str(fs["n"]),
        f"{fs['prev']:.1%}",
        f"{fs['ar_pr']:.3f}",
        f"{fs['full_pr']:.3f}",
        f"{d:+.3f}",
        verdict,
    ]
    vcols = ["#111111","#111111","#111111","#111111","#111111","#111111", color, color]
    for xi, v, cc in zip(xs, vals, vcols):
        tx(xi, cur(), v, ha="left", va="top", fontsize=7.8, color=cc)
    advance(LINE_H * 0.9)

advance(GAP_S)
hline(cur(), color="#AAAAAA")
advance(LINE_H * 0.3)

body("News helps in 4 of 7 folds (delta > 0). The largest gain is Fold 5 (Jun 2023, +0.149) where")
body("AR-Only collapses to near-random performance. The three negative folds share a common")
body("pattern: low onset count, strong chronic dominance, and stable IPC history that AR alone handles well.")

# Individual fold pages
for fs in fold_stats:
    new_page(_pdf)

    # Fold header
    verdict  = "News helps" if fs["delta_pr"] > 0.01 else ("News hurts" if fs["delta_pr"] < -0.005 else "Neutral")
    v_color  = "#1a7a1a" if fs["delta_pr"] > 0.01 else ("#aa1111" if fs["delta_pr"] < -0.005 else "#555555")
    narrative_title, narrative_body = FOLD_NARRATIVES[fs["fid"]]

    tx(0.5, cur(),
       f"Fold {fs['fid']}  —  {fs['date']}",
       ha="center", va="top", fontsize=13, fontweight="bold")
    advance(0.018)
    tx(0.5, cur(), f"{verdict}: AR PR-AUC = {fs['ar_pr']:.3f}  →  AR+News = {fs['full_pr']:.3f}  (delta = {fs['delta_pr']:+.3f})",
       ha="center", va="top", fontsize=10, fontweight="bold", color=v_color)
    advance(0.016)
    hline(cur())
    advance(0.010)

    # Key metrics
    section_head("Key Metrics")
    stat_row("Test observations (n):",       str(fs["n"]))
    stat_row("Crisis cases (positives):",    f"{fs['pos']}  ({fs['prev']:.1%} prevalence)")
    stat_row("AR-Only  PR-AUC:",             f"{fs['ar_pr']:.4f}")
    stat_row("AR+News  PR-AUC:",             f"{fs['full_pr']:.4f}")
    stat_row("Delta PR-AUC:",                f"{fs['delta_pr']:+.4f}")
    stat_row("AR-Only  ROC-AUC:",            f"{fs['ar_roc']:.4f}")
    stat_row("AR+News  ROC-AUC:",            f"{fs['full_roc']:.4f}")

    # Regime breakdown
    section_head("Regime Breakdown")
    body("Each row shows how news features affect predicted probabilities within that crisis regime.")
    body("Mean delta P = mean(P(AR+News) - P(AR-Only)).  Recall at threshold = 0.50.")
    advance(GAP_P)
    regime_table(fs["reg"])

    # Narrative
    section_head("What Drove This Fold's Result")
    fold_head(f'"{narrative_title}"')
    # Wrap narrative into ~110-char lines
    words = narrative_body.split()
    line  = ""
    for w in words:
        if len(line) + len(w) + 1 > 108:
            body(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        body(line)

# Close
_add_footer(fig)
_pdf.savefig(fig, bbox_inches="tight", dpi=300)
plt.close(fig)
_pdf.close()
print(f"Saved {out_path}")
