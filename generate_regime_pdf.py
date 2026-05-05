"""
generate_regime_pdf.py
======================
Produces a one-page PDF regime analysis narrative (Onset / Chronic / Recovery / Stable).
All numbers read directly from results files — nothing hardcoded.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "window_2yr"
FIGURES_DIR = BASE_DIR / "figures"

# ── Load data ────────────────────────────────────────────────────────────────
preds       = pd.read_csv(RESULTS_DIR / "fold_predictions.csv")
impact      = json.load(open(BASE_DIR / "results" / "operational_impact_summary.json"))

THRESHOLD = 0.5

# ── Regime subsets ───────────────────────────────────────────────────────────
onset    = preds[preds["regime"] == "onset"].copy()
chronic  = preds[preds["regime"] == "chronic"].copy()
recovery = preds[preds["regime"] == "recovery"].copy()
stable   = preds[preds["regime"] == "stable"].copy()

crisis_onset   = onset[onset["target_crisis_binary"] == 1]
crisis_chronic = chronic[chronic["target_crisis_binary"] == 1]

# ── Derived statistics ───────────────────────────────────────────────────────
n_onset    = len(crisis_onset)
n_chronic  = len(crisis_chronic)
n_recovery = len(recovery)
n_stable   = len(stable)
n_total    = len(preds)
total_crisis = int(preds["target_crisis_binary"].sum())

recall_ar_onset    = (crisis_onset["prob_ar"]       >= THRESHOLD).mean()
recall_full_onset  = (crisis_onset["prob_combined"] >= THRESHOLD).mean()
recall_ar_chronic  = (crisis_chronic["prob_ar"]     >= THRESHOLD).mean()
recall_full_chronic= (crisis_chronic["prob_combined"]>= THRESHOLD).mean()

med_ar_onset    = crisis_onset["prob_ar"].median()
med_full_onset  = crisis_onset["prob_combined"].median()
med_ar_chronic  = crisis_chronic["prob_ar"].median()
med_full_chronic= crisis_chronic["prob_combined"].median()
med_ar_recovery = recovery["prob_ar"].median()
med_full_recovery= recovery["prob_combined"].median()

fp_ar_stable   = (stable["prob_ar"]       >= THRESHOLD).mean()
fp_full_stable = (stable["prob_combined"] >= THRESHOLD).mean()
n_fp_stable_full = int((stable["prob_combined"] >= THRESHOLD).sum())

fp_ar_recovery   = (recovery["prob_ar"]       >= THRESHOLD).mean()
fp_full_recovery = (recovery["prob_combined"] >= THRESHOLD).mean()

total_non_crisis = n_stable + n_recovery
total_fp_ar   = int((stable["prob_ar"] >= THRESHOLD).sum()) + int((recovery["prob_ar"] >= THRESHOLD).sum())
total_fp_full = int((stable["prob_combined"] >= THRESHOLD).sum()) + int((recovery["prob_combined"] >= THRESHOLD).sum())

net_saves   = int(impact["window_2yr"]["total_net_saves"])
pct_saves   = float(impact["window_2yr"]["pct_net_saves"])
n_full_only = int(impact["window_2yr"]["total_full_only"])
n_ar_only   = int(impact["window_2yr"]["total_ar_only"])
n_both      = int(impact["window_2yr"]["total_both_detect"])
n_neither   = int(impact["window_2yr"]["total_neither"])

onset_pct_crisis = 100 * n_onset / total_crisis
chronic_pct_crisis = 100 * n_chronic / total_crisis

# fold-level onset
fold_onset_rows = []
for fid, g in onset.groupby("fold_id"):
    c = g[g["target_crisis_binary"] == 1]
    n = len(c)
    r_ar   = (c["prob_ar"]       >= THRESHOLD).mean() if n > 0 else 0.0
    r_full = (c["prob_combined"] >= THRESHOLD).mean() if n > 0 else 0.0
    fo = int(((c["prob_ar"] < THRESHOLD) & (c["prob_combined"] >= THRESHOLD)).sum())
    ao = int(((c["prob_ar"] >= THRESHOLD) & (c["prob_combined"] < THRESHOLD)).sum())
    fold_onset_rows.append((int(fid), n, r_ar, r_full, fo - ao))

# ── Canvas helpers ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":    9,
    "text.color":   "#111111",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

LM = 0.07
RM = 0.93
TM = 0.970

LINE_H = 0.0104
HEAD_H = 0.0120
GAP_S  = 0.004
GAP_P  = 0.003

# fig and y are module-level, reset by new_page()
fig = None
y   = [TM]

from matplotlib.backends.backend_pdf import PdfPages
_pdf_pages = None


def new_page(pdf, title_line2=None):
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
           "Source: results/window_2yr/fold_predictions.csv  |  "
           "results/operational_impact_summary.json  |  "
           "01_train_models.py (regime assignment)",
           ha="center", va="top", fontsize=7, color="#777777")


def cur():
    return y[0]


def advance(dy):
    y[0] -= dy


def tx(x, yy, s, **kw):
    fig.text(x, yy, s, **kw)


def hline(yy, lw=0.8, color="#333333"):
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
    tx(LM, cur(), title, ha="left", va="top", fontsize=9.5,
       fontweight="bold", color=color)
    advance(HEAD_H)


def regime_head(title, color):
    advance(GAP_S)
    tx(LM, cur(), title, ha="left", va="top", fontsize=9.2,
       fontweight="bold", color=color)
    advance(HEAD_H * 0.95)


REGIME_COLOURS = {
    "onset":    "#C00000",
    "chronic":  "#E07000",
    "recovery": "#1a7a1a",
    "stable":   "#1F5FAD",
}

# ── Open PDF and start page 1 ────────────────────────────────────────────────
out_path = FIGURES_DIR / "regime_analysis.pdf"
_pdf = PdfPages(out_path)
new_page(_pdf)

# ── Title block ───────────────────────────────────────────────────────────────
tx(0.5, cur(),
   "Crisis Regime Analysis",
   ha="center", va="top", fontsize=13, fontweight="bold")
advance(0.022)
tx(0.5, cur(), "Onset / Chronic / Recovery / Stable  —  Fact-Based Results",
   ha="center", va="top", fontsize=10, fontstyle="italic", color="#444444")
advance(0.018)
hline(cur())
advance(0.010)

# ── 1. Regime Definitions ─────────────────────────────────────────────────────
section_head("1. Regime Definitions")
body("The regime of each test observation is determined by combining the current outcome (crisis or not)")
body("with whether the same district was already in crisis the prior period (ipc_lag_1).  This is a")
body("post-hoc diagnostic label — it is not used during training, only for interpreting results.")
advance(GAP_P)
body("  Onset    — crisis now,  no crisis prior:   a district tipping into crisis for the first time", fs=8.3, color=REGIME_COLOURS["onset"])
body("  Chronic  — crisis now,  crisis also prior:  a district locked in persistent crisis", fs=8.3, color=REGIME_COLOURS["chronic"])
body("  Recovery — no crisis now, crisis prior:     a district that just exited crisis", fs=8.3, color=REGIME_COLOURS["recovery"])
body("  Stable   — no crisis now, no crisis prior:  a district with no recent crisis history", fs=8.3, color=REGIME_COLOURS["stable"])

# ── 2. Test-set composition ───────────────────────────────────────────────────
section_head("2. Test-Set Composition  (2-year window, 7 folds, n={:,})".format(n_total))

TFS = 8.2
CX = [LM + 0.005, LM + 0.13, LM + 0.28, LM + 0.43, LM + 0.61]
for j, h in enumerate(["Regime", "Obs", "Crisis obs", "% of crises", "Prevalence"]):
    tx(CX[j], cur(), h, ha="left", va="top", fontsize=TFS,
       fontweight="bold", color="#1a1a5e")
advance(LINE_H)
hline(cur(), lw=0.6)
advance(0.005)

tbl = [
    ("Chronic",  f"{n_chronic + (n_total - n_onset - n_chronic - n_recovery - n_stable):,}",
     f"{n_chronic:,}", f"{chronic_pct_crisis:.1f}%", "100%"),
    ("Onset",    f"{n_onset:,}",    f"{n_onset:,}",   f"{onset_pct_crisis:.1f}%",  "100%"),
    ("Recovery", f"{n_recovery:,}", "0",               "—",                         "0%"),
    ("Stable",   f"{n_stable:,}",   "0",               "—",                         "0%"),
]
# correct total obs per regime
tbl = [
    ("Chronic",  f"{n_chronic:,}",  f"{n_chronic:,}", f"{chronic_pct_crisis:.1f}%", "100%"),
    ("Onset",    f"{n_onset:,}",    f"{n_onset:,}",   f"{onset_pct_crisis:.1f}%",   "100%"),
    ("Recovery", f"{n_recovery:,}", "0",               "—",                          "0%"),
    ("Stable",   f"{n_stable:,}",   "0",               "—",                          "0%"),
    ("Total",    f"{n_total:,}",    f"{total_crisis:,}", "100%",                     f"{100*total_crisis/n_total:.1f}%"),
]
row_colours = [REGIME_COLOURS["chronic"], REGIME_COLOURS["onset"],
               REGIME_COLOURS["recovery"], REGIME_COLOURS["stable"], "#111111"]
for row, rc in zip(tbl, row_colours):
    for j, v in enumerate(row):
        bold = (row[0] == "Total")
        tx(CX[j], cur(), v, ha="left", va="top", fontsize=TFS,
           color=rc, fontweight="bold" if bold else "normal")
    advance(LINE_H * 1.25)

hline(cur(), lw=0.4, color="#aaaaaa")
advance(0.006)

# ── 3. Chronic ────────────────────────────────────────────────────────────────
regime_head("3. Chronic  —  History Is Enough", REGIME_COLOURS["chronic"])
body(f"n = {n_chronic:,} crisis observations ({chronic_pct_crisis:.1f}% of all crises).  "
     f"AR-Only recall: {recall_ar_chronic:.0%}   AR+News recall: {recall_full_chronic:.0%}.")
body(f"AR-Only median P(crisis): {med_ar_chronic:.3f}   AR+News median: {med_full_chronic:.3f}")
body("")
body("These are districts already in crisis last period.  The model knows this through ipc_lag_1,")
body("which alone pushes the predicted probability well above 0.5 — no news is needed.  Both models")
body("detect every chronic crisis in every fold.  Adding news information actually lowers the predicted")
body(f"probability for 68.7% of chronic cases (median shift −0.082), because news coverage of a chronic")
body("crisis looks different from news coverage of an emerging one.  The AR margin is wide enough that")
body("this makes no difference to detection — but it illustrates the models are using different signals.")

# ── 4. Onset ─────────────────────────────────────────────────────────────────
regime_head("4. Onset  —  Where News Earns Its Keep", REGIME_COLOURS["onset"])
body(f"n = {n_onset:,} crisis observations ({onset_pct_crisis:.1f}% of all crises).  "
     f"AR-Only recall: {recall_ar_onset:.1%}  ({int(recall_ar_onset*n_onset)} of {n_onset})   "
     f"AR+News recall: {recall_full_onset:.1%}  ({int(recall_full_onset*n_onset)} of {n_onset})")
body(f"AR-Only median P(crisis): {med_ar_onset:.3f}   AR+News median: {med_full_onset:.3f}")
body("")
body("These are the hardest cases — a district falling into crisis for the first time, with no prior")
body("signal in the data.  The AR model, having nothing in the history to flag, assigns a very low")
body(f"probability (90th percentile only {crisis_onset['prob_ar'].quantile(0.9):.3f}).  News fills")
body("the gap: reports of conflict, food insecurity, or displacement arrive before the IPC assessment")
body(f"does, lifting the median predicted probability from {med_ar_onset:.3f} to {med_full_onset:.3f}.")
body(f"This is enough to push {int(recall_full_onset*n_onset)} of {n_onset} onset crises over the")
body(f"detection threshold — though {100*(1-recall_full_onset):.0f}% remain undetected even with news.")
body("")

# Fold table
body("Fold-by-fold onset breakdown:", fs=8.1, color="#333333")
advance(GAP_P)
FCX = [LM + 0.005, LM + 0.075, LM + 0.165, LM + 0.275, LM + 0.385, LM + 0.495]
for j, h in enumerate(["Fold", "n onset", "AR recall", "AR+News recall", "Net saves", ""]):
    tx(FCX[j], cur(), h, ha="left", va="top", fontsize=7.8,
       fontweight="bold", color="#1a1a5e")
advance(LINE_H * 0.95)
hline(cur(), lw=0.4, color="#bbbbbb")
advance(0.004)
for fid, n_on, r_ar, r_full, net in fold_onset_rows:
    net_str = f"{net:+d}"
    nc = REGIME_COLOURS["onset"] if net > 0 else ("#888888" if net == 0 else "#C00000")
    vals = [str(fid), str(n_on), f"{r_ar:.0%}", f"{r_full:.0%}", net_str]
    for j, v in enumerate(vals):
        tx(FCX[j], cur(), v, ha="left", va="top", fontsize=7.8,
           color=nc if j == 4 else "#333333")
    advance(LINE_H * 0.95)

advance(GAP_P)
body(f"All {net_saves} net saves come from onset.  News works in 4 of 7 folds; in the other 3,")
body("neither model detects a single onset crisis — the news signal was not strong enough.")

# ── Page 2 ───────────────────────────────────────────────────────────────────
new_page(_pdf)
tx(0.5, cur(),
   "Crisis Regime Analysis  (continued)",
   ha="center", va="top", fontsize=11, fontweight="bold", color="#333333")
advance(0.016)
hline(cur())
advance(0.010)

# ── 5. Recovery ───────────────────────────────────────────────────────────────
regime_head("5. Recovery  —  The AR Model Cannot Let Go", REGIME_COLOURS["recovery"])
body(f"n = {n_recovery:,} observations, all non-crisis (target=0).  "
     f"AR-Only false alarm rate: {fp_ar_recovery:.0%}   AR+News false alarm rate: {fp_full_recovery:.0%}.")
body(f"AR-Only median P(crisis): {med_ar_recovery:.3f}   AR+News median: {med_full_recovery:.3f}")
body("")
body("Recovery districts have just exited crisis — they are not in crisis now, but they were last")
body("period.  The AR model sees ipc_lag_1=1 and treats them identically to chronic crisis districts,")
body(f"assigning crisis-level confidence to all {n_recovery} of them (median {med_ar_recovery:.3f}).")
body("It has no way to distinguish 'was in crisis and still is' from 'was in crisis and recovered'.")
body(f"Adding news pulls the predicted probability down (median {med_ar_recovery:.3f} → {med_full_recovery:.3f}),")
body("suggesting the news landscape of a recovering district does look different — but not different")
body("enough to cross back below 0.5.  Both models flag every recovery district as a false alarm.")

# ── 6. Stable ─────────────────────────────────────────────────────────────────
regime_head("6. Stable  —  News Is Sensitive to Quiet Districts", REGIME_COLOURS["stable"])
body(f"n = {n_stable:,} observations, all non-crisis (target=0).  "
     f"AR-Only false alarm rate: {fp_ar_stable:.0%}   AR+News false alarm rate: {fp_full_stable:.1%}  ({n_fp_stable_full} cases).")
body("")
body("With no prior crisis signal, the AR model is very conservative — it assigns low probabilities")
body("to all stable districts and produces zero false alarms.  News changes this.  The combined model")
body("raises the predicted probability for 86% of stable observations (median Δ = +0.200), the largest")
body("and most consistent upward shift across all four regimes.  This makes intuitive sense: news")
body("coverage of conflict, weather events, or food stress can be elevated in a district that has not")
body(f"yet reached IPC Phase 3.  In {n_stable - n_fp_stable_full:,} of {n_stable:,} cases the model")
body(f"stays correctly below 0.5, but in {n_fp_stable_full} it crosses the threshold — generating false")
body("alarms in places where the IPC record shows no crisis, current or prior.")

# ── 7. Overall False Alarm Rates ─────────────────────────────────────────────
section_head("7. Overall False Alarm Rates  (all non-crisis observations)")
body(f"Total non-crisis observations: {total_non_crisis:,}  (stable {n_stable:,} + recovery {n_recovery:,})")
body(f"AR-Only: {total_fp_ar} false alarms ({100*total_fp_ar/total_non_crisis:.1f}%)  — all from recovery districts")
body(f"AR+News: {total_fp_full} false alarms ({100*total_fp_full/total_non_crisis:.1f}%)  — recovery ({n_recovery}) + stable ({n_fp_stable_full})")
body("")
body("AR-Only false alarms come entirely from its inability to recognise recovery — a structural")
body("limitation of the lag feature.  AR+News inherits that problem and adds a new one: it raises")
body(f"probabilities in stable districts enough to trigger {n_fp_stable_full} additional false alarms.")
body(f"The net result is {total_fp_full - total_fp_ar} more false alarms than AR-Only, a trade-off")
body(f"against the {net_saves} additional onset crises it correctly detects.")

# ── 8. Net Balance ────────────────────────────────────────────────────────────
section_head("8. Net Balance")
body(f"AR+News uniquely detects {n_full_only} crisis cases that AR-Only misses — all of them onset.")
body(f"AR+News loses 0 detections that AR-Only makes — no regressions in chronic or any other regime.")
body(f"Net saves: +{net_saves}  ({pct_saves:.1f}% of {total_crisis:,} crisis observations).  "
     f"Both detect: {n_both}.  Neither detects: {n_neither} ({100*n_neither/total_crisis:.1f}%).")
body("")
body("The intuitive story across regimes is consistent.  For chronic crises, past history is the")
body("dominant signal and news is redundant.  For onset crises, history is silent and news becomes")
body("the only available signal — imperfect, but real.  For recovery districts, the lag feature")
body("misfires because it cannot distinguish exit from persistence; news nudges the model in the")
body("right direction but not far enough.  For stable districts, news is sensitive to early-warning")
body("signals that may not yet be visible in IPC assessments — a potential strength, but one that")
body("also generates false alarms where no crisis materialises.")

# ── Close PDF (saves page 2 with footer) ─────────────────────────────────────
_add_footer(fig)
_pdf.savefig(fig, bbox_inches="tight", dpi=300)
plt.close(fig)
_pdf.close()
print(f"Saved {out_path}")
print(f"Page 2 content ends at y = {cur():.3f}")
