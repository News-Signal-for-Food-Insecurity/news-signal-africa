"""
build_all.py — render the full GSSI-talk visualization suite from real data.

Ensures the News-only model predictions exist (generating them via the real
training pipeline if needed), then renders every figure to figures/presentation/
as vector PDF + 600-DPI PNG.

Run:  python build_all.py            # render everything
      python build_all.py --force-news   # also regenerate news-only predictions
"""
from __future__ import annotations

import argparse
import time

import make_news_only_predictions as news
import fig_pr_comparison
import fig_news_volume_map
import fig_ipc_map
import fig_nigeria_smallmultiples
import fig_datasources_map
import fig_temporal_heatmap
import fig_spatial_heatmap
import fig_null_histogram
import fig_predictions_scatter
import fig_deltap_violin
from viz_theme import OUTPUT_DIR, OUTPUT_DIR_UNTITLED, set_bare

# Figures grouped by the talk's narrative arc.
FIGURES = [
    ("Study scope (data sources)", fig_datasources_map.render),
    ("News-volume choropleth", fig_news_volume_map.render),
    ("IPC crisis-prevalence choropleth", fig_ipc_map.render),
    ("Nigeria quarterly small-multiples", fig_nigeria_smallmultiples.render),
    ("Temporal heterogeneity heatmap", fig_temporal_heatmap.render),
    ("Spatial heterogeneity heatmap", fig_spatial_heatmap.render),
    ("Precision-Recall build (3 states)", fig_pr_comparison.render),
    ("Null-distribution histograms", lambda: (fig_null_histogram.render("pr_auc"),
                                              fig_null_histogram.render("roc_auc"))),
    ("Predictions scatter by regime", fig_predictions_scatter.render),
    ("Delta-P violins by regime", fig_deltap_violin.render),
]


def _render_all() -> None:
    for name, fn in FIGURES:
        print(f"\n>> {name}")
        fn()


def main(force_news: bool = False) -> None:
    t0 = time.time()
    print("=" * 64)
    print("Building GSSI-talk visualization suite (real data, no hardcoding)")
    print("=" * 64)

    print("\n[1/3] Ensuring News-only model predictions exist...")
    news.generate(force=force_news)

    print("\n[2/3] Rendering TITLED figures (presentation)...")
    set_bare(False)
    _render_all()

    print("\n[3/3] Rendering UNTITLED figures (paper-ready, no headings)...")
    set_bare(True)
    _render_all()
    set_bare(False)

    for d in (OUTPUT_DIR, OUTPUT_DIR_UNTITLED):
        pdfs = sorted(d.glob("*.pdf"))
        pngs = sorted(d.glob("*.png"))
        print(f"\n{len(pdfs)} PDF + {len(pngs)} PNG  ->  {d}")
    print("\n" + "=" * 64)
    print(f"Done in {time.time() - t0:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-news", action="store_true",
                    help="regenerate the News-only predictions before rendering")
    main(force_news=ap.parse_args().force_news)
