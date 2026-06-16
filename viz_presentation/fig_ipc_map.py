"""
fig_ipc_map.py — "IPC data" crisis-prevalence choropleth (talk slide 16R).

Each study ADM2 district is shaded by the share of IPC periods in which it was in
crisis (IPC Phase >= 3), on the IPC yellow-orange-red scale. Prevalence from the
modelling dataset; geometry from GADM.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import data_access as da
from map_base import draw_choropleth
from viz_theme import INK, INK_SOFT, IPC_SEQ, apply_rcparams, is_bare, save_fig


def render() -> None:
    print("[fig] IPC crisis-prevalence choropleth")
    apply_rcparams()

    prev = da.district_crisis_prevalence()           # 0-100 %
    plot, africa = da.choropleth_frame(prev, "crisis_prev")

    # Fixed, interpretable prevalence bands (percent of periods in crisis).
    bins = [0, 10, 25, 50, 75, 100]
    labels = ["0 – 10 %", "10 – 25 %", "25 – 50 %", "50 – 75 %", "75 – 100 %"]

    fig, ax = plt.subplots(figsize=(8.0, 9.2))
    draw_choropleth(ax, plot, africa, "crisis_prev", bins, IPC_SEQ, labels,
                    "Crisis prevalence\n(% of IPC periods)")
    if not is_bare():
        ax.set_title("Food-Crisis Prevalence (IPC Phase ≥ 3), 2020–2024",
                     fontsize=16, fontweight="bold", color=INK, pad=10)
        fig.text(0.5, 0.02, "Source: FEWS NET / IPC classifications. District "
                 "boundaries: GADM v4.1 (ADM2). Crisis = IPC Phase ≥ 3.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "ipc_crisis_prevalence_choropleth")


if __name__ == "__main__":
    render()
