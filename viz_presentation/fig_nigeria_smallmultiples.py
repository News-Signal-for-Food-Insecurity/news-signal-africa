"""
fig_nigeria_smallmultiples.py — "Food Insecurity in Nigeria" small multiples (slide 17).

A 3x4 grid of Nigeria ADM2 maps, one per calendar quarter, shaded by the
food-security share of news coverage (green = low, red = high). Demonstrates the
spatio-temporal evolution of the signal. Coverage from monthly GDELT; geometry
from GADM. The quarters shown are logged (no silent truncation).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import data_access as da
from viz_theme import INK, INK_SOFT, apply_rcparams, is_bare, save_fig

N_ROWS, N_COLS = 3, 4                      # 12 quarterly panels, as in the talk
CMAP = "RdYlGn_r"                          # green (low) -> red (high)


def render() -> None:
    print("[fig] Nigeria quarterly small-multiples")
    apply_rcparams()

    quarterly = da.nigeria_quarterly_food_security() * 100.0   # -> %
    study, africa_ng = da.country_adm2_with_nearest_district("Nigeria")

    quarters = list(quarterly.columns)[: N_ROWS * N_COLS]
    print(f"  quarters shown: {quarters[0]} .. {quarters[-1]} "
          f"({len(quarters)} of {quarterly.shape[1]} available)")

    # Shared colour scale across all panels (robust upper bound = 95th pct).
    all_vals = quarterly[quarters].to_numpy().ravel()
    vmax = float(da.np.nanpercentile(all_vals, 95)) or 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12.5, 11.0))
    for ax, quarter in zip(axes.ravel(), quarters):
        value_by_district = quarterly[quarter]
        study["_val"] = study["nn_district"].map(value_by_district)
        africa_ng.plot(ax=ax, color="#F2F2F2", edgecolor="#BBBBBB",
                       linewidth=0.4, zorder=1)
        study.plot(ax=ax, column="_val", cmap=CMAP, norm=norm, linewidth=0.15,
                   edgecolor="white", zorder=2, missing_kwds={"color": "#E0E0E0"})
        ax.set_axis_off()
        qlabel = quarter.replace("Q", " Q")
        ax.set_title(qlabel, fontsize=12, fontweight="bold", color=INK, pad=2)

    # Hide any unused axes if fewer quarters than cells.
    for ax in axes.ravel()[len(quarters):]:
        ax.set_axis_off()

    if not is_bare():
        fig.suptitle("Food Insecurity in Nigeria — a Spatio-Temporal Process",
                     x=0.5, y=0.97, fontsize=18, fontweight="bold", color=INK)
        fig.text(0.5, 0.935, "Food-security share of news coverage by district and "
                 "quarter", ha="center", fontsize=12, color=INK_SOFT)

    # Shared colourbar.
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04,
                        pad=0.04, shrink=0.5)
    cbar.set_label("Food-security relative coverage (%)", fontsize=10,
                   color=INK)
    cbar.ax.tick_params(labelsize=9)

    if not is_bare():
        fig.text(0.5, 0.015, "Source: GDELT (food-security theme share). Boundaries: "
                 "GADM v4.1 (ADM2). Common colour scale across panels.",
                 ha="center", fontsize=8, color=INK_SOFT, fontstyle="italic")
    save_fig(fig, "nigeria_quarterly_food_security")


if __name__ == "__main__":
    render()
