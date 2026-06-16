"""
viz_theme.py — single source of design tokens for the GSSI-talk visualization suite.

Every figure in ``viz_presentation`` imports its colours, fonts, sizing and the
``save_fig`` helper from here, so the whole suite shares one visual system that
matches the talk (serif display type, teal/coral/purple model palette, blue
sequential maps, sage-green null bars, crimson accent).

Nothing in this module encodes a *result*; it holds styling constants only.
All numbers shown in figures come from the data via ``data_access.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # news-signal-africa/
OUTPUT_DIR = BASE_DIR / "figures" / "presentation"
OUTPUT_DIR_UNTITLED = BASE_DIR / "figures" / "presentation_untitled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# "Bare" mode — figure body + legends only (no titles / subtitles / captions),
# for dropping straight into a paper where the caption lives in the manuscript.
# Figures guard their heading/footer text with is_bare(); axis labels, tick
# labels, legends, panel labels and data annotations are always kept.
# ---------------------------------------------------------------------------
_BARE = {"on": False}


def set_bare(on: bool) -> None:
    _BARE["on"] = bool(on)
    if on:
        OUTPUT_DIR_UNTITLED.mkdir(parents=True, exist_ok=True)


def is_bare() -> bool:
    return _BARE["on"]

# ---------------------------------------------------------------------------
# Colour system (matches the talk's deck)
# ---------------------------------------------------------------------------
# Three models, distinguished on a hue axis (the talk's exact identity colours).
MODEL_COLOURS = {
    "News":      "#16A98C",   # teal-green   (News-only)
    "AR":        "#E0565B",   # coral-red    (autoregressive, drawn dashed)
    "AR + News": "#7D69C9",   # violet       (combined)
}
BASELINE_COLOUR = "#E0565B"   # prevalence baseline (dashed coral, as in slide 15)

# Soft plot-background tints used in the progressive PR build (slides 15/22/23).
PR_PANEL_TINTS = {
    "News":      "#EAF6F1",   # faint teal
    "AR":        "#FCEDEE",   # faint coral
    "AR + News": "#EEEAF7",   # faint violet
}

# Crisis regimes — reds for crisis states, greens for non-crisis (existing palette).
REGIME_COLOURS = {
    "onset":    "#E87070",
    "chronic":  "#C00000",
    "recovery": "#6CC56C",
    "stable":   "#1A7A1A",
}
REGIME_ORDER = ["onset", "chronic", "recovery", "stable"]
REGIME_SUBTITLES = {
    "onset":    "Transition\ninto crisis",
    "chronic":  "Persistent\ncrisis",
    "recovery": "Transition\nout of crisis",
    "stable":   "Persistent\nnon-crisis",
}

# Sequential blues for news heatmaps / news-volume choropleth (5 ordered steps).
BLUE_SEQ = ["#E9F0FA", "#C2D8F0", "#7BAFDE", "#2F7FC0", "#0B3D7A"]

# Yellow-orange-red for IPC crisis prevalence (FEWS NET / IPC convention).
IPC_SEQ = ["#FFF7BC", "#FED976", "#FC4E2A", "#BD0026", "#67000D"]

# Sage green for the null-distribution histogram bars (slide 26).
NULL_BAR_COLOUR = "#9DB389"
NULL_BAR_EDGE = "#7A9270"

# Neutral / accent ink.
INK = "#1A1A1A"            # primary text — high contrast on white (>15:1)
INK_SOFT = "#52525B"       # secondary text — still >7:1 on white
GRIDLINE = "#D9D9D9"
CRIMSON = "#A0211E"        # talk's accent for titles / emphasis
LAND_GREY = "#E8E8E8"      # off-study continent backdrop
BORDER_GREY = "#555555"

# Region grouping (kept identical to 06_paper_figures.py / generate_fig7_new.py).
COUNTRY_REGION = {
    "Burkina Faso": "West Africa", "Burundi": "East Africa",
    "Cameroon": "Central Africa", "Chad": "Central Africa",
    "Democratic Republic of the Congo": "Central Africa", "Ethiopia": "East Africa",
    "Kenya": "East Africa", "Madagascar": "Southern Africa",
    "Malawi": "Southern Africa", "Mali": "West Africa",
    "Mozambique": "Southern Africa", "Niger": "West Africa",
    "Nigeria": "West Africa", "Somalia": "East Africa",
    "South Sudan": "East Africa", "Sudan": "North Africa",
    "Uganda": "East Africa", "Zimbabwe": "Southern Africa",
}
REGION_ORDER = ["East Africa", "West Africa", "Central Africa",
                "North Africa", "Southern Africa"]

# Human-readable theme labels (column stem -> display).
THEME_LABELS = {
    "conflict": "Conflict", "displacement": "Displacement", "economic": "Economic",
    "food_security": "Food Security", "governance": "Governance", "health": "Health",
    "humanitarian": "Humanitarian", "weather": "Weather", "other": "Other",
}

# ---------------------------------------------------------------------------
# Typography + global rcParams
# ---------------------------------------------------------------------------
def _serif_stack() -> list[str]:
    """Prefer an elegant serif that is actually installed, else fall back."""
    installed = {f.name for f in fm.fontManager.ttflist}
    preferred = ["Georgia", "Charter", "Palatino Linotype", "Book Antiqua",
                 "Times New Roman", "Liberation Serif", "DejaVu Serif"]
    ordered = [f for f in preferred if f in installed]
    # Always keep DejaVu Serif as a guaranteed terminal fallback.
    if "DejaVu Serif" not in ordered:
        ordered.append("DejaVu Serif")
    return ordered


def apply_rcparams() -> None:
    """Apply the suite-wide matplotlib style. Call once before plotting."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _serif_stack(),
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#9A9A9A",
        "axes.linewidth": 0.9,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "#CCCCCC",
        "lines.linewidth": 2.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,       # embed TrueType so text stays editable/vector
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Output helper — every figure is written as BOTH vector PDF and 600-DPI PNG
# ---------------------------------------------------------------------------
def save_fig(fig, name: str, *, pad: float = 0.25) -> tuple[Path, Path]:
    """Save ``fig`` as <name>.pdf (vector) and <name>.png (600 DPI).

    Writes to OUTPUT_DIR, or OUTPUT_DIR_UNTITLED when bare mode is on.
    """
    out_dir = OUTPUT_DIR_UNTITLED if is_bare() else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=pad)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"  saved  {pdf_path.name}  +  {png_path.name}")
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Small shared drawing helpers
# ---------------------------------------------------------------------------
def style_axes(ax, *, grid_axis: str | None = "y") -> None:
    """Light, de-emphasised gridlines and clean spines."""
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def suptitle(fig, title: str, subtitle: str | None = None, *, x: float = 0.5,
             ha: str = "center") -> None:
    """Serif display title with optional muted subtitle, talk-styled.

    No-op in bare mode (paper figures carry their title in the manuscript).
    """
    if is_bare():
        return
    fig.suptitle(title, x=x, ha=ha, fontsize=18, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, 0.93, subtitle, ha=ha, fontsize=11.5, color=INK_SOFT)


def source_note(fig, text: str, *, x: float = 0.5, y: float = 0.005,
                ha: str = "center") -> None:
    """Footer attribution / data-provenance line. No-op in bare mode."""
    if is_bare():
        return
    fig.text(x, y, text, ha=ha, va="bottom", fontsize=8, color=INK_SOFT,
             fontstyle="italic")


def binned_cmap(colours: list[str], bins: list[float]):
    """Return (ListedColormap, BoundaryNorm) for a binned choropleth/heatmap."""
    cmap = ListedColormap(colours[: len(bins) - 1])
    norm = BoundaryNorm(bins, cmap.N)
    return cmap, norm


def add_north_arrow(ax, x: float = 0.08, y0: float = 0.12, y1: float = 0.19) -> None:
    """North arrow in axes-fraction coordinates (for maps)."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8,
                                mutation_scale=13))
    ax.text(x, y0 - 0.02, "N", transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color="#333333")
