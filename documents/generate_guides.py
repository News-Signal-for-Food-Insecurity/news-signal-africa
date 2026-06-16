"""
generate_guides.py
==================
Generates two standalone, premium-quality step-by-step guide PDFs for the
news-signal-africa repository:

  1. running-the-plots.pdf   — how to regenerate the paper figures
  2. accessing-the-data.pdf  — how to obtain and use the repository data

The PDFs are written into this ``documents/`` folder.

The layout is built with ReportLab's Platypus engine for professional
typography: a branded title band, numbered "step chips", shaded code
blocks with an accent bar, tinted callout boxes (note / warning / tip),
and a running footer with page numbers.

Run:
    python documents/generate_guides.py
"""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand palette ─────────────────────────────────────────────────────────────
NAVY = HexColor("#0E2A3B")     # deep heading ink
ACCENT = HexColor("#15658F")   # primary accent (steel blue)
ACCENT_DK = HexColor("#0C4666")
INK = HexColor("#23303A")      # body text
MUTED = HexColor("#6B7A85")    # secondary text
HAIRLINE = HexColor("#D5DEE4")
CODE_BG = HexColor("#0E2A3B")  # dark code panel
CODE_FG = HexColor("#E8F1F7")  # code text
CODE_ACCENT = HexColor("#43B0D9")
PAGE_BG_BAND = HexColor("#F4F8FA")

# Callout tints: (bar, fill, label)
CALLOUT = {
    "note":    (HexColor("#15658F"), HexColor("#EAF3F8"), "NOTE"),
    "warning": (HexColor("#C2620E"), HexColor("#FBF1E6"), "IMPORTANT"),
    "tip":     (HexColor("#2E7D52"), HexColor("#EAF5EE"), "TIP"),
}

MARGIN = 0.85 * inch
CONTENT_W = LETTER[0] - 2 * MARGIN


# ── Paragraph styles ──────────────────────────────────────────────────────────
def _styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["Title"] = ParagraphStyle(
        "Title", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=26, leading=30, textColor=HexColor("#FFFFFF"))
    s["Subtitle"] = ParagraphStyle(
        "Subtitle", parent=base["Normal"], fontName="Helvetica",
        fontSize=11.5, leading=16, textColor=HexColor("#CFE2EE"))
    s["Kicker"] = ParagraphStyle(
        "Kicker", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=9, leading=12, textColor=CODE_ACCENT, spaceAfter=4)
    s["H1"] = ParagraphStyle(
        "H1", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=NAVY)
    s["StepTitle"] = ParagraphStyle(
        "StepTitle", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=14, leading=18, textColor=NAVY)
    s["H2"] = ParagraphStyle(
        "H2", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=ACCENT_DK, spaceBefore=8,
        spaceAfter=2)
    s["Body"] = ParagraphStyle(
        "Body", parent=base["Normal"], fontName="Helvetica",
        fontSize=10.3, leading=15.5, textColor=INK, alignment=TA_LEFT,
        spaceAfter=6)
    s["Bullet"] = ParagraphStyle(
        "Bullet", parent=s["Body"], leftIndent=16, bulletIndent=4,
        spaceAfter=3.5)
    s["Code"] = ParagraphStyle(
        "Code", parent=base["Normal"], fontName="Courier",
        fontSize=9, leading=13.2, textColor=CODE_FG)
    s["CalloutLabel"] = ParagraphStyle(
        "CalloutLabel", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, textColor=NAVY, spaceAfter=2)
    s["CalloutBody"] = ParagraphStyle(
        "CalloutBody", parent=base["Normal"], fontName="Helvetica",
        fontSize=9.6, leading=14, textColor=INK)
    s["Footer"] = ParagraphStyle(
        "Footer", parent=base["Normal"], fontName="Helvetica",
        fontSize=8, leading=10, textColor=MUTED)
    return s


STYLES = _styles()


# ── Custom flowables ──────────────────────────────────────────────────────────
class HRule(Flowable):
    """A thin full-width horizontal rule."""

    def __init__(self, width: float, color=HAIRLINE, thickness: float = 0.8,
                 pad: float = 0.0):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness
        self.height = thickness + 2 * pad
        self.pad = pad

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.pad, self.width, self.pad)


def step_header(num: int, title: str) -> Flowable:
    """A numbered step chip (filled badge + title), kept with its content."""
    badge = ParagraphStyle(
        "Badge", fontName="Helvetica-Bold", fontSize=14, leading=16,
        textColor=HexColor("#FFFFFF"), alignment=1)
    cell = Table(
        [[Paragraph(str(num), badge), Paragraph(title, STYLES["StepTitle"])]],
        colWidths=[0.42 * inch, CONTENT_W - 0.42 * inch - 0.16 * inch])
    cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 0), (0, 0), 6),
        ("BOTTOMPADDING", (0, 0), (0, 0), 6),
        ("LEFTPADDING", (1, 0), (1, 0), 12),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
    ]))
    return cell


def code_block(code: str) -> Flowable:
    """A dark, shaded code panel with a left accent bar."""
    lines = code.strip("\n").split("\n")
    safe = "<br/>".join(
        ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
          .replace(" ", "&nbsp;")
        for ln in lines)
    body = Paragraph(safe, STYLES["Code"])
    t = Table([["", body]], colWidths=[0.09 * inch, CONTENT_W - 0.09 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), CODE_ACCENT),
        ("BACKGROUND", (1, 0), (1, 0), CODE_BG),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (1, 0), (1, 0), 14),
        ("RIGHTPADDING", (1, 0), (1, 0), 12),
        ("TOPPADDING", (1, 0), (1, 0), 11),
        ("BOTTOMPADDING", (1, 0), (1, 0), 11),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
    ]))
    return t


def callout(kind: str, text: str) -> Flowable:
    """A tinted callout box (note / warning / tip) with a colored left bar."""
    bar, fill, label = CALLOUT[kind]
    inner = [
        Paragraph(label, STYLES["CalloutLabel"]),
        Paragraph(text, STYLES["CalloutBody"]),
    ]
    t = Table([["", inner]], colWidths=[0.10 * inch, CONTENT_W - 0.10 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), bar),
        ("BACKGROUND", (1, 0), (1, 0), fill),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (1, 0), (1, 0), 12),
        ("RIGHTPADDING", (1, 0), (1, 0), 12),
        ("TOPPADDING", (1, 0), (1, 0), 9),
        ("BOTTOMPADDING", (1, 0), (1, 0), 9),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
    ]))
    return t


def bullets(items: list[str]) -> list[Flowable]:
    """Build a list of bullet Paragraphs."""
    return [Paragraph(it, STYLES["Bullet"], bulletText="•") for it in items]


# ── Document template with title band + footer ────────────────────────────────
class GuideDoc(BaseDocTemplate):
    """A two-frame document: a tall first-page frame that clears the title
    band, and a standard frame for continuation pages. Draws the branded
    header band on page 1 and a footer on every page.
    """

    def __init__(self, path: str, title: str, subtitle: str):
        super().__init__(
            path, pagesize=LETTER,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
            title=title, author="News Signal for Food Insecurity")
        self.doc_title = title
        self.doc_subtitle = subtitle
        self.band_h = 1.55 * inch

        first = Frame(
            MARGIN, MARGIN, CONTENT_W,
            LETTER[1] - MARGIN - self.band_h - 0.35 * inch,
            id="first")
        later = Frame(
            MARGIN, MARGIN, CONTENT_W,
            LETTER[1] - 2 * MARGIN, id="later")
        self.addPageTemplates([
            PageTemplate(id="first", frames=[first], onPage=self._first_page),
            PageTemplate(id="later", frames=[later], onPage=self._later_page),
        ])

    # ── page furniture ──────────────────────────────────────────────────────
    def _band(self, canvas) -> None:
        w, h = LETTER
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - self.band_h, w, self.band_h, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, h - self.band_h, w, 0.10 * inch, fill=1, stroke=0)
        # kicker + title + subtitle
        canvas.setFillColor(CODE_ACCENT)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN, h - 0.52 * inch,
                          "NEWS-SIGNAL-AFRICA  ·  REPOSITORY GUIDE")
        canvas.setFillColor(HexColor("#FFFFFF"))
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawString(MARGIN, h - 0.95 * inch, self.doc_title)
        canvas.setFillColor(HexColor("#CFE2EE"))
        canvas.setFont("Helvetica", 11)
        canvas.drawString(MARGIN, h - 1.26 * inch, self.doc_subtitle)
        canvas.restoreState()

    def _footer(self, canvas) -> None:
        w = LETTER[0]
        canvas.saveState()
        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.8)
        canvas.line(MARGIN, 0.62 * inch, w - MARGIN, 0.62 * inch)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(MARGIN, 0.45 * inch,
                          "News Signal for Food Insecurity")
        canvas.drawRightString(w - MARGIN, 0.45 * inch,
                               f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    def _first_page(self, canvas, doc) -> None:
        self._band(canvas)
        self._footer(canvas)

    def _later_page(self, canvas, doc) -> None:
        self._footer(canvas)


def _h1(text: str) -> list[Flowable]:
    """Section header with an underline rule."""
    return [
        Spacer(1, 6),
        Paragraph(text, STYLES["H1"]),
        Spacer(1, 3),
        HRule(CONTENT_W, color=HAIRLINE, thickness=1.0),
        Spacer(1, 6),
    ]


def _body(text: str) -> Paragraph:
    return Paragraph(text, STYLES["Body"])


def _build(path: Path, title: str, subtitle: str, story: list[Flowable]) -> None:
    """Render ``story`` into a GuideDoc, switching to the continuation
    template after the first frame fills."""
    doc = GuideDoc(str(path), title, subtitle)

    # After the first page, switch page template to the standard 'later' frame.
    flow = [NextPageTemplate("later")] + story
    doc.build(flow)


# ── Document 1: running the plots ─────────────────────────────────────────────
def build_running_the_plots(path: Path) -> None:
    """Write the figure-regeneration guide to ``path``."""
    s: list[Flowable] = []

    s += [_body(
        "The paper figures (Figures 1–7) are written as PDFs into the "
        "<b>figures/</b> folder; the Figure 8 onset/chronic report is written to "
        "<b>results/onset_chronic_analysis_report.pdf</b>. The model results behind "
        "them are already committed to the repository, so you do <b>not</b> need to "
        "retrain any models — you only run the figure scripts.")]
    s += [callout("warning",
        "Do <b>not</b> run <font face='Courier'>04_temporal_shuffle_test.py</font> "
        "to make the figures. That is the 1,000-permutation null test (~24 hours) "
        "and its output is already included in <font face='Courier'>results/</font>.")]
    s += [Spacer(1, 6)]

    s += [step_header(1, "Install Git LFS first")]
    s += [Spacer(1, 8), _body(
        "Three large files (the GADM map layers and the monthly GDELT parquet) "
        "are stored via Git Large File Storage. Install Git LFS <b>before</b> "
        "cloning, or Figure 1 (the choropleth maps) will break because the map "
        "files arrive as tiny pointer stubs.")]
    s += bullets([
        "<b>Windows:</b> download the installer from https://git-lfs.com and run it.",
        "<b>macOS:</b> brew install git-lfs",
        "<b>Linux:</b> sudo apt install git-lfs &nbsp;(or sudo yum install git-lfs)",
    ])
    s += [Spacer(1, 4), _body("Then enable it once per machine:"),
          code_block("git lfs install")]

    s += [Spacer(1, 12), step_header(2, "Clone the repository"), Spacer(1, 8)]
    s += [_body("Git LFS downloads the large files automatically during clone.")]
    s += [code_block(
        "git clone https://github.com/News-Signal-for-Food-Insecurity/news-signal-africa.git\n"
        "cd news-signal-africa")]
    s += [Spacer(1, 6), callout("tip",
        "Already cloned without LFS? You don't need to re-clone — just fetch the "
        "real files in place with <font face='Courier'>git lfs pull</font>.")]

    s += [Spacer(1, 12), step_header(3, "Install Python dependencies"), Spacer(1, 8)]
    s += [_body("Python 3.9 or later is required."),
          code_block("pip install -r requirements.txt")]
    s += [Spacer(1, 6), callout("note",
        "The geospatial packages (geopandas, shapely) are needed for the Figure 1 "
        "maps only and can be awkward to install on Windows. If they fail, every "
        "other figure still generates — Figure 1 is simply skipped.")]

    s += [Spacer(1, 12), step_header(4, "Generate the figures"), Spacer(1, 8)]
    s += [_body("Run the figure scripts from the repository root:")]
    s += [code_block(
        "python 06_paper_figures.py         # Figures 1-6\n"
        "python generate_fig7_new.py        # Figure 7a / 7b panels\n"
        "python 08_onset_chronic_report.py  # Figure 8 (onset vs. chronic)")]
    s += [_body(
        "Figures 1–7 land in the <b>figures/</b> folder. Script 08 writes its main "
        "report to <b>results/onset_chronic_analysis_report.pdf</b> (and its component "
        "panels to figures/).")]

    s += [Spacer(1, 12), step_header(5, "Verify the output"), Spacer(1, 8)]
    s += bullets([
        "Open the <b>figures/</b> folder (and <b>results/</b> for the Figure 8 "
        "report) and confirm the PDFs were updated.",
        "If Figure 1 is blank or missing, geopandas/shapely is not installed, or "
        "Git LFS did not fetch the .gpkg map files (re-run <font face='Courier'>git lfs pull</font>).",
        "Every other figure should render without the geospatial packages.",
    ])

    s += _h1("Troubleshooting")
    s += bullets([
        "<b>ModuleNotFoundError</b> — re-run <font face='Courier'>pip install -r requirements.txt</font> "
        "in the active environment.",
        "<b>Figure 1 empty / map files tiny</b> — Git LFS issue; run "
        "<font face='Courier'>git lfs install</font> then <font face='Courier'>git lfs pull</font>.",
        "<b>FileNotFoundError on results/</b> — run from the repository root, not "
        "from inside a subfolder.",
    ])

    _build(path, "Running the Plots",
           "Step-by-step guide to regenerating every paper figure", s)


# ── Document 2: accessing and using the data ──────────────────────────────────
def build_accessing_the_data(path: Path) -> None:
    """Write the data-access guide to ``path``."""
    s: list[Flowable] = []

    s += [_body(
        "All data is committed directly into the repository — there is no external "
        "download or database. You obtain it by cloning, and use it by reading the "
        "files with pandas (and geopandas for the maps).")]
    s += [Spacer(1, 6)]

    s += [step_header(1, "Install Git LFS, then clone"), Spacer(1, 8)]
    s += [_body("Three large files are stored via Git LFS and must be fetched "
                "with LFS installed first:")]
    s += bullets([
        "<font face='Courier'>DATA/raw/ml_dataset_monthly.parquet</font> &nbsp;(~35 MB)",
        "<font face='Courier'>DATA/shapefiles/gadm/africa_adm2_combined.gpkg</font> &nbsp;(~62 MB)",
        "<font face='Courier'>DATA/shapefiles/gadm/africa_adm0_basemap.gpkg</font> &nbsp;(~37 MB)",
    ])
    s += [code_block(
        "git lfs install\n"
        "git clone https://github.com/News-Signal-for-Food-Insecurity/news-signal-africa.git\n"
        "cd news-signal-africa")]
    s += [Spacer(1, 6), callout("tip",
        "Already cloned without LFS? Fetch the real files in place with "
        "<font face='Courier'>git lfs pull</font>.")]

    s += [Spacer(1, 12), step_header(2, "Know the file formats"), Spacer(1, 8)]
    s += bullets([
        "<b>DATA/dataset.parquet</b> — the main modelling table (Parquet, pandas).",
        "<b>DATA/raw/*.parquet, DATA/modelling/*.parquet</b> — source &amp; feature "
        "tables (Parquet).",
        "<b>DATA/filtering/*.csv, results/**/*.csv</b> — CSV.",
        "<b>DATA/*.json, results/**/*.json</b> — JSON metadata &amp; metrics.",
        "<b>DATA/shapefiles/gadm/*.gpkg</b> — GeoPackage (geopandas).",
    ])

    s += [Spacer(1, 12), step_header(3, "Load the main dataset"), Spacer(1, 8)]
    s += [_body(
        "<b>DATA/dataset.parquet</b> is the table every training script reads: "
        "3,245 rows covering 353 districts across 18 countries, Oct 2020 – Feb 2024. "
        "Each row is one district-period observation.")]
    s += [code_block(
        "import pandas as pd\n\n"
        "df = pd.read_parquet(\"DATA/dataset.parquet\")\n"
        "print(df.shape)             # (3245, ...)\n"
        "print(df.columns.tolist())\n"
        "print(df.head())")]
    s += [Paragraph("Key columns", STYLES["H2"])]
    s += bullets([
        "<b>Target:</b> <font face='Courier'>target_crisis_binary</font> — IPC Phase "
        "≥ 3 crisis 8 months ahead (about 28% positive).",
        "<b>AR features (5):</b> ipc_lag_1, ipc_persistence_2yr, spatial_lag, "
        "ipc_period, ipc_country.",
        "<b>News features (18):</b> {theme}_relative_coverage and {theme}_zscore for "
        "9 themes (conflict, displacement, economic, food_security, governance, "
        "health, humanitarian, weather, other).",
    ])

    s += _h1("Leakage caveat — read before reusing the z-scores")
    s += [_body(
        "The <font face='Courier'>{theme}_zscore</font> columns in dataset.parquet are "
        "<b>fold-aware</b>: they were standardised within each training window for the "
        "paper's CV split. If you build your own model with a different split, do "
        "<b>not</b> reuse these z-scores directly — recompute them from the raw monthly "
        "counts to avoid leakage:")]
    s += [code_block(
        "import pandas as pd\n\n"
        "monthly = pd.read_parquet(\"DATA/modelling/monthly_gdelt_features.parquet\")\n"
        "# standardise within your own training folds only")]

    s += [Spacer(1, 12), step_header(4, "Load the map layers (optional)"),
          Spacer(1, 8)]
    s += [_body("Join the GADM boundaries to dataset.parquet on the district/GADM "
                "identifier to build choropleths.")]
    s += [code_block(
        "import geopandas as gpd\n\n"
        "districts = gpd.read_file(\"DATA/shapefiles/gadm/africa_adm2_combined.gpkg\")\n"
        "countries = gpd.read_file(\"DATA/shapefiles/gadm/africa_adm0_basemap.gpkg\")")]

    s += [Spacer(1, 12), step_header(5, "Use the pre-computed results"),
          Spacer(1, 8)]
    s += [_body(
        "<b>results/</b> holds the committed model outputs as plain CSV/JSON — fold "
        "metrics, predictions, feature importance, and the null distribution — so you "
        "can analyse the findings without retraining anything.")]
    s += [code_block(
        "import pandas as pd, json\n\n"
        "folds      = pd.read_csv(\"results/window_2yr/fold_results.csv\")\n"
        "importance = pd.read_csv(\"results/window_2yr/feature_importance.csv\")\n"
        "summary    = json.load(open(\"results/window_2yr/metrics_summary.json\"))")]

    s += _h1("Reuse &amp; licensing")
    s += [_body(
        "Code is released under the <b>MIT License</b> (see the LICENSE file). The "
        "data is subject to <b>FEWSNET and GDELT terms of use</b>: you may freely read "
        "and analyse it, but if you redistribute it you must honour those upstream "
        "terms and cite the sources — FEWSNET IPC, GDELT GKG, and GADM v4.1.")]

    _build(path, "Accessing & Using the Data",
           "Step-by-step guide to obtaining and loading the repository data", s)


def main() -> None:
    """Generate both guide PDFs into this documents/ folder."""
    out_dir = Path(__file__).parent
    running = out_dir / "running-the-plots.pdf"
    data = out_dir / "accessing-the-data.pdf"
    build_running_the_plots(running)
    build_accessing_the_data(data)
    print(f"Wrote {running}")
    print(f"Wrote {data}")


if __name__ == "__main__":
    main()
