"""Shared figure style: colours validated for colour-vision deficiency, thin marks,
recessive axes, one y-axis per panel."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import OUT_FIG

# Cohort colours (categorical slots 1-2; pass CVD and contrast checks on white)
REC = "#2a78d6"   # Reception
YR6 = "#eb6834"   # Year 6
COHORT_COLOUR = {"Reception": REC, "Year 6": YR6}
AQUA = "#1baf7a"
CATEGORY_COLOUR = {  # fixed categorical order, adjacent pairs validated
    "underweight": "#2a78d6", "healthy": "#eb6834", "overweight": "#1baf7a",
    "obese": "#eda100", "severe": "#e87ba4",
}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

# Diverging blue <-> red with a neutral grey midpoint
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#b7d3f6", NEUTRAL, "#f2b8b5", "#e34948", "#9e2423"])
DIVERGING_R = DIVERGING.reversed()

MONTH_ABBR = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],  # Inter if installed
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 8,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "text.color": INK,
})

DOUBLE_COL_IN = 7.2   # ~183 mm
SINGLE_COL_IN = 3.5   # ~89 mm


def panel_label(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK,
            va="bottom", ha="left")


def titled(ax, letter, text):
    """Panel title with its letter, left-aligned (avoids letters colliding with titles)."""
    ax.set_title(f"{letter}   {text}" if letter else text, loc="left", fontsize=8.5, fontweight="semibold",
                 color=INK, pad=6)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)
