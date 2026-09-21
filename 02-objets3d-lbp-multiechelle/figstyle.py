"""Style commun des figures : palette validee, chrome discret, fond opaque.

Identique a parking-old/04-parking-lbp-multiechelle/figstyle.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # bleu, orange, aqua
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95")

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_SECOND,
    "axes.edgecolor": AXIS,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECOND,
    "ytick.labelcolor": INK_SECOND,
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def clean(ax, grid_axis="y"):
    """Retire les bordures superflues et ne garde qu'une grille discrete."""
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.grid(False)
    ax.grid(True, axis=grid_axis)
    ax.tick_params(length=0)
    return ax
