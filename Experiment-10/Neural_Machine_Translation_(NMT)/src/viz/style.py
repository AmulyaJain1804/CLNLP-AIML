"""One plotting style for every figure in the project (consistent palette + size)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots, fixed order (validated CVD-safe reference palette). Colour follows
# the entity: a model / language always gets the same colour in every figure.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"baseline": SLOTS[0], "backtranslation": SLOTS[1], "nllb_finetune": SLOTS[2]}
MODEL_LABELS = {"baseline": "Transformer baseline", "backtranslation": "+ Back-translation",
                "nllb_finetune": "NLLB-200 fine-tuned"}
SIDE_COLORS = {"en": SLOTS[0], "x": SLOTS[1]}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

FIGSIZE = (8, 4.5)
FIGSIZE_WIDE = (12, 4.5)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.figsize": FIGSIZE, "figure.dpi": 110, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.bbox": "tight",
    "font.family": ["DejaVu Sans", "Nirmala UI", "sans-serif"],
})


def save(fig, path: str | Path) -> None:
    """Save as both PNG and PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
