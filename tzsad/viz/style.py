"""Publication figure style: vector output, column-width legible, colorblind-safe."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Okabe-Ito, the standard colorblind-safe qualitative palette.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000")

#: Reserved semantics used consistently across every figure in the paper.
SEMANTIC = {
    "baseline": "#999999",
    "ours": "#0072B2",
    "oracle": "#000000",
    "random": "#BBBBBB",
    "danger": "#D55E00",
    "good": "#009E73",
}


def use_paper_style(base_font: float = 9.0) -> None:
    """Apply the paper's rcParams. Figures are sized for a single column."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "legend.fontsize": base_font - 1,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
    })


def save_figure(fig, path: str | Path, also_png: bool = True) -> list[Path]:
    """Save as vector PDF (and optionally PNG for slide decks)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path.with_suffix(".pdf")]
    fig.savefig(written[0])
    if also_png:
        written.append(path.with_suffix(".png"))
        fig.savefig(written[1])
    plt.close(fig)
    return written
