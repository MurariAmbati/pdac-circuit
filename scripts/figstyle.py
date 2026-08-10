from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "grey_mid": "#767676",
    "grey_dark": "#4D4D4D",
    "ink": "#272727",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

DEFAULT_ORDER = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                 PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"]]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    grid: bool = False
    font_family: tuple[str, ...] = field(
        default=("DejaVu Sans", "Helvetica", "Arial", "sans-serif"))


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    style = style or FigureStyle()
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.titlesize": style.font_size + 1,
        "axes.labelsize": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": style.grid,
        "text.color": PALETTE["ink"],
        "xtick.labelsize": style.font_size - 2,
        "ytick.labelsize": style.font_size - 2,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
        "legend.fontsize": style.font_size - 2,
        "grid.color": "#E6E8EC",
        "grid.linewidth": 1.0,
        "lines.linewidth": 2.5,
        "patch.linewidth": 1.5,
    })
    return style


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]
    return fig, axes


def annotate_bars(ax, bars, values=None, fmt="{:.3f}", fontsize=13, padding=3, weight="bold"):
    values = values if values is not None else [b.get_height() for b in bars]
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, padding),
                    ha="center", va="bottom", fontsize=fontsize, fontweight=weight,
                    color=PALETTE["ink"])


def tighten_ylim(ax, values, frac=0.45, floor=None):
    lo, hi = min(values), max(values)
    span = (hi - lo) or (abs(hi) or 1.0)
    low = lo - span * frac
    if floor is not None:
        low = max(floor, low)
    ax.set_ylim(low, hi + span * frac)


def ygrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E8EC", linewidth=1.0)
    ax.xaxis.grid(False)


def finalize_figure(fig, out_dir, name, formats=("png", "pdf"), dpi=300, pad=2.0, close=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pad is not None:
        fig.tight_layout(pad=pad)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi)
    if close:
        plt.close(fig)
    return f"{name}." + "/".join(formats)
