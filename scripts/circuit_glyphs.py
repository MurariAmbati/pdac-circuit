from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

BACKBONE_LW = 2.6


def backbone(ax, x0, x1, y, color, lw=BACKBONE_LW):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=1)


def promoter(ax, x, y, color, w=0.62, h=0.46, label=None, sub=None, ink="#272727"):
    ax.plot([x, x], [y, y + h], color=color, lw=2.4, solid_capstyle="round", zorder=3)
    ax.add_patch(FancyArrowPatch((x, y + h), (x + w, y + h), arrowstyle="-|>",
                                 mutation_scale=13, lw=2.4, color=color, zorder=3))
    if label:
        ax.text(x + w / 2, y + h + 0.20, label, ha="center", va="bottom",
                fontsize=10.5, color=ink, weight="bold")
    if sub:
        ax.text(x + w / 2, y - 0.24, sub, ha="center", va="top", fontsize=9, color="#767676")
    return x + w


def cds(ax, x, y, color, w=1.55, h=0.5, label=None, sub=None, ink="#272727", tip=0.28):
    pts = [(x, y - h / 2), (x + w - tip, y - h / 2), (x + w, y),
           (x + w - tip, y + h / 2), (x, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="#272727",
                         lw=1.3, zorder=3))
    if label:
        ax.text(x + (w - tip) / 2, y + 0.02, label, ha="center", va="center",
                fontsize=10.5, color="white", weight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y - h / 2 - 0.22, sub, ha="center", va="top",
                fontsize=9, color="#767676")
    return x + w


def terminator(ax, x, y, color, w=0.34, h=0.42, label=None, ink="#272727"):
    ax.plot([x + w / 2, x + w / 2], [y, y + h], color=color, lw=2.4, solid_capstyle="round", zorder=3)
    ax.plot([x, x + w], [y + h, y + h], color=color, lw=2.4, solid_capstyle="round", zorder=3)
    if label:
        ax.text(x + w / 2, y + h + 0.18, label, ha="center", va="bottom", fontsize=9.5, color=ink)
    return x + w


def cassette_part(ax, x, y, color, w, h=0.44, label=None, sub=None):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=color, edgecolor="#272727",
                           lw=1.2, zorder=3, joinstyle="round"))
    if label:
        ax.text(x + w / 2, y + 0.02, label, ha="center", va="center", fontsize=10,
                color="white", weight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y - h / 2 - 0.22, sub, ha="center", va="top",
                fontsize=9, color="#767676")
    return x + w


def dna_duplex(ax, x0, x1, y, color, gap=0.075, lw=2.0):
    ax.plot([x0, x1], [y + gap, y + gap], color=color, lw=lw, solid_capstyle="butt", zorder=1)
    ax.plot([x0, x1], [y - gap, y - gap], color=color, lw=lw, solid_capstyle="butt", zorder=1)


def dcas9(ax, cx, cy, color, edge="#272727", w=1.02, h=0.66):
    th = np.linspace(0.32 * np.pi, 1.68 * np.pi, 200)
    ax.add_patch(Polygon(np.column_stack([cx + w / 2 * np.cos(th), cy + h / 2 * np.sin(th)]),
                         closed=True, facecolor=color, edgecolor=edge, lw=1.5, zorder=5))


def guide_rna(ax, x0, y0, x1, y1, color, lw=2.0):
    t = np.linspace(0, 1, 120)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t + 0.10 * np.sin(np.pi * t * 3)
    ax.plot(xs, ys, color=color, lw=lw, zorder=6, solid_capstyle="round")


def repression(ax, x, y_from, y_to, color, bar=0.20, lw=2.2):
    ax.plot([x, x], [y_from, y_to], color=color, lw=lw, solid_capstyle="butt", zorder=6)
    ax.plot([x - bar, x + bar], [y_to, y_to], color=color, lw=lw + 0.6,
            solid_capstyle="round", zorder=6)


def curved(ax, p0, p1, color, rad=-0.36, ls="--", lw=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, linestyle=ls, zorder=4))
