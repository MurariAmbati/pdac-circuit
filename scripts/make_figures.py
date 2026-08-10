from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import matplotlib.pyplot as plt
from figstyle import (
    PALETTE,
    FigureStyle,
    annotate_bars,
    apply_publication_style,
    finalize_figure,
    tighten_ylim,
    ygrid,
)

SRC = Path("C:/Users/murar/pdac-circuit")
OUT = Path(__file__).resolve().parents[1] / "images"

INK = PALETTE["ink"]
BLUE = PALETTE["blue_main"]
BLUE2 = PALETTE["blue_secondary"]
GREEN = PALETTE["green_3"]
RED = PALETTE["red_strong"]
NEUT = PALETTE["neutral"]
GREY = PALETTE["grey_mid"]
TEAL = PALETTE["teal"]
VIOLET = PALETTE["violet"]

apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.2))


def load(rel):
    p = SRC / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name, pad=1.2):
    print("  " + finalize_figure(fig, OUT, name, pad=pad))


def box(ax, x, y, w, h, title, sub=None, detail=None, fill="#FFFFFF",
        edge=None, ts=12.5, ss=10.2, ds=9.2, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                linewidth=lw, edgecolor=edge or INK,
                                facecolor=fill, zorder=3))
    n = 1 + (sub is not None) + (detail is not None)
    ys = {1: [0.50], 2: [0.62, 0.32], 3: [0.72, 0.47, 0.22]}[n]
    ax.text(x + w / 2, y + h * ys[0], title, ha="center", va="center",
            fontsize=ts, fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * ys[1], sub, ha="center", va="center",
                fontsize=ss, color=PALETTE["grey_dark"], zorder=4)
    if detail:
        ax.text(x + w / 2, y + h * ys[-1], detail, ha="center", va="center",
                fontsize=ds, color=GREY, zorder=4, style="italic")


def arrow(ax, p0, p1, color=None, lw=2.0, ls="-", rad=0.0, style="-|>", ms=13):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color or INK, zorder=2,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def blunt(ax, tip, angle_deg=0, size=0.020, lw=3.0, color=None):
    import math
    a = math.radians(angle_deg)
    px, py = -math.sin(a), math.cos(a)
    ax.plot([tip[0] + px * size, tip[0] - px * size],
            [tip[1] + py * size, tip[1] - py * size],
            color=color or INK, linewidth=lw, solid_capstyle="butt", zorder=5)


def fig_scaleup():
    g = load("results/grna_cnn_kim_retrain.json")
    p = load("results/promoter_scaleup.json")
    e = load("results/enhancer_scaleup.json")
    a = load("results/promoter_gan_scaleup.json")
    panels = [
        ("gRNA on-target", "Spearman", g["shipped_ensemble"], g["deployed_ensemble"], "5,310", "18,142"),
        ("Promoter strength", "Spearman", p["baseline_shipped_ensemble"], p["scaleup_ensemble"], "60,000", "181,428"),
        ("Enhancer activity", "AUROC", e["baseline_shipped_auroc"], e["scaleup_auroc"], "20,000", "135,402"),
        ("Promoter generator", "p90 strength", a["baseline_p90"], a["scaleup_p90"], "12,000", "52,342"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), gridspec_kw={"wspace": 0.40})
    for ax, (title, metric, b, af, nb, na) in zip(axes, panels):
        bars = ax.bar([0, 1], [b, af], width=0.60, color=[NEUT, BLUE],
                      edgecolor="black", linewidth=2.0, hatch=["//", ""])
        tighten_ylim(ax, [b, af], frac=0.50, floor=0.0)
        annotate_bars(ax, bars, [b, af], fmt="{:.3f}", fontsize=12.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"capped\nn = {nb}", f"full\nn = {na}"], fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=13.5, pad=10)
        ax.set_ylabel(metric, fontsize=12.5)
        ygrid(ax)
        d = af - b
        ax.annotate(f"{d:+.4f}", (0.5, 0.97), xycoords="axes fraction", ha="center",
                    va="top", fontsize=12.5, fontweight="bold", color=PALETTE["grey_dark"])
    fig.suptitle("Held-out performance before and after removing the training-data caps",
                 fontsize=15.5, fontweight="bold", y=1.04)
    save(fig, "fig1_scaleup")


def fig_curves():
    pc = load("results/promoter_scaling_curve.json")
    ec = load("results/enhancer_scaling_curve.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"wspace": 0.24})

    ax = axes[0]
    n = [q["n_train"] for q in pc["points"]]
    ax.plot(n, [q["ensemble"] for q in pc["points"]], "-o", color=BLUE, lw=2.8, ms=8,
            mec="black", mew=1.6, label="ensemble", zorder=4)
    ax.plot(n, [q["cnn"] for q in pc["points"]], "--s", color=TEAL, lw=2.2, ms=7,
            mec="black", mew=1.4, label="sequence CNN", zorder=3)
    ax.plot(n, [q["rf"] for q in pc["points"]], ":^", color=RED, lw=2.2, ms=7,
            mec="black", mew=1.4, label="k-mer tree", zorder=2)
    ax.set_xlabel("training peaks")
    ax.set_ylabel("Spearman")
    ax.set_title("Promoter strength", fontweight="bold", fontsize=13.5)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(0.455, 0.552)
    ygrid(ax)

    ax = axes[1]
    n2 = [q["n_train"] for q in ec["points"]]
    ax.plot(n2, [q["auroc"] for q in ec["points"]], "-o", color=BLUE, lw=2.8, ms=8,
            mec="black", mew=1.6, zorder=3)
    ax.set_xlabel("training rows")
    ax.set_ylabel("AUROC")
    ax.set_title("Enhancer activity", fontweight="bold", fontsize=13.5)
    ax.set_ylim(0.7975, 0.8185)
    ygrid(ax)

    for ax in axes:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    fig.suptitle("Held-out performance against training-set size, fixed chr8 and chr9 test",
                 fontsize=15.5, fontweight="bold", y=1.03)
    save(fig, "fig2_scaling_curves")


def fig_grna_components():
    g = load("results/grna_cnn_kim_retrain.json")
    labels = ["sequence CNN", "gradient-boosted tree", "deployed ensemble"]
    before = [g["cnn_doench_only_prev"], 0.5250, g["shipped_ensemble"]]
    after = [g["cnn_doench_plus_kim"], g["gbm_doench_plus_kim"], g["deployed_ensemble"]]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    b1 = ax.bar(x - w / 2, before, w, label="Doench-2016 only, 17 genes",
                color=NEUT, edgecolor="black", linewidth=2.0, hatch="//")
    b2 = ax.bar(x + w / 2, after, w, label="Doench-2016 + Kim-2019, 18,142 guides",
                color=BLUE, edgecolor="black", linewidth=2.0)
    annotate_bars(ax, b1, before, fontsize=11.5, weight="normal")
    annotate_bars(ax, b2, after, fontsize=11.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel("Spearman on 688 held-out-gene guides")
    ax.set_ylim(0, 0.82)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("The convolutional component was the binding constraint",
                 fontweight="bold", fontsize=14, pad=12)
    ygrid(ax)
    save(fig, "fig3_grna_components")


def fig_h3k27ac():
    k = load("results/pdac_residual_foldchange_H3K27ac.json")
    tv = k["targets_vs_all_background"]
    items = sorted(k["per_target"].items(), key=lambda kv: kv[1]["log2_residual"])
    names = [g for g, _ in items]
    vals = [v["log2_residual"] for _, v in items]
    cols = [GREEN if v > 0 else PALETTE["red_2"] for v in vals]
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    bars = ax.barh(names, vals, color=cols, edgecolor="black", linewidth=1.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:+.2f}", (v, b.get_y() + b.get_height() / 2),
                    xytext=(7 if v > 0 else -7, 0), textcoords="offset points",
                    ha="left" if v > 0 else "right", va="center", fontsize=10.5,
                    color=PALETTE["grey_dark"])
    ax.axvline(0, color=INK, lw=2.2, zorder=4)
    ax.axvline(tv["background_mean_log2"], color=GREY, ls="--", lw=2.2, zorder=4,
               label=f"background mean {tv['background_mean_log2']:+.3f}  (n = {tv['n_background']:,})")
    ax.axvline(tv["target_mean_log2"], color=BLUE, ls="-", lw=2.6, zorder=4,
               label=f"target mean {tv['target_mean_log2']:+.3f}  (n = {tv['n_targets']})")
    ax.set_xlabel("log2 H3K27ac fold-change residual, PDAC against healthy pancreas")
    ax.set_title("Promoter H3K27ac at the prioritised targets",
                 fontweight="bold", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=10.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E8EC", linewidth=1.0)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlim(-3.2, 7.0)
    ax.annotate(f"Mann-Whitney one-sided p = {tv['mannwhitney_p_greater']:.4f}\n"
                f"{tv['target_frac_up']:.0%} of targets up against "
                f"{tv['background_frac_up']:.0%} of background",
                (0.985, 0.055), xycoords="axes fraction", ha="right", fontsize=11, color=INK)
    save(fig, "fig4_h3k27ac")


def fig_rac():
    r = load("results/rigorous_validation.json")["A_rac_vs_degree"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                             gridspec_kw={"width_ratios": [1.12, 1], "wspace": 0.26})
    ax = axes[0]
    names = ["attractor\ncollapse", "eigenvector\ncentrality", "network\ndegree"]
    vals = [r["auc_rac"], r["auc_eigenvector"], r["auc_degree"]]
    bars = ax.bar(names, vals, width=0.58, color=[RED, NEUT, BLUE],
                  edgecolor="black", linewidth=2.0)
    annotate_bars(ax, bars, vals, fontsize=12.5)
    ax.axhline(0.5, color=GREY, ls=":", lw=2.2)
    ax.annotate("chance", (-0.44, 0.503), fontsize=10.5, color=GREY, va="bottom")
    ax.set_ylim(0.45, 0.685)
    ax.set_ylabel("AUC, held-out essentiality")
    ax.set_title("The proposed score loses to network degree",
                 fontweight="bold", fontsize=13.5, pad=10)
    ygrid(ax)

    ax = axes[1]
    d = r["delta_auc_rac_minus_degree"]
    lo, hi = r["delta_auc_ci95_paired_bootstrap"]
    ax.errorbar([d], [0], xerr=[[d - lo], [hi - d]], fmt="o", color=RED, ms=13,
                mec="black", mew=1.8, capsize=9, elinewidth=2.8, capthick=2.8, zorder=3)
    ax.axvline(0, color=INK, ls="--", lw=2.2)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.27, 0.11)
    ax.set_xlabel(r"$\Delta$AUC   (collapse $-$ degree)")
    ax.set_title("The interval on the difference spans zero",
                 fontweight="bold", fontsize=13.5, pad=10)
    ax.annotate(f"{d:+.4f}   95% CI [{lo:+.3f}, {hi:+.3f}]\n"
                f"p = {r['delta_auc_p_two_sided']:.3f}   "
                f"n = {r['n_genes']} genes, {r['n_positive']} positives",
                (0.5, 0.26), xycoords="axes fraction", ha="center", fontsize=11.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E8EC", linewidth=1.0)
    fig.suptitle("Adversarial validation of the attractor-collapse claim",
                 fontsize=15.5, fontweight="bold", y=1.04)
    save(fig, "fig5_rac_validation")


def fig_transfer():
    p = load("results/enhancer_panc1_augment.json")
    ec = load("results/enhancer_scaling_curve.json")
    labels = ["pancreas to pancreas\nwithin domain", "pancreas to PANC-1\nforward transfer",
              "PANC-1 to pancreas\nreverse transfer"]
    vals = [p["pancreas_only_pancreas_test"], p["pancreas_only_panc1_test_xdomain"],
            ec["reverse_xdomain_panc1_to_pancreas"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, vals, width=0.56, color=[NEUT, BLUE, TEAL],
                  edgecolor="black", linewidth=2.0, hatch=["//", "", ".."])
    annotate_bars(ax, bars, vals, fontsize=12.5)
    ax.axhline(0.5, color=GREY, ls=":", lw=2.2)
    ax.annotate("chance", (-0.46, 0.508), fontsize=10.5, color=GREY, va="bottom")
    ax.set_ylim(0.48, 0.90)
    ax.set_ylabel("AUROC")
    ax.set_title("Enhancer grammar transfers in both directions",
                 fontweight="bold", fontsize=14, pad=12)
    ax.tick_params(axis="x", labelsize=11)
    ygrid(ax)
    save(fig, "fig6_cross_domain")


def fig_evidence_heatmap():
    d = load("results/rac_target_dossiers.json")
    layers = [
        ("disease_log2fc", "is_it_real", "disease log2FC"),
        ("depmap_essentiality", "is_it_real", "DepMap essentiality"),
        ("depmap_pdac_selectivity", "is_it_real", "PDAC selectivity"),
        ("cna_amplification_freq", "is_it_real", "CNA amplification"),
        ("cna_deletion_freq", "is_it_real", "CNA deletion"),
        ("protein_mean", "is_it_real", "protein abundance"),
        ("protein_detection_rate", "is_it_real", "protein detection"),
        ("h3k27ac_disease_residual_log2", "is_it_active", "H3K27ac residual"),
        ("hic_compartment_eigenvector", "is_it_active", "Hi-C compartment"),
    ]
    genes = [x["gene"] for x in d["dossiers"]]
    raw = np.full((len(genes), len(layers)), np.nan)
    for i, dos in enumerate(d["dossiers"]):
        for j, (key, grp, _) in enumerate(layers):
            v = dos.get(grp, {}).get(key)
            if isinstance(v, (int, float)):
                raw[i, j] = float(v)
    z = np.full_like(raw, np.nan)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        ok = ~np.isnan(col)
        if ok.sum() > 1 and np.nanstd(col) > 0:
            z[ok, j] = (col[ok] - np.nanmean(col)) / np.nanstd(col)
        elif ok.sum():
            z[ok, j] = 0.0
    z = np.clip(z, -2.0, 2.0)

    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    for i in range(len(genes)):
        for j in range(len(layers)):
            if np.isnan(z[i, j]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#EDEFF3",
                                       edgecolor="white", hatch="///", linewidth=0))
            else:
                ax.annotate(f"{raw[i, j]:.2f}", (j, i), ha="center", va="center",
                            fontsize=8.0,
                            color="white" if abs(z[i, j]) > 1.15 else PALETTE["grey_dark"])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([lay[2] for lay in layers], rotation=32, ha="right", fontsize=10.5)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=11)
    ax.set_xticks(np.arange(-.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(genes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
    cb.set_label("z-score within layer", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)
    ax.set_title("Multi-omic evidence per prioritised target, cells show the raw measurement",
                 fontweight="bold", fontsize=13.5, pad=14)
    ax.annotate("hatched cells were not measured in that layer",
                (1.0, -0.26), xycoords="axes fraction", ha="right", fontsize=10, color=GREY)
    save(fig, "fig7_evidence_heatmap")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14.6, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    data_specs = [
        ("TCGA-PAAD, GTEx", "expression, CNA"),
        ("FANTOM5", "209,374 peaks"),
        ("ENCODE pancreas", "470,874 actives"),
        ("Doench, Kim", "18,142 guides"),
        ("DepMap", "1,684 cell lines"),
    ]
    dw, dx0, dgap = 0.142, 0.020, 0.053
    for i, (lab, role) in enumerate(data_specs):
        x = dx0 + i * (dw + dgap)
        box(ax, x, 0.845, dw, 0.110, lab, role, fill="#F2F5FA",
            edge=PALETTE["grey_dark"], ts=10.8, ss=9.4, lw=1.6)

    mods = [
        ("I", "Target\nprioritisation", "MCDA over five layers", "1,639 TFs screened"),
        ("II", "Regulatory\nparts", "promoter r = 0.528", "enhancer AUROC 0.815"),
        ("V", "Guide\ndesign", "on-target r = 0.657", "CFD, genome-wide"),
        ("III / IV", "Circuit and\nsequence", "Hill ODE per circuit", "3,003 simulated"),
        ("VI", "Multi-objective\nscoring", "NSGA-II Pareto", "four objectives"),
    ]
    mw, mgap, mx0 = 0.142, 0.053, 0.020
    my, mh = 0.455, 0.225
    xs = [mx0 + i * (mw + mgap) for i in range(5)]
    for x, (num, name, sub, det) in zip(xs, mods):
        box(ax, x, my, mw, mh, name, sub, det, ts=12.2, ss=10.0, ds=9.0)
        ax.annotate(num, (x + 0.008, my + mh - 0.026), fontsize=9.5,
                    color=BLUE, fontweight="bold")
    edge_labels = ["targets", "parts", "guides", "circuits"]
    for i in range(4):
        arrow(ax, (xs[i] + mw, my + mh / 2), (xs[i + 1], my + mh / 2), lw=2.2)
        ax.annotate(edge_labels[i], ((xs[i] + mw + xs[i + 1]) / 2, my + mh / 2 + 0.024),
                    ha="center", fontsize=8.4, color=GREY, style="italic")
    for x in xs:
        arrow(ax, (x + mw / 2, 0.845), (x + mw / 2, my + mh), lw=1.8,
              color=PALETTE["grey_dark"])

    box(ax, xs[0], 0.175, mw, 0.155, "VIII  Attractor", "collapse score",
        "retracted, AUC 0.547", fill="#FBEEEE", edge=RED, ts=11.4, ss=9.6, ds=8.8)
    arrow(ax, (xs[0] + mw / 2, 0.330), (xs[0] + mw / 2, my), ls=(0, (4, 3)), color=RED, lw=2.0)
    box(ax, xs[1], 0.175, mw, 0.155, "VII  Generator", "WGAN-GP, 52,342",
        "4-mer JS 0.012", fill="#EFF6EF", edge=PALETTE["green_3"], ts=11.4, ss=9.6, ds=8.8)
    arrow(ax, (xs[1] + mw / 2, 0.330), (xs[1] + mw / 2, my), color=PALETTE["grey_dark"], lw=2.0)

    gx = xs[4] + mw / 2
    box(ax, xs[4] - 0.012, 0.175, mw + 0.024, 0.155, "Specificity gate",
        "genome-wide CFD", "0 of 4 guides clear", fill="#FBEEEE", edge=RED,
        ts=11.4, ss=9.6, ds=8.8)
    arrow(ax, (gx, my), (gx, 0.330), color=RED, lw=2.2)
    ax.annotate("certified\nnegative", (xs[4] + mw / 2, 0.140), fontsize=10.6,
                color=RED, fontweight="bold", ha="center", va="center")

    ax.annotate("Real public data, every corpus sha256-manifested",
                (0.028, 0.980), fontsize=11, color=PALETTE["grey_dark"], fontweight="bold")
    ax.annotate("Design path", (0.028, 0.702), fontsize=11,
                color=PALETTE["grey_dark"], fontweight="bold")
    ax.annotate("Auxiliary modules, and the gate that halts the run",
                (0.020, 0.140), fontsize=11, color=PALETTE["grey_dark"], fontweight="bold", va="center")
    ax.annotate("Solid arrows carry data forward and are labelled with what passes between stages. "
                "The dashed red edge marks the module whose\npredictive claim was withdrawn. Any stage "
                "lacking the evidence to proceed returns a certified negative rather than a default value.",
                (0.028, 0.085), fontsize=10.4, color=GREY, va="top")
    save(fig, "fig8_architecture", pad=0.8)


def fig_circuit():
    fig = plt.figure(figsize=(14.4, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.18], hspace=0.14)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate("a", (0.000, 0.93), fontsize=17, fontweight="bold", color=INK)
    ax.annotate("Delivered construct, two cassettes", (0.026, 0.93), fontsize=13,
                fontweight="bold", color=INK)

    for name, sub, x, w, col, tc in [
            ("synthetic promoter", "1,024 bp, generated", 0.028, 0.240, PALETTE["green_2"], INK),
            ("enhancer", "activity 0.815", 0.278, 0.120, PALETTE["green_1"], INK),
            ("dCas9-KRAB", "effector CDS", 0.408, 0.220, BLUE2, "white"),
            ("pA", "", 0.638, 0.070, NEUT, INK)]:
        ax.add_patch(Rectangle((x, 0.520), w, 0.170, facecolor=col,
                               edgecolor="black", linewidth=1.8, zorder=3))
        ax.annotate(name, (x + w / 2, 0.632), ha="center", va="center",
                    fontsize=11.2, fontweight="bold", color=tc, zorder=4)
        if sub:
            ax.annotate(sub, (x + w / 2, 0.567), ha="center", va="center",
                        fontsize=8.8, color="#E7EDF6" if tc == "white" else PALETTE["grey_dark"],
                        zorder=4)
    for name, sub, x, w, col, tc in [
            ("U6", "pol III", 0.028, 0.090, NEUT, INK),
            ("sgRNA spacer", "20 nt, on-target 0.657", 0.128, 0.240, VIOLET, "white"),
            ("scaffold", "", 0.378, 0.125, NEUT, INK)]:
        ax.add_patch(Rectangle((x, 0.195), w, 0.170, facecolor=col,
                               edgecolor="black", linewidth=1.8, zorder=3))
        ax.annotate(name, (x + w / 2, 0.307), ha="center", va="center",
                    fontsize=11.2, fontweight="bold", color=tc, zorder=4)
        if sub:
            ax.annotate(sub, (x + w / 2, 0.242), ha="center", va="center",
                        fontsize=8.8, color="#F3E8F2" if tc == "white" else PALETTE["grey_dark"],
                        zorder=4)
    ax.plot([0.028, 0.708], [0.605, 0.605], color=INK, lw=1.1, zorder=1)
    ax.plot([0.028, 0.503], [0.280, 0.280], color=INK, lw=1.1, zorder=1)
    ax.annotate("effector cassette", (0.726, 0.605), fontsize=10, va="center", color=GREY)
    ax.annotate("guide cassette", (0.521, 0.280), fontsize=10, va="center", color=GREY)
    ax.annotate("Sequence-optimised for GC content, cryptic splice sites, restriction sites "
                "and codon adaptation",
                (0.028, 0.060), fontsize=10.4, color=GREY)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.annotate("b", (0.000, 0.945), fontsize=17, fontweight="bold", color=INK)
    ax2.annotate("Regulatory action and kinetics", (0.026, 0.945), fontsize=13,
                 fontweight="bold", color=INK)

    box(ax2, 0.028, 0.470, 0.200, 0.240, "dCas9-KRAB", "effector complex",
        "guided by sgRNA", fill="#EAF0F8", edge=BLUE, ts=12, ss=10, ds=9)
    box(ax2, 0.330, 0.470, 0.200, 0.240, "GATA6 promoter", "target locus",
        "bivalent, poised", fill="#F2F5FA", ts=12, ss=10, ds=9)
    box(ax2, 0.632, 0.470, 0.200, 0.240, "GATA6 protein", "steady state x",
        "knockdown 0.647", ts=12, ss=10, ds=9)

    ax2.plot([0.228, 0.316], [0.590, 0.590], color=INK, lw=2.2, zorder=2)
    blunt(ax2, (0.322, 0.590), 0, size=0.052, lw=3.4)
    ax2.annotate("KRAB-mediated\nrepression", (0.272, 0.712), ha="center", fontsize=9.6,
                 color=GREY, style="italic")
    arrow(ax2, (0.530, 0.590), (0.630, 0.590), lw=2.2)
    ax2.annotate("transcription", (0.580, 0.646), ha="center", fontsize=9.6,
                 color=GREY, style="italic")
    ax2.add_patch(FancyArrowPatch((0.732, 0.466), (0.434, 0.466), arrowstyle="-",
                                  linewidth=2.2, color=RED, linestyle=(0, (5, 3)),
                                  connectionstyle="arc3,rad=-0.42", zorder=2))
    blunt(ax2, (0.432, 0.470), 90, size=0.034, lw=3.4, color=RED)
    ax2.annotate("negative feedback", (0.582, 0.298), ha="center", fontsize=10.2,
                 color=RED, style="italic")

    ax2.annotate(r"$\dfrac{dx_i}{dt} \;=\; \beta_i \prod_{a \in A_i} H^{+}(x_a)"
                 r"\prod_{r \in R_i} H^{-}(x_r) \;-\; \gamma_i x_i$",
                 (0.028, 0.150), fontsize=14.5, color=INK)
    ax2.annotate(r"$H^{+}(x)=x^{n}/(K^{n}+x^{n})$,   $H^{-}(x)=K^{n}/(K^{n}+x^{n})$,   "
                 r"$\beta$ set by promoter strength,   $n=2$",
                 (0.028, 0.040), fontsize=10.8, color=GREY)

    sx, sy, sw, sh = 0.858, 0.285, 0.142, 0.500
    ax2.add_patch(FancyBboxPatch((sx, sy), sw, sh,
                                 boxstyle="round,pad=0.006,rounding_size=0.02",
                                 facecolor="#F7F9FC", edgecolor=PALETTE["grey_dark"],
                                 linewidth=1.6, zorder=3))
    ax2.annotate("Pareto objectives", (sx + sw / 2, sy + sh - 0.045), ha="center",
                 fontsize=10.4, fontweight="bold", color=INK, zorder=4)
    for i, (k, v) in enumerate([("efficacy", 0.915), ("specificity", 0.794),
                                ("robustness", 1.000), ("safety", 0.877)]):
        y = sy + sh - 0.135 - i * 0.078
        ax2.annotate(k, (sx + 0.013, y), fontsize=9.6, color=PALETTE["grey_dark"], zorder=4)
        ax2.annotate(f"{v:.3f}", (sx + sw - 0.013, y), fontsize=9.6, ha="right",
                     color=INK, fontweight="bold", zorder=4)
    ax2.annotate("Pareto rank 0\nODE-stable", (sx + sw / 2, sy + 0.032), ha="center",
                 fontsize=8.8, color=GREY, style="italic", zorder=4)

    fig.suptitle("A designed CRISPRi circuit against GATA6, from construct to kinetics",
                 fontsize=15.5, fontweight="bold", y=0.982)
    save(fig, "fig9_circuit", pad=0.6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    fig_scaleup()
    fig_curves()
    fig_grna_components()
    fig_h3k27ac()
    fig_rac()
    fig_transfer()
    fig_evidence_heatmap()
    fig_architecture()
    fig_circuit()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
