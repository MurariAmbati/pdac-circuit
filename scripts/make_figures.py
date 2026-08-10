from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("C:/Users/murar/pdac-circuit")
OUT = Path(__file__).resolve().parents[1] / "images"

BLUE = "#2b6cb0"
GREEN = "#2f855a"
RED = "#c53030"
GREY = "#a0aec0"
DARK = "#1a202c"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": DARK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.9,
})


def load(rel):
    p = SRC / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return json.loads(p.read_text(encoding="utf-8"))


def ygrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def pad(lo, hi, frac=0.18):
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0
    return lo - span * frac, hi + span * frac


def fig_scaleup():
    grna = load("results/grna_cnn_kim_retrain.json")
    pro = load("results/promoter_scaleup.json")
    enh = load("results/enhancer_scaleup.json")
    gan = load("results/promoter_gan_scaleup.json")

    panels = [
        ("gRNA on-target", "Spearman", grna["shipped_ensemble"], grna["deployed_ensemble"], "5,310", "18,142"),
        ("Promoter strength", "Spearman", pro["baseline_shipped_ensemble"], pro["scaleup_ensemble"], "60,000", "181,428"),
        ("Enhancer activity", "AUROC", enh["baseline_shipped_auroc"], enh["scaleup_auroc"], "20,000", "135,402"),
        ("Promoter generator", "p90 strength", gan["baseline_p90"], gan["scaleup_p90"], "12,000", "52,342"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.6), gridspec_kw={"wspace": 0.42})
    for ax, (title, metric, before, after, nb, na) in zip(axes, panels):
        bars = ax.bar([0, 1], [before, after], width=0.62,
                      color=[GREY, BLUE], edgecolor=DARK, linewidth=1.6,
                      hatch=["//", ""])
        lo, hi = pad(min(before, after), max(before, after), 0.45)
        ax.set_ylim(max(0, lo), hi)
        for b, v in zip(bars, [before, after]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"capped\nn={nb}", f"full\nn={na}"], fontsize=9)
        ax.set_title(title, fontweight="bold", pad=9)
        ax.set_ylabel(metric)
        ygrid(ax)
        d = after - before
        ax.text(0.5, 0.955, f"{d:+.4f}", transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=GREEN if d > 0 else RED, fontweight="bold")
    fig.suptitle("Removing training-data caps: held-out performance before and after",
                 fontsize=12.5, fontweight="bold", y=1.06)
    save(fig, "fig1_scaleup")


def fig_curves():
    pc = load("results/promoter_scaling_curve.json")
    ec = load("results/enhancer_scaling_curve.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), gridspec_kw={"wspace": 0.26})

    ax = axes[0]
    n = [p["n_train"] for p in pc["points"]]
    ens = [p["ensemble"] for p in pc["points"]]
    cnn = [p["cnn"] for p in pc["points"]]
    ax.plot(n, ens, "-o", color=BLUE, lw=2.2, ms=6.5, mec=DARK, mew=1.1, label="ensemble", zorder=3)
    ax.plot(n, cnn, "--s", color=GREEN, lw=1.8, ms=5.5, mec=DARK, mew=1.0, label="CNN", zorder=2)
    ax.set_xlabel("training peaks")
    ax.set_ylabel("Spearman")
    ax.set_title("Promoter strength", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0.465, 0.545)
    ygrid(ax)

    ax = axes[1]
    n2 = [p["n_train"] for p in ec["points"]]
    au = [p["auroc"] for p in ec["points"]]
    ax.plot(n2, au, "-o", color=BLUE, lw=2.2, ms=6.5, mec=DARK, mew=1.1, zorder=3)
    ax.set_xlabel("training rows")
    ax.set_ylabel("AUROC")
    ax.set_title("Enhancer activity", fontweight="bold")
    ax.set_ylim(0.798, 0.818)
    ygrid(ax)

    for ax in axes:
        ax.xaxis.set_major_formatter(lambda v, p: f"{v / 1000:.0f}k")
    fig.suptitle("Performance versus training-set size, fixed held-out chr8/chr9 test",
                 fontsize=12.5, fontweight="bold", y=1.04)
    save(fig, "fig2_scaling_curves")


def fig_grna_components():
    g = load("results/grna_cnn_kim_retrain.json")
    labels = ["CNN", "GBM", "Ensemble"]
    before = [g["cnn_doench_only_prev"], 0.5250, g["shipped_ensemble"]]
    after = [g["cnn_doench_plus_kim"], g["gbm_doench_plus_kim"], g["deployed_ensemble"]]
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - w / 2, before, w, label="Doench-2016 only (17 genes)",
           color=GREY, edgecolor=DARK, linewidth=1.6, hatch="//")
    ax.bar(x + w / 2, after, w, label="+ Kim-2019 (18,142 guides)",
           color=BLUE, edgecolor=DARK, linewidth=1.6)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w / 2, b + .008, f"{b:.3f}", ha="center", fontsize=9.5)
        ax.text(i + w / 2, a + .008, f"{a:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman, held-out genes")
    ax.set_ylim(0, 0.78)
    ax.legend(loc="upper left")
    ax.set_title("gRNA on-target: the CNN was the bottleneck", fontweight="bold", pad=10)
    ygrid(ax)
    save(fig, "fig3_grna_components")


def fig_h3k27ac():
    k = load("results/pdac_residual_foldchange_H3K27ac.json")
    per = k["per_target"]
    tv = k["targets_vs_all_background"]
    items = sorted(per.items(), key=lambda kv: kv[1]["log2_residual"])
    names = [g for g, _ in items]
    vals = [v["log2_residual"] for _, v in items]
    cols = [GREEN if v > 0 else RED for v in vals]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.barh(names, vals, color=cols, edgecolor=DARK, linewidth=1.2)
    ax.axvline(0, color=DARK, lw=1.2)
    ax.axvline(tv["background_mean_log2"], color=GREY, ls="--", lw=1.8,
               label=f"background mean ({tv['background_mean_log2']:+.3f}, n={tv['n_background']:,})")
    ax.axvline(tv["target_mean_log2"], color=BLUE, ls="-", lw=2.0,
               label=f"target mean ({tv['target_mean_log2']:+.3f}, n={tv['n_targets']})")
    ax.set_xlabel("log2 H3K27ac fold-change residual, PDAC vs healthy pancreas")
    ax.set_title("Promoter H3K27ac at prioritised targets", fontweight="bold", pad=10)
    ax.legend(loc="lower right")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.tick_params(axis="y", labelsize=9)
    ax.text(0.985, 0.045, f"Mann-Whitney p = {tv['mannwhitney_p_greater']:.4f}",
            transform=ax.transAxes, ha="right", fontsize=10, fontweight="bold")
    save(fig, "fig4_h3k27ac")


def fig_rac():
    r = load("results/rigorous_validation.json")["A_rac_vs_degree"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.24})

    ax = axes[0]
    names = ["Attractor\ncollapse", "Eigenvector\ncentrality", "Network\ndegree"]
    vals = [r["auc_rac"], r["auc_eigenvector"], r["auc_degree"]]
    cols = [RED, GREY, BLUE]
    bars = ax.bar(names, vals, width=0.6, color=cols, edgecolor=DARK, linewidth=1.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .006, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color=DARK, ls=":", lw=1.4)
    ax.text(-0.42, 0.503, "chance", fontsize=9, color=DARK, va="bottom", ha="left")
    ax.set_ylim(0.45, 0.68)
    ax.set_ylabel("AUC, held-out essentiality")
    ax.set_title("The proposed score does not beat degree", fontweight="bold", pad=10)
    ygrid(ax)

    ax = axes[1]
    d = r["delta_auc_rac_minus_degree"]
    lo, hi = r["delta_auc_ci95_paired_bootstrap"]
    ax.errorbar([d], [0], xerr=[[d - lo], [hi - d]], fmt="o", color=RED,
                ms=10, mec=DARK, mew=1.3, capsize=7, elinewidth=2.2, capthick=2.2)
    ax.axvline(0, color=DARK, ls="--", lw=1.6)
    ax.set_yticks([])
    ax.set_xlim(-0.26, 0.10)
    ax.set_xlabel(r"$\Delta$AUC (collapse $-$ degree)")
    ax.set_title("95% CI spans zero", fontweight="bold", pad=10)
    ax.text(0.5, 0.30, f"{d:+.4f}  [{lo:+.3f}, {hi:+.3f}]\np = {r['delta_auc_p_two_sided']:.3f}",
            transform=ax.transAxes, ha="center", fontsize=10.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    fig.suptitle("Adversarial validation of the attractor-collapse claim",
                 fontsize=12.5, fontweight="bold", y=1.05)
    save(fig, "fig5_rac_validation")


def fig_transfer():
    p = load("results/enhancer_panc1_augment.json")
    ec = load("results/enhancer_scaling_curve.json")
    labels = ["pancreas → pancreas\n(within domain)", "pancreas → PANC-1\n(forward transfer)",
              "PANC-1 → pancreas\n(reverse transfer)"]
    vals = [p["pancreas_only_pancreas_test"], p["pancreas_only_panc1_test_xdomain"],
            ec["reverse_xdomain_panc1_to_pancreas"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, vals, width=0.58, color=[GREY, BLUE, GREEN],
                  edgecolor=DARK, linewidth=1.6, hatch=["//", "", ".."])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.3f}",
                ha="center", fontsize=10.5, fontweight="bold")
    ax.axhline(0.5, color=DARK, ls=":", lw=1.3)
    ax.set_ylim(0.5, 0.87)
    ax.set_ylabel("AUROC")
    ax.set_title("Enhancer grammar transfers between healthy pancreas and PDAC",
                 fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=9.5)
    ygrid(ax)
    save(fig, "fig6_cross_domain")



INK = "#111318"
RULE = 1.0


def _box(ax, x, y, w, h, label, sub=None, fill="#ffffff", fs=9.4, lw=RULE):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=lw, edgecolor=INK, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h * (0.60 if sub else 0.5), label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fs - 1.7, color="#4a5261", zorder=4)


def _arrow(ax, p0, p1, style="-|>", color=INK, lw=RULE, rad=0.0, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=2, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h, y = 0.148, 0.205, 0.455
    xs = [0.040, 0.226, 0.412, 0.598, 0.784]
    rows = [
        ("I", "Target\nprioritisation", "TCGA, GTEx, IntOGen"),
        ("II", "Regulatory\nparts", "promoter, enhancer"),
        ("V", "Guide\ndesign", "PAM, on-target, CFD"),
        ("III / IV", "Circuit and\nsequence", "Hill ODE, codon"),
        ("VI", "Pareto\nscoring", "NSGA-II, 4 objectives"),
    ]
    for x, (num, name, sub) in zip(xs, rows):
        _box(ax, x, y, w, h, name, sub, fs=9.8)
        ax.text(x + 0.007, y + h - 0.030, num, fontsize=8.0, color="#8892a4",
                fontweight="bold", zorder=4)
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    _box(ax, xs[0], 0.135, w, 0.165, "VIII  Attractor", "retracted predictor",
         fill="#f4f6fa", fs=9.4)
    _arrow(ax, (xs[0] + w / 2, 0.300), (xs[0] + w / 2, y), ls=(0, (3, 2)))
    _box(ax, xs[1], 0.135, w, 0.165, "VII  Generator", "WGAN-GP promoters",
         fill="#f4f6fa", fs=9.4)
    _arrow(ax, (xs[1] + w / 2, 0.300), (xs[1] + w / 2, y))

    _box(ax, 0.040, 0.795, 0.892, 0.135, "Real public data",
         "GRCh38, FANTOM5, ENCODE, DepMap, TCGA, GTEx, CPTAC, 4DN, dbSNP",
         fill="#f4f6fa", fs=10.2)
    for x in xs:
        _arrow(ax, (x + w / 2, 0.795), (x + w / 2, y + h))

    ax.text(0.932, 0.558, "certified\nnegative", fontsize=8.6, color="#5a6474",
            ha="left", va="center", style="italic")
    _arrow(ax, (xs[4] + w, y + h / 2), (0.928, y + h / 2))
    ax.text(0.040, 0.048,
            "Every stage returns a certified negative where the evidence to proceed is absent",
            fontsize=9.2, color="#5a6474", style="italic")
    save(fig, "fig7_architecture")


def fig_evidence_heatmap():
    d = load("results/rac_target_dossiers.json")
    layers = [
        ("disease_log2fc", "is_it_real", "Disease log2FC"),
        ("depmap_essentiality", "is_it_real", "DepMap essentiality"),
        ("depmap_pdac_selectivity", "is_it_real", "PDAC selectivity"),
        ("cna_amplification_freq", "is_it_real", "CNA amplification"),
        ("cna_deletion_freq", "is_it_real", "CNA deletion"),
        ("promoter_methylation_beta", "is_it_real", "Promoter methylation"),
        ("protein_mean", "is_it_real", "Protein abundance"),
        ("protein_detection_rate", "is_it_real", "Protein detection"),
        ("h3k27ac_disease_residual_log2", "is_it_active", "H3K27ac residual"),
        ("atac_disease_residual_log2", "is_it_active", "ATAC residual"),
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
    z = np.clip(z, -2.2, 2.2)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_facecolor("#eef1f5")
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(len(genes)):
        for j in range(len(layers)):
            if np.isnan(z[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#e3e7ed",
                                           edgecolor="white", hatch="///", linewidth=0))
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l[2] for l in layers], rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9.4)
    ax.set_xticks(np.arange(-.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(genes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("z-score within layer", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Multi-omic evidence per prioritised target", fontweight="bold", pad=12)
    ax.text(1.0, -0.335, "hatched cells are not measured in that layer", transform=ax.transAxes,
            ha="right", fontsize=8.6, color="#5a6474")
    save(fig, "fig8_evidence_heatmap")


def _repress_head(ax, tip, angle_deg, size=0.022, lw=RULE * 1.4):
    import math
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -dy, dx
    ax.plot([tip[0] + px * size, tip[0] - px * size],
            [tip[1] + py * size, tip[1] - py * size],
            color=INK, linewidth=lw, solid_capstyle="butt", zorder=5)


def fig_circuit():
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.030, 0.615, 0.190, 0.160, "Synthetic promoter", "GAN-generated library")
    _box(ax, 0.030, 0.335, 0.190, 0.160, "Enhancer", "activity model")
    _box(ax, 0.290, 0.475, 0.190, 0.160, "dCas9-KRAB", "CRISPRi effector")
    _box(ax, 0.290, 0.150, 0.190, 0.145, "sgRNA", "on-target model")
    _box(ax, 0.550, 0.475, 0.170, 0.160, "GATA6", "target locus", fill="#eef2f7")
    _box(ax, 0.550, 0.780, 0.170, 0.145, "Output", "steady state")

    _arrow(ax, (0.220, 0.695), (0.290, 0.600))
    _arrow(ax, (0.220, 0.415), (0.290, 0.510))
    _arrow(ax, (0.385, 0.295), (0.385, 0.475))
    _arrow(ax, (0.635, 0.635), (0.635, 0.780))

    ax.plot([0.480, 0.536], [0.555, 0.555], color=INK, linewidth=RULE, zorder=2)
    _repress_head(ax, (0.540, 0.555), 0, size=0.032)

    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((0.722, 0.852), (0.742, 0.555), arrowstyle="-",
                                 linewidth=RULE, color=INK, zorder=2,
                                 linestyle=(0, (4, 2)),
                                 connectionstyle="arc3,rad=-0.75"))
    _repress_head(ax, (0.734, 0.555), 0, size=0.030)
    ax.text(0.885, 0.690, "negative\nfeedback", fontsize=8.8, color="#4a5261",
            ha="center", va="center", style="italic")

    ax.text(0.550, 0.360,
            "efficacy 0.915     specificity 0.794\nrobustness 1.000     safety 0.877\n"
            "knockdown 0.647, ODE-stable",
            fontsize=8.9, color="#4a5261", va="top")

    ax.text(0.030, 0.045,
            r"$dx_i/dt \;=\; \beta_i \prod_{a} H^{+}(x_a)\,\prod_{r} H^{-}(x_r) \;-\; \gamma_i x_i$",
            fontsize=10.4, color=INK)
    ax.text(0.560, 0.055, "blunt heads denote repression, dashed denotes feedback",
            fontsize=8.8, color="#5a6474")
    ax.set_title("A designed circuit, GATA6 under negative feedback",
                 fontweight="bold", fontsize=12, pad=10, loc="left", x=0.030)
    save(fig, "fig9_circuit")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    fig_scaleup()
    fig_curves()
    fig_grna_components()
    fig_h3k27ac()
    fig_rac()
    fig_transfer()
    fig_architecture()
    fig_evidence_heatmap()
    fig_circuit()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
