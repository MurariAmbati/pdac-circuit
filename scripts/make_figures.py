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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    fig_scaleup()
    fig_curves()
    fig_grna_components()
    fig_h3k27ac()
    fig_rac()
    fig_transfer()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
