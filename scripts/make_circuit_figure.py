from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from figstyle import PALETTE, apply_publication_style, finalize_figure

_HERE = Path(__file__).resolve().parents[1]
SRC = _HERE if (_HERE / "results").is_dir() else Path("C:/Users/murar/pdac-circuit")
OUT = _HERE / "images"
sys.path.insert(0, str(SRC / "src"))

INK = PALETTE["ink"]
BLUE = PALETTE["blue_main"]
BLUE2 = PALETTE["blue_secondary"]
GREEN = PALETTE["green_3"]
RED = PALETTE["red_strong"]
GREY = PALETTE["grey_mid"]
NEUT = PALETTE["neutral"]
VIOLET = PALETTE["violet"]
TEAL = PALETTE["teal"]


def record():
    d = json.loads((SRC / "results" / "_rac_designed_constructs.json").read_text(encoding="utf-8"))
    r = d["payload"]["ranked_circuits"][0]
    c = json.loads((SRC / "results" / "_rac_designed_circuits.json").read_text(encoding="utf-8"))
    return r, c["payload"]["top_circuits"][0]


def trajectory(prom, enh, on_target, target_kd):
    from pdac_circuit.circuit.ode import ODEModel
    from pdac_circuit.pipeline.deep import _build_deep_circuit

    beta_syn = 0.4 + 1.6 * (prom * enh)
    beta_rep = 0.3 + 1.7 * on_target

    def kd_for(beta_tf):
        circ = _build_deep_circuit(beta_tf, beta_syn, beta_rep)
        ode = ODEModel.from_circuit(circ)
        sim = ode.simulate(t_span=(0.0, 80.0), n_points=600)
        idx = ode.index["TF"]
        xss = sim["x"][idx, -1] if sim["x"].shape[0] < sim["x"].shape[1] else sim["x"][:, -1][idx]
        g = circ.gene("TF")
        free = g.promoter.strength / max(g.tf.degradation, 1e-6) + g.tf.basal
        return float(np.clip(1.0 - xss / (free + 1e-9), 0.0, 1.0)), sim, ode, circ

    lo, hi = 0.30, 0.95
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        k, *_ = kd_for(mid)
        if k > target_kd:
            hi = mid
        else:
            lo = mid
    beta_tf = 0.5 * (lo + hi)
    k, sim, ode, circ = kd_for(beta_tf)
    return sim, ode, beta_tf, beta_syn, beta_rep, k


def panel_diagram(ax, r, c):
    img = mpimg.imread(str(OUT / "fig9_circuit_biorender.png"))
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.0, 1.045, "a", transform=ax.transAxes, fontsize=15, weight="bold",
            color=INK, va="bottom")
    ax.text(0.028, 1.045, "Delivered construct and its action at the target locus",
            transform=ax.transAxes, fontsize=12.5, weight="bold", color=INK, va="bottom")
    loc = r["enhancer"]["locus"]
    ax.text(0.0, -0.035,
            f"promoter strength {r['promoter']['strength']:.3f}   "
            f"enhancer activity {r['enhancer']['activity']:.3f}   "
            f"guide on-target {r['repressor_guide']['on_target']:.3f}   "
            f"CFD specificity {r['repressor_guide']['cfd_specificity']:.3f}   "
            f"{loc['chrom']}:{loc['start']:,}-{loc['end']:,}",
            transform=ax.transAxes, fontsize=10, color=GREY, va="top")
    ax.text(0.0, -0.085, "Diagram created with BioRender.com", transform=ax.transAxes,
            fontsize=9.5, color=GREY, va="top", style="italic")


def panel_c(ax, sim, ode, kd, betas):
    t = sim["t"]
    x = sim["x"]
    idx = ode.index
    order = {k: v for k, v in idx.items()}
    series = [("TF", INK, "GATA6 protein"), ("Repressor", RED, "dCas9-KRAB"),
              ("SynProm", GREEN, "synthetic promoter output")]
    for name, col, lab in series:
        if name not in order:
            continue
        row = x[order[name]] if x.shape[0] <= x.shape[1] else x[:, order[name]]
        ax.plot(t, row, color=col, lw=2.3, label=lab)
    tf = x[order["TF"]] if x.shape[0] <= x.shape[1] else x[:, order["TF"]]
    free = tf[0] if tf[0] > tf[-1] else float(np.max(tf))
    ax.axhline(tf[-1], color=GREY, ls=":", lw=1.4)
    ax.annotate("", xy=(72, tf[-1]), xytext=(72, free),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
    ax.text(70.0, (free + tf[-1]) / 2, f"knockdown\n{kd:.3f}", fontsize=10.5,
            color=INK, ha="right", va="center", weight="bold")
    ax.set_xlabel("time (arbitrary units)")
    ax.set_ylabel("concentration")
    ax.set_title("c  Simulated kinetics of this circuit", loc="left", fontsize=12.5, weight="bold")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    ax.grid(axis="y", color=NEUT, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.985, 0.04,
            rf"$\beta_{{TF}}$={betas[0]:.2f}  $\beta_{{syn}}$={betas[1]:.2f}  $\beta_{{rep}}$={betas[2]:.2f}",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=GREY)


def panel_d(ax, c):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.97, "c  Objectives", fontsize=12.5, weight="bold", color=INK, va="top")
    rows = [("efficacy", c["efficacy"]), ("specificity", c["specificity"]),
            ("robustness", c["robustness"]), ("safety", c["safety"])]
    y = 0.76
    for lab, v in rows:
        ax.text(0.0, y, lab, fontsize=10.5, color=INK, va="center")
        ax.add_patch(plt.Rectangle((0.42, y - 0.045), 0.44, 0.09,
                                   facecolor=NEUT, edgecolor="none"))
        ax.add_patch(plt.Rectangle((0.42, y - 0.045), 0.44 * float(v), 0.09,
                                   facecolor=TEAL, edgecolor="none"))
        ax.text(1.0, y, f"{v:.3f}", fontsize=10.5, color=INK, va="center", ha="right",
                weight="bold")
        y -= 0.165
    ax.text(0.0, 0.08, f"Pareto rank {c['pareto_rank']}, ODE-stable",
            fontsize=9.5, color=GREY, style="italic")


def main():
    apply_publication_style()
    r, c = record()
    sim, ode, b_tf, b_syn, b_rep, kd = trajectory(
        r["promoter"]["strength"], r["enhancer"]["activity"],
        r["repressor_guide"]["on_target"], c["tf_knockdown"])

    fig = plt.figure(figsize=(13.6, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0], width_ratios=[1.58, 1.0],
                          hspace=0.20, wspace=0.20,
                          left=0.055, right=0.975, top=0.955, bottom=0.065)
    ax = fig.add_subplot(gs[0, :])
    panel_diagram(ax, r, c)
    axc = fig.add_subplot(gs[1, 0])
    panel_c(axc, sim, ode, kd, (b_tf, b_syn, b_rep))
    axc.set_title("b  Simulated kinetics of this circuit", loc="left", fontsize=12.5,
                  weight="bold")
    axd = fig.add_subplot(gs[1, 1])
    panel_d(axd, c)
    finalize_figure(fig, OUT, "fig9_circuit", formats=("png", "pdf"), dpi=200, pad=None)
    print(f"  fig9_circuit  (GATA6, knockdown {kd:.3f}, beta_tf {b_tf:.3f})")


if __name__ == "__main__":
    main()
