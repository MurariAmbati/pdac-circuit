from __future__ import annotations

import json

import numpy as np

from ..core.paths import FIGURES, RESULTS

def _load(p):
    return json.loads((RESULTS / p).read_text(encoding="utf-8")) if (RESULTS / p).exists() else None

def fig_models():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prom=_load("promoter.train.json")
    enh=_load("enhancer.train.json")
    grna=_load("grna_ontarget.train.json")
    if not any([prom, enh, grna]):
        return None
    labels, vals, margins = [], [], []
    if prom:
        labels.append("promoter\n(Spearman)")
        vals.append(prom["spearman_ensemble"])
        margins.append(0.40)
    if enh:
        labels.append("enhancer\n(AUROC)")
        vals.append(enh["auroc"])
        margins.append(0.75)
    if grna:
        labels.append("gRNA on-target\n(Spearman)")
        vals.append(grna["spearman_ensemble"])
        margins.append(0.40)
    fig, ax = plt.subplots(figsize=(6, 4))
    x=np.arange(len(labels))
    ax.bar(x, vals, color="#2c7fb8", width=0.55, label="achieved")
    ax.plot(x, margins, "r_", markersize=28, markeredgewidth=2.5, label="pre-registered margin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("held-out performance")
    ax.set_title("Trained models (real held-out data) vs pre-registered margins")
    ax.legend()
    out=FIGURES / "fig_models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def fig_gan():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g=_load("promoter_gan.train.json")
    if not g:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["GAN vs real", "random vs real"], [g["js_gen_vs_real"], g["js_random_vs_real"]],
                color=["#31a354", "#bdbdbd"])
    axes[0].axhline(0.05, color="r", ls="--", label="realism threshold")
    axes[0].set_ylabel("4-mer Jensen-Shannon divergence")
    axes[0].set_title("GAN realism")
    axes[0].legend()
    if "pred_strength_gen_median" in g:
        axes[1].bar(["GAN-generated", "random DNA"],
                    [g["pred_strength_gen_median"], g["pred_strength_random_median"]],
                    color=["#31a354", "#bdbdbd"])
        axes[1].set_ylabel("predicted promoter strength (model)")
        axes[1].set_title(f"de-novo strength (uplift={g.get('strength_uplift', 0):.2f})")
    fig.suptitle("Module VII — WGAN-GP de-novo promoters")
    out=FIGURES / "fig_gan.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def fig_pareto(run="run_classical.json"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d=_load(run)
    if not d or not d.get("payload"):
        return None
    circuits=d["payload"].get("ranked_circuits", [])
    pts=[(c["sub_scores"]["efficacy"], c["sub_scores"]["safety"], c.get("pareto_rank", 9), c["target_tf"])
           for c in circuits if "sub_scores" in c]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    eff=[p[0] for p in pts]
    saf=[p[1] for p in pts]
    rank=[p[2] for p in pts]
    sc=ax.scatter(eff, saf, c=rank, cmap="viridis_r", s=90, edgecolor="k")
    for e, s, r, name in pts:
        ax.annotate(name, (e, s), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("efficacy")
    ax.set_ylabel("safety")
    ax.set_title(f"Module VI Pareto front ({d['payload'].get('subtype','?')} subtype)")
    plt.colorbar(sc, label="Pareto rank (0 = non-dominated)")
    out=FIGURES / "fig_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def fig_deep(run="deep_classical.json"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d=_load(run)
    if not d or not d.get("payload"):
        return None
    p=d["payload"]
    circuits=p.get("top_circuits", [])
    if not circuits:
        return None
    eff=[c["efficacy"] for c in circuits]
    saf=[c["safety"] for c in circuits]
    kd=[c["tf_knockdown"] for c in circuits]
    rank=[c["pareto_rank"] for c in circuits]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc=axes[0].scatter(eff, saf, c=rank, cmap="viridis_r", s=60, edgecolor="k", linewidth=0.4)
    axes[0].set_xlabel("efficacy")
    axes[0].set_ylabel("safety")
    axes[0].set_title(f"Deep design top circuits ({p['n_circuits']} total, {p['n_pareto_front0']} front-0)")
    plt.colorbar(sc, ax=axes[0], label="Pareto rank")
    axes[1].scatter(eff, kd, c=rank, cmap="viridis_r", s=60, edgecolor="k", linewidth=0.4)
    axes[1].set_xlabel("efficacy")
    axes[1].set_ylabel("TF knockdown (ODE steady state)")
    axes[1].set_title(f"{p.get('n_individually_simulated','?')} individually-simulated circuits")
    fig.suptitle(f"Deep circuit design — {p['subtype']} subtype")
    out=FIGURES / "fig_deep.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def fig_chromatin():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f=RESULTS / "chromatin_states.json"
    if not f.exists():
        return None
    import json as _json
    from collections import Counter

    r=_json.loads(f.read_text(encoding="utf-8"))
    states=r["states"]
    sc=Counter(s["state"] for s in states.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels, vals = zip(*sorted(sc.items(), key=lambda kv: -kv[1]))
    axes[0].barh(labels, vals, color="#756bb1")
    axes[0].set_xlabel("# target TFs")
    axes[0].set_title(f"Chromatin state of {r['n_loci']} target promoters\n(from {r['n_bams_used']} raw ChIP BAMs, marks: {','.join(r['marks'])})", fontsize=9)
    acts=sorted((s["activity_score"], s["bivalency"]) for s in states.values())
    a=[x[0] for x in acts]
    b=[x[1] for x in acts]
    axes[1].scatter(a, b, c=a, cmap="RdBu_r", s=40, edgecolor="k", linewidth=0.3)
    axes[1].set_xlabel("chromatin activity (repressed -1 .. +1 active)")
    axes[1].set_ylabel("bivalency (H3K4me3 ∧ H3K27me3)")
    axes[1].set_title("Base-resolution chromatin activity / bivalency", fontsize=9)
    fig.suptitle("Module VIII — deep ChIP-seq chromatin state (raw BAMs)")
    out=FIGURES / "fig_chromatin.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def _fig_attractor():
    from ..attractor.figure import make_attractor_figure

    return make_attractor_figure()

def _fig_chromatin_training():
    import subprocess
    import sys

    script=FIGURES.parent / "scripts" / "fig_chromatin_training.py"
    curve=RESULTS / "chromatin_training_curve.json"
    if not (script.exists() and curve.exists()):
        return None
    try:
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True, timeout=180)
    except Exception:
        return None
    out=FIGURES / "fig_chromatin_training.png"
    return out if out.exists() else None

def make_figures() -> list:
    FIGURES.mkdir(parents=True, exist_ok=True)
    made=[fig_models(), fig_gan(), fig_pareto("run_classical.json"), fig_pareto("run_basal.json"),
            fig_deep("deep_classical.json"), fig_deep("deep_basal.json"), fig_chromatin(), _fig_attractor(),
            _fig_chromatin_training()]
    return [str(m) for m in made if m]

if __name__ == "__main__":
    for f in make_figures():
        print("wrote", f)
