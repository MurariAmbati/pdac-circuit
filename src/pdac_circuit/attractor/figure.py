from __future__ import annotations

import json
from pathlib import Path

from ..core.paths import RESULTS

def make_attractor_figure(results_dir: Path | None = None,out: Path | None = None) -> Path | None:
    results_dir = results_dir or RESULTS
    out = out or (results_dir.parent / "figures" / "fig_attractor.png")
    vpath = results_dir / "attractor_validation.json"
    mpath = results_dir / "attractor_map.json"
    tpath = results_dir / "attractor_targets.json"
    if not (vpath.exists() and mpath.exists() and tpath.exists()):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val = json.loads(vpath.read_text())
    mp = json.loads(mpath.read_text())
    tg = json.loads(tpath.read_text())["targets"]

    fig,ax = plt.subplots(1,3,figsize=(15,5))

    sweep = val["threshold_sweep"]
    thr = [s["essential_threshold"] for s in sweep]
    auc = [s["auc_collapse"] for s in sweep]
    lo = [s["auc_collapse_ci95"][0] for s in sweep]
    hi = [s["auc_collapse_ci95"][1] for s in sweep]
    deg = [s["auc_degree"] for s in sweep]
    eig = [s["auc_eigencentrality"] for s in sweep]
    yerr = [[a - x for a,x in zip(auc,lo)],[x - a for a,x in zip(auc,hi)]]
    ax[0].errorbar(thr,auc,yerr=yerr,fmt="o-",capsize=4,color="#1f77b4",label="RAC collapse")
    ax[0].plot(thr,deg,"s--",color="#ff7f0e",label="degree centrality")
    ax[0].plot(thr,eig,"^:",color="#2ca02c",label="eigenvector centrality")
    ax[0].axhline(0.5,color="gray",lw=1,ls="-")
    ax[0].set_xlabel("DepMap essential threshold (−Chronos)")
    ax[0].set_ylabel("AUC (identify essential regulators)")
    ax[0].set_title("Out-of-modality validation\n(CRISPR held out of the fit)")
    ax[0].set_ylim(0.35,0.9)
    ax[0].legend(fontsize=8)

    tops = mp["top_collapse_nodes"][:15][::-1]
    names = [t["gene"] for t in tops]
    col = [t["collapse"] for t in tops]
    ess = [t["abs_essential"] if t["abs_essential"] is not None else 0.0 for t in tops]
    colors = ["#d62728" if e is not None and e > 0.4 else "#7f7f7f" for e in ess]
    ax[1].barh(range(len(names)),col,color=colors)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names,fontsize=8)
    ax[1].set_xlabel("attractor-collapse score")
    ax[1].set_title("Top collapse-driving regulators\n(red = DepMap-essential > 0.4)")

    tt = tg[:12][::-1]
    gn = [t["gene"] for t in tt]
    cv = [t["convergence_score"] for t in tt]
    driver = ["#9467bd" if t["intogen_driver"] else "#1f77b4" for t in tt]
    ax[2].barh(range(len(gn)),cv,color=driver)
    ax[2].set_yticks(range(len(gn)))
    ax[2].set_yticklabels(gn,fontsize=8)
    ax[2].set_xlabel("convergence score")
    ax[2].set_title("Convergent circuit targets\n(purple = IntOGen driver)")

    fig.suptitle("PDAC Regulatory Attractor Control (RAC) — real DepMap / GTEx / JASPAR / ENCODE",fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.96))
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=130)
    plt.close(fig)
    return out
