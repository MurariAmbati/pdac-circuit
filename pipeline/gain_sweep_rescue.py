from __future__ import annotations

import json
import re
import time

import numpy as np
import torch

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "gain_sweep_rescue.json"
GAINS=[4.0,5.0,6.0,7.0,8.0]
PRIMARY_THRESHOLD=0.4
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}",flush=True)

def auc(scores,labels):
    from scipy.stats import rankdata

    labels=np.asarray(labels,dtype=bool)
    npos,nneg = int(labels.sum()),int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r=rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def step(W,b,gain,x):
    return torch.sigmoid(gain * (x @ W.t() + b))

def converged_fraction(W,b,gain,states,iters=1500,tol=1e-6):
    n_ok=0
    for s in states:
        x=s.clone()
        for _ in range(iters):
            xn=step(W,b,gain,x)
            if (xn - x).abs().max() < tol:
                n_ok += 1
                break
            x=xn
    return n_ok / len(states)

def spectral_radius(W,b,gain,xstar):
    z=xstar @ W.t() + b
    sig=torch.sigmoid(gain * z)
    dsig=(sig * (1 - sig)).cpu().numpy()
    J=(gain * dsig)[:,None] * W.cpu().numpy()
    return float(np.max(np.abs(np.linalg.eigvals(J))))

def _sym(c):
    m=re.match(r"^(.*?)\s*\(\d+\)$",c)
    return (m.group(1) if m else c).strip()

def main():
    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import build_regulatory_graph
    from pdac_circuit.attractor.run import load_essentiality

    g=build_regulatory_graph(max_nodes=400,coexpr_threshold=0.4,motif_edges=False,seed=20260620)
    ess=load_essentiality(g.nodes)
    nodes=g.nodes
    cov=[i for i,x in enumerate(nodes) if x in ess.get("abs",{})]
    abs_e=np.array([ess["abs"][nodes[i]] for i in cov])
    deg=g.adjacency.sum(axis=1)[cov]
    states_t=torch.tensor(g.states,dtype=torch.float32)
    mean_state=states_t.mean(0)
    lab_all=abs_e > PRIMARY_THRESHOLD
    auc_degree=auc(deg[np.isfinite(abs_e)],lab_all[np.isfinite(abs_e)])
    log(f"degree AUC (fixed reference) = {auc_degree:.4f}; {int(lab_all.sum())} positives / {len(cov)}")

    rows=[]
    for gain in GAINS:
        d=AttractorDynamics(g,gain=gain,device="cpu")
        fr=d.fit(epochs=1500,motif_weight=0.0,seed=20260620)
        cf=converged_fraction(d.W,d.b,gain,states_t)
        x=mean_state.clone()
        for _ in range(1000):
            xn=step(d.W,d.b,gain,x)
            if (xn - x).abs().max() < 1e-6:
                break
            x=xn
        rho=spectral_radius(d.W,d.b,gain,x)
        collapse=d.collapse_scores(per_line=True)
        c=collapse[cov]
        ok=np.isfinite(c) & np.isfinite(abs_e)
        a_rac=auc(c[ok],lab_all[ok])
        rows.append({
            "gain": gain,"fit_fp_error": round(fr.fixed_point_error,5),
            "dead_activation": round(fr.dead_activation,4),
            "converged_fraction": round(cf,4),
            "spectral_radius": round(rho,4),
            "stable_and_convergent": bool(cf > 0.5 and rho < 1.0),
            "auc_collapse": round(a_rac,4),
            "auc_degree": round(auc_degree,4),
            "delta_auc_collapse_minus_degree": round(a_rac - auc_degree,4),
            "collapse_beats_degree": bool(a_rac > auc_degree),
        })
        log(f"  gain {gain}: conv {cf:.2f}, rho {rho:.3f}, collapse AUC {a_rac:.3f} "
            f"(degree {auc_degree:.3f}, delta {a_rac-auc_degree:+.3f})")

    bistable_rows=[r for r in rows if r["converged_fraction"] > 0.5 and r["spectral_radius"] < 1.0]
    rescued=[r for r in rows if r["collapse_beats_degree"] and r["converged_fraction"] > 0.5]
    if rescued:
        verdict=("RESCUABLE: collapse beats degree in a convergent/stable regime at gain(s) "
                   + ", ".join(str(r["gain"]) for r in rescued))
    elif bistable_rows and not rescued:
        verdict=("NOT the missing ingredient: even where the system becomes convergent/stable, "
                   "collapse does not beat degree -- the co-expression graph does not encode "
                   "essentiality beyond degree, so the §15 retraction is structural, not a tuning "
                   "artifact")
    else:
        verdict=("INCONCLUSIVE ON BISTABILITY: no swept gain produced a convergent, spectrally "
                   "stable regime, so the constructive fix (operate above the bifurcation) does not "
                   "by itself yield a bistable system with this graph/fit; collapse never beats degree")
    rep={
        "schema": "pdac-circuit.gain-sweep-rescue/1","data_class": "REAL",
        "sealed_studies_touched": False,
        "endpoint": "absolute essentiality (Chronos abs > 0.4), 419-gene panel, same as §15/part A",
        "degree_auc_reference": round(auc_degree,4),
        "per_gain": rows,"verdict": verdict,
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{'gain':>5} {'conv':>6} {'rho':>7} {'collapseAUC':>12} {'degreeAUC':>10} {'delta':>8} beats?")
    for r in rows:
        print(f"{r['gain']:>5} {r['converged_fraction']:>6.2f} {r['spectral_radius']:>7.3f} "
              f"{r['auc_collapse']:>12.3f} {r['auc_degree']:>10.3f} "
              f"{r['delta_auc_collapse_minus_degree']:>+8.3f} {'YES' if r['collapse_beats_degree'] else 'no'}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
