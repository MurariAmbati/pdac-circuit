from __future__ import annotations

import json
import time

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT = RESULTS / "conditional_collapse_signal.json"
PRIMARY_NODES,PRIMARY_TAU = 400,0.4
PRIMARY_THRESHOLD = 0.4
SEL_THRESHOLD = 0.15
T0 = time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}",flush=True)

def auc(scores,labels):
    from scipy.stats import rankdata

    labels = np.asarray(labels,dtype=bool)
    npos,nneg = int(labels.sum()),int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def collapse_for(max_nodes,tau,seed=20260620,epochs=1500):
    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import build_regulatory_graph

    g = build_regulatory_graph(max_nodes=max_nodes,coexpr_threshold=tau,motif_edges=False,seed=seed)
    d = AttractorDynamics(g)
    d.fit(epochs=epochs,motif_weight=0.0,seed=seed)
    return g,d.collapse_scores(per_line=True)

def degree_conditioned_concordance(score,deg,lab,caliper,rng,n_perm=2000):
    pos = np.flatnonzero(lab)
    neg = np.flatnonzero(~lab)
    if pos.size == 0 or neg.size == 0:
        return None
    dd = np.abs(deg[pos][:,None] - deg[neg][None,:])
    mask = dd <= caliper
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return None
    sp,sn = score[pos][:,None],score[neg][None,:]
    wins = (sp > sn).astype(float) + 0.5 * (sp == sn)
    obs = float(wins[mask].sum() / n_pairs)

    m = lab.size
    null = np.empty(n_perm)
    n_pos = int(lab.sum())
    for k in range(n_perm):
        perm = np.zeros(m,dtype=bool)
        perm[rng.choice(m,n_pos,replace=False)] = True
        pp,nn = np.flatnonzero(perm),np.flatnonzero(~perm)
        dm = np.abs(deg[pp][:,None] - deg[nn][None,:]) <= caliper
        npr = int(dm.sum())
        if npr == 0:
            null[k] = 0.5
            continue
        w = (score[pp][:,None] > score[nn][None,:]).astype(float) + 0.5 * (score[pp][:,None] == score[nn][None,:])
        null[k] = w[dm].sum() / npr
    p = float((null >= obs).mean())
    return {
        "caliper": float(caliper),
        "n_matched_pairs": n_pairs,
        "matched_auc": round(obs,4),
        "unconditioned_auc": round(auc(score,lab),4),
        "perm_p_one_sided": round(p,4),
        "null_mean": round(float(null.mean()),4),
        "null_p95": round(float(np.percentile(null,95)),4),
    }

def tertile_stratified(score,deg,eig,lab):
    order = np.argsort(deg)
    thirds = np.array_split(order,3)
    rows = []
    for name,idx in zip(("low_degree","mid_degree","high_degree"),thirds):
        rows.append({
            "band": name,
            "n": int(idx.size),
            "n_essential": int(lab[idx].sum()),
            "degree_range": [round(float(deg[idx].min()),3),round(float(deg[idx].max()),3)],
            "auc_collapse": round(auc(score[idx],lab[idx]),4) if 0 < lab[idx].sum() < idx.size else None,
            "auc_degree": round(auc(deg[idx],lab[idx]),4) if 0 < lab[idx].sum() < idx.size else None,
            "auc_eigenvector": round(auc(eig[idx],lab[idx]),4) if 0 < lab[idx].sum() < idx.size else None,
        })
    return rows

def analyse(tag,score,deg,eig,ess_vals,threshold,rng):
    lab = ess_vals > threshold
    npos = int(lab.sum())
    out = {"label": tag,"threshold": threshold,"n_genes": int(lab.size),"n_positive": npos,
           "positive_rate": round(npos / lab.size,4)}
    if npos < 5 or npos > lab.size - 5:
        out["status"] = f"underpowered (n_positive={npos}); conditional tests not run"
        out["unconditioned_auc_collapse"] = round(auc(score,lab),4) if 0 < npos < lab.size else None
        return out
    spread = float(np.percentile(deg,90) - np.percentile(deg,10)) or 1.0
    out["degree_conditioned"] = [
        r for c in (0.5,0.25,0.1)
        if (r := degree_conditioned_concordance(score,deg,lab,c * spread,rng)) is not None
    ]
    out["tertile_stratified"] = tertile_stratified(score,deg,eig,lab)
    return out

def main():
    from pdac_circuit.attractor.run import _eigencentrality,load_essentiality

    log(f"fitting primary config nodes={PRIMARY_NODES} tau={PRIMARY_TAU} (same as part A)")
    g,collapse = collapse_for(PRIMARY_NODES,PRIMARY_TAU)
    ess = load_essentiality(g.nodes)
    log(f"essentiality scope={ess.get('_scope')} covered={len(ess.get('abs',{}))}")

    nodes = g.nodes
    cov = [i for i,x in enumerate(nodes) if x in ess.get("abs",{})]
    c = collapse[cov]
    deg = g.adjacency.sum(axis=1)[cov]
    eig = _eigencentrality(g.adjacency)[cov]
    abs_e = np.array([ess["abs"][nodes[i]] for i in cov])
    sel_e = np.array([ess["sel"][nodes[i]] for i in cov])
    ok = np.isfinite(c) & np.isfinite(abs_e) & np.isfinite(sel_e)
    c,deg,eig,abs_e,sel_e = c[ok],deg[ok],eig[ok],abs_e[ok],sel_e[ok]
    log(f"genes analysed: {c.size}")

    rng = np.random.default_rng(20260717)
    rep = {
        "schema": "pdac-circuit.conditional-collapse/1","data_class": "REAL",
        "sealed_studies_touched": False,
        "config": {"nodes": PRIMARY_NODES,"tau": PRIMARY_TAU,
                   "note": "same primary configuration as rigorous_validation.py part A; extends the retraction"},
        "question": ("does collapse rank essential genes above non-essential ones AMONG genes of "
                     "comparable degree, where degree itself cannot separate them?"),
        "absolute_essentiality": analyse("absolute (Chronos, core+selective)",c,deg,eig,abs_e,PRIMARY_THRESHOLD,rng),
        "pdac_selective_essentiality": analyse("PDAC-selective (Chronos PDAC minus other)",c,deg,eig,sel_e,SEL_THRESHOLD,rng),
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")

    for key in ("absolute_essentiality","pdac_selective_essentiality"):
        a = rep[key]
        print(f"\n=== {a['label']}  (n={a['n_genes']}, positives={a['n_positive']}) ===")
        if "status" in a:
            print(f"  {a['status']}")
            continue
        print("  degree-conditioned concordance (matched pairs):")
        for r in a["degree_conditioned"]:
            print(f"    caliper {r['caliper']:6.2f}: matched AUC {r['matched_auc']:.3f} "
                  f"(uncond {r['unconditioned_auc']:.3f}), {r['n_matched_pairs']:>6} pairs, "
                  f"perm p={r['perm_p_one_sided']:.3f}, null95={r['null_p95']:.3f}")
        print("  degree-tertile stratified AUC:")
        for r in a["tertile_stratified"]:
            cc = "  n/a" if r["auc_collapse"] is None else f"{r['auc_collapse']:.3f}"
            dd = "  n/a" if r["auc_degree"] is None else f"{r['auc_degree']:.3f}"
            print(f"    {r['band']:12} n={r['n']:3} ess={r['n_essential']:2}  "
                  f"collapse {cc}  degree {dd}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
