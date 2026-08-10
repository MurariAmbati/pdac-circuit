from __future__ import annotations

import json
import re
import time

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "univariate_screen_modelfree.json"
GRN=RESULTS / "directed_grn.npz"
EDGE_T=0.9
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}",flush=True)

def _sym(c):
    m=re.match(r"^(.*?)\s*\(\d+\)$",c)
    return (m.group(1) if m else c).strip()

def pagerank(A,d=0.85,iters=200,tol=1e-9):
    n=A.shape[0]
    out=A.sum(1,keepdims=True)
    dangling=(out.ravel() == 0)
    T=np.divide(A,np.where(out == 0,1,out))
    r=np.full(n,1.0 / n)
    for _ in range(iters):
        rn=(1 - d) / n + d * (T.T @ r + r[dangling].sum() / n)
        if np.abs(rn - r).max() < tol:
            return rn
        r=rn
    return r

def rank_auc(x,y):
    from scipy.stats import rankdata
    y=np.asarray(y,dtype=bool)
    npos,nneg = int(y.sum()),int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r=rankdata(x)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def main():
    import pandas as pd
    from scipy.stats import spearmanr

    from pdac_circuit.attractor.graph import _depmap_expr_path,build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality,load_essentiality
    from pdac_circuit.data.misc import _pdac_model_ids

    log("rebuilding graph + features")
    g=build_regulatory_graph(max_nodes=400,coexpr_threshold=0.4,motif_edges=False,seed=20260620)
    nodes=g.nodes
    dat=np.load(GRN,allow_pickle=True)
    M=dat["M"]
    grn_nodes=list(dat["nodes"])
    gi={x: i for i,x in enumerate(grn_nodes)}
    sidx=np.array([gi[x] for x in nodes])
    M=M[np.ix_(sidx,sidx)]
    A=(M >= EDGE_T).astype(np.float64)

    ep=_depmap_expr_path()
    hdr=pd.read_csv(ep,nrows=0).columns.tolist()
    idc=hdr[0]
    s2c={}
    for c in hdr[1:]:
        s2c.setdefault(_sym(c),c)
    use=[idc] + [s2c[x] for x in nodes if x in s2c]
    full=pd.read_csv(ep,usecols=use,index_col=0)
    full.columns=[_sym(c) for c in full.columns]
    pdac_ids=set(_pdac_model_ids())
    is_p=full.index.isin(pdac_ids)
    raw,other = full[is_p],full[~is_p]
    rmean=np.array([float(raw[x].mean()) if x in raw.columns else np.nan for x in nodes])
    rvar=np.array([float(raw[x].var()) if x in raw.columns else np.nan for x in nodes])
    ediff=np.array([float(raw[x].mean() - other[x].mean())
                      if x in raw.columns and x in other.columns else np.nan for x in nodes])

    feats={
        "coexpr_degree": g.adjacency.sum(1),"eigenvector": _eigencentrality(g.adjacency),
        "out_strength": M.sum(1),"in_strength": M.sum(0),"out_degree": A.sum(1),
        "in_degree": A.sum(0),"pagerank": pagerank(A),"hits_hub": (A @ (A.sum(0))),
        "cna_amp": np.nan_to_num(g.cna_amp_freq),"cna_mean": np.nan_to_num(g.cna_mean),
        "methylation": np.nan_to_num(g.promoter_methylation,nan=0.05),
        "expr_mean_norm": g.states.mean(0),"expr_var_norm": g.states.var(0),
        "expr_mean_raw": np.nan_to_num(rmean),"expr_var_raw": np.nan_to_num(rvar),
        "disease_log2fc_LEVEL": g.disease_log2fc,
        "expr_pdac_minus_other_DIFFERENTIAL": np.nan_to_num(ediff),
    }

    ess=load_essentiality(nodes)
    absd=ess.get("abs",{})
    seld=ess.get("sel",{})
    cov=[i for i,x in enumerate(nodes)
           if x in absd and np.isfinite(absd[x]) and np.isfinite(seld.get(x,np.nan))]
    sel_e=np.array([seld[nodes[i]] for i in cov])
    y=(sel_e > 0.15).astype(int)
    log(f"genes {len(cov)}, selective positives {int(y.sum())}")

    rows=[]
    for k,v in feats.items():
        x=np.asarray(v,dtype=float)[cov]
        a=rank_auc(x,y)
        rho,p = spearmanr(x,sel_e)
        rows.append({"feature": k,"rank_auc": round(a,4),"effect_size_abs": round(abs(a - 0.5),4),
                     "direction": "higher => more selective" if a > 0.5 else "higher => less selective",
                     "spearman_rho_continuous": round(float(rho),4),
                     "spearman_p": round(float(p),5),
                     "significant_continuous_p05": bool(p < 0.05)})
    rows.sort(key=lambda r: -r["effect_size_abs"])

    ps=np.array([r["spearman_p"] for r in rows])
    m=len(ps)
    order=np.argsort(ps)
    bh=np.empty(m)
    bh[order]=np.minimum.accumulate((ps[order] * m / np.arange(1,m + 1))[::-1])[::-1]
    for r,q in zip(rows,np.clip(bh,0,1)):
        r["spearman_q_BH"]=round(float(q),5)
        r["survives_BH_q05"]=bool(q < 0.05)

    survivors=[r["feature"] for r in rows if r["survives_BH_q05"]]
    centrality_feats={"coexpr_degree","eigenvector","out_strength","in_strength",
                        "out_degree","in_degree","pagerank","hits_hub"}
    cent_survivors=[f for f in survivors if f in centrality_feats]

    rep={
        "schema": "pdac-circuit.univariate-modelfree/1","data_class": "REAL",
        "sealed_studies_touched": False,
        "why": ("fitted logistic OOF AUC is invariant to feature negation and, at 14 positives, drags "
                "genuinely-predictive features below 0.5 (demonstrated: synthetic feature with "
                "model-free AUC 0.562 gives OOF 0.424). Sub-0.5 fitted AUCs therefore carry no "
                "directional meaning; this screen is model-free."),
        "n_genes": len(cov),"n_selective_positive": int(y.sum()),
        "per_feature": rows,
        "bh_survivors_continuous": survivors,
        "centrality_features_surviving_BH": cent_survivors,
        "graph_peripheral_hypothesis": (
            "REFUTED" if not cent_survivors else f"partially supported by {cent_survivors}"),
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"{'feature':38}{'rankAUC':>9}{'|eff|':>8}{'rho':>9}{'p':>10}{'q(BH)':>9}  direction")
    for r in rows:
        print(f"{r['feature']:38}{r['rank_auc']:>9.4f}{r['effect_size_abs']:>8.4f}"
              f"{r['spearman_rho_continuous']:>9.3f}{r['spearman_p']:>10.4f}{r['spearman_q_BH']:>9.4f}"
              f"  {'^' if r['rank_auc']>0.5 else 'v'} {r['direction']}")
    print(f"\nBH survivors (continuous, q<0.05): {survivors or 'NONE'}")
    print(f"centrality features surviving BH: {cent_survivors or 'NONE'}")
    print(f"graph-peripheral hypothesis: {rep['graph_peripheral_hypothesis']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
