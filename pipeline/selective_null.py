from __future__ import annotations

import json
import re

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT = RESULTS / "selective_null.json"
GRN = RESULTS / "directed_grn.npz"
EDGE_T = 0.9
SEL_T = 0.15
N_PERM = 300

def _sym(c):
    m = re.match(r"^(.*?)\s*\(\d+\)$",c)
    return (m.group(1) if m else c).strip()

def pagerank(A,d=0.85,iters=200,tol=1e-9):
    n = A.shape[0]
    out = A.sum(1,keepdims=True)
    dangling = (out.ravel() == 0)
    T = np.divide(A,np.where(out == 0,1,out))
    r = np.full(n,1.0 / n)
    for _ in range(iters):
        rn = (1 - d) / n + d * (T.T @ r + r[dangling].sum() / n)
        if np.abs(rn - r).max() < tol:
            return rn
        r = rn
    return r

def cv_auc(X,y,seeds=(0,1,2,3,4),folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=s)
        oof = np.zeros(len(y))
        for tr,te in skf.split(X,y):
            mu,sd = X[tr].mean(0),X[tr].std(0)
            sd[sd == 0] = 1
            m = LogisticRegression(max_iter=3000).fit((X[tr] - mu) / sd,y[tr])
            oof[te] = m.predict_proba((X[te] - mu) / sd)[:,1]
        aucs.append(roc_auc_score(y,oof))
    return float(np.mean(aucs))

def main():
    from pdac_circuit.attractor.graph import build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality,load_essentiality

    g = build_regulatory_graph(max_nodes=400,coexpr_threshold=0.4,motif_edges=False,seed=20260620)
    nodes = g.nodes
    dat = np.load(GRN,allow_pickle=True)
    M = dat["M"]
    grn_nodes = list(dat["nodes"])
    gi = {x: i for i,x in enumerate(grn_nodes)}
    selm = np.array([gi[x] for x in nodes])
    M = M[np.ix_(selm,selm)]
    A = (M >= EDGE_T).astype(np.float64)
    import pandas as pd
    from pdac_circuit.attractor.graph import _depmap_expr_path
    from pdac_circuit.data.misc import _pdac_model_ids
    ep = _depmap_expr_path()
    hdr = pd.read_csv(ep,nrows=0).columns.tolist()
    idc = hdr[0]
    s2c = {}
    for c in hdr[1:]:
        s2c.setdefault(_sym(c),c)
    use = [idc] + [s2c[x] for x in nodes if x in s2c]
    raw = pd.read_csv(ep,usecols=use,index_col=0)
    raw.columns = [_sym(c) for c in raw.columns]
    raw = raw[raw.index.isin(set(_pdac_model_ids()))]
    raw_mean = np.nan_to_num(np.array([float(raw[x].mean()) if x in raw.columns else np.nan for x in nodes]))
    raw_var = np.nan_to_num(np.array([float(raw[x].var()) if x in raw.columns else np.nan for x in nodes]))
    feats = {
        "coexpr_degree": g.adjacency.sum(1),"eigenvector": _eigencentrality(g.adjacency),
        "out_strength": M.sum(1),"in_strength": M.sum(0),"out_degree": A.sum(1),
        "in_degree": A.sum(0),"pagerank": pagerank(A),"hits_hub": (A @ (A.sum(0))),
        "cna_amp": np.nan_to_num(g.cna_amp_freq),"cna_mean": np.nan_to_num(g.cna_mean),
        "methylation": np.nan_to_num(g.promoter_methylation,nan=0.05),
        "expr_mean_norm": g.states.mean(0),"expr_var_norm": g.states.var(0),
        "expr_mean_raw": raw_mean,"expr_var_raw": raw_var,
        "disease_log2fc": g.disease_log2fc,
    }
    ess = load_essentiality(nodes)
    seld = ess.get("sel",{})
    cov = [i for i,x in enumerate(nodes) if np.isfinite(seld.get(x,np.nan))]
    names = list(feats)
    Xall = np.column_stack([feats[k][cov] for k in names])
    y = (np.array([seld[nodes[i]] for i in cov]) > SEL_T).astype(int)

    obs = cv_auc(Xall,y)
    rng = np.random.default_rng(20260717)
    npos = int(y.sum())
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = np.zeros(len(y),dtype=int)
        yp[rng.choice(len(y),npos,replace=False)] = 1
        null[k] = cv_auc(Xall,yp)
    p = float((null >= obs).mean())
    rep = {
        "schema": "pdac-circuit.selective-null/1","data_class": "REAL",
        "sealed_studies_touched": False,
        "n_genes": len(cov),"n_selective_positive": npos,
        "observed_full_lr_cv_auc": round(float(obs),4),
        "null_permutations": N_PERM,
        "null_mean": round(float(null.mean()),4),"null_std": round(float(null.std()),4),
        "null_p95": round(float(np.percentile(null,95)),4),
        "perm_p_one_sided": round(p,4),
        "significant_at_0.05": bool(p < 0.05),
        "verdict": (
            f"REAL (underpowered): observed {obs:.3f} exceeds the permutation null "
            f"(mean {null.mean():.3f}, p95 {np.percentile(null,95):.3f}, p={p:.3f}); a multi-omic "
            f"model recovers PDAC-selective essentiality signal degree misses, though on only {npos} "
            f"positives so effect size is uncertain"
            if p < 0.05 else
            f"NOT SIGNIFICANT: observed {obs:.3f} is within the permutation null (mean {null.mean():.3f}, "
            f"p95 {np.percentile(null,95):.3f}, p={p:.3f}) -- with {npos} positives the apparent "
            f"selective ceiling is not distinguishable from chance; no feature set robustly recovers "
            f"selective vulnerability beyond degree, completing §15b"),
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(f"selective positives: {npos}/{len(cov)}")
    print(f"observed full-LR CV AUC: {obs:.4f}")
    print(f"permutation null: mean {null.mean():.4f}, std {null.std():.4f}, p95 {np.percentile(null,95):.4f}")
    print(f"perm p (one-sided): {p:.4f}  -> {'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")
    print(f"\nVERDICT: {rep['verdict']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
