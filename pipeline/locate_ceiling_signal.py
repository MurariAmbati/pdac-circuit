from __future__ import annotations

import json
import re

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "locate_ceiling_signal.json"
GRN=RESULTS / "directed_grn.npz"
PRIMARY_THRESHOLD=0.4
EDGE_T=0.9

def _sym(c):
    m=re.match(r"^(.*?)\s*\(\d+\)$", c)
    return (m.group(1) if m else c).strip()

def pagerank(A, d=0.85, iters=200, tol=1e-9):
    n=A.shape[0]
    out=A.sum(axis=1, keepdims=True)
    dangling=(out.ravel() == 0)
    T=np.divide(A, np.where(out == 0, 1, out))
    r=np.full(n, 1.0 / n)
    for _ in range(iters):
        rn=(1 - d) / n + d * (T.T @ r + r[dangling].sum() / n)
        if np.abs(rn - r).max() < tol:
            return rn
        r=rn
    return r

def cv_auc_1d(x, y, seeds=(0, 1, 2, 3, 4), folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    aucs=[]
    X=x.reshape(-1, 1)
    for s in seeds:
        skf=StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        oof=np.zeros(len(y))
        for tr, te in skf.split(X, y):
            mu, sd=X[tr].mean(), X[tr].std() or 1.0
            m=LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
            oof[te]=m.predict_proba((X[te] - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return max(float(np.mean(aucs)), 1 - float(np.mean(aucs)))

def main():
    from sklearn.linear_model import LogisticRegression

    from pdac_circuit.attractor.graph import build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality, load_essentiality

    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    nodes=g.nodes
    dat=np.load(GRN, allow_pickle=True)
    M=dat["M"]
    grn_nodes=list(dat["nodes"])
    gi={x: i for i, x in enumerate(grn_nodes)}
    sel=np.array([gi[x] for x in nodes])
    M=M[np.ix_(sel, sel)]
    A=(M >= EDGE_T).astype(np.float64)
    feats={
        "coexpr_degree": g.adjacency.sum(1), "eigenvector": _eigencentrality(g.adjacency),
        "out_strength": M.sum(1), "in_strength": M.sum(0),
        "out_degree": A.sum(1), "in_degree": A.sum(0),
        "pagerank": pagerank(A), "hits_hub": (A @ (A.sum(0))),
        "cna_amp": np.nan_to_num(g.cna_amp_freq), "cna_mean": np.nan_to_num(g.cna_mean),
        "methylation": np.nan_to_num(g.promoter_methylation, nan=0.05),
        "expr_mean": g.states.mean(0), "expr_var": g.states.var(0),
        "disease_log2fc_LEVEL": g.disease_log2fc,
    }
    ess=load_essentiality(nodes)
    absd=ess.get("abs", {})
    cov=[i for i, x in enumerate(nodes) if x in absd and np.isfinite(absd[x])]
    y=(np.array([absd[nodes[i]] for i in cov]) > PRIMARY_THRESHOLD).astype(int)
    names=list(feats)
    Xall=np.column_stack([feats[k][cov] for k in names])

    uni=sorted(((k, round(cv_auc_1d(feats[k][cov], y), 4)) for k in names),
                 key=lambda t: -t[1])
    mu, sd=Xall.mean(0), Xall.std(0)
    sd[sd == 0]=1
    lr=LogisticRegression(max_iter=5000, C=1.0).fit((Xall - mu) / sd, y)
    coefs=sorted(zip(names, [round(float(c), 3) for c in lr.coef_[0]]),
                   key=lambda t: -abs(t[1]))

    rep={
        "schema": "pdac-circuit.locate-ceiling/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "n_genes": len(y), "n_positive": int(y.sum()),
        "univariate_cv_auc_direction_agnostic": uni,
        "standardised_logistic_coefficients": coefs,
        "note": ("univariate AUC is direction-agnostic (a feature anti-correlated with essentiality "
                 "is as informative); coefficients are on standardised features from a single full fit"),
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"n={len(y)} positives={int(y.sum())}\n\nunivariate CV-AUC per feature (direction-agnostic):")
    for k, a in uni:
        print(f"  {k:22} {a:.3f}")
    print("\nstandardised logistic coefficients (|top|):")
    for k, c in coefs[:8]:
        print(f"  {k:22} {c:+.3f}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
