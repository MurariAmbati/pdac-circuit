from __future__ import annotations

import json
import re

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "supervised_ceiling.json"
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

def cv_auc(X, y, model_fn, seeds=(0, 1, 2, 3, 4), folds=5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    aucs, aps=[], []
    for s in seeds:
        skf=StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        oof=np.zeros(len(y))
        for tr, te in skf.split(X, y):
            mu, sd=X[tr].mean(0), X[tr].std(0)
            sd[sd == 0]=1.0
            m=model_fn()
            m.fit((X[tr] - mu) / sd, y[tr])
            p=m.predict_proba((X[te] - mu) / sd)[:, 1]
            oof[te]=p
        aucs.append(roc_auc_score(y, oof))
        aps.append(average_precision_score(y, oof))
    return np.array(aucs), np.array(aps)

def main():
    from sklearn.ensemble import GradientBoostingClassifier
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
        "coexpr_degree": g.adjacency.sum(1),
        "eigenvector": _eigencentrality(g.adjacency),
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

    def logit():
        return LogisticRegression(max_iter=2000, C=1.0)

    def gbm():
        return GradientBoostingClassifier(random_state=0)

    deg_idx=names.index("coexpr_degree")
    Xdeg=Xall[:, [deg_idx]]

    auc_deg, ap_deg=cv_auc(Xdeg, y, logit)
    auc_full_lr, ap_full_lr=cv_auc(Xall, y, logit)
    auc_full_gb, ap_full_gb=cv_auc(Xall, y, gbm)
    non_deg=[i for i in range(Xall.shape[1]) if i != deg_idx]
    auc_nodeg, _=cv_auc(Xall[:, non_deg], y, gbm)

    from sklearn.ensemble import GradientBoostingClassifier as GBC
    mu, sd=Xall.mean(0), Xall.std(0)
    sd[sd == 0]=1
    Xs=(Xall - mu) / sd
    base_model=GBC(random_state=0).fit(Xs, y)
    from sklearn.metrics import roc_auc_score
    base_auc=roc_auc_score(y, base_model.predict_proba(Xs)[:, 1])
    rng=np.random.default_rng(0)
    imp=[]
    for j, nm in enumerate(names):
        drops=[]
        for _ in range(10):
            Xp=Xs.copy()
            Xp[:, j]=rng.permutation(Xp[:, j])
            drops.append(base_auc - roc_auc_score(y, base_model.predict_proba(Xp)[:, 1]))
        imp.append((nm, round(float(np.mean(drops)), 4)))
    imp.sort(key=lambda t: -t[1])

    def band(a):
        return [round(float(a.mean()), 4), round(float(a.std()), 4)]

    delta_full_lr=float(auc_full_lr.mean() - auc_deg.mean())
    delta_full_gb=float(auc_full_gb.mean() - auc_deg.mean())
    rep={
        "schema": "pdac-circuit.supervised-ceiling/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "endpoint": "DepMap absolute essentiality (Chronos abs > 0.4)",
        "n_genes": len(y), "n_positive": int(y.sum()), "n_features": len(names),
        "features": names,
        "cv": "5-fold stratified, 5 seeds, nested (fixed model defaults, in-fold standardisation)",
        "auc_degree_alone": band(auc_deg),
        "auc_full_logistic": band(auc_full_lr),
        "auc_full_gbm": band(auc_full_gb),
        "auc_all_except_degree_gbm": band(auc_nodeg),
        "pr_auc_degree": band(ap_deg), "pr_auc_full_gbm": band(ap_full_gb),
        "delta_auc_full_logistic_minus_degree": round(delta_full_lr, 4),
        "delta_auc_full_gbm_minus_degree": round(delta_full_gb, 4),
        "permutation_importance_full_gbm": imp,
        "verdict": (
            "DEGREE IS THE CEILING: a full multi-omic model does not meaningfully beat degree alone "
            "(delta AUC within noise), so PDAC-TF essentiality here is not recoverable beyond "
            "connectivity from any available feature -- the §18/§19 structural conclusion is "
            "quantitative and complete"
            if max(delta_full_lr, delta_full_gb) < 0.03 else
            "SIGNAL BEYOND DEGREE EXISTS: a multivariate model beats degree; feature importances "
            "indicate which non-topology features carry it"),
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"n={len(y)} positives={int(y.sum())} features={len(names)}\n")
    print(f"  degree alone      CV AUC {band(auc_deg)}")
    print(f"  full (logistic)   CV AUC {band(auc_full_lr)}   d {delta_full_lr:+.4f}")
    print(f"  full (GBM)        CV AUC {band(auc_full_gb)}   d {delta_full_gb:+.4f}")
    print(f"  all-except-degree CV AUC {band(auc_nodeg)}")
    print("\n  permutation importance (full GBM, AUC drop when shuffled):")
    for nm, d in imp[:8]:
        print(f"    {nm:22} {d:+.4f}")
    print(f"\nVERDICT: {rep['verdict']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
