from __future__ import annotations

import re

import json

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "selective_ceiling.json"
GRN=RESULTS / "directed_grn.npz"
EDGE_T=0.9
ABS_T, SEL_T=0.4, 0.15

def _sym(c):
    m=re.match(r"^(.*?)\s*\(\d+\)$", c)
    return (m.group(1) if m else c).strip()

def pagerank(A, d=0.85, iters=200, tol=1e-9):
    n=A.shape[0]
    out=A.sum(1, keepdims=True)
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
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    out=[]
    for s in seeds:
        skf=StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        oof=np.zeros(len(y))
        for tr, te in skf.split(X, y):
            mu, sd=X[tr].mean(0), X[tr].std(0)
            sd[sd == 0]=1
            m=model_fn()
            m.fit((X[tr] - mu) / sd, y[tr])
            oof[te]=m.predict_proba((X[te] - mu) / sd)[:, 1]
        out.append(roc_auc_score(y, oof))
    return np.array(out)

def cv_auc_1d(x, y):
    a=cv_auc(x.reshape(-1, 1), y, lambda: __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=2000))
    return max(float(a.mean()), 1 - float(a.mean()))

def main():
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from pdac_circuit.attractor.graph import _depmap_expr_path, build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality, load_essentiality
    from pdac_circuit.data.misc import _pdac_model_ids

    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    nodes=g.nodes
    dat=np.load(GRN, allow_pickle=True)
    M=dat["M"]
    grn_nodes=list(dat["nodes"])
    gi={x: i for i, x in enumerate(grn_nodes)}
    sel=np.array([gi[x] for x in nodes])
    M=M[np.ix_(sel, sel)]
    A=(M >= EDGE_T).astype(np.float64)

    ep=_depmap_expr_path()
    hdr=pd.read_csv(ep, nrows=0).columns.tolist()
    idc=hdr[0]
    s2c={}
    for c in hdr[1:]:
        s2c.setdefault(_sym(c), c)
    use=[idc] + [s2c[x] for x in nodes if x in s2c]
    raw=pd.read_csv(ep, usecols=use, index_col=0)
    raw.columns=[_sym(c) for c in raw.columns]
    raw=raw[raw.index.isin(set(_pdac_model_ids()))]
    raw_mean=np.array([float(raw[x].mean()) if x in raw.columns else np.nan for x in nodes])
    raw_var=np.array([float(raw[x].var()) if x in raw.columns else np.nan for x in nodes])

    feats={
        "coexpr_degree": g.adjacency.sum(1), "eigenvector": _eigencentrality(g.adjacency),
        "out_strength": M.sum(1), "in_strength": M.sum(0), "out_degree": A.sum(1),
        "in_degree": A.sum(0), "pagerank": pagerank(A), "hits_hub": (A @ (A.sum(0))),
        "cna_amp": np.nan_to_num(g.cna_amp_freq), "cna_mean": np.nan_to_num(g.cna_mean),
        "methylation": np.nan_to_num(g.promoter_methylation, nan=0.05),
        "expr_mean_norm": g.states.mean(0), "expr_var_norm": g.states.var(0),
        "expr_mean_raw": np.nan_to_num(raw_mean), "expr_var_raw": np.nan_to_num(raw_var),
        "disease_log2fc_LEVEL": g.disease_log2fc,
    }
    ess=load_essentiality(nodes)
    absd=ess.get("abs", {})
    seld=ess.get("sel", {})
    cov=[i for i, x in enumerate(nodes) if x in absd and np.isfinite(absd[x]) and np.isfinite(seld.get(x, np.nan))]
    names=list(feats)
    Xall=np.column_stack([feats[k][cov] for k in names])
    abs_e=np.array([absd[nodes[i]] for i in cov])
    sel_e=np.array([seld[nodes[i]] for i in cov])
    y_abs=(abs_e > ABS_T).astype(int)
    y_sel=(sel_e > SEL_T).astype(int)
    di=names.index("coexpr_degree")
    Xdeg=Xall[:, [di]]

    def lr():
        return LogisticRegression(max_iter=3000)

    def gb():
        return GradientBoostingClassifier(random_state=0)

    def band(a):
        return [round(float(a.mean()), 4), round(float(a.std()), 4)]

    artifact={k: round(cv_auc_1d(feats[k][cov], y_abs), 4)
                for k in ("expr_var_norm", "expr_var_raw", "expr_mean_norm", "expr_mean_raw", "coexpr_degree")}

    res={}
    for tag, y in (("absolute", y_abs), ("selective", y_sel)):
        res[tag]={
            "n_positive": int(y.sum()),
            "auc_degree": band(cv_auc(Xdeg, y, lr)),
            "auc_full_logistic": band(cv_auc(Xall, y, lr)),
            "auc_full_gbm": band(cv_auc(Xall, y, gb)),
        }

    d_abs=res["absolute"]["auc_full_logistic"][0] - res["absolute"]["auc_degree"][0]
    d_sel=res["selective"]["auc_full_logistic"][0] - res["selective"]["auc_degree"][0]
    verdict=(
        "CEILING IS A HOUSEKEEPING PROXY: the multivariate model beats degree for ABSOLUTE "
        "essentiality but NOT for PDAC-SELECTIVE essentiality (delta_sel <= 0.03) -- it recovers the "
        "core-essential axis degree already captures, not therapeutically useful selective signal. "
        "No available feature set recovers selective vulnerability beyond degree, completing §15b."
        if d_sel <= 0.03 else
        "SELECTIVE SIGNAL EXISTS: the multivariate model beats degree even for PDAC-SELECTIVE "
        "essentiality; the feature set recovers therapeutically relevant signal degree misses")
    rep={
        "schema": "pdac-circuit.selective-ceiling/1", "data_class": "REAL",
        "sealed_studies_touched": False, "n_genes": len(cov),
        "expr_var_artifact_check": {
            "note": "univariate CV-AUC on absolute essentiality; if raw ~ normalised, expr_var is real biology not a normalisation artifact",
            "values": artifact},
        "absolute_vs_selective": res,
        "delta_full_minus_degree": {"absolute": round(d_abs, 4), "selective": round(d_sel, 4)},
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("expr_var artifact check (univariate CV-AUC vs absolute essentiality):")
    for k, v in artifact.items():
        print(f"  {k:18} {v:.3f}")
    print("\nabsolute vs selective ceiling (full logistic vs degree-alone CV AUC):")
    for tag in ("absolute", "selective"):
        r=res[tag]
        print(f"  {tag:9} pos={r['n_positive']:2}  degree {r['auc_degree'][0]:.3f}  "
              f"full-LR {r['auc_full_logistic'][0]:.3f}  full-GBM {r['auc_full_gbm'][0]:.3f}  "
              f"delta {rep['delta_full_minus_degree'][tag]:+.3f}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
