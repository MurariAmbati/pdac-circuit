from __future__ import annotations

import json
import re

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "selective_robustness.json"
GRN=RESULTS / "directed_grn.npz"
EDGE_T=0.9
CUTS=[0.10, 0.125, 0.15, 0.175, 0.20]
N_PERM=200
SEEDS=(0, 1, 2)

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

def cv_auc(X, y, seeds=SEEDS, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    aucs=[]
    for s in seeds:
        skf=StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        oof=np.zeros(len(y))
        for tr, te in skf.split(X, y):
            mu, sd=X[tr].mean(0), X[tr].std(0)
            sd[sd == 0]=1
            m=LogisticRegression(max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
            oof[te]=m.predict_proba((X[te] - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))

def main():
    import pandas as pd
    from pdac_circuit.attractor.graph import _depmap_expr_path, build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality, load_essentiality
    from pdac_circuit.data.misc import _pdac_model_ids

    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    nodes=g.nodes
    dat=np.load(GRN, allow_pickle=True)
    M=dat["M"]
    grn_nodes=list(dat["nodes"])
    gi={x: i for i, x in enumerate(grn_nodes)}
    selm=np.array([gi[x] for x in nodes])
    M=M[np.ix_(selm, selm)]
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
    raw_mean=np.nan_to_num(np.array([float(raw[x].mean()) if x in raw.columns else np.nan for x in nodes]))
    raw_var=np.nan_to_num(np.array([float(raw[x].var()) if x in raw.columns else np.nan for x in nodes]))
    feats={
        "coexpr_degree": g.adjacency.sum(1), "eigenvector": _eigencentrality(g.adjacency),
        "out_strength": M.sum(1), "in_strength": M.sum(0), "out_degree": A.sum(1),
        "in_degree": A.sum(0), "pagerank": pagerank(A), "hits_hub": (A @ (A.sum(0))),
        "cna_amp": np.nan_to_num(g.cna_amp_freq), "cna_mean": np.nan_to_num(g.cna_mean),
        "methylation": np.nan_to_num(g.promoter_methylation, nan=0.05),
        "expr_mean_norm": g.states.mean(0), "expr_var_norm": g.states.var(0),
        "expr_mean_raw": raw_mean, "expr_var_raw": raw_var, "disease_log2fc": g.disease_log2fc,
    }
    ess=load_essentiality(nodes)
    seld=ess.get("sel", {})
    cov=[i for i, x in enumerate(nodes) if np.isfinite(seld.get(x, np.nan))]
    Xall=np.column_stack([feats[k][cov] for k in feats])
    sel_e=np.array([seld[nodes[i]] for i in cov])

    rng=np.random.default_rng(20260717)
    rows=[]
    for cut in CUTS:
        y=(sel_e > cut).astype(int)
        npos=int(y.sum())
        if npos < 6 or npos > len(y) - 6:
            rows.append({"cut": cut, "n_positive": npos, "note": "too few/many positives"})
            continue
        obs=cv_auc(Xall, y)
        null=np.empty(N_PERM)
        for k in range(N_PERM):
            yp=np.zeros(len(y), dtype=int)
            yp[rng.choice(len(y), npos, replace=False)]=1
            null[k]=cv_auc(Xall, yp)
        p=float((null >= obs).mean())
        rows.append({"cut": cut, "n_positive": npos, "observed_auc": round(obs, 4),
                     "null_mean": round(float(null.mean()), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
                     "perm_p": round(p, 4), "significant": bool(p < 0.05)})
        print(f"  cut {cut:.3f}: n_pos={npos:2} obs {obs:.3f} null_mean {null.mean():.3f} "
              f"p95 {np.percentile(null,95):.3f} p={p:.3f} {'SIG' if p<0.05 else 'ns'}", flush=True)

    sig=[r for r in rows if r.get("significant")]
    n_tested=sum(1 for r in rows if "perm_p" in r)
    verdict=(
        f"ROBUST: significant at {len(sig)}/{n_tested} selective cuts -- the supervised selective "
        f"signal is not a single-threshold artifact (unlike the collapse hint, §15b), though it "
        f"remains underpowered (n_pos 6-20) and marginal"
        if len(sig) >= max(2, n_tested - 1) else
        f"FRAGILE / THRESHOLD-SENSITIVE: significant at only {len(sig)}/{n_tested} cuts -- like the "
        f"collapse selective hint (§15b), the supervised selective signal does not survive threshold "
        f"variation and should be treated as an artifact, not a finding")
    rep={
        "schema": "pdac-circuit.selective-robustness/1", "data_class": "REAL",
        "sealed_studies_touched": False, "n_genes": len(cov),
        "cuts_tested": CUTS, "n_perm": N_PERM, "seeds": list(SEEDS),
        "per_cut": rows, "n_significant": len(sig), "n_tested": n_tested, "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
