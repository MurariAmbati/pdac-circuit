from __future__ import annotations

import json
import re
import time

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "selective_confound_test.json"
GRN=RESULTS / "directed_grn.npz"
EDGE_T=0.9
SEL_T=0.15
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

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

def main():
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from pdac_circuit.attractor.graph import _depmap_expr_path, build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality, load_essentiality
    from pdac_circuit.data.misc import _pdac_model_ids

    def lr():
        return LogisticRegression(max_iter=3000)

    def gbm():
        return GradientBoostingClassifier(random_state=0)

    log("rebuilding graph + features (identical to selective_ceiling.py)")
    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    nodes=g.nodes
    dat=np.load(GRN, allow_pickle=True)
    M=dat["M"]
    grn_nodes=list(dat["nodes"])
    gi={x: i for i, x in enumerate(grn_nodes)}
    selidx=np.array([gi[x] for x in nodes])
    M=M[np.ix_(selidx, selidx)]
    A=(M >= EDGE_T).astype(np.float64)

    ep=_depmap_expr_path()
    hdr=pd.read_csv(ep, nrows=0).columns.tolist()
    idc=hdr[0]
    s2c={}
    for c in hdr[1:]:
        s2c.setdefault(_sym(c), c)
    use=[idc] + [s2c[x] for x in nodes if x in s2c]
    full=pd.read_csv(ep, usecols=use, index_col=0)
    full.columns=[_sym(c) for c in full.columns]
    pdac_ids=set(_pdac_model_ids())
    is_pdac=full.index.isin(pdac_ids)
    raw=full[is_pdac]
    other=full[~is_pdac]
    raw_mean=np.array([float(raw[x].mean()) if x in raw.columns else np.nan for x in nodes])
    raw_var=np.array([float(raw[x].var()) if x in raw.columns else np.nan for x in nodes])
    expr_diff=np.array([
        float(raw[x].mean() - other[x].mean()) if x in raw.columns and x in other.columns else np.nan
        for x in nodes])
    log(f"expr differential built over {int(is_pdac.sum())} PDAC vs {int((~is_pdac).sum())} other lines")

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
    EXPRESSION_FEATURES=["expr_mean_norm", "expr_var_norm", "expr_mean_raw", "expr_var_raw",
                           "disease_log2fc_LEVEL"]

    ess=load_essentiality(nodes)
    absd=ess.get("abs", {})
    seld=ess.get("sel", {})
    cov=[i for i, x in enumerate(nodes)
           if x in absd and np.isfinite(absd[x]) and np.isfinite(seld.get(x, np.nan))]
    names=list(feats)
    Xall=np.column_stack([feats[k][cov] for k in names])
    sel_e=np.array([seld[nodes[i]] for i in cov])
    y_sel=(sel_e > SEL_T).astype(int)
    ediff=np.nan_to_num(expr_diff[cov])
    log(f"genes {len(cov)}, selective positives {int(y_sel.sum())}")

    a_full_lr=cv_auc(Xall, y_sel, lr)
    a_full_gb=cv_auc(Xall, y_sel, gbm)
    log(f"1 REPRODUCE full model: logistic {a_full_lr.mean():.4f} (expect 0.651), "
        f"gbm {a_full_gb.mean():.4f} (expect 0.683)")
    reproduced=abs(a_full_lr.mean() - 0.651) < 0.02

    uni={}
    for k in names:
        a=cv_auc(np.asarray(feats[k])[cov].reshape(-1, 1), y_sel, lr)
        uni[k]=round(float(a.mean()), 4)
    a_ediff=cv_auc(ediff.reshape(-1, 1), y_sel, lr)
    uni["expr_pdac_minus_other_DIFFERENTIAL"]=round(float(a_ediff.mean()), 4)
    ranked=sorted(uni.items(), key=lambda kv: -kv[1])
    log("2+3 univariate-on-selective (top 6): " + ", ".join(f"{k}={v:.3f}" for k, v in ranked[:6]))

    keep=[i for i, k in enumerate(names) if k not in EXPRESSION_FEATURES]
    Xabl=Xall[:, keep]
    a_abl_lr=cv_auc(Xabl, y_sel, lr)
    a_abl_gb=cv_auc(Xabl, y_sel, gbm)
    log(f"4 ABLATION (no expression, {len(keep)} feats): logistic {a_abl_lr.mean():.4f}, "
        f"gbm {a_abl_gb.mean():.4f}")

    Xplus=np.column_stack([Xall, ediff])
    a_plus_lr=cv_auc(Xplus, y_sel, lr)

    ed=uni["expr_pdac_minus_other_DIFFERENTIAL"]
    best_expr_uni=max(uni[k] for k in EXPRESSION_FEATURES + ["expr_pdac_minus_other_DIFFERENTIAL"])
    if ed >= 0.62:
        verdict=("TAUTOLOGY-DOMINATED: the PDAC-vs-other expression differential alone reaches "
                   f"{ed:.3f}, i.e. 'selectively expressed => selectively essential' carries the claim")
    elif a_abl_lr.mean() >= 0.60:
        verdict=(f"SURVIVES ABLATION: with every expression feature removed the model still scores "
                   f"{a_abl_lr.mean():.3f}, so the selective signal is not an expression artifact")
    elif a_abl_lr.mean() <= 0.55:
        verdict=(f"EXPRESSION-CARRIED: ablating expression collapses the model to "
                   f"{a_abl_lr.mean():.3f}; the one positive of the investigation is carried by "
                   f"expression and must be re-described as such")
    else:
        verdict=(f"PARTIAL: ablated model {a_abl_lr.mean():.3f} -- expression carries most but not "
                   f"all of the selective signal")

    rep={
        "schema": "pdac-circuit.selective-confound/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "protocol": "identical to selective_ceiling.py (nested CV, 5 seeds, StratifiedKFold-5, in-fold standardisation)",
        "n_genes": len(cov), "n_selective_positive": int(y_sel.sum()), "selective_cut": SEL_T,
        "reproduce": {"full_logistic": [round(float(a_full_lr.mean()), 4), round(float(a_full_lr.std()), 4)],
                      "full_gbm": [round(float(a_full_gb.mean()), 4), round(float(a_full_gb.std()), 4)],
                      "expected_logistic": 0.651, "protocol_reproduced": bool(reproduced)},
        "univariate_on_selective": dict(ranked),
        "expression_differential_univariate": ed,
        "ablation_no_expression": {
            "n_features": len(keep), "features": [names[i] for i in keep],
            "logistic": [round(float(a_abl_lr.mean()), 4), round(float(a_abl_lr.std()), 4)],
            "gbm": [round(float(a_abl_gb.mean()), 4), round(float(a_abl_gb.std()), 4)]},
        "full_plus_differential_logistic": round(float(a_plus_lr.mean()), 4),
        "best_expression_univariate": best_expr_uni,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"reproduce full logistic {a_full_lr.mean():.4f} (expected 0.651) -> "
          f"{'OK' if reproduced else 'PROTOCOL MISMATCH'}")
    print("\nunivariate CV-AUC on SELECTIVE endpoint:")
    for k, v in ranked:
        tag="  <-- EXPRESSION" if k in EXPRESSION_FEATURES or "DIFFERENTIAL" in k else ""
        print(f"   {k:38} {v:.4f}{tag}")
    print(f"\nablation (expression removed, {len(keep)} feats): logistic {a_abl_lr.mean():.4f}, "
          f"gbm {a_abl_gb.mean():.4f}")
    print(f"full + differential: {a_plus_lr.mean():.4f}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
