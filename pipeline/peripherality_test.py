from __future__ import annotations

import json
import re
import time

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "peripherality_test.json"
GRN=RESULTS / "directed_grn.npz"
EDGE_T=0.9
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

def cv_auc(X, y, seeds=(0, 1, 2, 3, 4), folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    out=[]
    for s in seeds:
        skf=StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        oof=np.zeros(len(y))
        for tr, te in skf.split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0]=1
            m=LogisticRegression(max_iter=3000)
            m.fit((X[tr] - mu) / sd, y[tr])
            oof[te]=m.predict_proba((X[te] - mu) / sd)[:, 1]
        out.append(roc_auc_score(y, oof))
    return np.array(out)

def perm_null(X, y, n_perm=400, rng=None):
    rng=rng or np.random.default_rng(0)
    npos=int(y.sum())
    m=len(y)
    null=np.empty(n_perm)
    for k in range(n_perm):
        yp=np.zeros(m, dtype=int)
        yp[rng.choice(m, npos, replace=False)]=1
        null[k]=cv_auc(X, yp, seeds=(0,)).mean()
    return null

def main():
    import pandas as pd
    from scipy.stats import rankdata, spearmanr

    from pdac_circuit.attractor.graph import _depmap_expr_path, build_regulatory_graph
    from pdac_circuit.attractor.run import _eigencentrality, load_essentiality
    from pdac_circuit.data.misc import _pdac_model_ids

    log("rebuilding graph + features (identical protocol to Phase 5)")
    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    nodes=g.nodes
    dat=np.load(GRN, allow_pickle=True)
    M=dat["M"]
    grn_nodes=list(dat["nodes"])
    gi={x: i for i, x in enumerate(grn_nodes)}
    sidx=np.array([gi[x] for x in nodes])
    M=M[np.ix_(sidx, sidx)]
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
    raw=full[full.index.isin(set(_pdac_model_ids()))]
    expr_mean_raw=np.nan_to_num(np.array([float(raw[x].mean()) if x in raw.columns else np.nan
                                            for x in nodes]))

    cent={
        "pagerank": pagerank(A), "out_degree": A.sum(1), "hits_hub": (A @ (A.sum(0))),
        "coexpr_degree": g.adjacency.sum(1), "eigenvector": _eigencentrality(g.adjacency),
    }

    def z(v):
        v=np.asarray(v, dtype=float)
        s=v.std()
        return (v - v.mean()) / (s if s else 1.0)

    centrality=np.mean([z(v) for v in cent.values()], axis=0)
    peripherality=-centrality

    ess=load_essentiality(nodes)
    absd=ess.get("abs", {})
    seld=ess.get("sel", {})
    cov=[i for i, x in enumerate(nodes)
           if x in absd and np.isfinite(absd[x]) and np.isfinite(seld.get(x, np.nan))]
    sel_e=np.array([seld[nodes[i]] for i in cov])
    per=peripherality[cov]
    emr=expr_mean_raw[cov]
    log(f"genes {len(cov)}")

    rho_pe, p_pe = spearmanr(per, emr)
    log(f"2 PROXY: rho(peripherality, expr_mean_raw) = {rho_pe:+.3f} (p={p_pe:.2g})")

    y=(sel_e > 0.15).astype(int)
    a_per=cv_auc(per.reshape(-1, 1), y)
    a_expr=cv_auc(emr.reshape(-1, 1), y)
    a_both=cv_auc(np.column_stack([emr, per]), y)
    increment=float(a_both.mean() - a_expr.mean())
    rx=rankdata(emr)
    def resid(v):
        vr=rankdata(v)
        b=np.polyfit(rx, vr, 1)
        return vr - (b[0] * rx + b[1])
    rho_par, p_par = spearmanr(resid(per), resid(sel_e))
    log(f"1 peripherality univariate AUC {a_per.mean():.4f} | 3 expr {a_expr.mean():.4f} -> "
        f"expr+per {a_both.mean():.4f} (increment {increment:+.4f}) | 4 partial rho {rho_par:+.3f} p={p_par:.3f}")

    rng=np.random.default_rng(20260717)
    sweep=[]
    for cut in (0.10, 0.125, 0.15, 0.175, 0.20):
        yc=(sel_e > cut).astype(int)
        npos=int(yc.sum())
        if npos < 4 or npos > len(yc) - 4:
            sweep.append({"cut": cut, "n_positive": npos, "note": "too few positives"})
            continue
        obs=float(cv_auc(per.reshape(-1, 1), yc).mean())
        null=perm_null(per.reshape(-1, 1), yc, n_perm=400, rng=rng)
        sweep.append({"cut": cut, "n_positive": npos, "observed_auc": round(obs, 4),
                      "null_mean": round(float(null.mean()), 4),
                      "null_p95": round(float(np.percentile(null, 95)), 4),
                      "perm_p": round(float((null >= obs).mean()), 4),
                      "obs_minus_null": round(obs - float(null.mean()), 4)})
        log(f"5 cut {cut:.3f}: n={npos} obs {obs:.3f} null {null.mean():.3f} p={sweep[-1]['perm_p']:.3f}")

    aucs=[s["observed_auc"] for s in sweep if "observed_auc" in s]
    ps=[s["perm_p"] for s in sweep if "perm_p" in s]
    stable=bool(aucs and (max(aucs) - min(aucs) <= 0.10))
    sig_any=bool(ps and min(ps) < 0.05)

    if abs(rho_pe) > 0.5 and increment <= 0.02:
        verdict=(f"EXPRESSION-PROXY: peripherality correlates with expression at rho={rho_pe:+.2f} "
                   f"and adds {increment:+.3f} over it -- it is the expression channel restated")
    elif increment > 0.02 and p_par < 0.05 and stable:
        verdict=(f"REAL-INDEPENDENT: peripherality adds {increment:+.3f} over expression, partial "
                   f"rho={rho_par:+.2f} (p={p_par:.3f}), effect stable across cuts")
    elif not stable:
        verdict="ARTIFACT: observed AUC bounces across label cuts (the §15b signature)"
    elif not sig_any:
        verdict=(f"UNDERPOWERED: effect stable but no cut reaches p<0.05; increment over expression "
                   f"{increment:+.3f}, partial p={p_par:.3f}")
    else:
        verdict=(f"PARTIAL: increment {increment:+.3f}, partial p={p_par:.3f}, stable={stable}, "
                   f"sig at {sum(p<0.05 for p in ps)}/{len(ps)} cuts")

    rep={
        "schema": "pdac-circuit.peripherality/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "protocol": "identical to Phase 5 (nested CV, 5 seeds, StratifiedKFold-5, in-fold standardisation)",
        "n_genes": len(cov),
        "centralities_averaged": list(cent),
        "proxy_check": {"spearman_peripherality_vs_expr_mean_raw": round(float(rho_pe), 4),
                        "p": float(p_pe)},
        "primary_cut_0.15": {
            "n_positive": int(y.sum()),
            "auc_peripherality": [round(float(a_per.mean()), 4), round(float(a_per.std()), 4)],
            "auc_expr_only": [round(float(a_expr.mean()), 4), round(float(a_expr.std()), 4)],
            "auc_expr_plus_peripherality": [round(float(a_both.mean()), 4), round(float(a_both.std()), 4)],
            "increment_over_expression": round(increment, 4),
            "partial_spearman_given_expr": [round(float(rho_par), 4), round(float(p_par), 4)]},
        "threshold_sweep": sweep,
        "effect_stable_across_cuts": stable,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"proxy check: rho(peripherality, expr_mean_raw) = {rho_pe:+.3f}")
    print(f"peripherality alone {a_per.mean():.4f} | expr {a_expr.mean():.4f} -> "
          f"expr+per {a_both.mean():.4f}  (increment {increment:+.4f})")
    print(f"partial rho given expr: {rho_par:+.3f} (p={p_par:.3f})")
    print("\ncut     n   obs    null   p")
    for s in sweep:
        if "observed_auc" in s:
            print(f"{s['cut']:.3f} {s['n_positive']:3}  {s['observed_auc']:.3f}  "
                  f"{s['null_mean']:.3f}  {s['perm_p']:.3f}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
