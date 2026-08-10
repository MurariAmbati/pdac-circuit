from __future__ import annotations

import json
import re

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT = RESULTS / "substrate_signal_test.json"
GRN = RESULTS / "directed_grn.npz"
PRIMARY_THRESHOLD = 0.4
EDGE_T = 0.9

def auc(scores, labels):
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def matched_concordance(score, deg, lab, caliper, rng, n_perm=2000):
    pos, neg = np.flatnonzero(lab), np.flatnonzero(~lab)
    if pos.size == 0 or neg.size == 0:
        return None
    mask = np.abs(deg[pos][:, None] - deg[neg][None, :]) <= caliper
    npr = int(mask.sum())
    if npr == 0:
        return None
    sp, sn = score[pos][:, None], score[neg][None, :]
    obs = float(((sp > sn).astype(float) + 0.5 * (sp == sn))[mask].sum() / npr)
    m, npos = lab.size, int(lab.sum())
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.zeros(m, dtype=bool)
        perm[rng.choice(m, npos, replace=False)] = True
        pp, nn = np.flatnonzero(perm), np.flatnonzero(~perm)
        dm = np.abs(deg[pp][:, None] - deg[nn][None, :]) <= caliper
        q = int(dm.sum())
        if q == 0:
            null[k] = 0.5
            continue
        null[k] = (((score[pp][:, None] > score[nn][None, :]).astype(float)
                    + 0.5 * (score[pp][:, None] == score[nn][None, :]))[dm].sum() / q)
    return {"matched_auc": round(obs, 4), "n_pairs": npr,
            "perm_p": round(float((null >= obs).mean()), 4)}

def partial_spearman(x, y, z):
    from scipy.stats import rankdata, spearmanr

    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.column_stack([b, np.ones_like(b)])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex = resid(rx, rz[:, None])
    ey = resid(ry, rz[:, None])
    rho, p = spearmanr(ex, ey)
    return round(float(rho), 4), round(float(p), 4)

def pagerank(A, d=0.85, iters=200, tol=1e-9):
    n = A.shape[0]
    out = A.sum(axis=1, keepdims=True)
    dangling = (out.ravel() == 0)
    T = np.divide(A, np.where(out == 0, 1, out))
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        rn = (1 - d) / n + d * (T.T @ r + r[dangling].sum() / n)
        if np.abs(rn - r).max() < tol:
            r = rn
            break
        r = rn
    return r

def hits(A, iters=200, tol=1e-9):
    a = np.full(A.shape[0], 1.0 / A.shape[0])
    h = a.copy()
    for _ in range(iters):
        a_new = A.T @ h
        a_new /= (np.linalg.norm(a_new) or 1)
        h_new = A @ a_new
        h_new /= (np.linalg.norm(h_new) or 1)
        if np.abs(a_new - a).max() < tol and np.abs(h_new - h).max() < tol:
            a, h = a_new, h_new
            break
        a, h = a_new, h_new
    return h, a

def _sym(c):
    m = re.match(r"^(.*?)\s*\(\d+\)$", c)
    return (m.group(1) if m else c).strip()

def main():
    from pdac_circuit.attractor.run import load_essentiality

    dat = np.load(GRN, allow_pickle=True)
    M = dat["M"]
    nodes = list(dat["nodes"])
    coexpr_degree = dat["coexpr_degree"]

    A = (M >= EDGE_T).astype(np.float64)
    out_strength = M.sum(axis=1)
    in_strength = M.sum(axis=0)
    out_degree = A.sum(axis=1)
    in_degree = A.sum(axis=0)
    pr = pagerank(A)
    hub, auth = hits(A)

    ess = load_essentiality(nodes)
    absd = ess.get("abs", {})
    cov = [i for i, x in enumerate(nodes) if x in absd]
    abs_e = np.array([absd[nodes[i]] for i in cov])
    ok = np.isfinite(abs_e)
    idx = np.array(cov)[ok]
    abs_e = abs_e[ok]
    lab = abs_e > PRIMARY_THRESHOLD

    props = {
        "coexpr_degree_BASELINE": coexpr_degree[idx],
        "out_strength": out_strength[idx],
        "out_degree@0.9": out_degree[idx],
        "in_strength": in_strength[idx],
        "in_degree@0.9": in_degree[idx],
        "directed_pagerank": pr[idx],
        "hits_hub": hub[idx],
        "hits_authority": auth[idx],
    }
    base_deg = props["coexpr_degree_BASELINE"]
    spread = float(np.percentile(base_deg, 90) - np.percentile(base_deg, 10)) or 1.0
    caliper = 0.25 * spread
    rng = np.random.default_rng(20260717)

    auc_base = auc(base_deg, lab)
    rows = []
    for name, v in props.items():
        r = {"property": name, "raw_auc": round(auc(v, lab), 4)}
        if name != "coexpr_degree_BASELINE":
            mc = matched_concordance(v, base_deg, lab, caliper, rng)
            r["degree_matched"] = mc
            pr_rho, pr_p = partial_spearman(v, abs_e, base_deg)
            r["partial_spearman_given_coexpr_degree"] = {"rho": pr_rho, "p": pr_p}
            r["beats_degree"] = bool(mc and mc["matched_auc"] > 0.55 and mc["perm_p"] < 0.05
                                     and pr_p < 0.05)
        rows.append(r)

    gate_pass = [r["property"] for r in rows if r.get("beats_degree")]
    rep = {
        "schema": "pdac-circuit.substrate-signal/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "endpoint": "DepMap absolute essentiality (Chronos abs > 0.4)",
        "n_genes": int(idx.size), "n_positive": int(lab.sum()),
        "edge_threshold": EDGE_T, "coexpr_degree_auc": round(auc_base, 4),
        "gate_criterion": "matched AUC > 0.55 AND perm p < 0.05 AND partial p < 0.05 vs co-expr degree",
        "properties_passing_gate": gate_pass,
        "decision": ("PROCEED to Phase 3 (dynamics) — directed substrate carries signal beyond degree: "
                     + ", ".join(gate_pass)) if gate_pass else
                    ("STOP — no directed GRN property beats undirected co-expression degree after "
                     "controlling for it; the directed substrate also does not encode essentiality "
                     "beyond degree, strengthening the §18 structural conclusion"),
        "per_property": rows,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"n={idx.size}, positives={int(lab.sum())}, co-expr degree AUC={auc_base:.4f}\n")
    print(f"{'property':26} {'rawAUC':>7} {'matchedAUC':>11} {'permp':>7} {'partialrho':>11} {'p':>7} beats?")
    for r in rows:
        if r["property"] == "coexpr_degree_BASELINE":
            print(f"{r['property']:26} {r['raw_auc']:>7.3f} {'(baseline)':>11}")
            continue
        mc = r.get("degree_matched") or {}
        ps = r.get("partial_spearman_given_coexpr_degree") or {}
        print(f"{r['property']:26} {r['raw_auc']:>7.3f} {mc.get('matched_auc',float('nan')):>11.3f} "
              f"{mc.get('perm_p',float('nan')):>7.3f} {ps.get('rho',float('nan')):>11.3f} "
              f"{ps.get('p',float('nan')):>7.3f} {'YES' if r.get('beats_degree') else 'no'}")
    print(f"\nDECISION: {rep['decision']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
