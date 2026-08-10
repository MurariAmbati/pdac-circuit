from __future__ import annotations

import json
import time

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "selective_hint_dissection.json"
PRIMARY_NODES, PRIMARY_TAU, SEED = 400, 0.4, 20260620
PRIMARY_SEL_CUT=0.15
T0=time.time()

CURATED_SELECTIVE={"KLF5", "GATA6", "HNF1A", "HNF4A", "FOXA2", "FOXA1", "TP63", "RUNX3",
                     "MYBL2", "ELF3", "KLF4", "ONECUT2"}

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def auc(scores, labels):
    from scipy.stats import rankdata

    labels=np.asarray(labels, dtype=bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r=rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def matched_auc(score, deg, lab, caliper):
    pos, neg = np.flatnonzero(lab), np.flatnonzero(~lab)
    if pos.size == 0 or neg.size == 0:
        return None, 0
    mask=np.abs(deg[pos][:, None] - deg[neg][None, :]) <= caliper
    npr=int(mask.sum())
    if npr == 0:
        return None, 0
    w=(score[pos][:, None] > score[neg][None, :]).astype(float) + 0.5 * (score[pos][:, None] == score[neg][None, :])
    return float(w[mask].sum() / npr), npr

def perm_p(score, deg, lab, caliper, rng, n_perm=2000):
    obs, npr = matched_auc(score, deg, lab, caliper)
    if obs is None:
        return None, None, 0
    m, npos = lab.size, int(lab.sum())
    null=np.empty(n_perm)
    for k in range(n_perm):
        perm=np.zeros(m, dtype=bool)
        perm[rng.choice(m, npos, replace=False)]=True
        a, _ = matched_auc(score, deg, perm, caliper)
        null[k]=0.5 if a is None else a
    return obs, float((null >= obs).mean()), npr

def pctile(x):
    from scipy.stats import rankdata

    return (rankdata(x) - 1) / (len(x) - 1)

def main():
    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import build_regulatory_graph
    from pdac_circuit.attractor.run import load_essentiality

    log("fitting primary config (reproduces §15)")
    g=build_regulatory_graph(max_nodes=PRIMARY_NODES, coexpr_threshold=PRIMARY_TAU, motif_edges=False, seed=SEED)
    d=AttractorDynamics(g)
    d.fit(epochs=1500, motif_weight=0.0, seed=SEED)
    collapse=d.collapse_scores(per_line=True)
    ess=load_essentiality(g.nodes)

    nodes=g.nodes
    cov=[i for i, x in enumerate(nodes) if x in ess.get("abs", {})]
    names=np.array([nodes[i] for i in cov])
    c=collapse[cov]
    deg=g.adjacency.sum(axis=1)[cov]
    sel=np.array([ess["sel"][nodes[i]] for i in cov])
    abs_e=np.array([ess["abs"][nodes[i]] for i in cov])
    ok=np.isfinite(c) & np.isfinite(sel) & np.isfinite(abs_e)
    names, c, deg, sel, abs_e = names[ok], c[ok], deg[ok], sel[ok], abs_e[ok]
    log(f"genes: {c.size}; selective unconditioned AUC (cut {PRIMARY_SEL_CUT}) = "
        f"{auc(c, sel > PRIMARY_SEL_CUT):.3f}")

    spread=float(np.percentile(deg, 90) - np.percentile(deg, 10)) or 1.0
    caliper=0.25 * spread
    rng=np.random.default_rng(20260717)
    c_pct, d_pct = pctile(c), pctile(deg)

    sweep=[]
    for cut in (0.10, 0.125, 0.15, 0.175, 0.20):
        lab=sel > cut
        npos=int(lab.sum())
        if npos < 4 or npos > lab.size - 4:
            sweep.append({"cut": cut, "n_positive": npos, "matched_auc": None, "perm_p": None,
                          "note": "too few positives"})
            continue
        obs, p, npr = perm_p(c, deg, lab, caliper, rng)
        sweep.append({"cut": cut, "n_positive": npos, "matched_auc": None if obs is None else round(obs, 4),
                      "perm_p": None if p is None else round(p, 4), "n_matched_pairs": npr,
                      "unconditioned_auc": round(auc(c, lab), 4)})
        log(f"  cut {cut:.3f}: n_pos={npos} matched AUC {obs} p={p}")

    lab=sel > PRIMARY_SEL_CUT
    pos_idx=np.flatnonzero(lab)
    positives=[]
    for i in sorted(pos_idx, key=lambda j: -c_pct[j]):
        positives.append({
            "gene": str(names[i]),
            "selective_essentiality": round(float(sel[i]), 4),
            "absolute_essentiality": round(float(abs_e[i]), 4),
            "collapse_percentile": round(float(c_pct[i]), 3),
            "degree_percentile": round(float(d_pct[i]), 3),
            "curated_pdac_selective": bool(names[i] in CURATED_SELECTIVE),
            "informative_quadrant": bool(c_pct[i] >= 0.5 and d_pct[i] < 0.5),
        })

    base, _ = matched_auc(c, deg, lab, caliper)
    loo=[]
    for i in pos_idx:
        keep=np.ones(c.size, dtype=bool)
        keep[i]=False
        a, npr = matched_auc(c[keep], deg[keep], lab[keep], caliper)
        loo.append({"dropped": str(names[i]), "matched_auc_without": None if a is None else round(a, 4)})
    loo_vals=[x["matched_auc_without"] for x in loo if x["matched_auc_without"] is not None]

    n_informative=sum(p["informative_quadrant"] for p in positives)
    n_curated=sum(p["curated_pdac_selective"] for p in positives)

    verdict=_verdict(sweep, base, loo_vals, n_informative, len(positives), n_curated)
    rep={
        "schema": "pdac-circuit.selective-hint-dissection/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "config": {"nodes": PRIMARY_NODES, "tau": PRIMARY_TAU, "seed": SEED,
                   "primary_selective_cut": PRIMARY_SEL_CUT, "caliper": round(caliper, 3)},
        "primary_matched_auc": None if base is None else round(base, 4),
        "threshold_sensitivity": sweep,
        "positives_named": positives,
        "leave_one_positive_out": {"base_matched_auc": None if base is None else round(base, 4),
                                   "range_without_each": [round(min(loo_vals), 4), round(max(loo_vals), 4)] if loo_vals else None,
                                   "per_gene": loo},
        "degree_independent_standing": {
            "n_positives": len(positives), "n_in_informative_quadrant": int(n_informative),
            "n_curated_pdac_selective": int(n_curated),
            "note": "informative quadrant = high collapse (>=50th pct) AND low/mid degree (<50th pct); "
                    "only these can produce a degree-independent conditional signal"},
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n=== threshold sensitivity (degree-matched, caliper 0.25x spread) ===")
    for s in sweep:
        print(f"  cut {s['cut']:.3f}  n_pos={s['n_positive']:2}  "
              f"matched AUC {s['matched_auc']}  perm p {s['perm_p']}")
    print(f"\n=== positives at cut {PRIMARY_SEL_CUT} (collapse pct / degree pct) ===")
    for p in positives:
        flags=[]
        if p["curated_pdac_selective"]:
            flags.append("CURATED-PDAC")
        if p["informative_quadrant"]:
            flags.append("INFORMATIVE-QUADRANT")
        print(f"  {p['gene']:9} sel={p['selective_essentiality']:+.3f}  "
              f"collapse_pct={p['collapse_percentile']:.2f}  degree_pct={p['degree_percentile']:.2f}  "
              f"{' '.join(flags)}")
    print(f"\n=== leave-one-positive-out: base {base:.3f}, range without each "
          f"[{min(loo_vals):.3f}, {max(loo_vals):.3f}] ===" if loo_vals else "")
    print(f"informative-quadrant positives: {n_informative}/{len(positives)}; "
          f"curated PDAC-selective: {n_curated}/{len(positives)}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

def _verdict(sweep, base, loo_vals, n_informative, n_pos, n_curated):
    aucs=[s["matched_auc"] for s in sweep if s["matched_auc"] is not None]
    ps=[s["perm_p"] for s in sweep if s["perm_p"] is not None]
    stable=bool(aucs and (max(aucs) - min(aucs) <= 0.10))
    sig_any=bool(ps and min(ps) < 0.05)
    sig_most=bool(ps and sum(p < 0.05 for p in ps) >= max(1, len(ps) // 2))
    fragile=bool(loo_vals and base is not None and (base - min(loo_vals)) > 0.05)
    parts=[]
    parts.append("stable across thresholds" if stable else "threshold-sensitive")
    parts.append(f"significant at {sum(p<0.05 for p in ps)}/{len(ps)} cuts" if ps else "no p")
    parts.append("fragile to one gene" if fragile else "not driven by one gene")
    parts.append(f"{n_informative}/{n_pos} in informative quadrant")
    parts.append(f"{n_curated}/{n_pos} curated PDAC-selective")
    if stable and sig_most and not fragile and n_informative >= max(2, n_pos // 2):
        head="REAL-BUT-UNDERPOWERED: robust to threshold and LOO, biologically-standing positives"
    elif not stable or fragile or not sig_any:
        head="LIKELY ARTIFACT: does not survive its own robustness checks"
    else:
        head="INCONCLUSIVE: partial support, still exploratory"
    return head + " | " + "; ".join(parts)

if __name__ == "__main__":
    main()
