from __future__ import annotations

import json
import time

import numpy as np

from pdac_circuit.core.paths import RESULTS

OUT_NPZ=RESULTS / "directed_grn.npz"
OUT_JSON=RESULTS / "directed_grn.meta.json"
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def main():
    from pdac_circuit.attractor.graph import build_regulatory_graph
    from pdac_circuit.attractor.motif import load_jaspar_pwms, max_pwm_score, _promoter_seq

    log("building node set (same config as §15/§17: 400 nodes, tau 0.4)")
    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    nodes=g.nodes
    n=len(nodes)
    coexpr_degree=g.adjacency.sum(axis=1)

    pwms=load_jaspar_pwms()
    have_pwm=[i for i, x in enumerate(nodes) if x.upper() in pwms]
    log(f"{n} nodes; {len(have_pwm)} have a JASPAR PWM (potential regulators)")

    proms={}
    n_prom=0
    for j, x in enumerate(nodes):
        s=_promoter_seq(x)
        proms[j]=s
        n_prom += int(s is not None)
    log(f"{n_prom}/{n} nodes have an hg38 promoter sequence")

    M=np.zeros((n, n), dtype=np.float32)
    done=0
    for i in have_pwm:
        pwm=pwms[nodes[i].upper()]
        for j in range(n):
            if i == j:
                continue
            seq=proms.get(j)
            if seq:
                M[i, j]=max_pwm_score(pwm, seq)
        done += 1
        if done % 40 == 0:
            log(f"  scanned {done}/{len(have_pwm)} regulators; nonzero so far {int((M>0).sum())}")

    meta={
        "schema": "pdac-circuit.directed-grn/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "n_nodes": n, "n_regulators_with_pwm": len(have_pwm), "n_with_promoter": n_prom,
        "score_distribution": {
            "nonzero_entries": int((M > 0).sum()),
            "mean_nonzero": round(float(M[M > 0].mean()), 4) if (M > 0).any() else 0.0,
            "pctiles_of_nonzero": {p: round(float(np.percentile(M[M > 0], p)), 4)
                                   for p in (50, 75, 90, 95, 99)} if (M > 0).any() else {},
        },
        "edge_density_by_threshold": {
            str(t): int((M >= t).sum()) for t in (0.7, 0.8, 0.85, 0.9, 0.95)
        },
        "note": ("M[i,j] = best normalised JASPAR PWM hit of regulator i in target j's promoter; "
                 "directed TF->target; independent of DepMap CRISPR (leakage-free)."),
    }
    np.savez_compressed(OUT_NPZ, M=M, nodes=np.array(nodes, dtype=object),
                        coexpr_degree=coexpr_degree)
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"saved {OUT_NPZ} and {OUT_JSON}")
    print("\nscore pctiles (nonzero):", meta["score_distribution"]["pctiles_of_nonzero"])
    print("edge density by threshold:", meta["edge_density_by_threshold"])

if __name__ == "__main__":
    main()
