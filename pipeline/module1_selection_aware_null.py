from __future__ import annotations

import json

import numpy as np

from pdac_circuit.core.paths import RESULTS
from pdac_circuit.targeting.features import build_afm
from pdac_circuit.targeting.prioritize import CRITERIA, _prereg, _simplex_weights, criteria_matrix

OUT = RESULTS / "module1_selection_aware_null.json"
B = 2000
SEED = 20260620

def main():
    pre = _prereg()
    k = int(pre.get("recovery_at_k", 10))
    afm = build_afm()
    df = afm.table
    pool = df[df["frac_expressed"] >= 0.30].copy()
    C = criteria_matrix(pool)
    X = C[CRITERIA].to_numpy(dtype=float)
    cmask = pool["is_control"].to_numpy(dtype=bool)
    n, n_ctrl = X.shape[0], int(cmask.sum())
    if not np.isfinite(X).all():
        raise ValueError("non-finite criteria matrix would silently poison the comparisons")
    print(f"pool {n} genes | {n_ctrl} controls | recovery@k with k={k}", flush=True)

    grid = list(_simplex_weights())
    grid.sort(key=lambda w: float(np.var(list(w.values()))))
    W = np.array([[w[c] for c in CRITERIA] for w in grid], dtype=float)
    W = W / np.where(W.sum(1, keepdims=True) > 0, W.sum(1, keepdims=True), 1.0)
    S = X @ W.T
    order = np.argsort(-S, axis=0)
    ranks = np.empty_like(order)
    for j in range(S.shape[1]):
        ranks[order[:, j], j] = np.arange(n)
    topk = ranks < k
    print(f"grid {W.shape[0]} weight vectors", flush=True)

    def select_and_score(lab):
        rec = topk[lab].sum(0)
        j = int(np.argmax(rec))
        return int(rec[j]), -float(ranks[lab, j].mean()), j

    rec_obs, mr_obs, j_obs = select_and_score(cmask)
    print(f"observed: recovery@k={rec_obs}/{n_ctrl}  mean-rank stat={mr_obs:.2f}  "
          f"weights={ {c: round(float(v), 3) for c, v in zip(CRITERIA, W[j_obs])} }", flush=True)

    rng = np.random.default_rng(SEED)
    sel_rec = np.empty(B)
    sel_mr = np.empty(B)
    fix_rec = np.empty(B)
    fix_mr = np.empty(B)
    idx = np.arange(n)
    for b in range(B):
        lab = np.zeros(n, dtype=bool)
        lab[rng.choice(idx, size=n_ctrl, replace=False)] = True
        r, m, _ = select_and_score(lab)
        sel_rec[b], sel_mr[b] = r, m
        fix_rec[b] = topk[lab, j_obs].sum()
        fix_mr[b] = -float(ranks[lab, j_obs].mean())
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{B}", flush=True)

    for arr, nm in ((sel_rec, "sel_rec"), (sel_mr, "sel_mr"), (fix_rec, "fix_rec"), (fix_mr, "fix_mr")):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite null draws in {nm}: comparisons would silently be False")

    def p_of(null, obs):
        return float((1 + int((null >= obs).sum())) / (len(null) + 1))

    res = {
        "recovery_at_k": {
            "observed": rec_obs, "n_controls": n_ctrl, "k": k,
            "p_selection_aware": p_of(sel_rec, rec_obs), "p_naive_fixed_weights": p_of(fix_rec, rec_obs),
            "null_mean_selection_aware": round(float(sel_rec.mean()), 3),
            "null_mean_naive": round(float(fix_rec.mean()), 3),
        },
        "mean_rank_of_controls": {
            "observed": round(mr_obs, 3),
            "p_selection_aware": p_of(sel_mr, mr_obs), "p_naive_fixed_weights": p_of(fix_mr, mr_obs),
            "null_mean_selection_aware": round(float(sel_mr.mean()), 3),
            "null_mean_naive": round(float(fix_mr.mean()), 3),
        },
    }
    rep = {
        "schema": "pdac-circuit.module1-selection-aware-null/1", "data_class": "REAL",
        "sealed_studies_touched": False, "B": B, "seed": SEED,
        "n_pool": n, "n_grid": int(W.shape[0]),
        "selected_weights_observed": {c: round(float(v), 4) for c, v in zip(CRITERIA, W[j_obs])},
        "design": ("the MCDA weight grid search is re-run inside every permutation under that "
                   "replicate's shuffled labels, with the same recovery@k objective and the same "
                   "most-balanced tie-break; the fixed-weight null on identical draws is reported "
                   "alongside to quantify the inflation from selection reuse"),
        "results": res,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n=== selection-aware vs naive null ===")
    for nm, r in res.items():
        print(f"  {nm}: observed {r['observed']}")
        print(f"    naive (weights frozen)  p={r['p_naive_fixed_weights']:.4f}  null mean {r['null_mean_naive']}")
        print(f"    selection-aware         p={r['p_selection_aware']:.4f}  null mean {r['null_mean_selection_aware']}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
