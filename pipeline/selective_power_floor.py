from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import mannwhitneyu, norm

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "selective_power_floor.json"
N_TOTAL=419
N_FEATURES=17
OBSERVED={"expr_mean_raw": 0.7944, "expr_pdac_minus_other_DIFF": 0.6111,
            "best_centrality(eigenvector)": 0.5628, "coexpr_degree": 0.5271}
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

def power_at(n_pos, auc, n_total=N_TOTAL, n_sim=4000, alpha=0.05, seed=0):
    rng=np.random.default_rng(seed)
    n_neg=n_total - n_pos
    d=np.sqrt(2.0) * norm.ppf(auc)
    hits_nom=0
    hits_bh=0
    alpha_bh=alpha / N_FEATURES
    for _ in range(n_sim):
        pos=rng.standard_normal(n_pos) + d
        neg=rng.standard_normal(n_neg)
        p=mannwhitneyu(pos, neg, alternative="greater").pvalue
        hits_nom += p < alpha
        hits_bh += p < alpha_bh
    return hits_nom / n_sim, hits_bh / n_sim

def min_detectable(n_pos, target_power=0.8, bh=True, lo=0.50, hi=0.95):
    for _ in range(18):
        mid=(lo + hi) / 2
        pn, pb = power_at(n_pos, mid, n_sim=1500, seed=1)
        p=pb if bh else pn
        if p < target_power:
            lo=mid
        else:
            hi=mid
    return (lo + hi) / 2

def positives_needed(auc, target_power=0.8, bh=True, cap=400):
    for n in (10, 14, 20, 30, 40, 60, 80, 120, 160, 220, 300, cap):
        pn, pb = power_at(n, auc, n_sim=1200, seed=2)
        if (pb if bh else pn) >= target_power:
            return n
    return None

def main():
    log("simulating power grid")
    grid=[]
    for n_pos in (14, 20, 30, 50, 100):
        row={"n_positive": n_pos, "cells": []}
        for auc in (0.60, 0.65, 0.70, 0.75, 0.80):
            pn, pb = power_at(n_pos, auc, n_sim=3000, seed=3)
            row["cells"].append({"true_auc": auc, "power_nominal_a05": round(pn, 3),
                                 "power_BH_17tests": round(pb, 3)})
        grid.append(row)
        log(f"  n_pos={n_pos}: " + " ".join(
            f"{c['true_auc']}:{c['power_BH_17tests']:.2f}" for c in row["cells"]))

    log("bisecting minimum detectable effect")
    mde={}
    for n_pos in (14, 20, 30, 50, 100):
        mde[str(n_pos)]={
            "min_detectable_auc_BH80": round(min_detectable(n_pos, bh=True), 3),
            "min_detectable_auc_nominal80": round(min_detectable(n_pos, bh=False), 3)}
        log(f"  n_pos={n_pos}: MDE(BH,80%)={mde[str(n_pos)]['min_detectable_auc_BH80']}, "
            f"MDE(nominal,80%)={mde[str(n_pos)]['min_detectable_auc_nominal80']}")

    log("positives required for the effects actually observed")
    need={}
    for name, auc in OBSERVED.items():
        need[name]={"observed_rank_auc": auc,
                      "positives_needed_BH80": positives_needed(auc, bh=True),
                      "positives_needed_nominal80": positives_needed(auc, bh=False)}
        log(f"  {name} (AUC {auc}): BH80 needs {need[name]['positives_needed_BH80']}, "
            f"nominal80 needs {need[name]['positives_needed_nominal80']}")

    actual_bh=mde["14"]["min_detectable_auc_BH80"]
    rep={
        "schema": "pdac-circuit.selective-power-floor/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "design": {"n_total": N_TOTAL, "n_positive_actual": 14, "n_features_corrected": N_FEATURES,
                   "test": "Mann-Whitney rank comparison (matches the model-free univariate screen)"},
        "power_grid": grid,
        "minimum_detectable_effect": mde,
        "positives_required_for_observed_effects": need,
        "interpretation": {
            "min_detectable_auc_at_actual_design_BH80": actual_bh,
            "reading": (
                f"At the actual design (14 positives, 17 features, BH), only effects of rank-AUC "
                f">= {actual_bh:.2f} were detectable at 80% power. Effects weaker than that could "
                f"not have been seen regardless of whether they exist, so Phase 6b's 'nothing "
                f"survives BH' bounds the effect size rather than proving absence."),
        },
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print("power (BH-corrected across 17 features) by design size and true effect:")
    print(f"{'n_pos':>6} " + " ".join(f"{a:>7}" for a in (0.60, 0.65, 0.70, 0.75, 0.80)))
    for row in grid:
        print(f"{row['n_positive']:>6} " + " ".join(
            f"{c['power_BH_17tests']:>7.2f}" for c in row["cells"]))
    print("\nminimum detectable rank-AUC at 80% power:")
    for k, v in mde.items():
        print(f"  n_pos={k:>4}: BH {v['min_detectable_auc_BH80']:.3f}   "
              f"nominal {v['min_detectable_auc_nominal80']:.3f}")
    print("\npositives required for the effects actually observed:")
    for k, v in need.items():
        print(f"  {k:32} AUC {v['observed_rank_auc']:.3f} -> BH80 needs "
              f"{v['positives_needed_BH80']}, nominal80 needs {v['positives_needed_nominal80']}")
    print(f"\nREADING: {rep['interpretation']['reading']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
