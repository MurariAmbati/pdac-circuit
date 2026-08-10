from __future__ import annotations

import json

import numpy as np

from ..core.paths import RESULTS
from ..stats import (
    benjamini_hochberg,
    classify_cert,
    conformal_interval,
    empirical_coverage,
    permutation_p,
    tost_equivalence,
)

def fdr_control(*, m: int = 200, frac_null: float = 0.8, alpha: float = 0.05, reps: int = 200, seed: int = 0) -> float:
    rng=np.random.default_rng(seed)
    fdrs=[]
    n_alt=int(m * (1 - frac_null))
    for _ in range(reps):
        p=rng.uniform(0, 1, m)
        is_alt=np.zeros(m, dtype=bool)
        if n_alt > 0:
            idx=rng.choice(m, n_alt, replace=False)
            p[idx]=rng.beta(0.3, 8.0, n_alt)
            is_alt[idx]=True
        rej=benjamini_hochberg(p, alpha=alpha)["rejected"]
        n_rej=int(rej.sum())
        false_rej=int((rej & ~is_alt).sum())
        fdrs.append(false_rej / n_rej if n_rej > 0 else 0.0)
    return float(np.mean(fdrs))

def conformal_coverage(*, n_cal: int = 300, n_test: int = 300, level: float = 0.9, reps: int = 100, seed: int = 0) -> float:
    rng=np.random.default_rng(seed)
    covs=[]
    for _ in range(reps):
        cal_res=rng.normal(0, 1, n_cal)
        preds=rng.normal(0, 1, n_test)
        truth=preds + rng.normal(0, 1, n_test)
        ci=conformal_interval(cal_res, preds, level=level)
        covs.append(empirical_coverage(truth, ci["lo"], ci["hi"]))
    return float(np.mean(covs))

def permutation_typeI(*, n: int = 30, alpha: float = 0.05, reps: int = 300, B: int = 400, seed: int = 0) -> float:
    rng=np.random.default_rng(seed)
    values=rng.normal(0, 1, n)

    def stat(labels):
        labels=np.asarray(labels, dtype=float)
        return float(np.corrcoef(values, labels)[0, 1])

    rejections=0
    for r in range(reps):
        labels=rng.normal(0, 1, n)
        p=permutation_p(labels, stat, B=B, seed=r, two_sided=True)["p"]
        rejections += int(p < alpha)
    return rejections / reps

def cert_lattice(*, seed: int = 0) -> dict:
    rng=np.random.default_rng(seed)
    a=rng.normal(0, 1, 400)
    b=rng.normal(0, 1, 400)
    equiv=tost_equivalence(a, b, margin=0.5, alpha=0.05)
    cert_equiv=classify_cert(positive_significant=False, exceeds_margin=False, powered=True, tost_equivalent=equiv["equivalent"])
    c=rng.normal(0, 1, 400)
    d=rng.normal(2.0, 1, 400)
    diff=tost_equivalence(c, d, margin=0.5, alpha=0.05)
    cert_diff=classify_cert(positive_significant=True, exceeds_margin=True, powered=True, tost_equivalent=diff["equivalent"])
    return {
        "equiv_is_certified_negative": cert_equiv == "certified-negative",
        "different_is_not_certified_negative": cert_diff != "certified-negative",
        "cert_equiv": cert_equiv,
        "cert_diff": cert_diff,
    }

def run_all(*, fast: bool = False, write: bool = True) -> dict:
    reps=50 if fast else 200
    report={
        "schema": "pdac-circuit.validation/1",
        "fdr_control_0.05": fdr_control(alpha=0.05, reps=reps),
        "conformal_coverage_0.90": conformal_coverage(level=0.9, reps=reps // 2 or 1),
        "permutation_typeI_0.05": permutation_typeI(alpha=0.05, reps=min(reps, 200)),
        "cert_lattice": cert_lattice(),
    }
    cl=report["cert_lattice"]
    report["ok"]=bool(
        report["fdr_control_0.05"] <= 0.06
        and 0.86 <= report["conformal_coverage_0.90"] <= 0.96
        and report["permutation_typeI_0.05"] <= 0.10
        and cl["equiv_is_certified_negative"]
        and cl["different_is_not_certified_negative"]
    )
    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report

if __name__ == "__main__":
    import sys

    rep=run_all(fast="--fast" in sys.argv)
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["ok"] else 1)
