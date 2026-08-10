from __future__ import annotations

import numpy as np

from pdac_circuit.stats import (
    auroc,
    benjamini_hochberg,
    classify_cert,
    conformal_quantile,
    ndcg_at_k,
    permutation_p,
    program_cert,
    spearman,
    tost_equivalence,
)

def test_spearman_monotone_is_one():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12

def test_auroc_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert abs(auroc(y, s) - 1.0) < 1e-12
    assert abs(auroc(y, [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-12

def test_ndcg_top_ranked_is_one():
    assert abs(ndcg_at_k([3, 2, 1, 0], [3, 2, 1, 0], k=4) - 1.0) < 1e-12

def test_permutation_exact_small_n():
    vals = np.array([1.0, 2.0, 3.0, 4.0])

    def stat(labels):
        return float(np.corrcoef(vals, np.asarray(labels, dtype=float))[0, 1])

    res = permutation_p(vals.copy(), stat, exact=True)
    assert res["mode"] == "exact"
    assert abs(res["p"] - 1.0 / 24) < 1e-12

def test_bh_all_null_rejects_none():
    p = np.linspace(0.05, 0.95, 20)
    res = benjamini_hochberg(p, alpha=0.05)
    assert res["n_rejected"] == 0

def test_bh_strong_signal_rejects():
    p = np.array([1e-6, 1e-5, 1e-4] + [0.5] * 17)
    res = benjamini_hochberg(p, alpha=0.05)
    assert res["n_rejected"] >= 3

def test_conformal_quantile_coverage_property():
    s = np.arange(1, 101, dtype=float)
    q = conformal_quantile(s, level=0.9)
    assert q == 91.0

def test_cert_lattice_decisions():
    assert classify_cert(positive_significant=True, exceeds_margin=True, powered=True) == "real"
    assert classify_cert(tost_equivalent=True, powered=True) == "certified-negative"
    assert classify_cert(powered=False) == "underpowered"
    assert classify_cert(gate_ok=False) == "abstain"
    assert program_cert(["real", "abstain", "real"]) == "abstain"
    assert program_cert(["real", "certified-negative"]) == "certified-negative"

def test_tost_equivalent_when_same():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    assert tost_equivalence(a, b, margin=0.5)["equivalent"]
