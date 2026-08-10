from __future__ import annotations

import math

from pdac_circuit.core.contract import Verdict
from pdac_circuit.scoring import (
    CircuitScore,
    SubScores,
    crowding_distance,
    cpg_density,
    dominates,
    efficacy,
    fast_nondominated_sort,
    front_membership_probability,
    hla_binding_proxy,
    immunogenicity_risk,
    pareto_rank,
    safety,
    score_circuits,
    select_top,
    specificity,
)

def _sub(e, s, r, sa, intervals=None):
    return SubScores(efficacy=e, specificity=s, robustness=r, safety=sa, intervals=intervals or {})

def _circ(cid, e, s, r, sa, intervals=None):
    return CircuitScore(circuit_id=cid, sub=_sub(e, s, r, sa, intervals))

def test_dominates_equal_vectors_do_not_dominate():
    a = _sub(0.8, 0.8, 0.8, 0.8)
    b = _sub(0.8, 0.8, 0.8, 0.8)
    assert not dominates(a, b)
    assert not dominates(b, a)

def test_dominates_strictly_better_in_one():
    a = _sub(0.9, 0.8, 0.8, 0.8)
    b = _sub(0.8, 0.8, 0.8, 0.8)
    assert dominates(a, b)
    assert not dominates(b, a)

def test_dominates_mixed_neither():
    a = _sub(0.9, 0.6, 0.8, 0.8)
    b = _sub(0.7, 0.9, 0.8, 0.8)
    assert not dominates(a, b)
    assert not dominates(b, a)

def test_fast_nondominated_sort_known_fronts():
    A = _sub(0.90, 0.90, 0.90, 0.90)
    B = _sub(0.80, 1.00, 0.80, 0.80)
    C = _sub(0.70, 0.70, 0.70, 0.70)
    D = _sub(0.60, 0.60, 0.60, 0.60)
    E = _sub(0.75, 0.65, 0.70, 0.70)
    fronts = fast_nondominated_sort([A, B, C, D, E])
    assert fronts == [[0, 1], [2, 4], [3]]

def test_fast_nondominated_sort_empty():
    assert fast_nondominated_sort([]) == []

def test_crowding_distance_boundaries_inf_interior_analytic():
    P = [_sub(0.2, 0.5, 0.5, 0.5), _sub(0.5, 0.5, 0.5, 0.5), _sub(1.0, 0.5, 0.5, 0.5)]
    cd = crowding_distance([0, 1, 2], P)
    assert math.isinf(cd[0])
    assert math.isinf(cd[2])
    assert abs(cd[1] - 1.0) < 1e-12

def test_crowding_distance_two_points_all_inf():
    P = [_sub(0.2, 0.5, 0.5, 0.5), _sub(0.8, 0.5, 0.5, 0.5)]
    cd = crowding_distance([0, 1], P)
    assert math.isinf(cd[0]) and math.isinf(cd[1])

def test_safety_floor_excludes_unsafe_circuit():
    unsafe = _circ("UNSAFE", 0.99, 0.99, 0.99, 0.60)
    safe = _circ("SAFE", 0.80, 0.80, 0.80, 0.80)
    ranked = pareto_rank([unsafe, safe], safety_floor=0.70)
    by_id = {c.circuit_id: c for c in ranked}
    assert by_id["UNSAFE"].acceptable is False
    assert by_id["SAFE"].acceptable is True
    assert by_id["SAFE"].pareto_rank == 0
    assert by_id["UNSAFE"].pareto_rank > 0
    chosen = select_top(ranked, k=5)
    assert "UNSAFE" not in {c.circuit_id for c in chosen}
    assert "SAFE" in {c.circuit_id for c in chosen}

def test_front_membership_dominant_vs_knife_edge_deterministic():
    dom = _circ("DOM", 0.95, 0.95, 0.95, 0.95)
    knife = _circ(
        "KNIFE", 0.80, 0.80, 0.80, 0.80,
        intervals={"efficacy": (0.70, 0.90), "specificity": (0.70, 0.90)},
    )
    rival = _circ(
        "RIVAL", 0.80, 0.80, 0.80, 0.80,
        intervals={"efficacy": (0.70, 0.90), "specificity": (0.70, 0.90)},
    )
    circuits = [dom, knife, rival]
    pareto_rank(circuits, safety_floor=0.70)

    p1 = front_membership_probability(circuits, n_draws=500, seed=20260620, safety_floor=0.70)
    p2 = front_membership_probability(circuits, n_draws=500, seed=20260620, safety_floor=0.70)
    for k in p1:
        assert abs(p1[k] - p2[k]) < 1e-6

    assert p1["DOM"] > 0.99
    assert p1["KNIFE"] < p1["DOM"]
    assert p1["RIVAL"] < p1["DOM"]

def test_all_fail_safety_returns_certified_negative_abstain():
    c1 = _circ("C1", 0.90, 0.90, 0.90, 0.50)
    c2 = _circ("C2", 0.90, 0.90, 0.90, 0.68)
    c3 = _circ("C3", 0.90, 0.90, 0.90, 0.30)
    env = score_circuits([c1, c2, c3])
    assert env.verdict == Verdict.ABSTAIN
    assert env.cert == "certified-negative"
    assert env.payload is None
    assert env.audit["binding_constraint"].startswith("safety")
    miss = env.audit["closest_miss"]
    assert miss["circuit_id"] == "C2"
    assert miss["objective"] == "safety"
    assert abs(miss["gap"] - 0.02) < 1e-9

def test_passing_floors_returns_ok_envelope():
    good = _circ("GOOD", 0.80, 0.80, 0.80, 0.85)
    meh = _circ("MEH", 0.60, 0.70, 0.75, 0.75)
    env = score_circuits([good, meh], top_k=2, membership_draws=200)
    assert env.verdict == Verdict.OK
    assert env.cert == "real"
    assert env.payload["n_passing_floors"] >= 1
    assert "GOOD" in env.payload["front0_ids"]
    assert any("heuristic" in c.lower() for c in env.caveats)

def test_efficacy_geometric_mean_zero_tanks():
    assert efficacy(0.9, 0.9, 0.0) == 0.0
    assert abs(efficacy(0.8, 0.8, 0.8) - 0.8) < 1e-12
    assert abs(efficacy(0.25, 0.5, 1.0) - (0.25 * 0.5 * 1.0) ** (1 / 3)) < 1e-12

def test_safety_noisy_or():
    assert abs(safety(0.0, 0.0, 0.0) - 1.0) < 1e-12
    assert abs(safety(0.9, 0.0, 0.0) - 0.1) < 1e-12
    assert abs(safety(0.5, 0.5, 0.0) - 0.25) < 1e-12

def test_specificity_blend_and_leak_penalty():
    assert abs(specificity(1.0, 1.0) - 1.0) < 1e-12
    assert abs(specificity(0.8, 0.6) - 0.7) < 1e-12
    assert abs(specificity(0.4, 0.4, normal_leakiness=0.5) - 0.0) < 1e-12

def test_cpg_density_and_pamp_flag():
    out = cpg_density("CGCGCG")
    assert abs(out["density"] - 3 / 5) < 1e-12
    assert out["n_cpg"] == 3
    assert cpg_density("AAAGACGTTAAA")["pamp_flag"] is True
    assert cpg_density("AAAAAAAA")["pamp_flag"] is False

def test_hla_binding_proxy_anchor_motif():
    binder = hla_binding_proxy("ALAAAAAAV")
    assert binder["n_windows"] == 1
    assert binder["n_binders"] == 1
    nonbinder = hla_binding_proxy("AGAAAAAAG")
    assert nonbinder["n_binders"] == 0

def test_immunogenicity_risk_has_wide_interval_and_caveat():
    est = immunogenicity_risk(dna_seq="GACGTTCGCG", protein_seq="ALAAAAAAV")
    assert 0.0 <= est.risk <= 1.0
    assert est.lo <= est.risk <= est.hi
    assert (est.hi - est.lo) >= 0.3
    assert est.caveat == "heuristic proxy — not a clinical prediction"
