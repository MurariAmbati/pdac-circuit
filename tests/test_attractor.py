import numpy as np
import pytest


def test_pwm_scan_is_bounded_and_detects_planted_site():
    from pdac_circuit.attractor.motif import max_pwm_score

    pwm=np.log2(np.array([[0.90, 0.03, 0.04, 0.03]] * 6) / 0.25)
    perfect="AAAAAA" + "CGTCGT"
    absent="CGCGCGCGCGCG"
    s_hit=max_pwm_score(pwm, perfect)
    s_miss=max_pwm_score(pwm, absent)
    assert 0.0 <= s_miss <= s_hit <= 1.0
    assert s_hit > 0.9


def test_auc_matches_known_ranking():
    from pdac_circuit.attractor.run import _auc

    scores=np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    labels=np.array([False, False, False, False, True])
    assert _auc(scores, labels) == pytest.approx(1.0)
    assert _auc(-scores, labels) == pytest.approx(0.0)


def test_eigencentrality_ranks_hub_highest():
    from pdac_circuit.attractor.run import _eigencentrality

    adj=np.zeros((4, 4))
    adj[0, [1, 2, 3]]=1
    adj[[1, 2, 3], 0]=1
    ec=_eigencentrality(adj)
    assert ec.argmax() == 0


def test_bistable_dynamics_has_stable_high_and_dead_fixed_points():
    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import RegulatoryGraph

    n=6
    adj=np.ones((n, n)) - np.eye(n)
    states=np.full((3, n), 0.9, dtype=float)
    graph=RegulatoryGraph(
        nodes=[f"G{i}" for i in range(n)],
        adjacency=adj.astype(np.float32),
        signs=adj.astype(np.float32),
        motif_support=np.zeros((n, n), dtype=np.float32),
        states=states,
        line_ids=["a", "b", "c"],
        healthy_dir=-np.ones(n),
        disease_log2fc=np.ones(n),
        accessible=np.zeros(n),
        active_enhancer=np.zeros(n),
        cna_amp_freq=np.zeros(n),
        cna_mean=np.zeros(n),
        promoter_methylation=np.full(n, np.nan),
    )
    dyn=AttractorDynamics(graph, device="cpu")
    fit=dyn.fit(epochs=400, motif_weight=0.0)
    assert fit.fixed_point_error < 0.05
    collapse=dyn.collapse_scores()
    assert collapse.shape == (n,)
    assert np.all(collapse >= 0)


@pytest.mark.real_data
@pytest.mark.slow
def test_full_attractor_control_real_data(tmp_path):
    from pdac_circuit.attractor.run import run_attractor_control

    res=run_attractor_control(max_nodes=120, coexpr_threshold=0.4, motif_edges=False,
                                epochs=400, out_dir=tmp_path)
    assert res["graph"]["data_class"] == "REAL"
    assert res["graph"]["n_pdac_lines"] >= 8
    assert (tmp_path / "attractor_validation.json").exists()
    assert res["validation"]["primary"] is not None
