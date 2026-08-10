from __future__ import annotations

import numpy as np
import pytest

from pdac_circuit.circuit import (
    Circuit,
    assess,
    fragile_circuit,
    monostable_circuit,
    negative_feedback,
    negative_feedback_two_node,
    parameter_sweep,
    repressilator,
    robust_circuit,
    steady_state_within_tol,
    toggle_switch,
    viability_perm_null,
)
from pdac_circuit.circuit.ode import hill_activate,hill_repress

SEED = 20260620

def test_circuit_validate_catches_dangling_edge():
    c = Circuit("bad")
    c.add_gene("A")
    c.add_edge("A","B",sign=-1)
    with pytest.raises(ValueError):
        c.validate()
    assert not c.is_valid()

def test_circuit_activators_repressors():
    c = toggle_switch()
    assert c.repressors("A") == ["B"]
    assert c.repressors("B") == ["A"]
    assert c.activators("A") == []
    assert c.edge_sign("A","B") == -1

def test_edge_sign_must_be_signed():
    c = Circuit("s")
    c.add_gene("A")
    c.add_gene("B")
    with pytest.raises(ValueError):
        c.add_edge("A","B",sign=0)

def test_hill_terms_monotone():
    xs = np.linspace(0,5,50)
    a = np.array([hill_activate(x,K=0.5,n=2.0) for x in xs])
    r = np.array([hill_repress(x,K=0.5,n=2.0) for x in xs])
    assert np.all(np.diff(a) >= -1e-12) and a.min() >= 0 and a.max() <= 1
    assert np.all(np.diff(r) <= 1e-12) and r.min() >= 0 and r.max() <= 1
    assert abs(hill_activate(0.5,0.5,2.0) - 0.5) < 1e-12

def test_toggle_boolean_two_fixed_points():
    bm = toggle_switch().to_boolean()
    fps = bm.fixed_points("sync")
    assert len(fps) == 2
    assert set(fps) == {(True,False),(False,True)}

def test_toggle_boolean_basins_sum_to_2n():
    bm = toggle_switch().to_boolean()
    attr = bm.attractors("sync")
    n = len(bm.nodes)
    assert sum(a["basin"] for a in attr) == 2 ** n

def test_boolean_step_sync_deterministic():
    bm = toggle_switch().to_boolean()
    s0 = (True,False)
    assert bm.step_sync(s0) == bm.step_sync(s0)
    assert bm.step_sync((True,False)) == (True,False)

def test_repressilator_boolean_limit_cycle_sync():
    bm = repressilator().to_boolean()
    attr = bm.attractors("sync")
    cycles = [a for a in attr if a["length"] > 1]
    assert cycles,"repressilator (sync) must have a limit cycle"
    assert max(a["length"] for a in attr) > 1

def test_repressilator_boolean_limit_cycle_async():
    bm = repressilator().to_boolean()
    attr = bm.attractors("async")
    assert max(a["length"] for a in attr) > 1
    traj = bm.trajectory((False,True,False),steps=12,mode="async")
    assert len({t for t in traj}) > 1

def test_toggle_ode_bistable():
    ode = toggle_switch().to_ode()
    ss_a = ode.steady_state(x0=np.array([3.0,0.0]))
    ss_b = ode.steady_state(x0=np.array([0.0,3.0]))
    assert np.linalg.norm(ss_a - ss_b) > 0.5
    assert ode.is_stable(ss_a)
    assert ode.is_stable(ss_b)
    assert ss_a[0] > ss_a[1]
    assert ss_b[1] > ss_b[0]

def test_repressilator_ode_oscillates():
    ode = repressilator().to_ode()
    assert steady_state_within_tol(ode,t_span=(0.0,120.0),n_points=2400) is False
    sol = ode.simulate(t_span=(0.0,120.0),x0=np.array([1.0,0.3,0.6]),n_points=2400)
    t,x = sol["t"],sol["x"][0]
    late = x[len(x) // 2:]
    sig = late - late.mean()
    assert np.max(np.abs(sig)) > 0.5
    freqs = np.fft.rfftfreq(len(sig),d=(t[1] - t[0]))
    power = np.abs(np.fft.rfft(sig)) ** 2
    peak = freqs[1:][np.argmax(power[1:])]
    assert peak > 0.0

def test_negative_feedback_monostable_and_stable():
    ode = negative_feedback().to_ode()
    assert steady_state_within_tol(ode) is True
    x_ss = ode.steady_state()
    x_ss2 = ode.steady_state(x0=np.array([5.0]))
    assert np.linalg.norm(x_ss - x_ss2) < 1e-3
    assert ode.is_stable(x_ss)
    assert np.all(ode.eigenvalues(x_ss).real < 0)

def test_negative_feedback_two_node_settles_stable():
    ode = negative_feedback_two_node().to_ode()
    assert steady_state_within_tol(ode) is True
    x_ss = ode.steady_state()
    assert ode.is_stable(x_ss)
    assert np.all(ode.eigenvalues(x_ss).real < 0)

def test_parameter_sweep_in_unit_interval_with_wilson_ci():
    res = parameter_sweep(robust_circuit(),n=64,seed=SEED)
    assert 0.0 <= res["robustness"] <= 1.0
    ci = res["wilson_ci"]
    assert 0.0 <= ci["lo"] <= ci["hi"] <= 1.0
    assert ci["lo"] <= res["robustness"] <= ci["hi"] + 1e-9

def test_fragile_less_robust_than_robust():
    rob = parameter_sweep(robust_circuit(),n=128,seed=SEED)["robustness"]
    fra = parameter_sweep(fragile_circuit(),n=128,seed=SEED)["robustness"]
    assert fra < rob
    from pdac_circuit.circuit.stability import ROBUSTNESS_MIN

    assert rob >= ROBUSTNESS_MIN
    assert fra < ROBUSTNESS_MIN

def test_parameter_sweep_deterministic():
    a = parameter_sweep(robust_circuit(),n=96,seed=SEED)
    b = parameter_sweep(robust_circuit(),n=96,seed=SEED)
    assert abs(a["robustness"] - b["robustness"]) < 1e-6
    assert a["successes"] == b["successes"]
    assert abs(a["wilson_ci"]["lo"] - b["wilson_ci"]["lo"]) < 1e-6

def test_viability_perm_null_deterministic():
    a = viability_perm_null(robust_circuit(),B=40,seed=SEED)
    b = viability_perm_null(robust_circuit(),B=40,seed=SEED)
    assert abs(a["p"] - b["p"]) < 1e-12
    assert 0.0 < a["p"] <= 1.0

def test_assess_robust_is_real():
    rep = assess(robust_circuit(),sweep_n=64,perm_B=40,seed=SEED)
    assert rep["steady_state_ok"] is True
    assert rep["stability_ok"] is True
    assert rep["knife_edge"] is False
    assert rep["robustness"] >= rep["robustness_min"]
    assert rep["cert"] == "real"

def test_assess_oscillator_abstains():
    rep = assess(fragile_circuit(),sweep_n=64,perm_B=40,seed=SEED)
    assert rep["steady_state_ok"] is False
    assert rep["cert"] != "real"

def test_assess_knife_edge_abstains():
    c = Circuit("marginal")
    c.add_gene("A",basal=1.0,degradation=0.0)
    c.validate()
    ode = c.to_ode()
    x_ss = ode.steady_state(t_span=(0.0,5.0))
    assert ode.is_knife_edge(x_ss)
    assert not ode.is_stable(x_ss)
    rep = assess(c,sweep_n=32,perm_B=20,seed=SEED)
    assert rep["knife_edge"] is True
    assert rep["cert"] == "abstain"

def test_assess_monostable_is_real():
    rep = assess(monostable_circuit(),sweep_n=64,perm_B=20,seed=SEED)
    assert rep["cert"] == "real"
    assert rep["stability_ok"] is True
