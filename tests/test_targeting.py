from __future__ import annotations

import pytest

pytestmark=pytest.mark.real_data

@pytest.fixture(scope="module")
def result():
    from pdac_circuit.targeting import build_afm, prioritize_targets

    afm=build_afm()
    out, env=prioritize_targets(top_k=10, afm=afm)
    return afm, out, env

def test_afm_real_and_controls_present(result):
    afm, _, _=result
    assert afm.table.shape[0] > 1000
    assert int(afm.table["is_control"].sum()) >= 5

def test_known_drivers_enriched(result):
    _, out, env=result
    n=env.payload["numbers"]
    assert n["perm_p"] < 0.05, n
    assert n["recovery_top_quartile"] >= 4

def test_master_tfs_top_ranked(result):
    _, out, _=result
    order=list(out.table.index)
    for g in ("GATA6", "KLF5"):
        assert order.index(g) < 20, f"{g} should be top-ranked"

def test_cert_real_and_ok(result):
    _, _, env=result
    assert env.verdict.value == "OK"
    assert env.cert == "real"

def test_criteria_normalized(result):
    from pdac_circuit.targeting import criteria_matrix

    _, out, _=result
    C=criteria_matrix(out.table)
    for col in C.columns:
        v=C[col].to_numpy()
        assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9

def test_subtype_assignment(result):
    _, out, _=result
    assert set(out.table["subtype"].unique()) <= {"basal", "classical"}
    assert (out.table["subtype"] == "classical").sum() > 0
    assert (out.table["subtype"] == "basal").sum() > 0
