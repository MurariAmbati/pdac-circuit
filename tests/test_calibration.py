from __future__ import annotations

from pdac_circuit.validation.calibration import run_all


def test_calibration_report_ok():
    rep = run_all(fast=True, write=False)
    assert rep["fdr_control_0.05"] <= 0.06, rep
    assert 0.86 <= rep["conformal_coverage_0.90"] <= 0.96, rep
    assert rep["permutation_typeI_0.05"] <= 0.10, rep
    assert rep["cert_lattice"]["equiv_is_certified_negative"]
    assert rep["cert_lattice"]["different_is_not_certified_negative"]
    assert rep["ok"]
