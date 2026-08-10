from __future__ import annotations

import pytest

from pdac_circuit.grna import cfd_doench

pytestmark=pytest.mark.skipif(
    not cfd_doench.available(),
    reason="Doench-2016 CFD matrix not downloaded (data/raw/doench2016-cfd/)")

SPACER="ATCGATGCTGATGCTAGATA"
CASES=[
    ("ATCGATGCTGATGCTAGATA", "AGG", 1.000),
    ("ATCGATGCTGATGCTAGATA", "AAG", 0.259),
    ("ATCGATGCTGATGCTAGATA", "AGA", 0.069),
    ("ATCGATGCTGATGCTAGATA", "AGT", 0.016),
    ("TTCGATGCTGATGCTAGATA", "AGG", 1.000),
    ("TTCGATGCTGATGCTAGATG", "AGG", 0.765),
    ("TTCGATGCTAATCCTAGATG", "AGG", 0.301),
]

@pytest.mark.parametrize("protospacer,pam,expected", CASES)
def test_published_cas9_vectors(protospacer, pam, expected):
    assert round(cfd_doench.cfd_score(SPACER, protospacer, pam), 3) == expected

def test_matrix_shape():
    mm, pam=cfd_doench.load_matrices()
    keys=[k for k in mm if len(k) >= 3 and k[0] in "ACGT" and k[1] in "ACGT"]
    assert len(keys) == 240, "expected 12 mismatch types x 20 positions"
    assert len(pam) == 16, "expected 16 PAM dinucleotides"
    assert not [k for k in keys if k[0] == k[1]], "a self-pair is not a mismatch"

def test_key_order_is_spacer_then_protospacer():
    mm, _=cfd_doench.load_matrices()
    assert round(mm["AG20"], 4) == 0.7647
    assert round(mm["GA20"], 4) == 0.9375
    spacer="A" * 19 + "A"
    proto="A" * 19 + "G"
    assert round(cfd_doench.cfd_score(spacer, proto, "AGG"), 4) == 0.7647

def test_perfect_match_scores_pam_weight_only():
    assert cfd_doench.cfd_score(SPACER, SPACER, "AGG") == pytest.approx(1.0)
    assert cfd_doench.cfd_score(SPACER, SPACER, "AGT") == pytest.approx(0.016129032)

def test_specificity_aggregate_is_monotone():
    assert cfd_doench.cfd_specificity([]) == 1.0
    assert cfd_doench.cfd_specificity([1.0]) == pytest.approx(0.5)
    assert cfd_doench.cfd_specificity([1.0, 1.0]) < cfd_doench.cfd_specificity([1.0])

def test_validate_reports_reproduction():
    rep=cfd_doench.validate()
    assert rep["published_vectors_reproduced"] is True
    assert rep["n_mismatch_weights"] >= 240
