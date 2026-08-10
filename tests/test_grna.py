from __future__ import annotations

import pytest

from pdac_circuit.grna.offtarget import (
    HSU_WEIGHTS,
    cfd_specificity,
    cfd_style_score,
    mit_single_score,
)

PROTO="GACCCCGAGCGGGACGGCAA"

def test_perfect_match_is_one():
    assert mit_single_score(PROTO,PROTO) == 1.0
    assert abs(cfd_style_score(PROTO,PROTO,"GG") - 1.0) < 1e-12

def test_more_mismatches_lower_score():
    one=PROTO[:5] + ("A" if PROTO[5] != "A" else "C") + PROTO[6:]
    two=one[:10] + ("A" if one[10] != "A" else "C") + one[11:]
    assert mit_single_score(PROTO,one) > mit_single_score(PROTO,two)

def test_pam_proximal_mismatch_more_deleterious():
    distal="A" + PROTO[1:] if PROTO[0] != "A" else "C" + PROTO[1:]
    proximal=PROTO[:19] + ("A" if PROTO[19] != "A" else "C")
    assert cfd_style_score(PROTO,proximal) < cfd_style_score(PROTO,distal)

def test_cfd_hand_computed():
    p=13
    off=PROTO[:p] + ("A" if PROTO[p] != "A" else "C") + PROTO[p + 1 :]
    expected=1.0 - HSU_WEIGHTS[p]
    assert abs(cfd_style_score(PROTO,off,"GG") - expected) < 1e-9

def test_specificity_decreases_with_offtargets():
    assert cfd_specificity([]) == 1.0
    assert cfd_specificity([0.5]) > cfd_specificity([0.5,0.5])
    assert cfd_specificity([0.9,0.9]) < 0.4

def test_nag_pam_weaker_than_ngg():
    off=PROTO[:8] + ("A" if PROTO[8] != "A" else "C") + PROTO[9:]
    assert cfd_style_score(PROTO,off,"AG") < cfd_style_score(PROTO,off,"GG")

@pytest.mark.real_data
def test_pam_scan_real_genome():
    from pdac_circuit.grna.scan import enumerate_protospacers

    guides=enumerate_protospacers("chr7",5_527_100,5_527_400)
    assert len(guides) > 0
    for g in guides[:5]:
        assert len(g.protospacer) == 20 and g.pam.endswith("GG") and len(g.context) == 30
