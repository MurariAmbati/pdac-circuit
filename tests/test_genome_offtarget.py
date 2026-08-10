from __future__ import annotations

import numpy as np
import pytest

from pdac_circuit.grna.genome_offtarget import _COMP,_encode,_scan_block

GUIDE="GGAGATGATGACGATCTGCG"

def _rc(s: str) -> str:
    return "".join("ACGT"[b] for b in _COMP[_encode(s)][::-1])

def _enc_pair(guide):
    return {guide: _encode(guide)},{guide: _COMP[_encode(guide)][::-1].copy()}

def _hits(seq,guide=GUIDE,max_mm=4):
    e,r = _enc_pair(guide)
    return _scan_block(_encode(seq),e,r,max_mm)[guide]

def test_plus_strand_exact_site_found():
    hits=_hits("TTTT" + GUIDE + "AGG" + "TTTT")
    assert len(hits) == 1
    assert int(hits[0][1]) == 0
    assert int(hits[0][2]) == 0

def test_minus_strand_exact_site_found():
    seq="TTTT" + "CCT" + _rc(GUIDE) + "TTTT"
    hits=_hits(seq)
    assert len(hits) == 1
    assert int(hits[0][1]) == 0
    assert int(hits[0][2]) == 1

def test_mismatch_count_is_exact():
    mm=list(GUIDE)
    mm[0]="A" if mm[0] != "A" else "C"
    mm[5]="A" if mm[5] != "A" else "C"
    hits=_hits("TTTT" + "".join(mm) + "TGG" + "TTTT")
    assert len(hits) == 1
    assert int(hits[0][1]) == 2

def test_site_beyond_budget_is_not_reported():
    mm=list(GUIDE)
    for i in range(6):
        mm[i]="A" if mm[i] != "A" else "C"
    assert len(_hits("TTTT" + "".join(mm) + "TGG" + "TTTT",max_mm=4)) == 0

def test_non_ngg_pam_is_not_a_site():
    assert len(_hits("TTTT" + GUIDE + "TTT" + "TTTT")) == 0

def test_both_strands_found_in_one_pass():
    seq=("TTTT" + GUIDE + "AGG" + "TTTTTTTT" + "CCT" + _rc(GUIDE) + "TTTT")
    hits=_hits(seq)
    assert len(hits) == 2
    assert sorted(int(h[2]) for h in hits) == [0,1]
    assert all(int(h[1]) == 0 for h in hits)

def test_block_shorter_than_window_is_empty_not_an_error():
    assert _hits("ACGT").shape == (0,3)

@pytest.mark.parametrize("pam,expected",[("AGG",1),("CGG",1),("TGG",1),("GGG",1),
                                          ("AAG",0),("ACT",0)])
def test_ngg_pam_variants(pam,expected):
    assert len(_hits("TTTT" + GUIDE + pam + "TTTT")) == expected

def test_scanner_is_not_trivially_finding_everything():
    rng=np.random.default_rng(20260620)
    seq="".join("ACGT"[i] for i in rng.integers(0,4,size=200_000))
    hits=_hits(seq,max_mm=4)
    assert len(hits) < 50
