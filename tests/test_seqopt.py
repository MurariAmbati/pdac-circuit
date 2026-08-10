from __future__ import annotations

import itertools

from Bio.Seq import Seq

from pdac_circuit.core.paths import REGISTRY_JSON
from pdac_circuit.seqopt import (
    CircuitSeq,
    Feature,
    OptConfig,
    cai,
    mfe_window,
    minimize_5p_structure,
    normalize_gc,
    nussinov_mfe,
    optimize_circuit,
    remove_cryptic_sites,
    remove_sites,
    scan_cryptic_sites,
    scan_sites,
    windowed_gc,
    zuker_mfe,
)
from pdac_circuit.seqopt.codon import HUMAN_WEIGHTS, optimize_codons
from pdac_circuit.seqopt.types import CODON_TABLE, SYNONYMOUS

EGFP=(
    "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGC"
    "CACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACC"
    "ACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTAC"
    "CCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTC"
    "TTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAG"
    "CTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAAC"
    "GTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGC"
    "AGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCAC"
    "TACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTG"
    "ACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
)
EGFP_PROTEIN=str(Seq(EGFP).translate())

def _registry_module_iv() -> dict:
    import json

    with open(REGISTRY_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)["prereg"]["module_IV"]

def _cds(seq: str) -> CircuitSeq:
    return CircuitSeq(seq, [Feature("egfp", 0, len(seq), 1, "CDS", 0)])

def _translate(cs: CircuitSeq) -> str:
    return cs.translate_cds(cs.cds_features()[0])

def test_egfp_reference_is_valid():
    assert len(EGFP) % 3 == 0
    assert EGFP_PROTEIN.startswith("M")
    assert EGFP_PROTEIN.endswith("*")
    assert EGFP_PROTEIN[:-1].count("*") == 0

def test_nussinov_palindrome_stem_pairs():
    assert nussinov_mfe("GGGGAAAACCCC") >= 3

def test_nussinov_structure_free_polyA_zero_pairs():
    assert nussinov_mfe("AAAAAAAAAAAAAAAA") == 0

def test_zuker_stem_is_stable_polyA_is_not():
    assert zuker_mfe("GGGGGAAAACCCCC") < 0.0
    assert zuker_mfe("AAAAAAAAAAAAAAAA") == 0.0

def _plant_ecori_synonymous() -> tuple[str, int]:
    aa_idx=EGFP_PROTEIN.find("EF")
    assert aa_idx >= 0, "EGFP should contain a Glu-Phe pair"
    i=aa_idx * 3
    planted=EGFP[:i] + "GAATTC" + EGFP[i + 6:]
    assert str(Seq(planted).translate()) == EGFP_PROTEIN
    return planted, i

def test_remove_ecori_site_protein_unchanged():
    planted, _=_plant_ecori_synonymous()
    assert "EcoRI" in scan_sites(planted)
    cs=_cds(planted)
    removed=remove_sites(cs, ["EcoRI"])
    assert "EcoRI" in removed
    assert "EcoRI" not in scan_sites(cs.seq)
    assert _translate(cs) == EGFP_PROTEIN
    assert scan_sites(cs.seq) == {}

def test_optimize_circuit_removes_ecori_no_new_forbidden_sites():
    planted, _=_plant_ecori_synonymous()
    cs=_cds(planted)
    cfg=OptConfig.from_registry()
    out, report=optimize_circuit(cs, cfg)
    assert "EcoRI" in report.removed_restriction_sites
    assert scan_sites(out.seq) == {}
    assert _translate(out) == EGFP_PROTEIN
    assert report.protein_preserved

def _worst_codon(aa: str) -> str:
    return min(SYNONYMOUS[aa], key=lambda c: HUMAN_WEIGHTS[c])

def _rare_recode(seq: str) -> str:
    out=[]
    for i in range(0, len(seq), 3):
        codon=seq[i:i + 3]
        aa=CODON_TABLE[codon]
        out.append(codon if aa == "*" else _worst_codon(aa))
    return "".join(out)

def test_codon_optimization_increases_cai_above_floor():
    floor=_registry_module_iv()["cai_floor"]
    rare=_rare_recode(EGFP)
    assert str(Seq(rare).translate()) == EGFP_PROTEIN
    cai_before=cai(rare)
    assert cai_before < floor
    opt=optimize_codons(rare, HUMAN_WEIGHTS, frame=0)
    cai_after=cai(opt)
    assert cai_after > cai_before
    assert cai_after >= floor
    assert str(Seq(opt).translate()) == EGFP_PROTEIN

def test_optimize_circuit_lifts_rare_codons_above_floor():
    floor=_registry_module_iv()["cai_floor"]
    rare=_rare_recode(EGFP)
    cs=_cds(rare)
    out, report=optimize_circuit(cs, OptConfig.from_registry())
    assert report.cai_after["egfp"] > report.cai_before["egfp"]
    assert report.cai_after["egfp"] >= floor
    assert _translate(out) == EGFP_PROTEIN

def test_gc_normalization_brings_windows_into_band_or_reports():
    band=tuple(_registry_module_iv()["gc_band"])
    wg=windowed_gc(EGFP)
    assert any(g > band[1] for _, g in wg), "EGFP should have GC-rich windows out of band"

    cs=_cds(EGFP)
    out, report=optimize_circuit(cs, OptConfig.from_registry())
    assert _translate(out) == EGFP_PROTEIN
    if report.gc_windows_in_band:
        for _, g in windowed_gc(out.seq):
            assert band[0] <= g <= band[1]
        assert not any("gc_out_of_band" in u for u in report.unsatisfied)
    else:
        assert any("gc_out_of_band" in u for u in report.unsatisfied)

def test_normalize_gc_helper_preserves_protein():
    band=tuple(_registry_module_iv()["gc_band"])
    cs=_cds(EGFP)
    n_before=sum(1 for _, g in windowed_gc(cs.seq) if not band[0] <= g <= band[1])
    normalize_gc(cs, band, accept=lambda ci, feat, cand: True)
    n_after=sum(1 for _, g in windowed_gc(cs.seq) if not band[0] <= g <= band[1])
    assert n_after < n_before
    assert _translate(cs) == EGFP_PROTEIN

def test_minimize_5p_structure_raises_mfe_protein_unchanged():
    target=_registry_module_iv()["structure_target_mfe"]
    hairpin_utr="GGGGCGCGCGGCCGCAAGCGGCCGCGCGCCCC"
    construct=hairpin_utr + EGFP
    start=len(hairpin_utr)
    cs=CircuitSeq(
        construct,
        [
            Feature("utr5", 0, start, 1, "UTR5", None),
            Feature("egfp", start, len(construct), 1, "CDS", 0),
        ],
    )
    mfe0=mfe_window(cs.seq, start)
    assert mfe0 < target

    final=minimize_5p_structure(cs, start, target_mfe=target,
                                  accept=lambda ci, feat, cand: True)
    assert final > mfe0
    assert _translate(cs) == EGFP_PROTEIN

def _plant_cryptic_donor() -> tuple[str, int]:
    native={(h.kind, h.pos) for h in scan_cryptic_sites(EGFP)}
    best=None
    for w in range(0, len(EGFP) - 9, 3):
        codons=[EGFP[w + 3 * k:w + 3 * k + 3] for k in range(3)]
        aas=[CODON_TABLE[c] for c in codons]
        if "*" in aas:
            continue
        for combo in itertools.product(*[SYNONYMOUS[a] for a in aas]):
            trial=EGFP[:w] + "".join(combo) + EGFP[w + 9:]
            donors=[
                h for h in scan_cryptic_sites(trial)
                if h.kind == "donor" and (h.kind, h.pos) not in native
            ]
            if donors:
                top=max(donors, key=lambda h: h.score)
                if best is None or top.score > best[0]:
                    best=(top.score, w, trial, top.pos)
        if best and best[0] > 11.0:
            break
    assert best is not None, "should be able to plant a synonymous cryptic donor"
    return best[2], best[3]

def test_cryptic_splice_donor_detected_and_abolished():
    planted, donor_pos=_plant_cryptic_donor()
    assert str(Seq(planted).translate()) == EGFP_PROTEIN
    cs=_cds(planted)
    assert any(h.pos == donor_pos and h.kind == "donor" for h in scan_cryptic_sites(cs.seq))
    remove_cryptic_sites(cs, accept=lambda ci, feat, cand: True)
    assert not any(h.pos == donor_pos and h.kind == "donor" for h in scan_cryptic_sites(cs.seq))
    assert _translate(cs) == EGFP_PROTEIN

def test_optimize_circuit_abolishes_all_cryptic_sites():
    planted, _=_plant_cryptic_donor()
    cs=_cds(planted)
    out, report=optimize_circuit(cs, OptConfig.from_registry())
    assert report.splice_sites_after == 0
    assert not scan_cryptic_sites(out.seq)
    assert _translate(out) == EGFP_PROTEIN

def test_optimize_circuit_is_deterministic():
    planted, _=_plant_ecori_synonymous()
    out_a, _=optimize_circuit(_cds(planted), OptConfig.from_registry(seed=20260620))
    out_b, _=optimize_circuit(_cds(planted), OptConfig.from_registry(seed=20260620))
    assert out_a.seq == out_b.seq

def test_optimize_circuit_different_seed_still_valid():
    planted, _=_plant_ecori_synonymous()
    out, report=optimize_circuit(_cds(planted), OptConfig.from_registry(seed=12345))
    assert _translate(out) == EGFP_PROTEIN
    assert scan_sites(out.seq) == {}
    assert report.splice_sites_after == 0
