from __future__ import annotations

import numpy as np
import pytest

from pdac_circuit.signal.chromatin import (
    activity_unit,
    chromatin_features,
    classify_state,
    summit_features,
)

def test_classify_active_promoter():
    assert classify_state({"H3K4me3": 5.0, "H3K27ac": 3.0}) == "active_promoter"

def test_classify_active_enhancer():
    assert classify_state({"H3K4me1": 5.0, "H3K27ac": 4.0, "H3K4me3": 0.2}) == "active_enhancer"

def test_classify_poised_enhancer():
    assert classify_state({"H3K4me1": 5.0, "H3K27ac": 0.2}) == "poised_enhancer"

def test_classify_bivalent():
    assert classify_state({"H3K4me3": 4.0, "H3K27me3": 4.0}) == "bivalent_poised"

def test_classify_repressed_and_heterochromatin():
    assert classify_state({"H3K27me3": 5.0}) == "polycomb_repressed"
    assert classify_state({"H3K9me3": 5.0}) == "heterochromatin"

def test_classify_insulator_and_quiescent():
    assert classify_state({"CTCF": 5.0}) == "insulator"
    assert classify_state({"H3K27ac": 0.1}) == "quiescent"

def test_activity_score_sign():
    active=chromatin_features({"H3K27ac": 5.0, "H3K4me1": 4.0, "H3K27me3": 0.1})
    repressed=chromatin_features({"H3K27ac": 0.1, "H3K27me3": 5.0, "H3K9me3": 4.0})
    assert active["activity_score"] > 0.5
    assert repressed["activity_score"] < -0.3
    assert 0.0 <= activity_unit(active["activity_score"]) <= 1.0

def test_summit_features_on_peak():
    cov=np.concatenate([np.zeros(100), np.linspace(0, 50, 50), np.linspace(50, 0, 50), np.zeros(100)])
    s=summit_features(cov)
    assert s["summit"] == 50.0
    assert 140 <= s["summit_pos"] <= 160
    assert s["fwhm"] > 0

def test_bivalency_detected():
    biv=chromatin_features({"H3K4me3": 4.0, "H3K27me3": 4.0})
    assert biv["bivalency"] == 4.0

@pytest.mark.real_data
@pytest.mark.slow
def test_real_bam_coverage_enriched_at_active_gene():
    from pdac_circuit.core.paths import RAW
    from pdac_circuit.data.genes import promoter_window
    from pdac_circuit.signal.bamio import extract_coverage
    from pdac_circuit.signal.metadata import build_metadata

    meta=build_metadata()
    h3k27=[a for a, v in meta.items() if v.get("target") == "H3K27ac"]
    if not h3k27 or not (RAW / "encode-bulk" / f"{h3k27[0]}.bam").exists():
        pytest.skip("H3K27ac BAM not present")
    g=promoter_window("GATA6", up=2500, down=2500)
    ctrl={"chrom": "chr2", "start": 33_000_000, "end": 33_005_000}
    cov=extract_coverage(str(RAW / "encode-bulk" / f"{h3k27[0]}.bam"), [g, ctrl], frag_extend=150)
    assert cov[0].mean() > 3 * (cov[1].mean() + 0.1)
