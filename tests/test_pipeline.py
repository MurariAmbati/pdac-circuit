from __future__ import annotations

import json

import pytest

pytestmark=[pytest.mark.real_data, pytest.mark.slow]

def test_pipeline_end_to_end(tmp_path):
    from pdac_circuit.core.paths import MODELS
    from pdac_circuit.pipeline.orchestrator import run_pipeline

    if not (MODELS / "promoter.pt").exists():
        pytest.skip("models not trained")
    out=tmp_path / "run.json"
    rc=run_pipeline(subtype="classical", top_k=6, out=str(out))
    assert rc == 0
    art=json.loads(out.read_text(encoding="utf-8"))
    assert art["schema"] == "pdac-circuit.run/1"
    assert art["verdict"] in ("OK", "ABSTAIN")
    assert "RESEARCH USE ONLY" in art["ruo_banner"]
    if art["verdict"] == "OK":
        circuits=art["payload"]["ranked_circuits"]
        assert len(circuits) > 0
        c0=circuits[0]
        for key in ("target_tf", "promoter", "enhancer", "repressor_guide", "circuit", "sub_scores"):
            assert key in c0
        assert art["audit"]["head"] is not None
        assert len(art["audit"]["steps"]) >= 2

def test_pipeline_data_class_real(tmp_path):
    from pdac_circuit.core.paths import MODELS
    from pdac_circuit.pipeline.orchestrator import run_pipeline

    if not (MODELS / "promoter.pt").exists():
        pytest.skip("models not trained")
    out=tmp_path / "run_basal.json"
    run_pipeline(subtype="basal", top_k=5, out=str(out))
    art=json.loads(out.read_text(encoding="utf-8"))
    assert art["data_class"] in ("REAL", "POINTER")
