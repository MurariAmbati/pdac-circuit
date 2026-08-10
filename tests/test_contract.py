from __future__ import annotations

from pdac_circuit.core.contract import OutputEnvelope,Verdict
from pdac_circuit.core.governance import guard_emission,scan_for_clinical_directive
from pdac_circuit.core.provenance import GATED,POINTER,REAL,reduce_data_class

def test_envelope_ok_and_dataclass_reduction():
    env = OutputEnvelope.ok({"x": 1},data_classes=[REAL,POINTER,REAL])
    assert env.verdict == Verdict.OK
    assert env.data_class == POINTER
    assert env.to_dict()["payload"] == {"x": 1}

def test_abstain_hides_payload():
    env = OutputEnvelope.abstain("no real data for this step")
    assert env.verdict == Verdict.ABSTAIN
    assert env.to_dict()["payload"] is None
    assert env.caveats == ["no real data for this step"]

def test_reduce_data_class():
    assert reduce_data_class([REAL,REAL]) == REAL
    assert reduce_data_class([REAL,GATED]) == GATED
    assert reduce_data_class([]) == REAL

def test_clinical_directive_is_refused():
    assert scan_for_clinical_directive("administer 50mg to the patient") is not None
    env = OutputEnvelope.ok({"circuit": "..."})
    guarded = guard_emission(env,rendered_text="Clinicians should administer this construct.")
    assert guarded.verdict == Verdict.REFUSE

def test_descriptive_science_is_allowed():
    assert scan_for_clinical_directive("KRAS is a driver oncogene in PDAC") is None
    env = OutputEnvelope.ok({"circuit": "..."})
    guarded = guard_emission(env,rendered_text="KRAS is a driver oncogene; GATA6 marks the classical subtype.")
    assert guarded.verdict == Verdict.OK

def test_gated_payload_is_refused():
    env = OutputEnvelope.ok({"x": 1},data_classes=[GATED])
    guarded = guard_emission(env,rendered_text="result")
    assert guarded.verdict == Verdict.REFUSE
